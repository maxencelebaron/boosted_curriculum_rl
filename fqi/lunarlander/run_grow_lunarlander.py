"""Train a growable DQN on Gymnasium LunarLander-v3."""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
import importlib
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import tyro
import yaml

from fqi.lunarlander.dqn import LunarLanderDQN
from fqi.lunarlander.dqn_monitoring import (
    DQNMetricsMonitor,
    replay_batch_to_tensors,
)
from fqi.lunarlander.env import LunarLander
from fqi.lunarlander.run_dqn import (
    TrainingMetrics,
    _split_budget,
    save_training_metrics,
)

from mushroom_rl.approximators.parametric import TorchApproximator
from mushroom_rl.core import Core
from mushroom_rl.policy import EpsGreedy
from mushroom_rl.utils.dataset import compute_J
from mushroom_rl.utils.parameters import LinearParameter, Parameter

from fqi.utils.growing_network import pre_growth_optimize


torch.set_num_threads(1)


GROWTH_MODULES = {
    "random": "fqi.2_network_fqi_grow_randomly",
    "random-0": "fqi.2_network_fqi_grow_randomly",
    "svd": "fqi.3_network_fqi_optimizer_plus_svd",
    "als": "fqi.3_network_fqi_optimizer_plus_svd",
    "stagewise-als": "fqi.3_network_fqi_optimizer_plus_svd",
    "gromo_one_layer": "fqi.4_network_fqi_tiny_one_layer",
}


@dataclass
class Args:
    use_curriculum: bool = False
    """Train successively on increasing wind powers."""
    seed: int = 95
    """Random seed of the experiment."""
    n_timesteps: int = 1_500_000
    """Total number of environment steps."""
    n_eval_points: int = 200
    """Number of evaluation checkpoints."""
    n_test_episodes: int = 10
    """Number of episodes at each evaluation checkpoint."""
    train_freq: int = 4
    """Environment steps between DQN updates."""
    gradient_steps: int = 1
    """Adam updates per DQN training call."""
    exploration_fraction: float = 0.3
    """Fraction of training over which epsilon decreases linearly."""
    exploration_final_eps: float = 0.01
    """Minimum training epsilon."""
    learning_rate: float = 1e-3
    """Adam learning rate, also used as the K-FAC step size."""
    batch_size: int = 128
    """Replay-buffer batch size."""
    buffer_size: int = 100_000
    """Maximum replay-buffer size."""
    learning_starts: int = 1_000
    """Transitions collected before DQN updates begin."""
    target_update_interval: int = 1_000
    """DQN training calls between target-network synchronizations."""

    gravity: float = -10.0
    enable_wind: bool = True
    wind_powers: tuple[float, ...] = (0.0, 5.0, 10.0, 15.0)
    """Curriculum tasks; the last value defines the target task."""
    turbulence_power: float = 1.5
    use_cuda: bool = True

    growth_mode: str = "als"
    """Growth strategy: random, random-0, als, stagewise-als, or gromo_one_layer."""
    initial_hidden: int = 32
    """Initial width of the growable layer."""
    final_hidden: int = 64
    """Requested width after the final growth event."""
    n_growth_events: int = 8
    """Number of growth events when curriculum is disabled."""
    growth_start_step: int = 0
    """First growth step without curriculum; zero selects it automatically."""
    growth_end_step: int = 0
    """Last growth step without curriculum; zero selects it automatically."""
    pre_growth_steps: int = 10
    """Optimization steps on the fixed growth batch before growing."""
    grow_batch_size: int = 256
    """Replay transitions used by a growth event."""

    use_natural_gradient: bool = False
    """Use K-FAC instead of Adam during pre-growth optimization."""
    natural_gradient_damping: float = 1e-5
    natural_gradient_noise_variance: float = 1.0
    natural_gradient_eigenvalue_threshold: float = 1e-7
    kfac_retry_damping_multiplier: float = 10.0
    """Damping multiplier for the single non-finite K-FAC retry."""
    kfac_retry_step_size_multiplier: float = 0.1
    """Step-size multiplier for the single non-finite K-FAC retry."""

    numerical_threshold: float = 1e-6
    """Threshold to consider an eigenvalue as zero in the SVD"""
    statistical_threshold: float = 0.0
    """Threshold to decide how many singular values (number of neurons) to keep"""
    metric_monitoring_batch_size: int = 128
    """Fixed replay sample used to monitor representation metrics."""
    n_plasticity_measurements: int = 20
    """Number of evenly spaced plasticity measurements."""
    plasticity_n_steps: int = 100
    """number of gradient steps per probe task in plasticity measurement"""
    plasticity_n_tasks: int = 10
    """number of random probe tasks for plasticity measurement"""
    plasticity_n_samples: int = 512
    """number of replay buffer samples for plasticity measurement"""
    bellman_residual_threshold: float = 0.0
    """Skip growth when the pre-growth loss is below this value."""

    output_dir: str = "logs/dqn_lunarlander_grow"
    """Directory in which metrics are saved."""


def _task_wind_powers(args: Args) -> list[float]:
    if not args.wind_powers:
        raise ValueError("wind_powers must contain at least one value")
    if args.use_curriculum:
        return list(args.wind_powers)
    return [args.wind_powers[-1]]


def _growth_schedule(args: Args, task_steps: list[int]) -> list[int]:
    if args.use_curriculum:
        steps = []
        task_start = 0
        for task_index, n_steps_task in enumerate(task_steps):
            if task_index > 0:
                steps.append(task_start)
            steps.append(task_start + n_steps_task // 2)
            task_start += n_steps_task
        return steps

    if args.n_growth_events == 0:
        return []
    start = args.growth_start_step or (
        args.n_timesteps // (args.n_growth_events + 1)
    )
    end = args.growth_end_step or (
        args.n_timesteps * args.n_growth_events
        // (args.n_growth_events + 1)
    )
    return sorted(set(
        np.rint(np.linspace(start, end, args.n_growth_events))
        .astype(int)
        .tolist()
    ))


def _validate_args(args: Args, task_steps: list[int]):
    if args.growth_mode not in GROWTH_MODULES:
        raise ValueError(
            f"Unknown growth_mode {args.growth_mode!r}; "
            f"choose from {tuple(GROWTH_MODULES)}"
        )
    if args.n_timesteps < args.n_eval_points or args.n_eval_points < 1:
        raise ValueError("n_timesteps must be at least n_eval_points >= 1")
    if args.n_eval_points < len(task_steps):
        raise ValueError("n_eval_points must allow one evaluation per task")
    if min(task_steps) <= args.learning_starts:
        raise ValueError("learning_starts must be smaller than each task")
    if not 0.0 <= args.exploration_fraction <= 1.0:
        raise ValueError("exploration_fraction must be in [0, 1]")
    if args.n_growth_events < 0:
        raise ValueError("n_growth_events must be non-negative")
    schedule = _growth_schedule(args, task_steps)
    if schedule and args.final_hidden <= args.initial_hidden:
        raise ValueError("final_hidden must be greater than initial_hidden")
    if args.pre_growth_steps < 0:
        raise ValueError("pre_growth_steps must be non-negative")
    if args.grow_batch_size < 1:
        raise ValueError("grow_batch_size must be positive")
    if not 0.0 < args.exploration_final_eps <= 1.0:
        raise ValueError("exploration_final_eps must be in (0, 1]")
    if args.kfac_retry_damping_multiplier <= 1.0:
        raise ValueError("kfac_retry_damping_multiplier must be greater than 1")
    if not 0.0 < args.kfac_retry_step_size_multiplier < 1.0:
        raise ValueError(
            "kfac_retry_step_size_multiplier must be between 0 and 1"
        )

    expected_events = (
        2 * len(task_steps) - 1
        if args.use_curriculum
        else args.n_growth_events
    )
    if len(schedule) != expected_events:
        raise ValueError("growth steps must be distinct")
    if schedule and schedule[-1] > args.n_timesteps:
        raise ValueError("growth steps must remain within training")
    if schedule:
        if args.use_curriculum and any(
            n_steps // 2 <= args.learning_starts
            for n_steps in task_steps
        ):
            raise ValueError(
                "each task midpoint must occur after learning_starts"
            )
        if not args.use_curriculum and schedule[0] <= args.learning_starts:
            raise ValueError("first growth must occur after learning_starts")

    target_widths = np.rint(np.linspace(
        args.initial_hidden,
        args.final_hidden,
        len(schedule) + 1,
    )).astype(int)
    if schedule and np.any(np.diff(target_widths) <= 0):
        raise ValueError(
            "initial_hidden and final_hidden are too close for the "
            "number of growth events"
        )


def _reset_adam(torch_approximator, learning_rate):
    torch_approximator._optimizer = optim.Adam(
        torch_approximator.network.parameters(), lr=learning_rate
    )


def _save_growth_results(log_dir, seed, controller: GrowthController):
    controller.save_events()
    controller.metrics_monitor.save(log_dir, seed)


class GrowthController:
    """Grow from a batch sampled from DQN replay."""

    def __init__(self, args: Args, task_steps: list[int], log_dir: Path):
        self.args = args
        self.events_path = log_dir / f"growth-events-{args.seed}.yaml"
        self.module = importlib.import_module(GROWTH_MODULES[args.growth_mode])
        self.steps = _growth_schedule(args, task_steps)
        self.target_widths = np.rint(np.linspace(
            args.initial_hidden,
            args.final_hidden,
            len(self.steps) + 1,
        )).astype(int).tolist()[1:]
        self.next_event = 0
        self.current_task_index = 0
        self.training_environment_steps = 0
        self.events = []
        self.metrics_monitor = DQNMetricsMonitor(
            n_eval_points=args.n_eval_points,
            monitoring_batch_size=args.metric_monitoring_batch_size,
            n_plasticity_measurements=args.n_plasticity_measurements,
            plasticity_n_samples=args.plasticity_n_samples,
            plasticity_n_steps=args.plasticity_n_steps,
            plasticity_n_tasks=args.plasticity_n_tasks,
            learning_rate=args.learning_rate,
        )

    def save_events(self):
        """Atomically persist completed growth decisions."""
        temporary_path = self.events_path.with_suffix(".yaml.tmp")
        with temporary_path.open("w", encoding="utf-8") as stream:
            yaml.safe_dump(self.events, stream, sort_keys=False)
        temporary_path.replace(self.events_path)

    def _record_event(self, event):
        self.events.append(event)
        self.save_events()

    def record_environment_step(self):
        self.training_environment_steps += 1

    def start_task(self, task_index):
        self.current_task_index = int(task_index)
        self.metrics_monitor.start_task()

    def apply_scheduled_growth(self, agent):
        while (
            self.next_event < len(self.steps)
            and self.training_environment_steps >= self.steps[self.next_event]
        ):
            self._grow(
                agent,
                self.steps[self.next_event],
                self.target_widths[self.next_event]
            )
            self.next_event += 1

    @staticmethod
    def _restore_pre_growth_state(network, optimizer, network_state,
                                  optimizer_state):
        network.load_state_dict(network_state)
        optimizer.load_state_dict(optimizer_state)
        network.zero_grad(set_to_none=True)

    def _pre_growth_with_retry(self, online, network, states, actions,
                               td_targets):
        network_state = copy.deepcopy(network.state_dict())
        optimizer_state = copy.deepcopy(online._optimizer.state_dict())

        def optimize(damping):
            return pre_growth_optimize(
                network=network,
                states=states,
                actions=actions,
                td_targets=td_targets,
                optimizer=online._optimizer,
                n_steps=self.args.pre_growth_steps,
                use_natural_gradient=self.args.use_natural_gradient,
                natural_gradient_damping=damping,
                natural_gradient_noise_variance=(
                    self.args.natural_gradient_noise_variance
                ),
                natural_gradient_eigenvalue_threshold=(
                    self.args.natural_gradient_eigenvalue_threshold
                ),
            )

        try:
            _, final_loss, losses = optimize(
                self.args.natural_gradient_damping
            )
            return final_loss, losses, False, []
        except FloatingPointError as first_error:
            self._restore_pre_growth_state(
                network, online._optimizer, network_state, optimizer_state
            )
            original_step_size = online._optimizer.param_groups[0]["lr"]
            retry_damping = max(
                self.args.natural_gradient_damping
                * self.args.kfac_retry_damping_multiplier,
                self.args.natural_gradient_eigenvalue_threshold * 10,
            )
            retry_step_size = (
                original_step_size
                * self.args.kfac_retry_step_size_multiplier
            )
            online._optimizer.param_groups[0]["lr"] = retry_step_size
            print(
                "WARNING: non-finite K-FAC pre-growth update; retrying "
                f"event {self.next_event} with damping={retry_damping:g} "
                f"and step_size={retry_step_size:g}. Error: {first_error}"
            )
            try:
                _, final_loss, losses = optimize(retry_damping)
                online._optimizer.param_groups[0]["lr"] = original_step_size
                return final_loss, losses, True, [str(first_error)]
            except FloatingPointError as retry_error:
                self._restore_pre_growth_state(
                    network, online._optimizer, network_state,
                    optimizer_state,
                )
                return None, [], True, [
                    str(first_error), str(retry_error)
                ]

    def _grow(self, agent, scheduled_step, target_width):
        online = agent.approximator.model
        network = online.network
        hidden_before = network.encoder_size
        neurons_to_add = max(0, target_width - hidden_before)
        states, actions, td_targets = replay_batch_to_tensors(
            agent, agent._replay_memory.get(self.args.grow_batch_size)
        )
        pre_growth_losses = []
        pre_growth_retry_used = False

        if self.args.pre_growth_steps:
            final_loss, pre_growth_losses, pre_growth_retry_used, errors = (
                self._pre_growth_with_retry(
                    online, network, states, actions, td_targets
                )
            )
            if final_loss is None:
                _reset_adam(online, self.args.learning_rate)
                self._record_event({
                    "task_index": self.current_task_index,
                    "scheduled_step": int(scheduled_step),
                    "actual_step": int(self.training_environment_steps),
                    "hidden_before": int(hidden_before),
                    "hidden_after": int(hidden_before),
                    "neurons_added": 0,
                    "skipped": True,
                    "skip_reason": "nonfinite_kfac_after_retry",
                    "pre_growth_retry_used": True,
                    "pre_growth_errors": errors,
                    "pre_growth_losses": [],
                })
                print(
                    "WARNING: growth event "
                    f"{self.next_event} at step {scheduled_step} skipped "
                    "after two non-finite K-FAC attempts"
                )
                return
            if final_loss < self.args.bellman_residual_threshold:
                _reset_adam(online, self.args.learning_rate)
                self._record_event({
                    "task_index": self.current_task_index,
                    "scheduled_step": int(scheduled_step),
                    "actual_step": int(self.training_environment_steps),
                    "hidden_before": int(hidden_before),
                    "hidden_after": int(hidden_before),
                    "neurons_added": 0,
                    "skipped": True,
                    "skip_reason": "bellman_residual_below_threshold",
                    "pre_growth_retry_used": pre_growth_retry_used,
                    "pre_growth_losses": [float(x) for x in pre_growth_losses],
                })
                return

        singular_values = []
        if self.args.growth_mode in ("random", "random-0"):
            new_network = self.module.grow_network(
                network,
                hidden_before + neurons_to_add,
                zero_fan_out=(self.args.growth_mode == "random-0"),
            )
            online.network = new_network
        elif self.args.growth_mode in ("svd", "als", "stagewise-als"):
            als_method = (
                "stagewise"
                if self.args.growth_mode == "stagewise-als"
                else "als"
            )
            new_network, singular_values = self.module.grow_network_als(
                network,
                states,
                actions,
                td_targets,
                d_a=neurons_to_add,
                numerical_threshold=self.args.numerical_threshold,
                method=als_method,
            )
            online.network = new_network
        elif self.args.growth_mode == "gromo_one_layer":
            self.module.grow_network_gromo(
                network,
                states,
                actions,
                td_targets,
                maximum_added_neurons=neurons_to_add,
                numerical_threshold=self.args.numerical_threshold,
                statistical_threshold=self.args.statistical_threshold,
            )

        _reset_adam(online, self.args.learning_rate)

        # The old target has a different width. Rebuilding it also performs
        # the target synchronization required immediately after growth.
        target = agent.target_approximator.model
        target.network = copy.deepcopy(online.network)
        _reset_adam(target, self.args.learning_rate)

        hidden_after = online.network.encoder_size
        self._record_event({
            "task_index": self.current_task_index,
            "scheduled_step": int(scheduled_step),
            "actual_step": int(self.training_environment_steps),
            "hidden_before": int(hidden_before),
            "hidden_after": int(hidden_after),
            "neurons_added": int(hidden_after - hidden_before),
            "skipped": False,
            "pre_growth_retry_used": pre_growth_retry_used,
            "pre_growth_losses": [float(x) for x in pre_growth_losses],
            "singular_values": [float(x) for x in singular_values],
        })
        if hidden_after > hidden_before:
            self.metrics_monitor.register_growth(
                agent=agent,
                event_index=self.next_event,
                start=hidden_before,
                end=hidden_after,
                step=self.training_environment_steps,
            )
        print(
            f"Growth at step {self.training_environment_steps}: "
            f"{hidden_before} -> {hidden_after} neurons"
        )

    def monitor(self, agent, evaluation_index):
        self.metrics_monitor.monitor_evaluation(
            agent=agent,
            step=self.training_environment_steps,
            evaluation_index=evaluation_index,
            task_index=self.current_task_index,
        )


class GrowingLunarLanderDQN(LunarLanderDQN):
    def __init__(self, *args, growth_controller=None, **kwargs):
        self.growth_controller = growth_controller
        super().__init__(*args, **kwargs)

    def _fit_standard(self, dataset):
        super()._fit_standard(dataset)
        if (
            self.growth_controller is not None
            and self._replay_memory.initialized
        ):
            self.growth_controller.apply_scheduled_growth(self)


def experiment(args: Args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.use_cuda:
        torch.cuda.manual_seed_all(args.seed)

    wind_powers = _task_wind_powers(args)
    n_tasks = len(wind_powers)
    task_steps = _split_budget(args.n_timesteps, n_tasks)
    task_evaluations = _split_budget(args.n_eval_points, n_tasks)
    _validate_args(args, task_steps)

    log_dir = Path(args.output_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    with open(log_dir / f"config-{args.seed}.yaml", "w", encoding="utf-8") as stream:
        yaml.safe_dump(asdict(args), stream, sort_keys=False)

    mdps = [
        LunarLander(
            gravity=args.gravity,
            enable_wind=args.enable_wind and wind_power > 0.0,
            wind_power=wind_power,
            turbulence_power=args.turbulence_power,
        )
        for wind_power in wind_powers
    ]
    for task_index, mdp in enumerate(mdps):
        mdp.seed(args.seed + task_index)
    controller = GrowthController(args, task_steps, log_dir)

    optimizer = {
        "class": optim.Adam,
        "params": {"lr": args.learning_rate},
    }
    approximator_params = {
        "network": controller.module.DQNNetwork,
        "input_shape": mdps[0].info.observation_space.shape,
        "output_shape": (mdps[0].info.action_space.n,),
        "n_actions": mdps[0].info.action_space.n,
        "hidden_size": args.initial_hidden,
        "loss": F.mse_loss,
        "optimizer": optimizer,
        "use_cuda": args.use_cuda,
    }
    algorithm_params = {
        "batch_size": args.batch_size,
        "target_update_frequency": args.target_update_interval,
        "initial_replay_size": args.learning_starts,
        "max_replay_size": args.buffer_size,
    }

    test_epsilon = Parameter(value=0.0)
    policy = EpsGreedy(Parameter(value=1.0))
    agent = GrowingLunarLanderDQN(
        mdps[0].info,
        policy,
        TorchApproximator,
        gradient_steps=args.gradient_steps,
        growth_controller=controller,
        approximator_params=approximator_params,
        **algorithm_params,
    )

    training_metrics = TrainingMetrics()

    def training_step_callback(dataset):
        training_metrics(dataset)
        controller.record_environment_step()

    returns = []
    evaluation_steps = []
    evaluation_task_indices = []
    task_boundaries = [0]
    evaluation_index = 0

    try:
        for task_index, (mdp, n_steps_task, n_evals_task) in enumerate(
            zip(mdps, task_steps, task_evaluations)
        ):
            agent.reset_for_new_task()
            controller.start_task(task_index)

            n_explore = max(
                1, int(n_steps_task * args.exploration_fraction)
            )
            epsilon = LinearParameter(
                value=1.0,
                threshold_value=args.exploration_final_eps,
                n=n_explore,
            )
            training_core = Core(
                agent, mdp, callback_step=training_step_callback
            )
            evaluation_core = Core(agent, mdp)

            for n_steps_block in _split_budget(
                n_steps_task, n_evals_task
            ):
                evaluation_index += 1
                policy.set_epsilon(epsilon)
                training_metrics.start_block()
                training_core.learn(
                    n_steps=n_steps_block,
                    n_steps_per_fit=args.train_freq,
                    quiet=True,
                )
                controller.monitor(agent, evaluation_index)

                policy.set_epsilon(test_epsilon)
                dataset = evaluation_core.evaluate(
                    n_episodes=args.n_test_episodes,
                    quiet=True,
                )
                returns.append(np.mean(compute_J(
                    dataset, mdp.info.gamma
                )))
                evaluation_steps.append(
                    controller.training_environment_steps
                )
                evaluation_task_indices.append(task_index)

            task_boundaries.append(
                controller.training_environment_steps
            )
    finally:
        for mdp in mdps:
            mdp.stop()

    np.save(log_dir / f"J-{args.seed}.npy", np.asarray(returns))
    task_arrays = {
        "evaluation_steps": evaluation_steps,
        "evaluation_task_indices": evaluation_task_indices,
        "task_boundaries": task_boundaries,
        "task_wind_powers": wind_powers,
        "task_timesteps": task_steps,
        "task_evaluations": task_evaluations,
    }
    for name, values in task_arrays.items():
        np.save(
            log_dir / f"{name}-{args.seed}.npy",
            np.asarray(values),
        )
    save_training_metrics(str(log_dir), args.seed, training_metrics, agent)
    _save_growth_results(str(log_dir), args.seed, controller)
    return returns


if __name__ == "__main__":
    cli_args = tyro.cli(Args)
    result = experiment(cli_args)
    print("J:", result)
