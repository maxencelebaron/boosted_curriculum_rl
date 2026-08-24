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
from fqi.lunarlander.run_dqn import TrainingMetrics, save_training_metrics

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
    "gromo_one_layer": "fqi.4_network_fqi_tiny_one_layer",
}


@dataclass
class Args:
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
    wind_power: float = 15.0
    turbulence_power: float = 1.5
    use_cuda: bool = True

    growth_mode: str = "svd"
    """Growth strategy: random, random-0, svd, or gromo_one_layer."""
    initial_hidden: int = 32
    """Initial width of the growable layer."""
    final_hidden: int = 64
    """Requested width after the final growth event."""
    n_growth_events: int = 8
    """Number of growth events."""
    growth_start_step: int = 0
    """First growth step; zero selects n_timesteps/(n_events+1)."""
    growth_end_step: int = 0
    """Last growth step; zero selects n_timesteps*n_events/(n_events+1)."""
    pre_growth_steps: int = 10
    """Optimization steps on the fixed growth batch before growing."""
    grow_batch_size: int = 256
    """Replay transitions used by a growth event."""

    use_natural_gradient: bool = False
    """Use K-FAC instead of Adam during pre-growth optimization."""
    natural_gradient_damping: float = 1e-5
    natural_gradient_noise_variance: float = 1.0
    natural_gradient_eigenvalue_threshold: float = 1e-7

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


def _growth_schedule(args: Args) -> list[int]:
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


def _validate_args(args: Args):
    if args.growth_mode not in GROWTH_MODULES:
        raise ValueError(
            f"Unknown growth_mode {args.growth_mode!r}; "
            f"choose from {tuple(GROWTH_MODULES)}"
        )
    if args.n_timesteps < args.n_eval_points or args.n_eval_points < 1:
        raise ValueError("n_timesteps must be at least n_eval_points >= 1")
    if args.n_growth_events < 0:
        raise ValueError("n_growth_events must be non-negative")
    if args.n_growth_events > 0 and args.final_hidden <= args.initial_hidden:
        raise ValueError("final_hidden must be greater than initial_hidden")
    if args.pre_growth_steps < 0:
        raise ValueError("pre_growth_steps must be non-negative")
    if args.grow_batch_size < 1:
        raise ValueError("grow_batch_size must be positive")
    if not 0.0 < args.exploration_final_eps <= 1.0:
        raise ValueError("exploration_final_eps must be in (0, 1]")

    schedule = _growth_schedule(args)
    if len(schedule) != args.n_growth_events:
        raise ValueError("growth steps must be distinct")
    if schedule and (schedule[0] <= args.learning_starts or schedule[-1] > args.n_timesteps):
        raise ValueError(
            "growth steps must be after learning_starts and within training"
        )


def _reset_adam(torch_approximator, learning_rate):
    torch_approximator._optimizer = optim.Adam(
        torch_approximator.network.parameters(), lr=learning_rate
    )


def _save_growth_results(log_dir, seed, controller: GrowthController):
    with open(
        os.path.join(log_dir, f"growth-events-{seed}.yaml"),
        "w",
        encoding="utf-8",
    ) as stream:
        yaml.safe_dump(controller.events, stream, sort_keys=False)

    controller.metrics_monitor.save(log_dir, seed)


class GrowthController:
    """Grow from a batch sampled from DQN replay."""

    def __init__(self, args: Args):
        self.args = args
        self.module = importlib.import_module(GROWTH_MODULES[args.growth_mode])
        self.steps = _growth_schedule(args)
        self.target_widths = np.rint(np.linspace(
            args.initial_hidden,
            args.final_hidden,
            args.n_growth_events + 1,
        )).astype(int).tolist()[1:]
        self.next_event = 0
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

    def record_environment_step(self):
        self.training_environment_steps += 1

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

    def _grow(self, agent, scheduled_step, target_width):
        online = agent.approximator.model
        network = online.network
        hidden_before = network.encoder_size
        neurons_to_add = max(0, target_width - hidden_before)
        states, actions, td_targets = replay_batch_to_tensors(
            agent, agent._replay_memory.get(self.args.grow_batch_size)
        )
        pre_growth_losses = []

        if self.args.pre_growth_steps:
            _, final_loss, pre_growth_losses = pre_growth_optimize(
                network=network,
                states=states,
                actions=actions,
                td_targets=td_targets,
                optimizer=online._optimizer,
                n_steps=self.args.pre_growth_steps,
                use_natural_gradient=self.args.use_natural_gradient,
                natural_gradient_damping=self.args.natural_gradient_damping,
                natural_gradient_noise_variance=(
                    self.args.natural_gradient_noise_variance
                ),
                natural_gradient_eigenvalue_threshold=(
                    self.args.natural_gradient_eigenvalue_threshold
                ),
            )
            if final_loss < self.args.bellman_residual_threshold:
                _reset_adam(online, self.args.learning_rate)
                self.events.append({
                    "scheduled_step": int(scheduled_step),
                    "actual_step": int(self.training_environment_steps),
                    "hidden_before": int(hidden_before),
                    "hidden_after": int(hidden_before),
                    "neurons_added": 0,
                    "skipped": True,
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
        elif self.args.growth_mode == "svd":
            new_network, singular_values = self.module.grow_network_svd(
                network,
                states,
                actions,
                td_targets,
                d_a=neurons_to_add,
                numerical_threshold=self.args.numerical_threshold,
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
        self.events.append({
            "scheduled_step": int(scheduled_step),
            "actual_step": int(self.training_environment_steps),
            "hidden_before": int(hidden_before),
            "hidden_after": int(hidden_after),
            "neurons_added": int(hidden_after - hidden_before),
            "skipped": False,
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
    _validate_args(args)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.use_cuda:
        torch.cuda.manual_seed_all(args.seed)

    log_dir = Path(args.output_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    with open(log_dir / f"config-{args.seed}.yaml", "w", encoding="utf-8") as stream:
        yaml.safe_dump(asdict(args), stream, sort_keys=False)

    mdp = LunarLander(
        gravity=args.gravity,
        enable_wind=args.enable_wind,
        wind_power=args.wind_power,
        turbulence_power=args.turbulence_power,
    )
    mdp.seed(args.seed)
    controller = GrowthController(args)

    optimizer = {
        "class": optim.Adam,
        "params": {"lr": args.learning_rate},
    }
    approximator_params = {
        "network": controller.module.DQNNetwork,
        "input_shape": mdp.info.observation_space.shape,
        "output_shape": (mdp.info.action_space.n,),
        "n_actions": mdp.info.action_space.n,
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

    n_explore = max(
        1, int(args.n_timesteps * args.exploration_fraction)
    )
    epsilon = LinearParameter(
        value=1.0,
        threshold_value=args.exploration_final_eps,
        n=n_explore,
    )
    test_epsilon = Parameter(value=0.0)
    policy = EpsGreedy(epsilon)
    agent = GrowingLunarLanderDQN(
        mdp.info,
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

    training_core = Core(agent, mdp, callback_step=training_step_callback)
    evaluation_core = Core(agent, mdp)
    steps_per_eval = args.n_timesteps // args.n_eval_points
    returns = []

    try:
        for evaluation_index in range(1, args.n_eval_points + 1):
            policy.set_epsilon(epsilon)
            training_metrics.start_block()
            training_core.learn(
                n_steps=steps_per_eval,
                n_steps_per_fit=args.train_freq,
                quiet=True,
            )
            controller.monitor(agent, evaluation_index)

            policy.set_epsilon(test_epsilon)
            dataset = evaluation_core.evaluate(
                n_episodes=args.n_test_episodes,
                quiet=True,
            )
            returns.append(np.mean(compute_J(dataset, mdp.info.gamma)))
    finally:
        mdp.stop()

    np.save(log_dir / f"J-{args.seed}.npy", np.asarray(returns))
    save_training_metrics(str(log_dir), args.seed, training_metrics, agent)
    _save_growth_results(str(log_dir), args.seed, controller)
    return returns


if __name__ == "__main__":
    cli_args = tyro.cli(Args)
    result = experiment(cli_args)
    print("J:", result)
