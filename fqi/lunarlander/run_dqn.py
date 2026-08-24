"""DQN baseline for Gymnasium LunarLander-v3."""

from dataclasses import asdict, dataclass
import os

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import tyro
import yaml

from fqi.lunarlander.dqn import BoostedLunarLanderDQN, LunarLanderDQN
from fqi.lunarlander.dqn_monitoring import DQNMetricsMonitor
from fqi.lunarlander.env import LunarLander
from fqi.network_fqi_lunarlander import DQNNetwork
from mushroom_rl.approximators.parametric import TorchApproximator
from mushroom_rl.core import Core
from mushroom_rl.policy import EpsGreedy
from mushroom_rl.utils.dataset import compute_J
from mushroom_rl.utils.parameters import LinearParameter, Parameter


torch.set_num_threads(1)


@dataclass
class Args:
    use_curriculum: bool = False
    """Train successively on increasing wind powers."""
    use_boosting: bool = False
    """Add one residual Q-network at each task."""
    seed: int = 95
    """Random seed of the experiment."""
    n_timesteps: int = 1_500_000
    """Total environment steps per experiment."""
    n_eval_points: int = 200
    """Number of evaluation checkpoints."""
    n_test_episodes: int = 10
    """Number of episodes at each evaluation checkpoint."""
    train_freq: int = 4
    """Number of environment steps between training calls."""
    gradient_steps: int = 1
    """Number of gradient updates per training call."""
    exploration_fraction: float = 0.3
    """Fraction of training over which epsilon is annealed."""
    exploration_final_eps: float = 0.01
    """Final epsilon value."""
    learning_rate: float = 1e-3
    """Adam learning rate."""
    batch_size: int = 128
    """Replay-buffer batch size."""
    buffer_size: int = 100_000
    """Maximum replay-buffer size."""
    learning_starts: int = 1_000
    """Number of transitions collected before learning starts."""
    target_update_interval: int = 1_000
    """Number of training calls between target-network updates."""

    gravity: float = -10.0
    """LunarLander gravity."""
    enable_wind: bool = True
    """Whether wind is enabled."""
    wind_powers: tuple[float, ...] = (0.0, 5.0, 10.0, 15.0)
    """Curriculum tasks; the last value defines the target task."""
    turbulence_power: float = 1.5
    """Turbulence power of the target task."""

    use_cuda: bool = True
    """Whether to train the neural network on CUDA."""
    metric_monitoring_batch_size: int = 128
    """Fixed replay sample used for feature rank monitoring."""
    n_plasticity_measurements: int = 20
    """Number of uniformly spaced plasticity measurements."""
    plasticity_n_steps: int = 100
    """Gradient steps per plasticity probe task."""
    plasticity_n_tasks: int = 10
    """Number of random plasticity probe tasks."""
    plasticity_n_samples: int = 512
    """Fresh replay samples used by each plasticity measurement."""
    output_dir: str = "logs/dqn_lunarlander"
    """Directory where results are saved."""


class TrainingMetrics:
    """Collect rewards and completed-episode lengths during training only."""

    def __init__(self):
        self.step_rewards = []
        self.episode_returns = []
        self.episode_lengths = []
        self._episode_return = 0.0
        self._episode_length = 0

    def start_block(self):
        # Core starts a new episode at every call to learn().
        self._episode_return = 0.0
        self._episode_length = 0

    def __call__(self, dataset):
        transition = dataset[0]
        reward = float(transition[2])
        self.step_rewards.append(reward)
        self._episode_return += reward
        self._episode_length += 1
        if transition[5]:
            self.episode_returns.append(self._episode_return)
            self.episode_lengths.append(self._episode_length)
            self._episode_return = 0.0
            self._episode_length = 0


def rolling_mean(values, window):
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return values
    cumulative = np.cumsum(np.r_[0.0, values])
    indices = np.arange(len(values))
    starts = np.maximum(0, indices - window + 1)
    return (cumulative[indices + 1] - cumulative[starts]) / (
        indices - starts + 1
    )


def _split_budget(total, n_parts):
    """Split an integer budget exactly and as evenly as possible."""
    quotient, remainder = divmod(total, n_parts)
    return [
        quotient + (index < remainder)
        for index in range(n_parts)
    ]


def _task_wind_powers(args):
    if not args.wind_powers:
        raise ValueError("wind_powers must contain at least one value")
    if args.use_curriculum:
        return list(args.wind_powers)
    return [args.wind_powers[-1]] * len(args.wind_powers)


def _validate_args(args, n_tasks):
    if args.n_timesteps < n_tasks:
        raise ValueError("n_timesteps must allow at least one step per task")
    if args.n_eval_points < n_tasks:
        raise ValueError("n_eval_points must allow one evaluation per task")
    if not 0.0 <= args.exploration_fraction <= 1.0:
        raise ValueError("exploration_fraction must be in [0, 1]")
    if not 0.0 < args.exploration_final_eps <= 1.0:
        raise ValueError("exploration_final_eps must be in (0, 1]")

    shortest_task = min(_split_budget(args.n_timesteps, n_tasks))
    if args.learning_starts >= shortest_task:
        raise ValueError("learning_starts must be smaller than each task")


def save_training_metrics(log_dir, seed, metrics, agent):
    reward_steps = np.arange(1, len(metrics.step_rewards) + 1)
    step_rewards = np.asarray(metrics.step_rewards)
    smoothed_step_rewards = rolling_mean(step_rewards, 500)

    episode_indices = np.arange(1, len(metrics.episode_returns) + 1)
    episode_returns = np.asarray(metrics.episode_returns)
    episode_lengths = np.asarray(metrics.episode_lengths)
    smoothed_episode_returns = rolling_mean(episode_returns, 50)
    smoothed_episode_lengths = rolling_mean(episode_lengths, 50)

    gradient_loss_steps = np.asarray(agent.training_loss_steps)
    gradient_losses = np.asarray(agent.training_losses)
    loss_steps, inverse = np.unique(gradient_loss_steps, return_inverse=True)
    losses = np.zeros(len(loss_steps), dtype=float)
    loss_counts = np.zeros(len(loss_steps), dtype=int)
    np.add.at(losses, inverse, gradient_losses)
    np.add.at(loss_counts, inverse, 1)
    losses /= loss_counts
    smoothed_losses = rolling_mean(losses, 50)

    arrays = {
        "training_reward_steps": reward_steps,
        "training_rewards_raw": step_rewards,
        "training_rewards": smoothed_step_rewards,
        "episode_indices": episode_indices,
        "episode_returns_raw": episode_returns,
        "episode_returns": smoothed_episode_returns,
        "episode_lengths_raw": episode_lengths,
        "episode_lengths": smoothed_episode_lengths,
        "loss_steps": loss_steps,
        "losses_raw": losses,
        "losses": smoothed_losses,
    }
    for name, values in arrays.items():
        np.save(os.path.join(log_dir, "%s-%d.npy" % (name, seed)), values)


def train_dqn(seed, log_dir, args):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if args.use_cuda:
        torch.cuda.manual_seed_all(seed)

    wind_powers = _task_wind_powers(args)
    n_tasks = len(wind_powers)
    _validate_args(args, n_tasks)
    task_steps = _split_budget(args.n_timesteps, n_tasks)
    task_evaluations = _split_budget(args.n_eval_points, n_tasks)

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
        mdp.seed(seed + task_index)

    optimizer = {
        "class": optim.Adam,
        "params": dict(lr=args.learning_rate),
    }
    approximator_params = dict(
        network=DQNNetwork,
        input_shape=mdps[0].info.observation_space.shape,
        output_shape=(mdps[0].info.action_space.n,),
        n_actions=mdps[0].info.action_space.n,
        loss=F.mse_loss,
        optimizer=optimizer,
        use_cuda=args.use_cuda,
    )
    if args.use_boosting:
        approximator_params.update(
            n_models=n_tasks,
            prediction="sum",
        )
    algorithm_params = dict(
        batch_size=args.batch_size,
        target_update_frequency=args.target_update_interval,
        initial_replay_size=args.learning_starts,
        max_replay_size=args.buffer_size,
    )

    test_epsilon = Parameter(value=0.0)
    policy = EpsGreedy(Parameter(value=1.0))

    agent_class = (
        BoostedLunarLanderDQN
        if args.use_boosting
        else LunarLanderDQN
    )
    agent = agent_class(
        mdps[0].info,
        policy,
        TorchApproximator,
        gradient_steps=args.gradient_steps,
        approximator_params=approximator_params,
        **algorithm_params,
    )
    metrics = TrainingMetrics()
    monitor = DQNMetricsMonitor(
        n_eval_points=args.n_eval_points,
        monitoring_batch_size=args.metric_monitoring_batch_size,
        n_plasticity_measurements=args.n_plasticity_measurements,
        plasticity_n_samples=args.plasticity_n_samples,
        plasticity_n_steps=args.plasticity_n_steps,
        plasticity_n_tasks=args.plasticity_n_tasks,
        learning_rate=args.learning_rate,
    )
    returns = []
    evaluation_steps = []
    evaluation_task_indices = []
    task_boundaries = [0]
    evaluation_index = 0

    try:
        for task_index, (mdp, n_steps_task, n_evals_task) in enumerate(
            zip(mdps, task_steps, task_evaluations)
        ):
            if args.use_boosting:
                agent.set_curriculum_idx_and_reset(task_index)
            else:
                agent.reset_for_new_task()
            monitor.start_task()

            n_explore = max(
                1, int(n_steps_task * args.exploration_fraction)
            )
            epsilon = LinearParameter(
                value=1.0,
                threshold_value=args.exploration_final_eps,
                n=n_explore,
            )
            training_core = Core(agent, mdp, callback_step=metrics)
            evaluation_core = Core(agent, mdp)

            for n_steps_block in _split_budget(
                n_steps_task, n_evals_task
            ):
                evaluation_index += 1
                policy.set_epsilon(epsilon)
                metrics.start_block()
                training_core.learn(
                    n_steps=n_steps_block,
                    n_steps_per_fit=args.train_freq,
                    quiet=True,
                )
                global_step = len(metrics.step_rewards)
                monitor.monitor_evaluation(
                    agent,
                    step=global_step,
                    evaluation_index=evaluation_index,
                    task_index=task_index,
                )

                policy.set_epsilon(test_epsilon)
                test_dataset = evaluation_core.evaluate(
                    n_episodes=args.n_test_episodes,
                    quiet=True,
                )
                returns.append(np.mean(compute_J(
                    test_dataset, mdp.info.gamma
                )))
                evaluation_steps.append(global_step)
                evaluation_task_indices.append(task_index)

            task_boundaries.append(len(metrics.step_rewards))
    finally:
        for mdp in mdps:
            mdp.stop()

    np.save(os.path.join(log_dir, "J-%d.npy" % seed), returns)
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
            os.path.join(log_dir, "%s-%d.npy" % (name, seed)),
            np.asarray(values),
        )
    save_training_metrics(log_dir, seed, metrics, agent)
    monitor.save(log_dir, seed)
    return returns


def experiment(seed, log_dir, args):
    return train_dqn(seed, log_dir, args)


if __name__ == "__main__":
    args = tyro.cli(Args)
    os.makedirs(args.output_dir, exist_ok=True)

    returns = experiment(args.seed, args.output_dir, args)

    config_path = os.path.join(
        args.output_dir, "config-%d.yaml" % args.seed
    )
    with open(config_path, "w") as stream:
        yaml.safe_dump(
            asdict(args), stream, default_flow_style=False,
            sort_keys=False,
        )

    print("J: ", returns)
