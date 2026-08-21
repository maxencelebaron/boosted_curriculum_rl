"""DQN baseline for Gymnasium LunarLander-v3."""

from dataclasses import asdict, dataclass
import os

from joblib import Parallel, delayed
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import tyro
import yaml

from fqi.lunarlander.dqn import LunarLanderDQN
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
    n_jobs: int = 1
    """Number of parallel jobs."""
    n_exp: int = 3
    """Number of experiments (seeds)."""
    n_timesteps: int = 1_500_000
    """Total environment steps per experiment."""
    n_eval_points: int = 50
    """Number of evaluation checkpoints."""
    n_test_episodes: int = 10
    """Number of episodes at each evaluation checkpoint."""
    train_freq: int = 4
    """Number of environment steps between training calls."""
    gradient_steps: int = 1
    """Number of gradient updates per training call."""
    exploration_fraction: float = 0.5
    """Fraction of training over which epsilon is annealed."""
    exploration_final_eps: float = 0.01
    """Final epsilon value."""
    learning_rate: float = 5e-4
    """Adam learning rate."""
    batch_size: int = 64
    """Replay-buffer batch size."""
    buffer_size: int = 10_000
    """Maximum replay-buffer size."""
    learning_starts: int = 1_000
    """Number of transitions collected before learning starts."""
    target_update_interval: int = 1_000
    """Number of training calls between target-network updates."""
    gravity: float = -10.0
    """LunarLander gravity."""
    enable_wind: bool = True
    """Whether wind is enabled."""
    wind_power: float = 15.0
    """Wind power of the target task."""
    turbulence_power: float = 1.5
    """Turbulence power of the target task."""
    use_cuda: bool = True
    """Whether to train the neural network on CUDA."""
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

    mdp = LunarLander(
        gravity=args.gravity,
        enable_wind=args.enable_wind,
        wind_power=args.wind_power,
        turbulence_power=args.turbulence_power,
    )
    mdp.seed(seed)

    optimizer = {
        "class": optim.Adam,
        "params": dict(lr=args.learning_rate),
    }
    approximator_params = dict(
        network=DQNNetwork,
        input_shape=mdp.info.observation_space.shape,
        output_shape=(mdp.info.action_space.n,),
        n_actions=mdp.info.action_space.n,
        loss=F.mse_loss,
        optimizer=optimizer,
        use_cuda=args.use_cuda,
    )
    algorithm_params = dict(
        batch_size=args.batch_size,
        target_update_frequency=args.target_update_interval,
        initial_replay_size=args.learning_starts,
        max_replay_size=args.buffer_size,
    )

    steps_per_eval = args.n_timesteps // args.n_eval_points
    if steps_per_eval < 1:
        raise ValueError("n_timesteps must be at least n_eval_points")
    n_explore = int(args.n_timesteps * args.exploration_fraction)
    epsilon = LinearParameter(
        value=1.0,
        threshold_value=args.exploration_final_eps,
        n=max(1, n_explore),
    )
    test_epsilon = Parameter(value=0.0)
    policy = EpsGreedy(epsilon)

    agent = LunarLanderDQN(
        mdp.info,
        policy,
        TorchApproximator,
        gradient_steps=args.gradient_steps,
        approximator_params=approximator_params,
        **algorithm_params,
    )
    metrics = TrainingMetrics()
    training_core = Core(agent, mdp, callback_step=metrics)
    evaluation_core = Core(agent, mdp)
    returns = []

    for _ in range(args.n_eval_points):
        policy.set_epsilon(epsilon)
        metrics.start_block()
        training_core.learn(
            n_steps=steps_per_eval,
            n_steps_per_fit=args.train_freq,
            quiet=True,
        )

        policy.set_epsilon(test_epsilon)
        test_dataset = evaluation_core.evaluate(
            n_episodes=args.n_test_episodes,
            quiet=True,
        )
        returns.append(np.mean(compute_J(test_dataset, mdp.info.gamma)))

    np.save(os.path.join(log_dir, "J-%d.npy" % seed), returns)
    save_training_metrics(log_dir, seed, metrics, agent)
    return returns


def experiment(seed, log_dir, args):
    return train_dqn(seed, log_dir, args)


if __name__ == "__main__":
    args = tyro.cli(Args)
    os.makedirs(args.output_dir, exist_ok=True)

    out = Parallel(n_jobs=args.n_jobs)(
        delayed(experiment)(95 + k, args.output_dir, args)
        for k in range(args.n_exp)
    )
    returns = np.asarray(out)
    np.save(os.path.join(args.output_dir, "J.npy"), returns)

    with open(os.path.join(args.output_dir, "config.yaml"), "w") as stream:
        yaml.dump(asdict(args), stream, default_flow_style=False,
                  sort_keys=False)

    print("J: ", np.mean(returns, axis=0))
