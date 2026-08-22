"""Evaluate a pretrained greedy DQN policy on LunarLander-v3.

The checkpoint is expected to contain a PyTorch state_dict with layers named
``fc1``, ``fc2`` and ``fc3``.  A single final checkpoint cannot reproduce an
evaluation curve over training; this script instead plots episode returns and
their cumulative mean for the fixed policy.
"""

import argparse
from pathlib import Path

import gymnasium as gym
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn


BASE_DIR = Path(__file__).resolve().parent


class PretrainedQNetwork(nn.Module):
    def __init__(self, activation):
        super().__init__()
        self.fc1 = nn.Linear(8, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 4)
        self.activation = activation

    def forward(self, state):
        x = self.activation(self.fc1(state))
        x = self.activation(self.fc2(x))
        return self.fc3(x)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--n-episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=95)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--activation", choices=("relu", "tanh"),
                        default="relu")
    parser.add_argument("--gravity", type=float, default=-10.0)
    parser.add_argument("--enable-wind", action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument("--wind-power", type=float, default=15.0)
    parser.add_argument("--turbulence-power", type=float, default=1.5)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"),
                        default="auto")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=BASE_DIR / "logs" / "pretrained_dqn",
    )
    return parser.parse_args()


def select_device(requested):
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(requested)


def load_model(checkpoint, activation_name, device):
    activation = torch.relu if activation_name == "relu" else torch.tanh
    model = PretrainedQNetwork(activation).to(device)
    state_dict = torch.load(checkpoint, map_location=device, weights_only=True)
    if not isinstance(state_dict, dict):
        raise TypeError("The checkpoint does not contain a state_dict")
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def evaluate(model, env, n_episodes, seed, gamma, device):
    discounted_returns = np.empty(n_episodes, dtype=np.float64)
    raw_returns = np.empty(n_episodes, dtype=np.float64)
    episode_lengths = np.empty(n_episodes, dtype=np.int64)

    for episode in range(n_episodes):
        state, _ = env.reset(seed=seed + episode)
        terminated = truncated = False
        discounted_return = 0.0
        raw_return = 0.0
        step = 0

        while not (terminated or truncated):
            state_tensor = torch.as_tensor(
                state, dtype=torch.float32, device=device
            ).unsqueeze(0)
            with torch.no_grad():
                action = int(model(state_tensor).argmax(dim=1).item())
            state, reward, terminated, truncated, _ = env.step(action)
            raw_return += reward
            discounted_return += gamma ** step * reward
            step += 1

        discounted_returns[episode] = discounted_return
        raw_returns[episode] = raw_return
        episode_lengths[episode] = step

    return discounted_returns, raw_returns, episode_lengths


def cumulative_mean(values):
    return np.cumsum(values) / np.arange(1, len(values) + 1)


def save_results(output_dir, discounted_returns, raw_returns, lengths):
    output_dir.mkdir(parents=True, exist_ok=True)
    episode_indices = np.arange(1, len(discounted_returns) + 1)
    mean_returns = cumulative_mean(discounted_returns)

    np.save(output_dir / "evaluation_returns.npy", discounted_returns)
    np.save(output_dir / "evaluation_returns_cumulative_mean.npy", mean_returns)
    np.save(output_dir / "undiscounted_returns.npy", raw_returns)
    np.save(output_dir / "episode_lengths.npy", lengths)
    np.save(output_dir / "episode_indices.npy", episode_indices)

    figure, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(episode_indices, discounted_returns, alpha=0.35, linewidth=1,
            label="Discounted return per episode")
    ax.plot(episode_indices, mean_returns, linewidth=1.5,
            label="Cumulative mean")
    ax.set(
        xlabel="Evaluation episode",
        ylabel="Discounted return",
        title="Pretrained DQN evaluation",
    )
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0),
              borderaxespad=0)
    figure.tight_layout()
    figure.savefig(output_dir / "evaluation_returns.pdf",
                   bbox_inches="tight")
    plt.close(figure)


def main():
    args = parse_args()
    if args.n_episodes < 1:
        raise ValueError("n_episodes must be positive")
    if not 0.0 <= args.gamma <= 1.0:
        raise ValueError("gamma must be between 0 and 1")

    device = select_device(args.device)
    model = load_model(args.checkpoint, args.activation, device)
    env = gym.make(
        "LunarLander-v3",
        continuous=False,
        gravity=args.gravity,
        enable_wind=args.enable_wind,
        wind_power=args.wind_power,
        turbulence_power=args.turbulence_power,
    )
    try:
        discounted_returns, raw_returns, lengths = evaluate(
            model, env, args.n_episodes, args.seed, args.gamma, device
        )
    finally:
        env.close()

    save_results(args.output_dir, discounted_returns, raw_returns, lengths)
    print(f"Mean discounted return: {discounted_returns.mean():.2f}")
    print(f"Standard deviation: {discounted_returns.std():.2f}")
    print(f"Mean raw return: {raw_returns.mean():.2f}")
    print(f"Saved results to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
