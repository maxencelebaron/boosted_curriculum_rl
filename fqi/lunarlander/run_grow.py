from __future__ import annotations

import importlib
import tyro
import pathlib
import pickle
import yaml
from dataclasses import dataclass, asdict

import numpy as np
import torch
from joblib import Parallel, delayed
from tqdm import trange

from fqi.neural_fqi import BoostedNeuralFQI, NeuralFQI
from fqi.car_on_hill.solver import solve_car_on_hill
from fqi.utils.growing_network import pre_growth_optimize, compute_metrics

from mushroom_rl.core import Core, Logger
from mushroom_rl.environments.car_on_hill import CarOnHill
from mushroom_rl.policy import EpsGreedy
from mushroom_rl.utils.dataset import compute_J, parse_dataset
from mushroom_rl.utils.parameters import Parameter


# Maps growth_mode → dotted module path
GROWTH_MODULES = {
    "random":          "fqi.2_network_fqi_grow_randomly",
    "svd":             "fqi.3_network_fqi_optimizer_plus_svd",
    "gromo_one_layer": "fqi.4_network_fqi_tiny_one_layer",
}


@dataclass
class Args:
    # Experiment
    n_exp: int = 10
    """number of independent experiment seeds"""
    n_jobs: int = 1
    """number of parallel jobs (joblib). Keep at 1 when using GPU."""
    use_curriculum: bool = False
    """if set, trains sequentially on ms=[0.8, 1.0, 1.2]"""
    use_boosting: bool = False
    """if set, uses BoostedNeuralFQI (curriculum boosting)"""
    monitor_loss: bool = False
    """track per-epoch loss and Q-error"""
    growth_mode: str = "svd"
    """growth strategy: random | svd | gromo_one_layer """

    # FQI training
    iters_per_env: int = 20
    """FQI iterations per environment"""
    lr: float = 1e-3
    """learning rate"""
    n_epochs: int = 20
    """training epochs per FQI iteration"""
    batch_size: int = 32
    """mini-batch size for FQI training"""

    feature_rank_n_states: int = 2_000
    """number of states to collect for periodic feature rank monitoring"""

    # Network growth
    initial_hidden: int = 128
    """initial encoder hidden size"""
    final_hidden: int = 256
    """target encoder hidden size after all growth events"""
    n_growth_events: int = 2
    """total number of growth events"""
    growth_start: int = 0
    """global iteration of first growth event (0 = auto: total_iters / (n+1))"""
    growth_end: int = 0
    """global iteration of last growth event (0 = auto: total_iters * n / (n+1))"""
    pre_growth_steps: int = 10
    """gradient steps to update current weights before growing"""
    grow_batch_size: int = 1000
    """batch size for the growth computation"""
    bellman_residual_threshold: float = 0.0
    """skip growth if pre-growth final loss is below this"""
    numerical_threshold: float = 1e-6
    """threshold for near-zero singular values"""
    statistical_threshold: float = 0.0
    """gromo sub-selection threshold"""


def _ms_for_args(args: Args) -> list[float]:
    if args.use_curriculum:
        return [0.8, 1.0, 1.2]
    if args.use_boosting:
        return [1.2, 1.2, 1.2]
    return [1.2]


def _growth_schedule(n_events: int, start: int, end: int) -> set:
    """
    Returns the set of global iteration indices at which growth should occur.
    Events are evenly spaced between start and end (both inclusive).

    Example: n_events=6, start=10, end=20 → {10, 12, 14, 16, 18, 20}
    """
    return set(np.round(np.linspace(start, end, n_events)).astype(int))


def _compute_grow_batch(
    dataset,
    regressor,
    gamma: float,
    grow_batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Prepare a mini-batch for the growth step.

    agent.fit() computes TD targets internally without exposing them.
    This function recomputes them on a random subset of the dataset
    so they can be passed to the grow functions (grow_network_*).

    Steps:
      1. Parse the full dataset -> (s, a, r, s', absorbing)
      2. Randomly subsample grow_batch_size transitions
      3. Compute td = r + gamma * max_a Q(s'; theta) with the current network

    Returns (states, actions, td_targets) as PyTorch tensors.
    """
    states, actions, rewards, next_states, absorbing, _ = parse_dataset(dataset)
    n = len(states)
    idx = np.random.choice(n, min(grow_batch_size, n), replace=False)

    s = states[idx]
    a = actions[idx].reshape(-1).astype(int)
    r = rewards[idx]
    s_next = next_states[idx]
    done = absorbing[idx]

    q_next = regressor.predict(s_next)
    q_max = q_next.max(axis=1)
    td = r + gamma * (1 - done) * q_max

    device = next(regressor._model.parameters()).device
    # All tensors are built from numpy → requires_grad=False by construction.
    # The no_grad wrapper in the recomputation block after pre_growth_optimize

    return (
        torch.FloatTensor(s).to(device),
        torch.LongTensor(a).to(device),
        torch.FloatTensor(td).to(device),
        torch.FloatTensor(s_next).to(device),
        torch.FloatTensor(r).to(device),
        torch.FloatTensor(done).to(device),
    )


def _grow_step(
    growth_mode: str,
    module,
    regressor,
    states: torch.Tensor,
    actions: torch.Tensor,
    td_targets: torch.Tensor,
    next_states: torch.Tensor,
    rewards: torch.Tensor,
    done: torch.Tensor,
    gamma: float,
    args: Args,
    neurons_per_step: int,
) -> dict:
    """
    Optionally update the current weights, then grow according to growth_mode.
    Resets the optimizer after any structural change.

    neurons_per_step = (final_hidden - initial_hidden) // n_growth_events

    Returns a dict with keys: grew, neurons_added, pre_growth_losses.
    """
    q_network = regressor._model
    hidden_size_before = regressor._model.encoder_size
    pre_growth_losses = []

    if args.pre_growth_steps > 0:
        if regressor._optimizer is None:
            regressor._optimizer = torch.optim.Adam(
                q_network.parameters(), lr=args.lr
            )
        initial_loss, final_loss, pre_growth_losses = pre_growth_optimize(
            q_network,
            states,
            actions,
            td_targets,
            regressor._optimizer,
            args.pre_growth_steps,
        )
        if final_loss < args.bellman_residual_threshold:
            return {
                "grew": False,
                "neurons_added": 0,
                "pre_growth_losses": pre_growth_losses
            }
        # Recompute TD targets with the updated network weights before growing
        with torch.no_grad():
            q_next_fresh = q_network(next_states).max(dim=1)[0]
            td_targets = rewards + gamma * (1 - done) * q_next_fresh

    if growth_mode == "random":
        new_h = q_network.encoder[0].out_features + neurons_per_step
        new_net = module.grow_network(q_network, new_h)
        regressor._model = new_net
        regressor._optimizer = None

    elif growth_mode == "svd":
        new_net, svd_singular_values = module.grow_network_svd(
            q_network,
            states,
            actions,
            td_targets,
            d_a=neurons_per_step,
            numerical_threshold=args.numerical_threshold,
        )
        regressor._model = new_net
        regressor._optimizer = None

    elif growth_mode == "gromo_one_layer":
        module.grow_network_gromo(
            q_network, states, actions, td_targets,
            maximum_added_neurons=neurons_per_step,
            numerical_threshold=args.numerical_threshold,
            statistical_threshold=args.statistical_threshold,
        )
        regressor._optimizer = None

    hidden_size_after = regressor._model.encoder_size
    neurons_added = hidden_size_after - hidden_size_before
    return {"grew": neurons_added > 0, "neurons_added": neurons_added, "pre_growth_losses": pre_growth_losses}
