"Tiny -> all the layers"

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from gromo.modules.linear_growing_module import LinearGrowingModule

from mushroom_rl.utils.dataset import parse_dataset


def feature_rank(features: np.ndarray, epsilon: float = 0.01):
    pass

def srank(singular_values: np.ndarray, delta: float = 0.01) -> int:
    pass

def principal_angle_cosines(phi: np.ndarray, phi_new: np.ndarray) -> np.ndarray:
    pass

def bellman_residual_correlation(phi_new: np.ndarray, R: np.ndarray) -> float:
    pass


# ── Adapted: both encoder and q_head are LinearGrowingModule ──────────────────
# Mirrors 6_dqn_minatar_tiny_all_layer.QNetwork where:
#   conv  → Conv2dGrowingModule  (dropped here, no conv in FQI)
#   encoder → LinearGrowingModule(previous_module=conv)   becomes LinearGrowingModule(allow_growing=True)
#   q_head  → LinearGrowingModule(previous_module=encoder)
#
# grow_layer_gromo is called twice:
#   1. downstream_layer=q_network.encoder → grows the input→encoder connection
#      (adds neurons to encoder, grows encoder.in_features side — but input is fixed at state_dim)
#      NOTE: in the pure MLP case, input dim is fixed so only encoder output grows.
#      Therefore: downstream_layer=q_network.q_head grows encoder output (encoder adds neurons).
#   2. downstream_layer=q_network.encoder grows the input→encoder connection.
#      This is NOT meaningful when input dim is fixed (state_dim=2).
#      So file 6 effectively grows only the encoder output (same as file 4),
#      with q_head as downstream. Growing both directions is reserved for Conv→encoder case.
#
# Decision: keep structure symmetric with DQN file 6 but only call grow_layer_gromo
# once (downstream=q_head), unless a second hidden layer is added.

class Q_Network(nn.Module):
    """
    encoder: LinearGrowingModule(state_dim → hidden_size, allow_growing=True)
             + ReLUDerivativeOneAtZero
    q_head:  LinearGrowingModule(hidden_size → n_actions, previous_module=encoder)

    Both layers registered as LinearGrowingModule so gromo can grow encoder output
    and optionally the input→encoder connection (if a pre-encoder layer is added).
    """

    def __init__(self, state_dim: int, n_actions: int, hidden_size: int = 128):
        pass

    def encode(self, state: torch.Tensor) -> torch.Tensor:
        pass

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        pass


class NeuralRegressor:
    def __init__(self, state_dim: int, n_actions: int, hidden_size: int = 128):
        pass

    def predict(self, state, **kwargs):
        pass

    def fit(self, state, action, q, n_epochs: int, lr=1e-4,
            batch_size=32, reinit=False, **kwargs):
        pass

    def encode(self, state) -> np.ndarray:
        pass


# ── Same as 2_network_fqi ──────────────────────────────────────────────────────

def pre_growth_optimize(
    network: Q_Network,
    states: torch.Tensor,
    actions: torch.Tensor,
    td_targets: torch.Tensor,
    optimizer,
    n_steps: int,
) -> tuple[float, float]:
    pass


def measure_plasticity(
    network: Q_Network,
    x: torch.Tensor,
    optimizer_class=optim.Adam,
    optimizer_params: dict = None,
    n_steps: int = 2_000,
    n_tasks: int = 10,
) -> float:
    pass


# ── Near-identical to 6_dqn_minatar_tiny_all_layer.grow_layer_gromo ───────────
# Only change: data.observations replaced by states tensor passed directly.
# No conv or env references. Everything else (gromo API calls) is identical.

def grow_layer_gromo(
    q_network: Q_Network,
    states: torch.Tensor,
    actions: torch.Tensor,
    td_targets: torch.Tensor,
    downstream_layer: LinearGrowingModule,
    scaling_factor: float = 1.0,
    maximum_added_neurons: int | None = None,
    numerical_threshold: float = 1e-6,
    statistical_threshold: float = 0,
) -> torch.Tensor | None:
    """
    Grow one layer via gromo's optimal neuron criterion.

    Pass the layer DOWNSTREAM of the one to grow:
      - grow encoder output: downstream_layer=q_network.q_head
      (growing input→encoder not meaningful here since state_dim is fixed)

    Returns eigenvalues_extension or None.
    """
    pass
