import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from mushroom_rl.utils.dataset import parse_dataset

from utils.growing_network import (
    feature_rank,
    srank,
    measure_plasticity,
    principal_angle_cosines,
    bellman_residual_correlation,
    pre_growth_optimize,
)


class Q_Network(nn.Module):
    """
    Growable MLP
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.h1 = nn.Sequential(
            nn.Linear(2, 64),
            nn.Sigmoid()
        )
        self.encoder = nn.Sequential(
            nn.Linear(64, hidden_size),
            nn.Sigmoid()
        )
        self.q_head = nn.Linear(hidden_size, 2)

    def encode(self, state: torch.Tensor) -> torch.Tensor:
        return self.encoder(self.h1(state.float()))

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.q_head(self.encode(state))


class NeuralRegressor:
    """
    Same fit/predict interface as network_fqi.NeuralRegressor.
    Additions: encode() for feature monitoring, _model is swappable after growth.
    """

    def __init__(self, hidden_size: int = 16, **kwargs):
        self._model = Q_Network(hidden_size)
        self._optimizer = None
        self._loss_fn = nn.MSELoss(reduction='sum')
        self._is_fitted = False
        self.last_loss_history = []
        self.epoch_callback = None

    def predict(self, state, **kwargs):
        if not self._is_fitted:
            return np.zeros((len(state), 2))
        s = torch.FloatTensor(state)
        with torch.no_grad():
            self._model.eval()
            return self._model(s).numpy()

    def fit(
        self,
        state,
        action,
        q,
        n_epochs: int,
        lr=1e-3,
        batch_size=32,
        reinit=False,
        **kwargs
    ):
        if reinit:
            for m in self._model.modules():
                if hasattr(m, 'reset_parameters'):
                    m.reset_parameters()
            self._optimizer = None

        if self._optimizer is None:
            self._optimizer = torch.optim.Adam(
                self._model.parameters(),
                lr=lr
            )

        s = torch.FloatTensor(state)
        a = torch.LongTensor(action.reshape(-1))
        t = torch.FloatTensor(q)

        loader = DataLoader(
            TensorDataset(s, a, t),
            batch_size=batch_size, shuffle=True
        )
        self._is_fitted = True
        self.last_loss_history = []
        self._model.train()
        for epoch in range(n_epochs):
            epoch_loss = 0.
            for sb, ab, tb in loader:
                self._optimizer.zero_grad()
                q_pred = self._model(sb)
                q_pred_a = q_pred[torch.arange(len(ab)), ab]
                loss = self._loss_fn(q_pred_a, tb)
                loss.backward()
                self._optimizer.step()
                epoch_loss += loss.item()
            avg_loss = epoch_loss / len(loader)
            self.last_loss_history.append(avg_loss)
            if self.epoch_callback is not None:
                self.epoch_callback()
                self._model.train()
            print(f"  epoch {epoch+1}/{n_epochs}  loss={avg_loss:.6f}")

    def encode(self, state) -> np.ndarray:
        s = torch.FloatTensor(state)
        with torch.no_grad():
            self._model.eval()
            return self._model.encode(s).numpy()


# ── Adapted from 5_dqn_minatar_full_optimizer_all_layer.grow_full_optimizer ────
# Key changes vs DQN:
#   - no Conv expansion (no n_new_conv / new_conv_ch logic)
#   - encoder grows only (Linear input → encoder hidden)
#   - new encoder rows and matching q_head columns zeroed, trained on residual
#   - states/actions/td_targets passed as tensors directly

def grow_full_optimizer(
    q_network: Q_Network,
    states: torch.Tensor,
    actions: torch.Tensor,
    td_targets: torch.Tensor,
    n_steps: int,
    n_new_hidden: int,
    learning_rate: float,
) -> None:
    """
    Expand encoder in-place (+n_new_hidden neurons), zero-init new weights,
    then train ONLY the new parameters on the Bellman residual for n_steps.

    Simpler than DQN version: only one layer expanded (no conv block).
    New q_head columns that connect to new encoder neurons are also zero-init.
    """
    pass
