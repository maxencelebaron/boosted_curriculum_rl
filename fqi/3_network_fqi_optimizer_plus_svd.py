import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from mushroom_rl.utils.dataset import parse_dataset

from fqi.utils.growing_network import (
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
            nn.Linear(2, 445),
            nn.Tanh()
        )
        self.encoder = nn.Sequential(
            nn.Linear(445, hidden_size),
            nn.Tanh()
        )
        self.q_head = nn.Linear(hidden_size, 2)

    @property
    def encoder_size(self) -> int:
        return self.encoder[0].out_features

    def encode(self, state: torch.Tensor) -> torch.Tensor:
        return self.encoder(self.h1(state.float()))

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.q_head(self.encode(state))


class NeuralRegressor:
    """
    Same fit/predict interface as network_fqi.NeuralRegressor.
    Additions: encode() for feature monitoring, _model is swappable after growth.
    """

    def __init__(self, hidden_size: int, **kwargs):
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


def grow_network_svd(
    old_net: Q_Network,
    states: torch.Tensor,
    actions: torch.Tensor,
    td_targets: torch.Tensor,
    d_a: int,
    numerical_threshold: float = 1e-6,
) -> tuple[Q_Network, np.ndarray]:
    """
    Grow encoder from old_h to old_h + d_a neurons using SVD.

    Φ_prev = U Σ V^T (output of h1, the fixed layer before the encoder)
    W_a  = Σ^(-1/2) V[:, :d_a]^T   (top-d_a right singular vectors of phi_prev)
    R[k, i] = y_k - Q*(s_k, i)  if a_k = i  (true Bellman residual)
    R[k, i] = 0 - Q*(s_k, i)    if a_k ≠ i  (target fixed at 0)
    θ_a  = R^T · U[:, :d_a] · Σ^(-1/2)  (n_actions, d_a)
    """
    old_h = old_net.q_head.in_features

    with torch.no_grad():
        # Φ_prev: output of h1 (analog of conv+flatten in DQN)
        phi_prev = old_net.h1(states.float()).cpu().numpy()

        # R[k, i]: Bellman residual for action i in sample k
        q_all_np = old_net(states).cpu().numpy()  # (B, n_actions)
        R = -q_all_np.copy()  # target=0 residuals for all actions
        td_target_np = td_targets.cpu().numpy()
        actions_np = actions.squeeze().cpu().numpy().astype(int)
        R[np.arange(len(actions_np)), actions_np] = (
            td_target_np - q_all_np[np.arange(len(actions_np)), actions_np]
        )

    # SVD of Φ_prev augmented with ones (weight + bias)
    phi_aug = np.concatenate(
        [phi_prev, np.ones((phi_prev.shape[0], 1))],
        axis=1
    )
    U, S, Vt = np.linalg.svd(phi_aug, full_matrices=False)
    k = len(S)  # number of non null singular values (can be lower than d_a)
    if k < d_a:
        print(
            f"Warning: d_a={d_a} > rank(Φ_prev)={k}, the number of non null\
            singular values, initializing {k} new neurons"
        )
    d_a_svd = min(d_a, k)  # number of initialized new neurons with SVD

    # Σ^{-1/2} with convention 0^(-1) = 0
    sigma_half_plus = np.where(S[:d_a_svd] > numerical_threshold, 1.0 / np.sqrt(S[:d_a_svd]), 0.0)

    W_a = sigma_half_plus[:, None] * Vt[:d_a_svd, :-1]
    b_a = sigma_half_plus * Vt[:d_a_svd, -1]

    # θ_a = R^T · U[:, :d_a_svd] · Σ^{-1/2}  shape: (n_actions, d_a_svd)
    theta_a = (R.T @ U[:, :d_a_svd]) * sigma_half_plus

    # Build new network and wire in the analytically initialized weights
    new_net = Q_Network(hidden_size=old_h + d_a_svd)
    with torch.no_grad():
        new_net.h1[0].weight.copy_(old_net.h1[0].weight)
        new_net.h1[0].bias.copy_(old_net.h1[0].bias)

        new_net.encoder[0].weight[:old_h, :].copy_(old_net.encoder[0].weight)
        new_net.encoder[0].weight[old_h:, :].copy_(torch.as_tensor(W_a, dtype=torch.float32))
        new_net.encoder[0].bias[:old_h].copy_(old_net.encoder[0].bias)
        new_net.encoder[0].bias[old_h:].copy_(torch.as_tensor(b_a, dtype=torch.float32))

        new_net.q_head.weight[:, :old_h].copy_(old_net.q_head.weight)
        new_net.q_head.weight[:, old_h:].copy_(torch.as_tensor(theta_a, dtype=torch.float32))
        new_net.q_head.bias.copy_(old_net.q_head.bias)

    return new_net, S[:d_a_svd]
