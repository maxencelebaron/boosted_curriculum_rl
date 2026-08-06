import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from gromo.modules.linear_growing_module import LinearGrowingModule

from mushroom_rl.utils.dataset import parse_dataset

from fqi.utils.growing_network import (
    feature_rank,
    srank,
    measure_plasticity,
    principal_angle_cosines,
    bellman_residual_correlation,
    pre_growth_optimize,
)

from fqi.utils.growing_network import (
    line_search
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
        self.encoder = LinearGrowingModule(
            in_features=445,
            out_features=hidden_size,
            post_layer_function=nn.Tanh(),
            name="encoder",
        )
        self.q_head = LinearGrowingModule(
            in_features=hidden_size,
            out_features=2,
            previous_module=self.encoder,
            name="q_head",
        )

    @property
    def encoder_size(self) -> int:
        return self.encoder.out_features

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
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model.to(device)
        self._optimizer = None
        self._loss_fn = nn.MSELoss(reduction='sum')
        self._is_fitted = False
        self.last_loss_history = []
        self.epoch_callback = None

    @property
    def _device(self):
        return next(self._model.parameters()).device

    def predict(self, state, **kwargs):
        if not self._is_fitted:
            return np.zeros((len(state), 2))
        print(f"[NeuralRegressor] device: {self._device}")
        s = torch.FloatTensor(state).to(self._device)
        with torch.no_grad():
            self._model.eval()
            return self._model(s).cpu().numpy()

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

        s = torch.FloatTensor(state).to(self._device)
        a = torch.LongTensor(action.reshape(-1)).to(self._device)
        t = torch.FloatTensor(q).to(self._device)

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
        s = torch.FloatTensor(state).to(self._device)
        with torch.no_grad():
            self._model.eval()
            return self._model.encode(s).cpu().numpy()


def grow_network_gromo(
    q_network: Q_Network,
    states: torch.Tensor,
    actions: torch.Tensor,
    td_targets: torch.Tensor,
    maximum_added_neurons: int | None = None,
    numerical_threshold: float = 1e-6,
    statistical_threshold: float = 0,
) -> torch.Tensor | None:
    """
    Grow encoder output in-place via gromo (q_head is the downstream layer).
    Returns eigenvalues_extension or None.
    """
    loss_sum = nn.MSELoss(reduction="sum")
    q_head = q_network.q_head

    _actions = actions.unsqueeze(1)

    with torch.no_grad():
        old_val_optimal = q_network(states).gather(1, _actions).squeeze()
        residu = td_targets - old_val_optimal

    q_head.init_computation()
    q_network.eval()

    q_network.zero_grad()
    old_val_optimal = q_network(states).gather(1, _actions).squeeze()
    loss = loss_sum(residu, old_val_optimal)
    loss.backward()
    q_head.update_computation()

    # Apply tiny to align delta_s with the residual gradient
    q_head.compute_optimal_updates(
        numerical_threshold=numerical_threshold,
        statistical_threshold=0,
        maximum_added_neurons=maximum_added_neurons,
        compute_delta=True,
        use_covariance=True,
        use_projection=True,
    )
    q_head.reset_computation()
    q_network.encoder.store_input = False

    _device = next(q_network.parameters()).device

    q_head.set_scaling_factor(1.0)
    if q_head.eigenvalues_extension is not None:
        q_head.sub_select_optimal_added_parameters(
            threshold=statistical_threshold,
            zeros_if_not_enough=False,
            zeros_fan_in=True,
            zeros_fan_out=False,
        )
        q_head.normalize_optimal_updates(normalization_type="weird_normalization")
    eigenvalues = q_head.eigenvalues_extension

    dataset = torch.utils.data.TensorDataset(states, td_targets)
    grow_dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=len(td_targets), shuffle=False
    )

    with torch.no_grad():
        initial_q = q_network(states).gather(1, _actions).squeeze()
    initial_loss = F.mse_loss(initial_q, td_targets).item()

    def _loss_fn(q_vals, targets):
        return F.mse_loss(q_vals.gather(1, _actions).squeeze(), targets)

    line_search(
        model=q_network,
        layer=q_head,
        dataloader=grow_dataloader,
        loss_function=_loss_fn,
        initial_loss=initial_loss,
        first_order_improvement=q_head.first_order_improvement,
        device=_device,
    )

    if q_head.scaling_factor.item() > 1e-5:
        q_head.apply_change()
        q_head.delete_update()
    else:
        q_head.delete_update()

    return eigenvalues
