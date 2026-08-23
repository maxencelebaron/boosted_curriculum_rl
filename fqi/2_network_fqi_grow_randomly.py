"Natural Gradient or Backpropagation or anything & random and random - 0 initialization"

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from mushroom_rl.utils.dataset import parse_dataset


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


class DQNNetwork(nn.Module):
    """Growable DQN MLP with dimensions inferred from the environment."""

    def __init__(
        self,
        input_shape,
        output_shape,
        hidden_size: int,
        first_hidden_size: int = 128,
        second_hidden_size: int = 128,
        **kwargs,
    ):
        del kwargs
        super().__init__()
        self.input_shape = tuple(input_shape)
        self.output_shape = tuple(output_shape)
        self.first_hidden_size = first_hidden_size
        self.second_hidden_size = second_hidden_size
        self.h1 = nn.Sequential(
            nn.Linear(self.input_shape[0], first_hidden_size),
            nn.ReLU(),
            nn.Linear(first_hidden_size, second_hidden_size),
            nn.ReLU(),
        )
        self.encoder = nn.Sequential(
            nn.Linear(second_hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.q_head = nn.Linear(hidden_size, self.output_shape[0])

    @property
    def encoder_size(self) -> int:
        return self.encoder[0].out_features

    def encode(self, state: torch.Tensor) -> torch.Tensor:
        return self.encoder(self.h1(state.float()))

    def forward(self, state: torch.Tensor, action=None) -> torch.Tensor:
        q = self.q_head(self.encode(state))
        if action is not None:
            q = q.gather(1, action.long().reshape(-1, 1)).squeeze(1)
        return q

    def new_with_hidden_size(self, hidden_size: int):
        return type(self)(
            self.input_shape,
            self.output_shape,
            hidden_size=hidden_size,
            first_hidden_size=self.first_hidden_size,
            second_hidden_size=self.second_hidden_size,
        )


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


def grow_network(
    old_net: Q_Network,
    new_hidden: int,
    zero_fan_out: bool = False,
) -> Q_Network:
    """
    Expand encoder from old_h to new_hidden.
    When zero_fan_out is True, connections from the new neurons to q_head
    are initialized to zero, making the growth function-preserving.
    The NeuralRegressor optimizer must be reset after calling this.
    """
    old_h = old_net.encoder[0].out_features

    if hasattr(old_net, "new_with_hidden_size"):
        new_net = old_net.new_with_hidden_size(new_hidden)
    else:
        new_net = Q_Network(hidden_size=new_hidden)
    new_net.to(next(old_net.parameters()).device)

    with torch.no_grad():
        new_net.h1.load_state_dict(old_net.h1.state_dict())

        # encoder: copy old rows, new rows keep random init
        new_net.encoder[0].weight[:old_h].copy_(old_net.encoder[0].weight)
        new_net.encoder[0].bias[:old_h].copy_(old_net.encoder[0].bias)

        # q_head: copy old columns; optionally zero the new fan-out columns
        new_net.q_head.weight[:, :old_h].copy_(old_net.q_head.weight)
        if zero_fan_out:
            new_net.q_head.weight[:, old_h:].zero_()
        new_net.q_head.bias.copy_(old_net.q_head.bias)

    return new_net
