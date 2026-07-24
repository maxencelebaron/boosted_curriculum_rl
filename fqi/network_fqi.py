import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class Q_Network(nn.Module):
    """
    MLP for Q-value approximation.

    Takes only the state as input and outputs one Q-value per action.
    mushroom-rl's QApproximatorSimple handles action selection on top
    of this output.

    Architecture: state 2 -> 32 -> 32 -> 2.
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 32),
            nn.Sigmoid(),
            nn.Linear(32, 32),
            nn.Sigmoid(),
            nn.Linear(32, 2),
        )

    def forward(self, state):
        return self.net(state)


class NeuralRegressor():
    """
    Wrapper around Q_Network that exposes the fit/predict interface expected
    by mushroom-rl's QApproximatorSimple.

    The optimizer is managed here and is created lazily on the first fit() call
    so that lr comes from fit_params.
    """

    def __init__(self, **kwargs):
        self._model = Q_Network()
        self._optimizer = None
        self._loss_fn = nn.MSELoss(reduction='sum')
        self._is_fitted = False
        self.last_loss_history = []
        self.epoch_callback = None

    def predict(self, state, **kwargs):
        """Return Q-values for all actions, shape (N, n_actions)."""
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
        lr=1e-4,
        batch_size=32,
        reinit=False,
        **kwargs
    ):
        """
        Train on (state, action, target) triples.
        Loss is applied only to the output neuron of the taken action.
        """
        if reinit:
            for layer in self._model.net:
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()
            self._optimizer = None

        if self._optimizer is None:
            self._optimizer = torch.optim.Adam(self._model.parameters(), lr=lr)

        s = torch.FloatTensor(state)
        a = torch.LongTensor(action.reshape(-1))
        t = torch.FloatTensor(q)

        loader = DataLoader(
            TensorDataset(s, a, t),
            batch_size=batch_size,
            shuffle=True
        )
        self._is_fitted = True
        self.last_loss_history = []
        self._model.train()
        for epoch in range(n_epochs):
            epoch_loss = 0.
            for sb, ab, tb in loader:
                self._optimizer.zero_grad()
                q_pred = self._model(sb)                       # (batch, 2)
                q_pred_a = q_pred[torch.arange(len(ab)), ab]  # (batch,)
                # on choisit la sortie du réseau correspondant à l'action
                # effectivement prise dans le dataset
                loss = self._loss_fn(q_pred_a, tb)
                loss.backward()
                self._optimizer.step()
                epoch_loss += loss.item()
            avg_loss = epoch_loss / len(loader)
            self.last_loss_history.append(avg_loss)
            if self.epoch_callback is not None:
                self.epoch_callback()
            print(f"  epoch {epoch+1}/{n_epochs}  loss={avg_loss:.6f}")
