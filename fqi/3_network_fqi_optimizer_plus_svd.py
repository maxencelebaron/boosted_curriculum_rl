"Natural Gradient or Backpropagation & SVD"

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from mushroom_rl.utils.dataset import parse_dataset

from fqi.utils.activations import ReLUDerivativeOneAtZeroFunctorch


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
    """Growable DQN MLP"""

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
            ReLUDerivativeOneAtZeroFunctorch(),
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


def grow_network_svd(
    old_net: Q_Network,
    states: torch.Tensor,
    actions: torch.Tensor,
    td_targets: torch.Tensor,
    d_a: int,
    numerical_threshold: float = 1e-6,
) -> tuple[Q_Network, np.ndarray]:
    """
    Grow the encoder from ``old_h`` to ``old_h + d_a_svd`` neurons.

    The new neurons are initialized to approximate the Bellman residuals of
    the actions observed in the batch. We assume that the activation of the
    added neurons is locally linear around zero:

        sigma(z) ≈ z,

    corresponding to sigma(0) = 0 and sigma'(0) = 1.

    Let

        phi_k = old_net.h1(s_k),
        x_k   = [phi_k, 1],

    where the last coordinate accounts for the encoder bias, and let

        r_k = td_target_k - Q_old(s_k, a_k)

    denote the Bellman residual of the sampled action.

    For each action ``a``, we solve independently

        M[a] = argmin_m sum_{k : a_k = a} (r_k - m x_k)^2.

    The solution is computed using the truncated pseudoinverse of the feature
    matrix associated with action ``a``. Singular values smaller than
    ``numerical_threshold`` are discarded using an absolute threshold.

    Samples associated with other actions impose no artificial target on
    action ``a``. If an action is absent from the batch, its row in ``M`` is
    set to zero, following the minimum-norm convention.

    After constructing all rows of ``M``, we compute its truncated SVD:

        M ≈ U_d Sigma_d V_d^T,

    and use the balanced factorization

        theta_a      = U_d Sigma_d^(1/2),
        [W_a, b_a]   = Sigma_d^(1/2) V_d^T.

    Therefore,

        theta_a @ [W_a, b_a] ≈ M.

    The factorization is exact if ``d_a`` is at least the numerical rank of
    ``M``. Otherwise, it is the best rank-``d_a`` approximation of ``M`` in
    Frobenius norm.

    Parameters
    ----------
    old_net:
        Network to enlarge. All existing parameters are preserved.
    states:
        Batch of input states.
    actions:
        Integer action selected for each state.
    td_targets:
        Scalar TD target associated with each transition.
    d_a:
        Maximum number of neurons to add.
    numerical_threshold:
        Absolute threshold below which singular values are treated as zero.

    Returns
    -------
    new_net:
        Enlarged network with analytically initialized new neurons.
    singular_values:
        Singular values of ``M`` retained in the factorization.
    """
    if d_a < 0:
        raise ValueError(f"d_a must be nonnegative, got {d_a}")

    if numerical_threshold < 0:
        raise ValueError(
            "numerical_threshold must be nonnegative, "
            f"got {numerical_threshold}"
        )

    old_h = old_net.q_head.in_features
    n_actions = old_net.q_head.out_features

    reference_parameter = next(old_net.parameters())
    device = reference_parameter.device
    dtype = reference_parameter.dtype

    states_model = states.to(device=device, dtype=dtype)

    with torch.no_grad():
        # Fixed features preceding the encoder.
        phi_prev = (
            old_net.h1(states_model)
            .detach()
            .cpu()
            .numpy()
        )

        # Predictions of the original network for every action.
        q_all = (
            old_net(states_model)
            .detach()
            .cpu()
            .numpy()
        )

    actions_np = (
        actions.detach()
        .reshape(-1)
        .cpu()
        .numpy()
        .astype(np.int64)
    )
    td_targets_np = (
        td_targets.detach()
        .reshape(-1)
        .cpu()
        .numpy()
    )

    batch_size = phi_prev.shape[0]

    if actions_np.shape[0] != batch_size:
        raise ValueError(
            "states and actions must contain the same number of samples"
        )

    if td_targets_np.shape[0] != batch_size:
        raise ValueError(
            "states and td_targets must contain the same number of samples"
        )

    if np.any(actions_np < 0) or np.any(actions_np >= n_actions):
        raise ValueError(
            f"Action indices must belong to [0, {n_actions - 1}]"
        )

    # Augment the fixed features with a constant coordinate for the bias:
    #
    #     X[k] = [phi_prev[k], 1].
    #
    X = np.concatenate(
        [
            phi_prev,
            np.ones(
                (batch_size, 1),
                dtype=phi_prev.dtype,
            ),
        ],
        axis=1,
    )

    # Only the Q-value of the sampled action enters the Bellman residual.
    selected_q = q_all[
        np.arange(batch_size),
        actions_np,
    ]
    residuals = td_targets_np - selected_q

    # M has one regression row per action. Rows corresponding to actions
    # absent from the batch remain equal to zero.
    M = np.zeros(
        (n_actions, X.shape[1]),
        dtype=X.dtype,
    )

    for action in range(n_actions):
        action_mask = actions_np == action

        if not np.any(action_mask):
            continue

        X_action = X[action_mask]
        residuals_action = residuals[action_mask]

        # Explicit SVD of the action-specific feature matrix:
        #
        #     X_action = U_a diag(S_a) Vt_a.
        #
        U_a, S_a, Vt_a = np.linalg.svd(
            X_action,
            full_matrices=False,
        )

        # Truncated pseudoinverse with an absolute cutoff.
        S_a_plus = np.zeros_like(S_a)
        retained = S_a > numerical_threshold
        S_a_plus[retained] = 1.0 / S_a[retained]

        # Minimum-norm least-squares solution:
        #
        #     M[action]
        #         = X_action^† residuals_action
        #         = V_a diag(S_a^+) U_a^T residuals_action.
        #
        M[action] = (
            Vt_a.T
            @ (
                S_a_plus
                * (U_a.T @ residuals_action)
            )
        )

    # Factorize the residual operator M.
    U, S, Vt = np.linalg.svd(
        M,
        full_matrices=False,
    )

    numerical_rank = int(
        np.sum(S > numerical_threshold)
    )
    d_a_svd = min(d_a, numerical_rank)

    if d_a > numerical_rank:
        print(
            f"Warning: d_a={d_a} > rank(M)={numerical_rank}; "
            f"initializing {d_a_svd} new neurons."
        )

    # Balanced factorization:
    #
    #     M_d = theta_a @ [W_a, b_a].
    #
    sqrt_S = np.sqrt(S[:d_a_svd])

    theta_a = (
        U[:, :d_a_svd]
        * sqrt_S[None, :]
    )

    encoder_augmented = (
        sqrt_S[:, None]
        * Vt[:d_a_svd, :]
    )

    W_a = encoder_augmented[:, :-1]  # transpos
    b_a = encoder_augmented[:, -1]

    # Construct the enlarged network.
    if hasattr(old_net, "new_with_hidden_size"):
        new_net = old_net.new_with_hidden_size(
            old_h + d_a_svd
        )
    else:
        new_net = Q_Network(
            hidden_size=old_h + d_a_svd
        )

    new_net.to(device=device, dtype=dtype)

    with torch.no_grad():
        # Preserve the fixed feature extractor.
        new_net.h1.load_state_dict(
            old_net.h1.state_dict()
        )

        # Preserve the existing encoder neurons.
        new_net.encoder[0].weight[:old_h].copy_(
            old_net.encoder[0].weight
        )
        new_net.encoder[0].bias[:old_h].copy_(
            old_net.encoder[0].bias
        )

        # Initialize the added encoder neurons.
        new_net.encoder[0].weight[old_h:].copy_(
            torch.as_tensor(
                W_a,
                device=device,
                dtype=dtype,
            )
        )
        new_net.encoder[0].bias[old_h:].copy_(
            torch.as_tensor(
                b_a,
                device=device,
                dtype=dtype,
            )
        )

        # Preserve the existing output weights.
        new_net.q_head.weight[:, :old_h].copy_(
            old_net.q_head.weight
        )

        # Connect the added neurons to the output.
        new_net.q_head.weight[:, old_h:].copy_(
            torch.as_tensor(
                theta_a,
                device=device,
                dtype=dtype,
            )
        )

        # Preserve the existing output bias.
        new_net.q_head.bias.copy_(
            old_net.q_head.bias
        )

    return new_net, S[:d_a_svd]
