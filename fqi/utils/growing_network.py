import copy
from collections.abc import Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import logging
import numpy as np
from torchmetrics import Metric

from gromo.containers.growing_container import GrowingModule
from gromo.utils.utils import global_device
from gromo.utils.training_utils import (
    AverageMeter,
    DummyMetric,
    enumerate_dataloader
)
from .natural_gradient import KFAC, KFACConfig


def feature_rank(features: np.ndarray, epsilon: float = 0.01):
    """
    Number of singular values of (1/√n)·φ(X) above epsilon where φ(X) is
    the feature matrix.
    """
    phi = features / np.sqrt(features.shape[0])
    singular_values = np.linalg.svd(phi, compute_uv=False)
    rank = int(np.sum(singular_values > epsilon))
    return rank, singular_values


def srank(singular_values: np.ndarray, delta: float = 0.01) -> int:
    """
    Effective rank: smallest k such that the top-k singular values account
    for at least (1 - delta) of the total singular value mass.

    Operates directly on pre-computed singular values (sorted descending),
    so no extra SVD is needed when called after feature_rank().
    """
    total = singular_values.sum()
    if total == 0:
        return 0
    cumulative_ratio = np.cumsum(singular_values) / total
    k = int(np.searchsorted(cumulative_ratio, 1.0 - delta, side="left")) + 1
    return min(k, len(singular_values))


def measure_plasticity(
    network: nn.Module,
    x: torch.Tensor,
    optimizer_class=optim.Adam,
    optimizer_params: dict = None,
    n_steps: int = 2_000,
    n_tasks: int = 10,
) -> float:
    """
    Plasticity P(θ) = mean(b_k) - mean(l_k) over n_tasks random probe tasks.

    For each task k:
        - sample random init ω₀
        - compute targets y = a + sin(1e5 · f(x; ω₀))  where a = E[f(x; θ_t)]
        - b_k = Var(y)  (target variance, measures task difficulty)
        - fine-tune a probe copy θ_probe of θ_t for n_steps on MSE(f(x; θ_probe), y)
        - l_k = MSE(f(x; θ_probe), y) after fine-tuning  (residual loss)

    Higher P -> more plastic (can reduce loss further relative to task difficulty).
    """
    if optimizer_params is None:
        optimizer_params = {"lr": 1e-4}

    # anchor: mean output of current network, no grad  (n_actions,)
    with torch.no_grad():
        a = network(x).mean(dim=0)  # (n_actions,)

    b_list, l_list = [], []
    for _ in range(n_tasks):
        # ω_0 sampled from the same distribution as θ_0 via reset_parameters()
        omega_0 = copy.deepcopy(network)
        for m in omega_0.modules():
            if hasattr(m, 'reset_parameters'):
                m.reset_parameters()

        # targets y = a + sin(1e5 · f(x; ω_0))
        with torch.no_grad():
            y = a + torch.sin(1e5 * omega_0(x))  # (n_samples, n_actions)

        b_k = y.var().item()
        b_list.append(b_k)

        # fine-tune a copy of θ_t on this task
        probe = copy.deepcopy(network)
        opt = optimizer_class(probe.parameters(), **optimizer_params)
        for _ in range(n_steps):
            loss = F.mse_loss(probe(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()

        with torch.no_grad():
            l_k = F.mse_loss(probe(x), y).item()
        l_list.append(l_k)

    return float(np.mean(b_list) - np.mean(l_list))


def principal_angle_cosines(phi: np.ndarray, phi_new: np.ndarray) -> np.ndarray:
    """
    Cosines of principal angles between column spaces of phi and phi_new.
    Values near 1 = aligned (new neurons replicate existing features).
    Values near 0 = orthogonal (new neurons learn independent features).
    """
    U, _, _ = np.linalg.svd(phi, full_matrices=False)
    V, _, _ = np.linalg.svd(phi_new, full_matrices=False)
    return np.linalg.svd(U.T @ V, compute_uv=False)


def bellman_residual_correlation(phi_new: np.ndarray, R: np.ndarray) -> float:
    """
    Correlation between new features and accumulated Bellman residuals.
    High value = new neurons capture unexplained residual.
    """
    return float(np.linalg.norm(phi_new.T @ R, 'fro'))


def normalized_bellman_residual_correlation(
    phi_new: np.ndarray,
    residual: np.ndarray,
    epsilon: float = 1e-12,
) -> float:
    """Scale-invariant alignment between new features and TD residuals."""
    if epsilon <= 0:
        raise ValueError("epsilon must be strictly positive")
    numerator = np.linalg.norm(phi_new.T @ residual, "fro")
    denominator = (
        np.linalg.norm(phi_new, "fro")
        * np.linalg.norm(residual, "fro")
        + epsilon
    )
    return float(numerator / denominator)


def compute_metrics(
    model: nn.Module,
    monitoring_states: torch.Tensor,
    monitoring_actions: torch.Tensor,
    monitoring_targets: torch.Tensor,
    feature_split: int = 0,
) -> dict:
    """
    Compute feature rank, srank, old/new neuron split metrics.

    monitoring_targets should be fresh TD targets for the monitoring states,
    recomputed from the current network at each call.

    Returns a dict of scalar metrics. Keys present unconditionally:
        rank, rank_ratio, srank, srank_ratio.
    Keys present only when feature_split > 0:
        rank_old, rank_new, rank_ratio_old, rank_ratio_new,
        srank_old, srank_new, srank_ratio_old, srank_ratio_new,
        angles_min, angles_max, angles_mean, brc.
    """
    device = next(model.parameters()).device
    mon_states = monitoring_states.to(device)
    mon_actions = monitoring_actions.to(device)
    mon_targets = monitoring_targets.to(device)

    was_training = model.training
    model.eval()
    with torch.no_grad():
        features = model.encode(mon_states).cpu().numpy()
        q_pred = (
            model(mon_states)
            .gather(1, mon_actions.unsqueeze(1))
            .squeeze()
            .cpu()
            .numpy()
        )
    bellman_residual = float(np.mean((mon_targets.cpu().numpy() - q_pred) ** 2))

    rank, svs = feature_rank(features)
    feature_dim = features.shape[1]
    sr = srank(svs)

    result = {
        "bellman_residual": bellman_residual,
        "rank": rank,
        "rank_ratio": rank / feature_dim,
        "srank": sr,
        "srank_ratio": sr / feature_dim,
    }

    if 0 < feature_split < feature_dim:
        phi = features[:, :feature_split]
        phi_new = features[:, feature_split:]

        rank_old, svs_old = feature_rank(phi)
        rank_new, svs_new = feature_rank(phi_new)
        sr_old = srank(svs_old)
        sr_new = srank(svs_new)
        cosines = principal_angle_cosines(phi, phi_new)

        with torch.no_grad():
            q_brc = model(mon_states).gather(1, mon_actions.unsqueeze(1)).squeeze().cpu().numpy()
        r_brc = (mon_targets.cpu().numpy() - q_brc).reshape(-1, 1)
        brc = bellman_residual_correlation(phi_new, r_brc)
        brc_normalized = normalized_bellman_residual_correlation(
            phi_new, r_brc
        )

        result.update({
            "rank_old": rank_old,
            "rank_new": rank_new,
            "rank_ratio_old": rank_old / phi.shape[1],
            "rank_ratio_new": rank_new / phi_new.shape[1],
            "srank_old": sr_old,
            "srank_new": sr_new,
            "srank_ratio_old": sr_old / phi.shape[1],
            "srank_ratio_new": sr_new / phi_new.shape[1],
            "angles_min": float(cosines.min()),
            "angles_max": float(cosines.max()),
            "angles_mean": float(cosines.mean()),
            "brc": brc,
            "brc_normalized": brc_normalized,
        })

    model.train(was_training)
    return result


def pre_growth_optimize(
    network: nn.Module,
    states: torch.Tensor,
    actions: torch.Tensor,
    td_targets: torch.Tensor,
    optimizer,
    n_steps: int,
    use_natural_gradient: bool = False,
    natural_gradient_damping: float = 1e-5,
    natural_gradient_noise_variance: float = 1.0,
    natural_gradient_eigenvalue_threshold: float = 1e-7,
) -> tuple[float, float, list[float]]:
    """
    Optimize the Bellman loss on a fixed batch before deciding whether to grow.

    Optimizes L = (1/n)||T^π Q - Φ(W)·θ^T||² jointly over all network weights.

    By default, regular backpropagation through ``optimizer`` is used.  When
    ``use_natural_gradient`` is true, each step applies the block-diagonal K-FAC
    update from equations (23)--(27) of the accompanying Tiny K-FAC document.
    The learning rate is read from the optimizer's first parameter group. The
    A small positive damping enables the stable Cholesky solve. Passing zero
    selects the theoretical, thresholded Moore--Penrose path.

    Returns (initial_loss, final_loss, loss_history). The network and optimizer
    are updated in place, so W* and θ* are retained for the subsequent growth step.
    """
    if n_steps < 1:
        raise ValueError("n_steps must be at least 1.")
    if use_natural_gradient and not optimizer.param_groups:
        raise ValueError("The optimizer must contain a parameter group.")

    _actions = actions.reshape(-1, 1)
    kfac = None
    if use_natural_gradient:
        kfac = KFAC(
            network,
            KFACConfig(
                noise_variance=natural_gradient_noise_variance,
                damping=natural_gradient_damping,
                eigenvalue_threshold=natural_gradient_eigenvalue_threshold,
            ),
        )
    loss_history = []
    loss = None
    for _ in range(n_steps):
        optimizer.zero_grad()
        if use_natural_gradient:
            assert kfac is not None
            old_val, loss, updates = kfac.compute_updates(
                states, actions, td_targets
            )
            kfac.apply_updates_(updates, step_size=optimizer.param_groups[0]["lr"])
        else:
            phi = network.encode(states)
            old_val = network.q_head(phi).gather(1, _actions).squeeze()
            loss = F.mse_loss(td_targets, old_val)
            loss.backward()
            optimizer.step()
        reported_loss = F.mse_loss(td_targets, old_val)
        loss_history.append(reported_loss.item())

    with torch.no_grad():
        final_values = network(states).gather(1, _actions).squeeze()
        final_loss = F.mse_loss(td_targets, final_values).item()
    return loss_history[0], final_loss, loss_history


@torch.no_grad()
def evaluate_layer(
    model: nn.Module,
    layer: GrowingModule,
    dataloader: torch.utils.data.DataLoader,
    loss_function: nn.Module | Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    metrics: Metric | None = None,
    batch_limit: int | None = None,
    dataloader_seed: int | None = None,
    device: torch.device = torch.device("cpu"),
) -> tuple[float, float]:
    """
    Evaluate the extended forward pass of the growing layer in the context of
    the full model, using a dataloader of (observations, targets) pairs.

    Performs the chain:
        h1=linear → encoder.extended_forward → q_head.extended_forward

    Parameters
    ----------
    model : nn.Module
        Full Q-network (must expose .h1, .encoder, .q_head).
    layer : GrowingModule
        The growing layer being tested. Not called directly here — the
        scaling_factor set on it is picked up by the extended_forward chain.
    dataloader : torch.utils.data.DataLoader
        Yields (observations, targets) batches.
    loss_function : nn.Module | Callable
        Applied to (q_vals, targets). Must have reduction="mean".
    metrics : Metric | None, optional
        Auxiliary metric. Default is None.
    batch_limit : int | None, optional
        Maximum number of batches. None = no limit.
    dataloader_seed : int | None, optional
        Seed for the dataloader generator.
    device : torch.device, optional
        Device to use. Default is cpu.

    Returns
    -------
    tuple[float, float]
        (average_loss, metrics_value).
    """
    assert (
        not isinstance(loss_function, nn.Module) or loss_function.reduction == "mean"
    ), "The loss function should be averaged over the batch"

    loss_meter = AverageMeter()
    if metrics is None:
        metrics = DummyMetric()
    else:
        metrics.reset()
        metrics = metrics.to(device)

    model.eval()
    for _, (x, y) in enumerate_dataloader(
        dataloader, dataloader_seed=dataloader_seed, batch_limit=batch_limit
    ):
        x, y = x.to(device), y.to(device)
        flat_x = model.h1(x.float())
        main_enc, ext_enc = model.encoder.extended_forward(flat_x)
        y_pred, _ = model.q_head.extended_forward(main_enc, x_ext=ext_enc)
        loss = loss_function(y_pred, y)
        loss_meter.update(loss, x.size(0))
        metrics.update(y_pred, y)

    return loss_meter.compute().item(), metrics.compute().item()


def line_search(
    model: nn.Module,
    layer: GrowingModule,
    dataloader: torch.utils.data.DataLoader,
    dataloader_seed: int | None = None,
    loss_function: nn.Module = nn.MSELoss(reduction="sum"),
    aux_loss_function: Metric | None = None,
    batch_limit: int = -1,
    initial_loss: float | None = None,
    initial_aux_loss: float | None = None,
    first_order_improvement: float | torch.Tensor = 1,
    alpha: float = 0.1,
    beta: float = 0.5,
    t0: float | None = None,
    max_gamma: float | None = None,
    extended_search: bool = False,
    max_iter: int = 20,
    epsilon: float = 1e-7,
    verbose: bool = False,
    device: torch.device | None = None,
) -> tuple[float, float, float, list[float], list[float], list[float]]:
    """
    Perform line search to find optimal scaling factor for the currently updated layer.

    This function performs a backtracking line search with Armijo condition to find
    the optimal scaling factor (gamma) for the newly added neurons in the currently
    updated layer. The search operates on the square root of gamma (t) for numerical
    stability, where gamma = t^2.

    Parameters
    ----------
    layer : GrowingModule
        An updated layer that has a scaling_factor attribute.
    dataloader : torch.utils.data.DataLoader
        DataLoader to evaluate the layer on during the line search.
    dataloader_seed : int | None, optional
        Seed for shuffling the dataloader during evaluation. If None, no reseeding is done.
        Default is None.
    loss_function : nn.Module, optional
        Loss function to minimize. Should use reduction="sum".
        Default is nn.MSELoss(reduction="sum").
    aux_loss_function : Metric | None, optional
        A Metric instance to track auxiliary metrics (e.g., accuracy).
        Default is None.
    batch_limit : int, optional
        Maximum number of batches to use for evaluation. Use -1 for no limit.
        Default is -1.
    initial_loss : float | None, optional
        Initial loss at gamma=0. If None, it will be computed.
        Default is None.
    initial_aux_loss : float | None, optional
        Initial auxiliary loss at gamma=0. If None, will be replaced by 0.
        Default is None.
    first_order_improvement : float | torch.Tensor, optional
        Expected first-order improvement in loss. Used to initialize the search
        and for the Armijo condition.
        Default is 1.
    alpha : float, optional
        Armijo condition parameter. The sufficient decrease condition is:
        loss < initial_loss - alpha * gamma * first_order_improvement.
        Default is 0.1.
    beta : float, optional
        Step size reduction factor. The actual factor applied is sqrt(beta) since
        we work with t = sqrt(gamma). Typical values: 0.5 for aggressive, 0.8 for conservative.
        Default is 0.5.
    t0 : float | None, optional
        Initial value for t (sqrt of initial gamma). If None, computed as:
        t0 = sqrt(2 * initial_loss / first_order_improvement).
        Default is None.
    max_gamma : float | None, optional
        Maximum allowable gamma value. If None, no maximum is enforced.
    extended_search : bool, optional
        If True, extends the search in both directions (increase and decrease gamma)
        to find a local minimum. If False, only performs backtracking.
        Default is True.
    max_iter : int, optional
        Maximum number of iterations for the line search.
        Default is 100.
    epsilon : float, optional
        Minimum value for t (sqrt of gamma) below which search stops.
        Actual threshold is sqrt(epsilon).
        Default is 1e-7.
    verbose : bool, optional
        If True, prints detailed information about each tested gamma value.
        Default is False.
    device : torch.device | None, optional
        Device to perform computations on. If None, uses global_device().
        Default is None.

    Returns
    -------
    tuple[float, float, float, list[float], list[float], list[float]]
        A tuple containing:
        - gamma (float): The optimal scaling factor (t^2).
        - loss (float): The loss at the optimal gamma.
        - aux_loss (float): The auxiliary loss at the optimal gamma.
        - gammas (list[float]): List of all tested gamma values.
        - losses (list[float]): List of losses corresponding to tested gammas.
        - aux_losses (list[float]): List of auxiliary losses corresponding to tested gammas.

    Raises
    ------
    AssertionError
        If layer is None.

    Notes
    -----
    - The function works with t = sqrt(gamma) for numerical stability.
    - The Armijo condition ensures sufficient decrease in the loss function.
    - The extended_search option allows finding better local minima by exploring
      beyond the first acceptable point.
    - The scaling_factor of the currently_updated_layer is set to the optimal
      value (t, not gamma) at the end of the function.

    Examples
    --------
    >>> from tools.metrics import TopKAccuracy
    >>> gamma, loss, aux_loss, gammas, losses, aux_losses = line_search(
    ...     layer=growing_layer,
    ...     dataloader=train_loader,
    ...     loss_function=nn.CrossEntropyLoss(reduction="sum"),
    ...     aux_loss_function=TopKAccuracy(k=1),
    ...     initial_loss=100.0,
    ...     first_order_improvement=5.0,
    ...     alpha=0.1,
    ...     beta=0.5,
    ...     verbose=True
    ... )
    """
    logger = logging.getLogger(__name__)
    assert layer is not None, "No currently updated layer"

    if device is None:
        device = global_device()

    gammas = []
    losses = []
    aux_losses = []
    beta = np.sqrt(beta)
    epsilon = np.sqrt(epsilon)
    if isinstance(first_order_improvement, torch.Tensor):
        first_order_improvement = first_order_improvement.item()
    if isinstance(initial_loss, torch.Tensor):
        initial_loss = initial_loss.item()

    def test_gamma(sqrt_gamma):
        layer.set_scaling_factor(sqrt_gamma)
        loss, aux_loss = evaluate_layer(
            model=model,
            layer=layer,
            dataloader=dataloader,
            dataloader_seed=dataloader_seed,
            loss_function=loss_function,
            metrics=aux_loss_function,
            batch_limit=batch_limit,
            device=device,
        )
        gammas.append(sqrt_gamma**2)
        losses.append(loss)
        aux_losses.append(aux_loss)
        if verbose:
            logger.info(
                f"gamma n° {len(gammas)}: {sqrt_gamma**2:.3e} -> Loss: {loss:.6e} (aux_loss: {aux_loss * 100:.2f}%)"
            )
        return loss, aux_loss

    if initial_loss is None:
        logger.warning("Initial loss is not provided, computing it")
        initial_loss, initial_aux_loss = test_gamma(0.0)
        logger.info(
            f"Initial loss: {initial_loss:.3e} (aux_loss: {initial_aux_loss * 100:.2f}%)"
        )
    gammas.append(0.0)
    losses.append(initial_loss)
    initial_aux_loss = initial_aux_loss if initial_aux_loss is not None else 0.0
    aux_losses.append(initial_aux_loss)
    if verbose:
        logger.info(
            f"gamma n° {len(gammas)}: {0.0:.3e} -> Loss: {initial_loss:.3e} (aux_loss: {initial_aux_loss * 100:.2f}%)"
        )

    def under_bound(sqrt_gamma: float, loss: float):
        return loss < initial_loss - alpha * sqrt_gamma**2 * first_order_improvement

    # gamma = t ** 2
    if t0 is None:
        t = np.sqrt(2 * (initial_loss / first_order_improvement))
    else:
        t = np.sqrt(t0)
    if max_gamma is not None:
        max_t = np.sqrt(max_gamma)
        t = min(t, max_t)
    l0, l0_aux = test_gamma(t)
    l1, l1_aux = l0, l0_aux
    i = 0
    if under_bound(t, l0):
        if extended_search:
            go = t / beta < max_t
            while go:
                l0, l0_aux = l1, l1_aux
                t /= beta
                l1, l1_aux = test_gamma(t)
                go = l1 < l0 and i < max_iter and t < max_t
                i += 1
            t *= beta
        layer.set_scaling_factor(t)
    else:
        go = True
        while go:
            l0, l0_aux = l1, l1_aux
            t *= beta
            l1, l1_aux = test_gamma(t)
            go = (
                ((not under_bound(t, l1)) or (l1 < l0 and extended_search))
                and i < max_iter
                and t > epsilon
            )
            i += 1
        t /= beta
        layer.set_scaling_factor(t)

    # select best gamma found
    min_loss = float("inf")
    best_idx = -1
    for idx, loss in enumerate(losses):
        if loss < min_loss:
            min_loss = loss
            best_idx = idx
    t = np.sqrt(gammas[best_idx])
    layer.set_scaling_factor(t)
    l0 = losses[best_idx]
    l0_aux = aux_losses[best_idx]

    if verbose:
        logger.info(
            f"Line search completed: optimal gamma = {t**2:.3e}, loss = {l0:.3e}, aux_loss = {l0_aux * 100:.2f}%"
        )

    return t**2, l0, l0_aux, gammas, losses, aux_losses
