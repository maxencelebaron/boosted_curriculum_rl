"""K-FAC natural gradient for the Tiny Q-networks."""

from collections.abc import Iterable
from dataclasses import dataclass

import torch
from torch import nn


def trainable_linear_layers(network: nn.Module) -> list[nn.Linear]:
    """Return trainable affine layers in module/forward order."""
    return [
        module for module in network.modules()
        if isinstance(module, nn.Linear) and module.weight.requires_grad
    ]


def register_linear_hooks(
    layers: Iterable[nn.Linear],
) -> tuple[
    dict[nn.Linear, torch.Tensor],
    dict[nn.Linear, torch.Tensor],
    list[torch.utils.hooks.RemovableHandle],
]:
    """Capture layer inputs and preactivations during one forward pass."""
    inputs: dict[nn.Linear, torch.Tensor] = {}
    outputs: dict[nn.Linear, torch.Tensor] = {}
    handles = []

    def capture(layer: nn.Linear, args: tuple[torch.Tensor, ...], output: torch.Tensor):
        inputs[layer] = args[0]
        outputs[layer] = output

    for layer in layers:
        handles.append(layer.register_forward_hook(capture))
    return inputs, outputs, handles


def remove_hooks(handles: Iterable[torch.utils.hooks.RemovableHandle]) -> None:
    """Remove PyTorch hooks."""
    for handle in handles:
        handle.remove()


def output_sensitivities(
    selected_predictions: torch.Tensor,
    layer_outputs: Iterable[torch.Tensor],
) -> list[torch.Tensor]:
    r"""Compute ``r_l = grad_{a_l} Q(s,a)`` for every sample and layer."""
    sensitivities = torch.autograd.grad(
        selected_predictions.sum(),
        list(layer_outputs),
        retain_graph=True,
        create_graph=False,
        allow_unused=True,
    )
    if any(sensitivity is None for sensitivity in sensitivities):
        raise RuntimeError("A selected linear layer does not affect the Q predictions.")
    return list(sensitivities)


def append_bias_coordinate(inputs: torch.Tensor, has_bias: bool) -> torch.Tensor:
    """Append a homogeneous coordinate for an affine-layer bias."""
    if not has_bias:
        return inputs
    ones = torch.ones(*inputs.shape[:-1], 1, device=inputs.device, dtype=inputs.dtype)
    return torch.cat((inputs, ones), dim=-1)


def kfac_factors(
    layer_inputs: torch.Tensor,
    sensitivities: torch.Tensor,
    noise_variance: float = 1.0,
    accumulation_dtype: torch.dtype = torch.float32,
    check_finite: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Estimate ``S_{l-1}`` and ``Gamma_l`` from equation (26)."""
    if noise_variance <= 0:
        raise ValueError("noise_variance must be strictly positive.")
    layer_inputs = layer_inputs.detach().to(dtype=accumulation_dtype)
    sensitivities = sensitivities.detach().to(dtype=accumulation_dtype)
    batch_size = layer_inputs.shape[0]
    s_factor = 2 * layer_inputs.T @ layer_inputs / batch_size
    gamma_factor = sensitivities.T @ sensitivities / (
        batch_size * noise_variance
    )
    s_factor = (s_factor + s_factor.T) / 2
    gamma_factor = (gamma_factor + gamma_factor.T) / 2
    if check_finite and (
        not torch.isfinite(s_factor).all()
        or not torch.isfinite(gamma_factor).all()
    ):
        raise FloatingPointError("A K-FAC factor contains NaN or Inf values.")
    return s_factor, gamma_factor


def solve_psd(
    matrix: torch.Tensor,
    rhs: torch.Tensor,
    damping: float,
    eigenvalue_threshold: float = 1e-7,
) -> torch.Tensor:
    """Solve a positive-semidefinite system with a spectral fallback."""
    if damping < 0:
        raise ValueError("damping must be non-negative.")
    if eigenvalue_threshold <= 0:
        raise ValueError("eigenvalue_threshold must be strictly positive.")

    matrix = (matrix + matrix.T) / 2
    if damping > 0:
        matrix = matrix + damping * torch.eye(
            matrix.shape[0], device=matrix.device, dtype=matrix.dtype
        )
        cholesky, info = torch.linalg.cholesky_ex(matrix)
        if bool(torch.all(info == 0)):
            return torch.cholesky_solve(rhs, cholesky)

    eigenvalues, eigenvectors = torch.linalg.eigh(matrix)
    projected_rhs = eigenvectors.T @ rhs
    if damping == 0:
        # The theoretical path uses a thresholded Moore--Penrose inverse.
        inverse_eigenvalues = torch.where(
            eigenvalues > eigenvalue_threshold,
            eigenvalues.reciprocal(),
            torch.zeros_like(eigenvalues),
        )
    else:
        # Damped fallback remains strictly invertible despite round-off errors.
        inverse_eigenvalues = eigenvalues.clamp_min(eigenvalue_threshold).reciprocal()
    return eigenvectors @ (inverse_eigenvalues.unsqueeze(1) * projected_rhs)


def precondition_kfac_matrix(
    matrix: torch.Tensor,
    gamma_factor: torch.Tensor,
    s_factor: torch.Tensor,
    damping: float = 1e-5,
    eigenvalue_threshold: float = 1e-7,
) -> torch.Tensor:
    r"""Compute ``Gamma^-1 matrix S^-1`` without a Kronecker matrix.

    For positive damping, Cholesky solves avoid forming either inverse. With
    zero damping, Moore--Penrose pseudoinverses preserve equation (27), even
    when a factor is singular.
    """
    left_solved = solve_psd(
        gamma_factor, matrix, damping, eigenvalue_threshold
    )
    return solve_psd(
        s_factor, left_solved.T, damping, eigenvalue_threshold
    ).T


@torch.no_grad()
def apply_kfac_updates_(
    updates: dict[nn.Parameter, torch.Tensor],
    step_size: float,
) -> None:
    """Apply previously computed descent directions to their parameters."""
    if step_size < 0:
        raise ValueError("step_size must be non-negative.")
    for parameter, update in updates.items():
        parameter.add_(update.to(device=parameter.device, dtype=parameter.dtype), alpha=step_size)


def compute_kfac_updates(
    network: nn.Module,
    states: torch.Tensor,
    actions: torch.Tensor,
    targets: torch.Tensor,
    noise_variance: float = 1.0,
    damping: float = 1e-5,
    accumulation_dtype: torch.dtype = torch.float32,
    eigenvalue_threshold: float = 1e-7,
    check_finite: bool = True,
    layers: Iterable[nn.Linear] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[nn.Parameter, torch.Tensor]]:
    """Compute, but do not apply, all block-diagonal K-FAC updates.

    Curvature vectors only build ``Gamma``. The training gradient is computed
    independently by backpropagating the Gaussian negative log-likelihood.
    """
    if noise_variance <= 0:
        raise ValueError("noise_variance must be strictly positive.")
    layers = trainable_linear_layers(network) if layers is None else list(layers)
    if not layers:
        raise ValueError("K-FAC requires at least one trainable linear layer.")
    if check_finite:
        for name, tensor in (("states", states), ("targets", targets)):
            if not torch.isfinite(tensor).all():
                raise FloatingPointError(
                    f"K-FAC input '{name}' contains NaN or Inf values."
                )
    layer_names = {
        layer: name for name, layer in network.named_modules()
        if isinstance(layer, nn.Linear)
    }

    inputs, outputs, handles = register_linear_hooks(layers)
    try:
        q_values = network(states)
    finally:
        remove_hooks(handles)
    layers = [layer for layer in layers if layer in outputs]
    if not layers:
        raise RuntimeError("No trainable linear layer participated in the forward pass.")
    predictions = q_values.gather(1, actions.reshape(-1, 1)).squeeze(1)
    if check_finite and not torch.isfinite(predictions).all():
        raise FloatingPointError(
            "K-FAC selected predictions contain NaN or Inf values."
        )
    sensitivities = output_sensitivities(
        predictions, [outputs[layer] for layer in layers]
    )
    loss = torch.nn.functional.mse_loss(
        predictions, targets.reshape(-1)
    ) / noise_variance
    if check_finite and not torch.isfinite(loss):
        raise FloatingPointError("The K-FAC training loss is NaN or Inf.")
    network.zero_grad(set_to_none=True)
    loss.backward()

    updates: dict[nn.Parameter, torch.Tensor] = {}
    for layer, sensitivity in zip(layers, sensitivities):
        extended_inputs = append_bias_coordinate(
            inputs[layer].detach(), has_bias=layer.bias is not None
        ).to(dtype=accumulation_dtype)
        s_factor, gamma_factor = kfac_factors(
            extended_inputs,
            sensitivity.detach(),
            noise_variance=noise_variance,
            accumulation_dtype=accumulation_dtype,
            check_finite=check_finite,
        )
        if layer.weight.grad is None or (layer.bias is not None and layer.bias.grad is None):
            raise RuntimeError("A K-FAC layer has no training-loss gradient.")
        gradient = layer.weight.grad.detach().to(dtype=accumulation_dtype)
        if layer.bias is not None:
            gradient = torch.cat(
                (gradient, layer.bias.grad.detach().to(dtype=accumulation_dtype)[:, None]),
                dim=1,
            )
        if check_finite and not torch.isfinite(gradient).all():
            layer_name = layer_names.get(layer, "<unnamed linear layer>")
            nan_count = int(torch.isnan(gradient).sum().item())
            inf_count = int(torch.isinf(gradient).sum().item())
            raise FloatingPointError(
                f"K-FAC gradient for layer '{layer_name}' contains "
                f"{nan_count} NaN and {inf_count} Inf values."
            )
        # compute_updates returns a descent direction, hence the leading minus.
        update = -precondition_kfac_matrix(
            gradient,
            gamma_factor,
            s_factor,
            damping=damping,
            eigenvalue_threshold=eigenvalue_threshold,
        )
        if check_finite and not torch.isfinite(update).all():
            layer_name = layer_names.get(layer, "<unnamed linear layer>")
            raise FloatingPointError(
                f"K-FAC update for layer '{layer_name}' contains NaN or Inf "
                "values."
            )
        weight_columns = layer.weight.shape[1]
        updates[layer.weight] = update[:, :weight_columns]
        if layer.bias is not None:
            updates[layer.bias] = update[:, weight_columns]

    return predictions.detach(), loss.detach(), updates


@dataclass(frozen=True)
class KFACConfig:
    """Configuration of the K-FAC preconditioner."""

    noise_variance: float = 1.0
    damping: float = 1e-5
    accumulation_dtype: torch.dtype = torch.float32
    eigenvalue_threshold: float = 1e-7
    check_finite: bool = True

    def __post_init__(self) -> None:
        if self.noise_variance <= 0:
            raise ValueError("noise_variance must be strictly positive.")
        if self.damping < 0:
            raise ValueError("damping must be non-negative.")
        if self.eigenvalue_threshold <= 0:
            raise ValueError("eigenvalue_threshold must be strictly positive.")


class KFAC:
    """Lightweight K-FAC orchestrator for Q-networks.

    Unlike a stateful training optimizer, this class does not retain factors,
    decompositions, gradients, or batch activations between calls. It only
    centralizes layer selection and configuration around the independently
    testable K-FAC functions in this module.
    """

    def __init__(
        self,
        network: nn.Module,
        config: KFACConfig | None = None,
        layers: Iterable[nn.Linear] | None = None,
    ) -> None:
        self.network = network
        self.config = KFACConfig() if config is None else config
        self.layers = (
            trainable_linear_layers(network) if layers is None else list(layers)
        )
        if not self.layers:
            raise ValueError("K-FAC requires at least one trainable linear layer.")

    def compute_updates(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        targets: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[nn.Parameter, torch.Tensor]]:
        """Compute fresh K-FAC directions for one minibatch without applying them."""
        return compute_kfac_updates(
            network=self.network,
            states=states,
            actions=actions,
            targets=targets,
            noise_variance=self.config.noise_variance,
            damping=self.config.damping,
            accumulation_dtype=self.config.accumulation_dtype,
            eigenvalue_threshold=self.config.eigenvalue_threshold,
            check_finite=self.config.check_finite,
            layers=self.layers,
        )

    def apply_updates_(
        self,
        updates: dict[nn.Parameter, torch.Tensor],
        step_size: float,
    ) -> None:
        """Apply directions returned by :meth:`compute_updates`."""
        apply_kfac_updates_(updates, step_size=step_size)
        if self.config.check_finite:
            for name, parameter in self.network.named_parameters():
                if not torch.isfinite(parameter).all():
                    raise FloatingPointError(
                        f"K-FAC produced a non-finite parameter '{name}'."
                    )
