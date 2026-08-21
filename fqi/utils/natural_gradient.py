"""Utilities for empirical natural-gradient updates.

The Fisher matrix used here is the Gauss--Newton/empirical Fisher matrix of
the selected scalar Q-values.  Computations are performed in sample space,
which is normally much smaller than parameter space.
"""

from collections.abc import Iterable

import torch
from torch import nn


def flatten_tensors(tensors: Iterable[torch.Tensor]) -> torch.Tensor:
    """Flatten and concatenate tensors into a single vector."""
    tensors = list(tensors)
    if not tensors:
        return torch.empty(0)
    return torch.cat([tensor.reshape(-1) for tensor in tensors])


def prediction_jacobian(
    predictions: torch.Tensor,
    parameters: Iterable[nn.Parameter],
) -> torch.Tensor:
    """Return the Jacobian of scalar predictions with respect to parameters.

    Parameters which do not affect a particular prediction contribute a row
    of zeros.  ``predictions`` may have any shape; it is flattened by sample.
    """
    parameters = list(parameters)
    if not parameters:
        raise ValueError("Natural gradient requires at least one parameter.")

    flat_predictions = predictions.reshape(-1)
    rows = []
    for prediction in flat_predictions:
        gradients = torch.autograd.grad(
            prediction,
            parameters,
            retain_graph=True,
            allow_unused=True,
        )
        rows.append(
            flatten_tensors(
                torch.zeros_like(parameter) if gradient is None else gradient
                for parameter, gradient in zip(parameters, gradients)
            )
        )
    return torch.stack(rows)


def natural_gradient_direction(
    jacobian: torch.Tensor,
    residual: torch.Tensor,
    damping: float = 1e-4,
) -> torch.Tensor:
    r"""Compute ``J^T (J J^T + damping I)^-1 residual``.

    With zero damping this uses the Moore--Penrose least-squares solution
    ``J^dagger residual``.  Positive damping improves stability when the
    empirical Fisher is singular or poorly conditioned.
    """
    if damping < 0:
        raise ValueError("damping must be non-negative.")
    residual = residual.reshape(-1).to(device=jacobian.device, dtype=jacobian.dtype)
    if jacobian.shape[0] != residual.numel():
        raise ValueError("The Jacobian and residual must have the same sample count.")

    if damping == 0:
        return torch.linalg.lstsq(jacobian, residual.unsqueeze(1)).solution.squeeze(1)

    gram = jacobian @ jacobian.T
    gram.diagonal().add_(damping)
    coefficients = torch.linalg.solve(gram, residual)
    return jacobian.T @ coefficients


@torch.no_grad()
def apply_parameter_update(
    parameters: Iterable[nn.Parameter],
    update: torch.Tensor,
    step_size: float,
) -> None:
    """Add a flattened update to parameters in their iteration order."""
    if step_size < 0:
        raise ValueError("step_size must be non-negative.")
    parameters = list(parameters)
    expected_size = sum(parameter.numel() for parameter in parameters)
    if update.numel() != expected_size:
        raise ValueError(f"Expected an update of size {expected_size}, got {update.numel()}.")

    offset = 0
    for parameter in parameters:
        size = parameter.numel()
        parameter.add_(update[offset:offset + size].view_as(parameter), alpha=step_size)
        offset += size


def natural_gradient_step(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    parameters: Iterable[nn.Parameter],
    step_size: float,
    damping: float = 1e-4,
) -> torch.Tensor:
    """Apply one empirical natural-gradient step and return its direction."""
    parameters = list(parameters)
    jacobian = prediction_jacobian(predictions, parameters)
    residual = targets.reshape(-1).detach() - predictions.reshape(-1).detach()
    direction = natural_gradient_direction(jacobian.detach(), residual, damping=damping)
    apply_parameter_update(parameters, direction, step_size=step_size)
    return direction
