"""Preference-conditioned scalarization functions."""

import torch
from torch import Tensor


def smooth_tchebycheff(
    y: Tensor,
    lambdas: Tensor,
    ideal_point: Tensor | float,
    tau: float,
    mask: Tensor | None = None,
) -> Tensor:
    """Compute masked smooth Tchebycheff values with a stable log-sum-exp."""
    if y.shape != lambdas.shape:
        raise ValueError(f"y and lambdas must have the same shape: {y.shape} != {lambdas.shape}")
    if tau <= 0:
        raise ValueError("tau must be positive")

    ideal = torch.as_tensor(ideal_point, device=y.device, dtype=y.dtype)
    values = lambdas * (y - ideal) / tau
    if mask is not None:
        mask = torch.as_tensor(mask, device=y.device, dtype=torch.bool)
        while mask.ndim < values.ndim:
            mask = mask.unsqueeze(-2)
        mask = torch.broadcast_to(mask, values.shape)
        if not mask.any(dim=-1).all():
            raise ValueError("Every sample must contain at least one valid objective")
        values = values.masked_fill(~mask, float("-inf"))

    return tau * torch.logsumexp(values, dim=-1, keepdim=True)
