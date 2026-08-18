"""Preference sampling helpers."""

import torch
from torch import Tensor


def sample_preferences(
    batch_size: int,
    num_preferences: int,
    m: int,
    device: str | torch.device,
    method: str = "dirichlet",
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Sample points on the ``m``-objective probability simplex."""
    if m < 1:
        raise ValueError("m must be positive")
    if method == "grid":
        if m != 2:
            raise ValueError("grid preference sampling is only supported for m=2")
        grid = torch.linspace(0, 1, num_preferences, device=device, dtype=dtype)
        prefs = torch.stack((grid, 1 - grid), dim=-1)
        return prefs.unsqueeze(0).expand(batch_size, -1, -1).clone()
    if method != "dirichlet":
        raise ValueError(f"Unknown preference sampling method: {method}")

    concentration = torch.ones(m, device=device, dtype=dtype)
    return torch.distributions.Dirichlet(concentration).sample(
        (batch_size, num_preferences)
    )


def sample_padded_preferences(
    objective_mask: Tensor,
    num_preferences: int,
    method: str = "dirichlet",
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Sample preferences independently and scatter them into padded dimensions."""
    if objective_mask.ndim == 1:
        objective_mask = objective_mask.unsqueeze(0)
    objective_mask = objective_mask.bool()
    batch_size, max_m = objective_mask.shape
    result = torch.zeros(
        batch_size, num_preferences, max_m,
        device=objective_mask.device, dtype=dtype,
    )
    for b in range(batch_size):
        valid = objective_mask[b].nonzero(as_tuple=True)[0]
        prefs = sample_preferences(1, num_preferences, len(valid), objective_mask.device, method, dtype)[0]
        result[b, :, valid] = prefs
    return result
