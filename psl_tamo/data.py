"""Build augmented STCH prediction batches without changing TAMO interfaces."""

from typing import Optional
import random

import torch
from torch import Tensor

from data.base.masking import gather_by_indices, generate_dim_mask
from .preferences import sample_padded_preferences
from .scalarization import smooth_tchebycheff


def _sample_nc(x_dim: int, min_nc: int, max_nc: int, warmup: bool) -> int:
    """Mirror TAMO's context-size schedule without importing the GP stack."""
    x_dim = int(x_dim)
    scale_factor = 2 if 1 < x_dim <= 3 else (4 if x_dim > 3 else 1)
    upper = int(max_nc * scale_factor)
    return upper if warmup else random.randint(min_nc, upper)


def prepare_stch_prediction_batches(
    x: Tensor,
    y: Tensor,
    valid_x_counts: Tensor | int,
    valid_y_counts: Tensor | int,
    dim_scatter_mode: str,
    min_nc: int,
    max_nc: int,
    nc_fixed: Optional[int] = None,
    warmup: bool = True,
    tau: float = 0.1,
    ideal_point: Tensor | float = -1.0,
    preference_method: str = "dirichlet",
):
    """Convert ``(x, y)`` into ``([x, lambda], STCH(y|lambda))`` batches."""
    batch_size, num_points, max_x_dim = x.shape
    max_y_dim = y.shape[-1]
    x_mask, x_indices = generate_dim_mask(
        max_x_dim, x.device, valid_x_counts, dim_scatter_mode
    )
    objective_mask, y_indices = generate_dim_mask(
        max_y_dim, y.device, valid_y_counts, dim_scatter_mode
    )
    x = gather_by_indices(x, x_indices)
    y = gather_by_indices(y, y_indices)

    if x_mask.ndim == 1:
        x_mask = x_mask.unsqueeze(0).expand(batch_size, -1)
    if objective_mask.ndim == 1:
        objective_mask = objective_mask.unsqueeze(0).expand(batch_size, -1)

    lambdas = sample_padded_preferences(
        objective_mask, num_points, method=preference_method, dtype=y.dtype
    )
    scalar_y = smooth_tchebycheff(y, lambdas, ideal_point, tau, objective_mask)
    z = torch.cat((x, lambdas.to(x.dtype)), dim=-1)

    if nc_fixed is None:
        counts = valid_x_counts if isinstance(valid_x_counts, Tensor) else torch.tensor([valid_x_counts], device=x.device)
        nc = _sample_nc(torch.max(counts), min_nc, max_nc, warmup)
    else:
        nc = nc_fixed
    if not 0 <= nc < num_points:
        raise ValueError(f"Context size {nc} must be smaller than {num_points}")

    perm = torch.randperm(num_points, device=x.device)
    context_idx, target_idx = perm[:nc], perm[nc:]
    z_mask = torch.cat((x_mask.bool(), objective_mask.bool()), dim=-1)
    s_mask = torch.ones((batch_size, 1), device=x.device, dtype=torch.bool)
    return (
        z[:, context_idx], scalar_y[:, context_idx],
        z[:, target_idx], scalar_y[:, target_idx], z_mask, s_mask,
    )
