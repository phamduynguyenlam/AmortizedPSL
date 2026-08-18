"""Differentiable task-local PSL optimization through a frozen TAMO predictor."""

from collections.abc import Sequence

import torch
from torch import Tensor, nn

from model.layers import GMMPredictionHead
from .preferences import sample_padded_preferences


def update_psl_inner_loop(
    psl_models: Sequence[nn.Module],
    psl_optimizers: Sequence[torch.optim.Optimizer],
    tamo_model: nn.Module,
    z_ctx: Tensor,
    s_ctx: Tensor,
    z_mask: Tensor,
    objective_mask: Tensor,
    num_preferences: int,
    num_steps: int,
    preference_method: str = "dirichlet",
) -> float:
    """Update only phi while retaining gradients through TAMO with respect to z."""
    requires_grad = [p.requires_grad for p in tamo_model.parameters()]
    for parameter in tamo_model.parameters():
        parameter.requires_grad_(False)

    losses = []
    try:
        for _ in range(num_steps):
            for b, (psl, optimizer) in enumerate(zip(psl_models, psl_optimizers)):
                optimizer.zero_grad()
                lambdas = sample_padded_preferences(
                    objective_mask[b:b + 1], num_preferences,
                    method=preference_method, dtype=z_ctx.dtype,
                )
                x = psl(lambdas[0]).unsqueeze(0)
                z_query = torch.cat((x, lambdas), dim=-1)
                output = tamo_model.predict(
                    x_ctx=z_ctx[b:b + 1], y_ctx=s_ctx[b:b + 1],
                    x_tar=z_query, x_mask=z_mask[b:b + 1],
                    y_mask=torch.ones((1, 1), device=z_ctx.device, dtype=torch.bool),
                    read_cache=False,
                )
                loss = GMMPredictionHead.expected_value(output).mean()
                loss.backward()
                optimizer.step()
                losses.append(loss.detach())
    finally:
        for parameter, flag in zip(tamo_model.parameters(), requires_grad):
            parameter.requires_grad_(flag)

    return torch.stack(losses).mean().item() if losses else 0.0
