"""Task-local PSL optimization through a frozen objective surrogate."""

from collections.abc import Sequence

import torch
from torch import Tensor, nn

from data.base.masking import restore_by_mask
from .preferences import sample_padded_preferences
from .scalarization import expected_smooth_tchebycheff_loss


def update_psl_inner_loop(
    psl_models: Sequence[nn.Module],
    psl_optimizers: Sequence[torch.optim.Optimizer],
    objective_model: nn.Module,
    x_ctx: Tensor,
    y_ctx: Tensor,
    x_mask: Tensor,
    objective_mask: Tensor,
    ideal_point: Tensor,
    tau: float,
    num_preferences: int,
    num_steps: int,
    preference_method: str = "dirichlet",
    utility_beta: float = 0.0,
    compact_psl_dimensions: bool = False,
) -> float:
    """Update only phi while gradients flow through ``x -> objective_model``.

    Existing gradients on the objective model (from its NLL training loss) are
    preserved. Its parameters are temporarily frozen, but autograd remains
    enabled so PSL receives gradients with respect to its generated decisions.
    """
    if len(psl_models) != x_ctx.shape[0]:
        raise ValueError("PSL model count must equal the task batch size")

    requires_grad = [p.requires_grad for p in objective_model.parameters()]
    was_training = objective_model.training
    for parameter in objective_model.parameters():
        parameter.requires_grad_(False)
    objective_model.eval()

    losses = []
    try:
        for _ in range(num_steps):
            for b, (psl, optimizer) in enumerate(zip(psl_models, psl_optimizers)):
                optimizer.zero_grad()
                lambdas = sample_padded_preferences(
                    objective_mask[b:b + 1], num_preferences,
                    method=preference_method, dtype=x_ctx.dtype,
                )
                lambda_input = lambdas[0]
                if compact_psl_dimensions:
                    lambda_input = lambda_input[:, objective_mask[b]]
                x_generated = psl(lambda_input)
                if compact_psl_dimensions:
                    x_generated = restore_by_mask(
                        x_generated, x_mask[b], dim=-1
                    )
                x = x_generated.unsqueeze(0)
                if utility_beta != 0.0:
                    predicted_y = objective_model.predictive_utility(
                        x_ctx=x_ctx[b:b + 1],
                        y_ctx=y_ctx[b:b + 1],
                        x_tar=x,
                        x_mask=x_mask[b:b + 1],
                        y_mask=objective_mask[b:b + 1],
                        beta=utility_beta,
                    )
                else:
                    predicted_y = objective_model.predictive_mean(
                        x_ctx=x_ctx[b:b + 1],
                        y_ctx=y_ctx[b:b + 1],
                        x_tar=x,
                        x_mask=x_mask[b:b + 1],
                        y_mask=objective_mask[b:b + 1],
                    )
                loss = expected_smooth_tchebycheff_loss(
                    y=predicted_y,
                    lambdas=lambdas,
                    ideal_point=ideal_point[b:b + 1].unsqueeze(1),
                    tau=tau,
                    mask=objective_mask[b:b + 1],
                )
                loss.backward()
                optimizer.step()
                losses.append(loss.detach())
    finally:
        for parameter, flag in zip(objective_model.parameters(), requires_grad):
            parameter.requires_grad_(flag)
        objective_model.train(was_training)

    return torch.stack(losses).mean().item() if losses else 0.0
