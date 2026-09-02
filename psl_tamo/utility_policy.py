"""Meta-policy rollout with a task-local PSL-MOBO preference-to-x MLP."""

from dataclasses import dataclass
from typing import Optional, Sequence

import torch
from torch import Tensor, nn

from data.base.masking import restore_by_mask
from model import TAMO
from .preferences import sample_padded_preferences
from .scalarization import expected_smooth_tchebycheff_loss


@dataclass
class PreferencePolicyAction:
    indices: Tensor
    log_probs: Tensor
    entropy: Tensor
    logits: Optional[Tensor]


def augment_with_preferences(
    x_ctx: Tensor,
    x_tar: Tensor,
    x_mask: Tensor,
    objective_mask: Tensor,
    method: str = "dirichlet",
) -> tuple[Tensor, Tensor, Tensor]:
    """Append nuisance preferences for objective-head meta-training."""
    lambda_ctx = sample_padded_preferences(
        objective_mask, x_ctx.shape[1], method, x_ctx.dtype
    )
    lambda_tar = sample_padded_preferences(
        objective_mask, x_tar.shape[1], method, x_tar.dtype
    )
    return (
        torch.cat((x_ctx, lambda_ctx), dim=-1),
        torch.cat((x_tar, lambda_tar), dim=-1),
        torch.cat((x_mask, objective_mask), dim=-1),
    )


def infer_history_preferences(
    psl_models: Sequence[nn.Module],
    x_ctx: Tensor,
    lambda_star: Tensor,
    x_mask: Tensor,
    objective_mask: Tensor,
    num_steps: int = 10,
    lr: float = 5e-2,
    reconstruction_weight: float = 1.0,
    reference_weight: float = 1.0,
) -> tuple[Tensor, float]:
    """Fit lambda for each old x using ||x-h(lambda)|| + ||lambda-lambda*||."""
    inferred, losses = [], []
    for b, psl in enumerate(psl_models):
        active_y = objective_mask[b]
        active_x = x_mask[b]
        reference = lambda_star[b, :, active_y].clamp_min(1e-8)
        logits = reference.log().detach().clone().requires_grad_(True)
        optimizer = torch.optim.Adam([logits], lr=lr)
        requires_grad = [p.requires_grad for p in psl.parameters()]
        for parameter in psl.parameters():
            parameter.requires_grad_(False)
        try:
            for _ in range(num_steps):
                optimizer.zero_grad()
                compact_lambda = torch.softmax(logits, dim=-1)
                reconstructed_x = psl(compact_lambda)
                reconstruction = torch.linalg.vector_norm(
                    reconstructed_x - x_ctx[b, :, active_x], dim=-1
                ).mean()
                reference_loss = torch.linalg.vector_norm(
                    compact_lambda - reference, dim=-1
                ).mean()
                loss = (
                    reconstruction_weight * reconstruction
                    + reference_weight * reference_loss
                )
                loss.backward()
                optimizer.step()
            compact_lambda = torch.softmax(logits, dim=-1).detach()
            losses.append(loss.detach())
        finally:
            for parameter, flag in zip(psl.parameters(), requires_grad):
                parameter.requires_grad_(flag)
        inferred.append(restore_by_mask(compact_lambda, active_y, dim=-1))
    mean_loss = torch.stack(losses).mean().item() if losses else 0.0
    return torch.stack(inferred), mean_loss


def update_preference_to_x_mlp(
    psl_models: Sequence[nn.Module],
    psl_optimizers: Sequence[torch.optim.Optimizer],
    model: nn.Module,
    x_ctx: Tensor,
    y_ctx: Tensor,
    history_preferences: Tensor,
    x_mask: Tensor,
    objective_mask: Tensor,
    ideal_point: Tensor,
    tau: float,
    beta: float,
    num_preferences: int,
    num_steps: int,
    preference_method: str = "dirichlet",
) -> float:
    """Update only h_theta through the frozen TAMO objective utility head."""
    model_flags = [p.requires_grad for p in model.parameters()]
    was_training = model.training
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    losses = []
    try:
        z_ctx = torch.cat((x_ctx, history_preferences), dim=-1)
        z_mask = torch.cat((x_mask, objective_mask), dim=-1)
        for _ in range(num_steps):
            for b, (psl, optimizer) in enumerate(zip(psl_models, psl_optimizers)):
                optimizer.zero_grad()
                lambdas = sample_padded_preferences(
                    objective_mask[b:b + 1], num_preferences,
                    preference_method, x_ctx.dtype,
                )
                compact_lambda = lambdas[0, :, objective_mask[b]]
                compact_x = psl(compact_lambda)
                x = restore_by_mask(compact_x, x_mask[b], dim=-1).unsqueeze(0)
                z_tar = torch.cat((x, lambdas), dim=-1)
                utilities = model.predictive_utility(
                    x_ctx=z_ctx[b:b + 1], y_ctx=y_ctx[b:b + 1],
                    x_tar=z_tar, x_mask=z_mask[b:b + 1],
                    y_mask=objective_mask[b:b + 1], beta=beta,
                )
                loss = expected_smooth_tchebycheff_loss(
                    utilities, lambdas, ideal_point[b:b + 1].unsqueeze(1),
                    tau, objective_mask[b:b + 1],
                )
                loss.backward()
                optimizer.step()
                losses.append(loss.detach())
    finally:
        for parameter, flag in zip(model.parameters(), model_flags):
            parameter.requires_grad_(flag)
        model.train(was_training)
    return torch.stack(losses).mean().item() if losses else 0.0


def select_preference_with_policy(
    model: TAMO,
    z_ctx: Tensor,
    y_ctx: Tensor,
    z_candidates: Tensor,
    z_mask: Tensor,
    objective_mask: Tensor,
    t: int,
    T: int,
    use_budget: bool = True,
    epsilon: float = 1.0,
) -> PreferencePolicyAction:
    """Let TAMO's learned categorical policy select the next preference."""
    batch_size = z_ctx.shape[0]
    result = model.action(
        x_ctx=z_ctx, y_ctx=y_ctx, x_mask=z_mask, y_mask=objective_mask,
        query_chunks=z_candidates.unsqueeze(1),
        query_x_mask=z_mask.unsqueeze(1), t=t, T=T,
        use_budget=use_budget, epsilon=epsilon, return_logits=True,
        read_cache=False, write_cache=False, auto_clear_cache=True,
    )
    return PreferencePolicyAction(
        indices=result[1].reshape(batch_size),
        log_probs=result[2].reshape(batch_size),
        entropy=result[3].reshape(batch_size),
        logits=result[4],
    )
