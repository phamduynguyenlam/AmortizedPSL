"""Preference-space acquisition for the utility PSL-TAMO variant.

This module deliberately contains no Pareto-set network.  It scores the
``(lambda, h_phi(lambda))`` pairs produced by a task-local PSL MLP with the
objective-wise TAMO surrogate, then selects a preference rather than searching
the decision domain directly.
"""

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .scalarization import smooth_tchebycheff


@dataclass
class UtilityPreferenceAction:
    """Result of selecting one candidate in preference space."""

    indices: Tensor
    preferences: Tensor
    decisions: Tensor
    objective_utilities: Tensor
    stch: Tensor
    entropy: Tensor


def predict_objective_utilities(
    objective_model: nn.Module,
    x_ctx: Tensor,
    y_ctx: Tensor,
    x_candidates: Tensor,
    x_mask: Tensor,
    objective_mask: Tensor,
    beta: float,
) -> Tensor:
    """Predict one differentiable LCB utility for every objective."""
    return objective_model.predictive_utility(
        x_ctx=x_ctx,
        y_ctx=y_ctx,
        x_tar=x_candidates,
        x_mask=x_mask,
        y_mask=objective_mask,
        beta=beta,
    )


def select_preference_by_utility(
    objective_model: nn.Module,
    x_ctx: Tensor,
    y_ctx: Tensor,
    x_candidates: Tensor,
    preferences: Tensor,
    x_mask: Tensor,
    objective_mask: Tensor,
    ideal_point: Tensor,
    tau: float,
    beta: float = 0.5,
    epsilon: float = 0.0,
    temperature: float = 1.0,
) -> UtilityPreferenceAction:
    """Select ``lambda`` using objective utilities of ``h_phi(lambda)``.

    With ``epsilon=0`` this returns the candidate minimizing smooth
    Tchebycheff. Otherwise it samples from ``softmax(-STCH / temperature)``
    with probability ``epsilon`` and uses the greedy candidate with probability
    ``1-epsilon``.
    """
    if not 0.0 <= epsilon <= 1.0:
        raise ValueError("epsilon must be in [0, 1]")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    if x_candidates.shape[:2] != preferences.shape[:2]:
        raise ValueError("Candidates and preferences must share [batch, count]")

    utilities = predict_objective_utilities(
        objective_model=objective_model,
        x_ctx=x_ctx,
        y_ctx=y_ctx,
        x_candidates=x_candidates,
        x_mask=x_mask,
        objective_mask=objective_mask,
        beta=beta,
    )
    stch = smooth_tchebycheff(
        y=utilities,
        lambdas=preferences,
        ideal_point=ideal_point.unsqueeze(1),
        tau=tau,
        mask=objective_mask,
    ).squeeze(-1)
    probabilities = torch.softmax(-stch / temperature, dim=-1)
    greedy = stch.argmin(dim=-1)
    sampled = torch.multinomial(probabilities, num_samples=1).squeeze(-1)
    explore = torch.rand_like(greedy, dtype=torch.float32) < epsilon
    indices = torch.where(explore, sampled, greedy)

    batch = torch.arange(x_candidates.shape[0], device=x_candidates.device)
    selected_preferences = preferences[batch, indices].unsqueeze(1)
    selected_decisions = x_candidates[batch, indices].unsqueeze(1)
    entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=-1)
    return UtilityPreferenceAction(
        indices=indices,
        preferences=selected_preferences,
        decisions=selected_decisions,
        objective_utilities=utilities,
        stch=stch,
        entropy=entropy,
    )
