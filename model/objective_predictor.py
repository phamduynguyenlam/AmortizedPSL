"""Objective-wise surrogate used by APSL.

The implementation deliberately reuses TAMO's original prediction stack so the
objective surrogate keeps the same dimension-wise embeddings, transformer, and
GMM semantics without duplicating that architecture.
"""

from dataclasses import replace

from torch import Tensor

from model.layers import GMMPredictionHead
from .tamo import TAMO, TAMOConfig


class ObjectiveValuePredictor(TAMO):
    """TAMO predictor specialized to true decision/objective observations."""

    def encode_history(
        self,
        x_ctx: Tensor,
        y_ctx: Tensor,
        x_mask: Tensor,
        y_mask: Tensor,
    ):
        """Encode observed ``(x, y)`` pairs for another decoder branch."""
        tokens, x_ids, y_ids = self._make_tokens(
            x=x_ctx,
            y=y_ctx,
            x_mask=x_mask,
            y_mask=y_mask,
        )
        # Every token here belongs to the observed history, hence full
        # self-attention is intended.  Passing an all-False square mask makes
        # some PyTorch versions incorrectly infer ``is_causal=True``; the
        # custom context-prefix layer then rightfully fails because it has no
        # causal mask.  ``mask=None`` is both semantically correct and portable.
        return (
            self.transformer_block(tokens, mask=None),
            x_ids,
            y_ids,
        )

    def predictive_mean(
        self,
        x_ctx: Tensor,
        y_ctx: Tensor,
        x_tar: Tensor,
        x_mask: Tensor,
        y_mask: Tensor,
    ) -> Tensor:
        """Return differentiable objective means ``E[f(x) | H]``."""
        output = self.predict(
            x_ctx=x_ctx,
            y_ctx=y_ctx,
            x_tar=x_tar,
            x_mask=x_mask,
            y_mask=y_mask,
            read_cache=False,
        )
        return GMMPredictionHead.expected_value(output)

    def predictive_utility(
        self,
        x_ctx: Tensor,
        y_ctx: Tensor,
        x_tar: Tensor,
        x_mask: Tensor,
        y_mask: Tensor,
        beta: float = 0.0,
    ) -> Tensor:
        """Return an objective-wise optimistic utility for minimization.

        ``beta=0`` gives the posterior mixture mean. Positive ``beta`` uses
        the lower-confidence-bound convention from the original PSL paper,
        ``E[f_i(x)] - beta * Std[f_i(x)]``. The operation remains fully
        differentiable with respect to ``x_tar``.
        """
        output = self.predict(
            x_ctx=x_ctx,
            y_ctx=y_ctx,
            x_tar=x_tar,
            x_mask=x_mask,
            y_mask=y_mask,
            read_cache=False,
        )
        mean = GMMPredictionHead.expected_value(output)
        if beta == 0.0:
            return mean
        return mean - float(beta) * GMMPredictionHead.std(output)


def build_objective_predictor(
    scalar_tamo_config: TAMOConfig,
    max_x_dim: int,
    max_y_dim: int,
) -> ObjectiveValuePredictor:
    """Build the objective surrogate with TAMO's architecture hyperparameters."""
    config = replace(
        scalar_tamo_config,
        max_x_dim=max_x_dim,
        max_y_dim=max_y_dim,
    )
    return ObjectiveValuePredictor(config)
