"""Objective-wise surrogate used by PSL-TAMO.

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
