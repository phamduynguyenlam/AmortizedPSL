"""History-conditioned Amortized Pareto Set Learning head."""

from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor, nn


_FF_MULTIPLIER = 4


class AmortizedParetoSetHead(nn.Module):
    """Predict ``x = h(H, lambda)`` one decision dimension at a time.

    History features and dimension IDs come from the objective predictor.  The
    head embeds each objective preference, lets it attend to encoded history,
    then decodes every decision dimension with one shared scalar MLP.
    """

    def __init__(
        self,
        dim_mlp: int,
        dim_attn: int,
        nhead: int,
        dropout: float,
        num_preference_layers: int,
        num_cross_layers: int,
        dim_hidden: int,
        depth: int,
        max_x_dim: int,
        max_y_dim: int,
        x_lower: Sequence[float] | Tensor | float,
        x_upper: Sequence[float] | Tensor | float,
    ):
        super().__init__()
        self.max_x_dim = max_x_dim
        self.max_y_dim = max_y_dim

        self.preference_embedder = nn.Linear(1, dim_mlp)
        self.preference_in = self._projection(dim_mlp, dim_attn)
        self.preference_out = self._projection(dim_attn, dim_mlp)
        preference_layer = nn.TransformerEncoderLayer(
            d_model=dim_attn,
            nhead=nhead,
            dim_feedforward=_FF_MULTIPLIER * dim_attn,
            dropout=dropout,
            batch_first=True,
            activation="relu",
        )
        self.preference_encoder = nn.TransformerEncoder(
            preference_layer, num_layers=num_preference_layers
        )

        self.query_in = self._projection(dim_mlp, dim_attn)
        self.history_in = self._projection(dim_mlp, dim_attn)
        self.cross_out = self._projection(dim_attn, dim_mlp)
        cross_layer = nn.TransformerDecoderLayer(
            d_model=dim_attn,
            nhead=nhead,
            dim_feedforward=_FF_MULTIPLIER * dim_attn,
            dropout=dropout,
            batch_first=True,
            activation="relu",
        )
        self.history_cross_attention = nn.TransformerDecoder(
            cross_layer, num_layers=num_cross_layers
        )

        self.psl_token = nn.Parameter(torch.randn(1, dim_mlp))
        self.output_query_in = self._projection(dim_mlp, dim_attn)
        self.output_memory_in = self._projection(dim_mlp, dim_attn)
        self.output_attention = nn.MultiheadAttention(
            embed_dim=dim_attn,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.output_out = self._projection(dim_attn, dim_mlp)
        self.output_norm = nn.LayerNorm(dim_mlp)
        self.output_head = self._mlp(dim_mlp, dim_hidden, 1, depth)

        lower = torch.as_tensor(x_lower, dtype=torch.float32)
        upper = torch.as_tensor(x_upper, dtype=torch.float32)
        self.register_buffer("x_lower", lower.expand(max_x_dim).clone())
        self.register_buffer("x_upper", upper.expand(max_x_dim).clone())
        if torch.any(self.x_upper <= self.x_lower):
            raise ValueError("Every APSL upper bound must exceed its lower bound")

    @staticmethod
    def _projection(dim_in: int, dim_out: int) -> nn.Module:
        return nn.Linear(dim_in, dim_out) if dim_in != dim_out else nn.Identity()

    @staticmethod
    def _mlp(dim_in: int, dim_hidden: int, dim_out: int, depth: int) -> nn.Module:
        if depth < 1:
            raise ValueError("APSL head depth must be at least one")
        if depth == 1:
            return nn.Linear(dim_in, dim_out)
        layers: list[nn.Module] = [nn.Linear(dim_in, dim_hidden), nn.ReLU()]
        for _ in range(depth - 2):
            layers.extend((nn.Linear(dim_hidden, dim_hidden), nn.ReLU()))
        layers.append(nn.Linear(dim_hidden, dim_out))
        return nn.Sequential(*layers)

    @staticmethod
    def _masked_mean(values: Tensor, mask: Tensor) -> Tensor:
        weights = mask.to(values.dtype).unsqueeze(-1)
        return (values * weights).sum(dim=-2) / weights.sum(dim=-2).clamp_min(1.0)

    def forward(
        self,
        history_tokens: Tensor,
        preferences: Tensor,
        x_ids: Tensor,
        objective_ids: Tensor,
        x_mask: Tensor,
        objective_mask: Tensor,
    ) -> Tensor:
        """Return bounded decisions with shape ``[B, Nq, max_x_dim]``."""
        batch_size, num_preferences, objective_dim = preferences.shape
        if objective_dim != self.max_y_dim:
            raise ValueError(
                f"Expected {self.max_y_dim} preference dimensions, got {objective_dim}"
            )

        # e_lambda followed by within-preference self-attention.  TAMO's
        # dimension encoder uses multiplicative positional IDs, so APSL follows
        # the same convention and shares the objective IDs.
        preference_dims = self.preference_embedder(preferences.unsqueeze(-1))
        preference_dims = preference_dims.reshape(
            batch_size * num_preferences, objective_dim, -1
        )
        preference_padding = (~objective_mask).unsqueeze(1).expand(
            -1, num_preferences, -1
        ).reshape(batch_size * num_preferences, objective_dim)
        preference_dims = self.preference_encoder(
            self.preference_in(preference_dims),
            src_key_padding_mask=preference_padding,
        )
        preference_dims = self.preference_out(preference_dims).reshape(
            batch_size, num_preferences, objective_dim, -1
        )
        preference_dims = preference_dims * objective_ids[
            None, None, :objective_dim, :
        ]
        preference_tokens = self._masked_mean(
            preference_dims,
            objective_mask.unsqueeze(1).expand(-1, num_preferences, -1),
        )

        # Each lambda independently cross-attends to the encoded history H'.
        query = self.query_in(preference_tokens).reshape(
            batch_size * num_preferences, 1, -1
        )
        memory = self.history_in(history_tokens).unsqueeze(1).expand(
            -1, num_preferences, -1, -1
        ).reshape(batch_size * num_preferences, history_tokens.shape[1], -1)
        preference_context = self.history_cross_attention(query, memory)
        preference_context = self.cross_out(preference_context).reshape(
            batch_size, num_preferences, -1
        )

        # For every x_j, l_lambda attends to M_PSL,j = [z_PSL, p_x^(j)].
        decision_dim = x_ids.shape[0]
        output_query = preference_context.unsqueeze(2).expand(
            -1, -1, decision_dim, -1
        ).reshape(batch_size * num_preferences * decision_dim, 1, -1)
        task_tokens = self.psl_token.expand(decision_dim, -1)
        output_memory = torch.stack((task_tokens, x_ids[:decision_dim]), dim=1)
        output_memory = output_memory[None, None].expand(
            batch_size, num_preferences, -1, -1, -1
        ).reshape(batch_size * num_preferences * decision_dim, 2, -1)
        decoded, _ = self.output_attention(
            self.output_query_in(output_query),
            self.output_memory_in(output_memory),
            self.output_memory_in(output_memory),
            need_weights=False,
        )
        decoded = self.output_norm(
            output_query + self.output_out(decoded)
        ).squeeze(1)
        unit_x = torch.sigmoid(self.output_head(decoded)).reshape(
            batch_size, num_preferences, decision_dim
        )
        decisions = self.x_lower[:decision_dim] + unit_x * (
            self.x_upper[:decision_dim] - self.x_lower[:decision_dim]
        )
        return torch.where(
            x_mask.unsqueeze(1), decisions, torch.zeros_like(decisions)
        )


def build_apsl_head(
    tamo_config,
    max_x_dim: int,
    max_y_dim: int,
    x_range,
) -> AmortizedParetoSetHead:
    """Build a deterministic APSL head using TAMO architecture widths."""
    return AmortizedParetoSetHead(
        dim_mlp=tamo_config.dim_mlp,
        dim_attn=tamo_config.dim_attn,
        nhead=tamo_config.nhead,
        dropout=tamo_config.dropout,
        num_preference_layers=tamo_config.num_layers_encoder,
        num_cross_layers=tamo_config.num_layers_decoder,
        dim_hidden=tamo_config.dim_hidden,
        depth=tamo_config.depth,
        max_x_dim=max_x_dim,
        max_y_dim=max_y_dim,
        x_lower=x_range[0],
        x_upper=x_range[1],
    )
