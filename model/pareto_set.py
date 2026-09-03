"""Task-specific Pareto-set MLP used only by Utility-PSL-TAMO."""

from typing import Optional, Sequence

import torch
from torch import Tensor, nn


class ParetoSetMLP(nn.Module):
    """Map an objective preference to a bounded decision vector."""

    def __init__(
        self,
        preference_dim: int,
        decision_dim: int,
        hidden_dim: int = 128,
        depth: int = 3,
        x_lower: Optional[Sequence[float] | Tensor | float] = None,
        x_upper: Optional[Sequence[float] | Tensor | float] = None,
    ):
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be at least 1")

        layers = []
        dim_in = preference_dim
        for _ in range(depth - 1):
            layers.extend((nn.Linear(dim_in, hidden_dim), nn.ReLU()))
            dim_in = hidden_dim
        layers.extend((nn.Linear(dim_in, decision_dim), nn.Sigmoid()))
        self.network = nn.Sequential(*layers)

        lower = torch.as_tensor(0.0 if x_lower is None else x_lower).float()
        upper = torch.as_tensor(1.0 if x_upper is None else x_upper).float()
        self.register_buffer("x_lower", lower.expand(decision_dim).clone())
        self.register_buffer("x_upper", upper.expand(decision_dim).clone())
        if torch.any(self.x_upper <= self.x_lower):
            raise ValueError("Every upper bound must exceed its lower bound")

    def forward(self, lambdas: Tensor) -> Tensor:
        unit_x = self.network(lambdas)
        return self.x_lower + unit_x * (self.x_upper - self.x_lower)
