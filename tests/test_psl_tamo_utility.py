"""Contracts for the objective-utility PSL-TAMO flow."""

import numpy as np
import torch
from torch import nn
from types import SimpleNamespace

from model import ParetoSetMLP, TAMOConfig, build_objective_predictor
from model.layers import GMMPredictionHead
from psl_tamo.psl_inner_loop import update_psl_inner_loop
from psl_tamo.utility_selection import select_preference_by_utility
from psl_tamo.utility_evaluation import run_utility_psl_optimization


def _tiny_objective_model():
    return build_objective_predictor(
        TAMOConfig(
            max_x_dim=2,
            max_y_dim=2,
            dim_mlp=8,
            dim_attn=8,
            nhead=2,
            num_layers_backbone=1,
            num_layers_encoder=1,
            num_layers_decoder=1,
            dim_hidden=8,
            depth=2,
            num_components=2,
        ),
        max_x_dim=2,
        max_y_dim=2,
    )


def test_objective_utility_is_gmm_lcb_and_differentiable():
    model = _tiny_objective_model().eval()
    x_ctx = torch.randn(1, 4, 2)
    y_ctx = torch.randn(1, 4, 2)
    x_tar = torch.randn(1, 3, 2, requires_grad=True)
    mask = torch.ones(1, 2, dtype=torch.bool)

    output = model.predict(x_ctx, y_ctx, x_tar, mask, mask)
    expected = (
        GMMPredictionHead.expected_value(output)
        - 0.5 * GMMPredictionHead.std(output)
    )
    utility = model.predictive_utility(
        x_ctx, y_ctx, x_tar, mask, mask, beta=0.5
    )
    torch.testing.assert_close(utility, expected)
    utility.sum().backward()
    assert x_tar.grad is not None and torch.isfinite(x_tar.grad).all()


def test_expected_stch_updates_only_pareto_set_mlp_with_lcb():
    model = _tiny_objective_model()
    model.zero_grad(set_to_none=True)
    psl = ParetoSetMLP(2, 2, hidden_dim=256, depth=3)
    optimizer = torch.optim.Adam(psl.parameters(), lr=1e-3)
    mask = torch.ones(1, 2, dtype=torch.bool)
    x_ctx = torch.randn(1, 5, 2)
    y_ctx = torch.randn(1, 5, 2)

    loss = update_psl_inner_loop(
        psl_models=[psl],
        psl_optimizers=[optimizer],
        objective_model=model,
        x_ctx=x_ctx,
        y_ctx=y_ctx,
        x_mask=mask,
        objective_mask=mask,
        ideal_point=y_ctx.amin(dim=1),
        tau=0.1,
        num_preferences=10,
        num_steps=1,
        utility_beta=0.5,
    )
    assert np.isfinite(loss)
    assert any(parameter.grad is not None for parameter in psl.parameters())
    assert all(parameter.grad is None for parameter in model.parameters())
    assert all(parameter.requires_grad for parameter in model.parameters())


class _AnalyticObjectiveUtility(nn.Module):
    """Small differentiable two-objective utility used to test selection."""

    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.tensor(0.0))

    def predictive_utility(self, x_tar, **kwargs):
        x = x_tar[..., :1] + 0.0 * self.anchor
        return torch.cat((x.square(), (x - 1.0).square()), dim=-1)


def test_selection_operates_on_preferences_and_their_mlp_decisions():
    model = _AnalyticObjectiveUtility()
    preferences = torch.tensor([[[0.9, 0.1], [0.5, 0.5], [0.1, 0.9]]])
    decisions = torch.tensor([[[0.0], [0.5], [1.0]]])
    objective_mask = torch.ones(1, 2, dtype=torch.bool)
    action = select_preference_by_utility(
        objective_model=model,
        x_ctx=torch.zeros(1, 2, 1),
        y_ctx=torch.zeros(1, 2, 2),
        x_candidates=decisions,
        preferences=preferences,
        x_mask=torch.ones(1, 1, dtype=torch.bool),
        objective_mask=objective_mask,
        ideal_point=torch.zeros(1, 2),
        tau=0.1,
        beta=0.5,
        epsilon=0.0,
    )
    expected_index = action.stch.argmin(dim=-1)
    torch.testing.assert_close(action.indices, expected_index)
    torch.testing.assert_close(
        action.preferences.squeeze(1), preferences[0, expected_index]
    )
    torch.testing.assert_close(
        action.decisions.squeeze(1), decisions[0, expected_index]
    )


def test_tiny_utility_rollout_runs_end_to_end():
    class FakeFunction:
        x_dim = 1
        y_dim = 2
        max_hv = 1.0

        @staticmethod
        def _evaluate(x):
            return torch.cat((x.square(), (x - 1.0).square()), dim=-1)

        def init(self, batch_size, num_initial_points, device, **kwargs):
            x = torch.linspace(0.1, 0.9, num_initial_points, device=device)
            x = x.view(1, -1, 1).expand(batch_size, -1, -1).clone()
            y = self._evaluate(x)
            hv = torch.full((batch_size,), 0.1, device=device)
            return x, y, hv, None

        def transform_outputs(self, outputs, **kwargs):
            return outputs

        def step(self, x_new, x_ctx, y_ctx, **kwargs):
            y_new = self._evaluate(x_new)
            x_ctx = torch.cat((x_ctx, x_new), dim=1)
            y_ctx = torch.cat((y_ctx, y_new), dim=1)
            hv = torch.full(
                (x_ctx.shape[0],), 0.1 * x_ctx.shape[1], device=x_ctx.device
            )
            return x_ctx, y_ctx, hv, None

    class TinyOptimizationConfig:
        batch_size = 1
        num_initial_points = 2
        regret_type = "ratio"
        epsilon = 0.0

        @staticmethod
        def sample_T():
            return 2

    result = run_utility_psl_optimization(
        model=_AnalyticObjectiveUtility(),
        test_function=FakeFunction(),
        data_config=SimpleNamespace(
            max_x_dim=1, max_y_dim=2, x_range=[0.0, 1.0], y_range=[0.0, 1.0]
        ),
        optimization_config=TinyOptimizationConfig(),
        psl_config={
            "hidden_dim": 8,
            "depth": 2,
            "lr": 1e-3,
            "init_steps": 1,
            "update_steps": 1,
            "num_train_preferences": 4,
            "num_policy_preferences": 5,
            "log_every": 1,
        },
        scalarization_config={"tau": 0.1, "ideal_point": "observed_min"},
        utility_config={"beta": 0.5, "temperature": 1.0},
        device="cpu",
        seed=0,
        log=lambda _: None,
    )
    assert result.hv.shape == (1, 3)
    assert result.x_queries.shape == (1, 2, 1)
    assert result.preferences.shape == (1, 2, 2)
