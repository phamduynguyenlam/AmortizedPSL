"""Contracts for the objective-utility PSL-TAMO flow."""

import torch
from types import SimpleNamespace

from model import ParetoSetMLP, TAMOConfig, build_objective_predictor
from model.layers import GMMPredictionHead
from psl_tamo.utility_evaluation import run_utility_psl_optimization
import psl_tamo.utility_forwards as utility_forwards
from psl_tamo.utility_policy import (
    infer_history_preferences,
    select_preference_with_policy,
    update_preference_to_x_mlp,
)


def _tiny_objective_model(max_x_dim=2):
    return build_objective_predictor(
        TAMOConfig(
            max_x_dim=max_x_dim,
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
        max_x_dim=max_x_dim,
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


def test_expected_stch_updates_only_mlp_through_augmented_utility_head():
    model = _tiny_objective_model(max_x_dim=4)
    model.zero_grad(set_to_none=True)
    psl = ParetoSetMLP(2, 2, hidden_dim=256, depth=3)
    optimizer = torch.optim.Adam(psl.parameters(), lr=1e-3)
    mask = torch.ones(1, 2, dtype=torch.bool)
    x_ctx = torch.randn(1, 5, 2)
    y_ctx = torch.randn(1, 5, 2)
    history_lambda = torch.softmax(torch.randn(1, 5, 2), dim=-1)

    loss = update_preference_to_x_mlp(
        psl_models=[psl],
        psl_optimizers=[optimizer],
        model=model,
        x_ctx=x_ctx,
        y_ctx=y_ctx,
        history_preferences=history_lambda,
        x_mask=mask,
        objective_mask=mask,
        ideal_point=y_ctx.amin(dim=1),
        tau=0.1,
        beta=0.5,
        num_preferences=10,
        num_steps=1,
    )
    assert torch.isfinite(torch.tensor(loss))
    assert any(parameter.grad is not None for parameter in psl.parameters())
    assert all(parameter.grad is None for parameter in model.parameters())
    assert all(parameter.requires_grad for parameter in model.parameters())


def test_inverse_preferences_and_policy_log_prob_contract():
    psl = ParetoSetMLP(2, 1, hidden_dim=8, depth=2)
    mask_x = torch.ones(1, 1, dtype=torch.bool)
    mask_y = torch.ones(1, 2, dtype=torch.bool)
    reference = torch.softmax(torch.randn(1, 3, 2), dim=-1)
    inferred, loss = infer_history_preferences(
        [psl], torch.rand(1, 3, 1), reference, mask_x, mask_y,
        num_steps=2,
    )
    assert inferred.shape == reference.shape
    torch.testing.assert_close(inferred.sum(dim=-1), torch.ones(1, 3))
    assert torch.isfinite(torch.tensor(loss))

    model = _tiny_objective_model(max_x_dim=3)
    z_ctx = torch.randn(1, 3, 3)
    y_ctx = torch.randn(1, 3, 2)
    candidates = torch.randn(1, 5, 3)
    action = select_preference_with_policy(
        model, z_ctx, y_ctx, candidates,
        torch.ones(1, 3, dtype=torch.bool), mask_y, 1, 2,
    )
    assert action.log_probs.requires_grad
    action.log_probs.sum().backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


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
        use_time_budget = True

        @staticmethod
        def sample_T():
            return 2

    result = run_utility_psl_optimization(
        model=_tiny_objective_model(max_x_dim=3),
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
            "inverse_steps": 1,
            "log_every": 1,
        },
        scalarization_config={"tau": 0.1, "ideal_point": "observed_min"},
        utility_config={"beta": 0.5},
        device="cpu",
        seed=0,
        log=lambda _: None,
    )
    assert result.hv.shape == (1, 3)
    assert result.x_queries.shape == (1, 2, 1)
    assert result.preferences.shape == (1, 2, 2)


def test_meta_training_rollout_produces_reinforce_gradient(monkeypatch):
    class FakeDiscreteEnvironment:
        def __init__(self, data_config, batch_size, d, **kwargs):
            self.batch_size = batch_size
            self.num_points = d
            self.x_mask = torch.ones(data_config.max_x_dim, dtype=torch.bool)
            self.y_mask = torch.ones(data_config.max_y_dim, dtype=torch.bool)
            base = torch.linspace(-1, 1, d)
            self._x = torch.stack((base, base.square(), base.sin()), dim=-1)
            self._x = self._x.unsqueeze(0).expand(batch_size, -1, -1).clone()
            self._y = torch.stack((base.square(), (base - 0.5).square()), dim=-1)
            self._y = self._y.unsqueeze(0).expand(batch_size, -1, -1).clone()
            self.y_mins = self._y.min(dim=1).values

        def init(self, num_initial_points, **kwargs):
            indices = torch.arange(num_initial_points).view(1, -1, 1)
            indices = indices.expand(self.batch_size, -1, -1)
            return self.step(indices, None, None, compute_regret=False)

        def step(self, index_new, x_ctx=None, y_ctx=None, compute_regret=True, **kwargs):
            x_new = torch.gather(
                self._x, 1, index_new.expand(-1, -1, self._x.shape[-1])
            )
            y_new = torch.gather(
                self._y, 1, index_new.expand(-1, -1, self._y.shape[-1])
            )
            x_ctx = x_new if x_ctx is None else torch.cat((x_ctx, x_new), dim=1)
            y_ctx = y_new if y_ctx is None else torch.cat((y_ctx, y_new), dim=1)
            regret = (
                torch.full((self.batch_size,), 1.0 / y_ctx.shape[1]).numpy()
                if compute_regret else None
            )
            return x_ctx, y_ctx, None, regret

    monkeypatch.setattr(
        utility_forwards, "GPSampleFunction", FakeDiscreteEnvironment
    )
    model = _tiny_objective_model(max_x_dim=5)
    data = SimpleNamespace(max_x_dim=3, max_y_dim=2, x_range=[-1.0, 1.0])
    opt = SimpleNamespace(
        batch_size=2, num_samples=1, num_query_points=10,
        use_grid_sampling=False, num_initial_points=3,
        regret_type="norm_ratio", use_time_budget=True, epsilon=1.0,
    )
    loss = SimpleNamespace(
        use_cumulative_rewards=True, discount_factor=0.99,
        batch_standardize=True, clip_rewards=True,
    )
    result = utility_forwards.optimization_forward_utility_psl(
        model, data, opt, loss,
        {
            "hidden_dim": 8, "depth": 2, "init_steps": 1,
            "update_steps": 1, "inverse_steps": 1,
            "num_train_preferences": 4, "num_policy_preferences": 4,
            "lr": 1e-3,
        },
        {"tau": 0.1}, {"beta": 0.5}, T=2, device="cpu",
    )
    assert torch.isfinite(result[0])
    result[0].backward()
    assert any(parameter.grad is not None for parameter in model.parameters())
