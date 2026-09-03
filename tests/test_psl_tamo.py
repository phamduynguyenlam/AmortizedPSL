"""Fast sanity tests for PSL-TAMO's representation and gradient contracts."""

import torch
import numpy as np
from types import SimpleNamespace

from forwards import prediction_forward
from model import (
    TAMO,
    TAMOConfig,
    build_apsl_head,
    build_objective_predictor,
)
from psl_tamo.data import prepare_stch_prediction_batches
from psl_tamo.forwards import (
    compute_apsl_loss,
    generate_apsl_solutions,
    project_psl_to_pool,
    select_next_preference,
)
import psl_tamo.forwards as psl_forwards
from psl_tamo.scalarization import smooth_tchebycheff


def _tiny_tamo():
    return TAMO(
        TAMOConfig(
            max_x_dim=5,
            max_y_dim=1,
            dim_mlp=16,
            dim_attn=16,
            nhead=2,
            num_layers_backbone=1,
            num_layers_encoder=1,
            num_layers_decoder=1,
            dim_hidden=16,
            depth=2,
            num_components=3,
        )
    )


def _tiny_apsl():
    model = _tiny_tamo()
    model.objective_predictor = build_objective_predictor(
        model.config, max_x_dim=3, max_y_dim=2
    )
    model.apsl_head = build_apsl_head(
        model.config, max_x_dim=3, max_y_dim=2, x_range=[-1.0, 1.0]
    )
    return model


def test_scalarization_mask_and_gradient():
    y = torch.randn(2, 7, 3, requires_grad=True)
    lambdas = torch.softmax(torch.randn_like(y), dim=-1)
    mask = torch.tensor([[True, True, False], [True, True, True]])
    scalar = smooth_tchebycheff(y, lambdas, -1.0, 0.1, mask)
    assert scalar.shape == (2, 7, 1)
    assert torch.isfinite(scalar).all()
    scalar.sum().backward()
    assert y.grad is not None

    changed_padding = y.detach().clone()
    changed_padding[0, :, 2] = 1e6
    scalar_changed = smooth_tchebycheff(changed_padding, lambdas, -1.0, 0.1, mask)
    torch.testing.assert_close(scalar[0].detach(), scalar_changed[0])


def test_augmented_prediction_batch():
    x, y = torch.randn(2, 20, 3), torch.randn(2, 20, 2)
    batch = prepare_stch_prediction_batches(
        x, y, torch.tensor([2, 3]), torch.tensor([1, 2]),
        "random_k", 2, 8, nc_fixed=5,
    )
    zc, sc, zt, st, z_mask, s_mask = batch
    assert zc.shape == (2, 5, 5)
    assert sc.shape == (2, 5, 1)
    assert zt.shape == (2, 15, 5)
    assert st.shape == (2, 15, 1)
    assert z_mask.shape == (2, 5)
    assert s_mask.shape == (2, 1) and s_mask.all()


def test_scalar_prediction_forward():
    model = _tiny_tamo()
    batch_size, nc, nt = 2, 4, 3
    zc, sc = torch.randn(batch_size, nc, 5), torch.randn(batch_size, nc, 1)
    z_mask = torch.ones(batch_size, 5, dtype=torch.bool)
    s_mask = torch.ones(batch_size, 1, dtype=torch.bool)
    zt, st = torch.randn(batch_size, nt, 5), torch.randn(batch_size, nt, 1)
    loss, mse, _ = prediction_forward(model, zc, sc, zt, st, z_mask, s_mask)
    assert torch.isfinite(loss) and mse.shape == (1,)
    loss.backward()


def test_objective_head_nll_and_apsl_gradient_isolation():
    model = _tiny_apsl()
    objective_model = model.objective_predictor
    batch_size, nc, nt = 2, 4, 3
    xc, yc = torch.randn(batch_size, nc, 3), torch.randn(batch_size, nc, 2)
    xt, yt = torch.randn(batch_size, nt, 3), torch.randn(batch_size, nt, 2)
    x_mask = torch.ones(batch_size, 3, dtype=torch.bool)
    objective_mask = torch.ones(batch_size, 2, dtype=torch.bool)

    loss, mse, _ = prediction_forward(
        objective_model, xc, yc, xt, yt, x_mask, objective_mask
    )
    assert torch.isfinite(loss) and mse.shape == (2,)
    loss.backward()
    assert any(parameter.grad is not None for parameter in objective_model.parameters())

    model.zero_grad(set_to_none=True)
    apsl_loss = compute_apsl_loss(
        model=model,
        x_ctx=xc[:1],
        y_ctx=yc[:1],
        x_mask=x_mask[:1],
        objective_mask=objective_mask[:1],
        ideal_point=yc[:1].min(dim=1).values,
        tau=0.1,
        num_preferences=8,
        preference_method="dirichlet",
    )
    apsl_loss.backward()
    assert torch.isfinite(apsl_loss)
    assert any(parameter.grad is not None for parameter in model.apsl_head.parameters())
    assert all(parameter.grad is None for parameter in objective_model.parameters())
    assert all(parameter.requires_grad for parameter in objective_model.parameters())


def test_apsl_head_is_history_conditioned_and_bounded():
    model = _tiny_apsl()
    x_ctx = torch.randn(2, 4, 3)
    y_ctx = torch.randn(2, 4, 2)
    x_mask = torch.ones(2, 3, dtype=torch.bool)
    y_mask = torch.ones(2, 2, dtype=torch.bool)
    preferences = torch.softmax(torch.randn(2, 5, 2), dim=-1)
    decisions = generate_apsl_solutions(
        model, x_ctx, y_ctx, preferences, x_mask, y_mask
    )
    assert decisions.shape == (2, 5, 3)
    assert (decisions >= -1.0).all() and (decisions <= 1.0).all()


def test_projection_is_unique_and_avoids_used_points():
    pool = torch.linspace(0, 1, 10).view(1, 10, 1)
    pool = torch.cat((pool, pool.square()), dim=-1)
    x_cont = pool[:, [0, 0, 1, 1]].clone()
    indices, projected, distances = project_psl_to_pool(
        x_cont, pool, torch.ones(2, dtype=torch.bool), used_indices=torch.tensor([[0, 1]])
    )
    assert indices.shape == (1, 4)
    assert indices.unique().numel() == 4
    assert not torch.isin(indices, torch.tensor([0, 1])).any()
    assert projected.shape == (1, 4, 2) and torch.isfinite(distances).all()


def test_action_wrapper_maps_to_candidate_and_keeps_logp_gradient():
    model = _tiny_tamo()
    z_ctx, s_ctx = torch.randn(2, 4, 5), torch.randn(2, 4, 1)
    candidates = torch.randn(2, 6, 5)
    z_mask = torch.ones(2, 5, dtype=torch.bool)
    result = select_next_preference(model, z_ctx, s_ctx, candidates, z_mask, 1, 3)
    assert ((0 <= result.indices) & (result.indices < 6)).all()
    assert result.log_probs.requires_grad
    assert torch.isfinite(result.entropy).all()


def test_tiny_rollout_end_to_end(monkeypatch):
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
            x_new = torch.gather(self._x, 1, index_new.expand(-1, -1, self._x.shape[-1]))
            y_new = torch.gather(self._y, 1, index_new.expand(-1, -1, self._y.shape[-1]))
            x_ctx = x_new if x_ctx is None else torch.cat((x_ctx, x_new), dim=1)
            y_ctx = y_new if y_ctx is None else torch.cat((y_ctx, y_new), dim=1)
            regret = np.full(self.batch_size, 1.0 / y_ctx.shape[1]) if compute_regret else None
            return x_ctx, y_ctx, None, regret

    monkeypatch.setattr(psl_forwards, "GPSampleFunction", FakeDiscreteEnvironment)
    model = _tiny_apsl()
    data = SimpleNamespace(max_x_dim=3, max_y_dim=2, x_range=[-1.0, 1.0])
    opt = SimpleNamespace(
        batch_size=2, num_samples=1, num_query_points=10,
        use_grid_sampling=False, num_initial_points=3, regret_type="norm_ratio",
        use_time_budget=True, epsilon=1.0,
    )
    loss = SimpleNamespace(
        use_cumulative_rewards=True, discount_factor=0.99,
        batch_standardize=True, clip_rewards=True,
    )
    result = psl_forwards.optimization_forward_psl(
        model, data, opt, loss,
        {"loss_weight": 0.5, "num_train_preferences": 4,
         "num_policy_preferences": 4},
        {"tau": 0.1}, T=2, device="cpu",
    )
    assert torch.isfinite(result[0]) and torch.isfinite(result[1])
    (result[0] + result[1]).backward()
    assert any(parameter.grad is not None for parameter in model.parameters())
