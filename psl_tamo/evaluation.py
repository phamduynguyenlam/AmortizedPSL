"""Deployment-time PSL-TAMO rollout on a real test function."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Mapping

import torch
from torch import Tensor

from data.base.masking import restore_by_mask
from model import ParetoSetMLP, TAMO
from utils.log import format_duration
from .forwards import select_next_preference
from .preferences import sample_padded_preferences
from .psl_inner_loop import update_psl_inner_loop
from .scalarization import smooth_tchebycheff


@dataclass
class PSLRolloutResult:
    """Tensors produced by a PSL-TAMO deployment rollout."""

    hv: Tensor
    x_context: Tensor
    y_context: Tensor
    x_queries: Tensor
    y_queries: Tensor
    preferences: Tensor
    entropy: Tensor
    psl_loss: Tensor

    def to_cpu_dict(self) -> dict[str, Tensor]:
        return {
            key: value.detach().cpu()
            for key, value in vars(self).items()
        }


def _active_mask(max_dim: int, active_dim: int, device: str) -> Tensor:
    if active_dim > max_dim:
        raise ValueError(f"Active dimension {active_dim} exceeds model limit {max_dim}")
    mask = torch.zeros(max_dim, device=device, dtype=torch.bool)
    mask[:active_dim] = True
    return mask


def _pad_for_model(values: Tensor, mask: Tensor) -> Tensor:
    return restore_by_mask(values, mask, dim=-1)


def _format_batch(values: Tensor) -> str:
    return ", ".join(f"{value:.6f}" for value in values.detach().cpu().tolist())


def run_psl_optimization(
    model: TAMO,
    test_function,
    data_config,
    optimization_config,
    psl_config: Mapping[str, Any],
    scalarization_config: Mapping[str, Any],
    device: str,
    seed: int,
    log: callable = print,
) -> PSLRolloutResult:
    """Run ``N_init + T`` function evaluations and report TAMO-compatible HV.

    Hypervolume is always computed from unscaled, true objective observations by
    ``test_function.compute_hv``--the same implementation used by TAMO's
    ``evaluate.run_optimization``. Scaling is limited to surrogate/PSL inputs.
    """
    objective_model = getattr(model, "objective_predictor", None)
    if objective_model is None:
        raise ValueError("PSL-TAMO checkpoint must contain an objective predictor")

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    batch_size = optimization_config.batch_size
    num_initial = optimization_config.num_initial_points
    horizon = optimization_config.sample_T()
    max_x_dim = data_config.max_x_dim
    max_y_dim = data_config.max_y_dim
    x_mask_1d = _active_mask(max_x_dim, test_function.x_dim, device)
    y_mask_1d = _active_mask(max_y_dim, test_function.y_dim, device)
    x_mask = x_mask_1d.unsqueeze(0).expand(batch_size, -1)
    objective_mask = y_mask_1d.unsqueeze(0).expand(batch_size, -1)
    z_mask = torch.cat((x_mask, objective_mask), dim=-1)

    # TestFunction operates in compact real dimensions. Model tensors are padded.
    x_context, y_context, hv, _ = test_function.init(
        input_bounds=data_config.x_range,
        batch_size=batch_size,
        num_initial_points=num_initial,
        regret_type=optimization_config.regret_type,
        compute_hv=True,
        compute_regret=False,
        device=device,
        seed=seed,
    )
    x_model = _pad_for_model(x_context, x_mask_1d)
    y_scaled = test_function.transform_outputs(
        outputs=y_context, output_bounds=data_config.y_range
    )
    y_model = _pad_for_model(y_scaled, y_mask_1d)

    preference_method = psl_config.get("preference_method", "dirichlet")
    lambdas_context = sample_padded_preferences(
        objective_mask, num_initial, preference_method, x_model.dtype
    )
    ideal_point = torch.full(
        (batch_size, max_y_dim),
        float(scalarization_config.get("ideal_point", -1.0)),
        device=device,
        dtype=y_model.dtype,
    )
    tau = float(scalarization_config.get("tau", 0.1))
    z_context = torch.cat((x_model, lambdas_context), dim=-1)
    s_context = smooth_tchebycheff(
        y_model, lambdas_context, ideal_point.unsqueeze(1), tau, objective_mask
    )

    lower, upper = data_config.x_range
    psl_models = [
        ParetoSetMLP(
            preference_dim=max_y_dim,
            decision_dim=max_x_dim,
            hidden_dim=psl_config.get("hidden_dim", 128),
            depth=psl_config.get("depth", 3),
            x_lower=lower,
            x_upper=upper,
        ).to(device)
        for _ in range(batch_size)
    ]
    psl_optimizers = [
        torch.optim.Adam(psl.parameters(), lr=psl_config.get("lr", 1e-3))
        for psl in psl_models
    ]

    model = model.to(device)
    model.eval()
    psl_loss = update_psl_inner_loop(
        psl_models=psl_models,
        psl_optimizers=psl_optimizers,
        objective_model=objective_model,
        x_ctx=x_model,
        y_ctx=y_model,
        x_mask=x_mask,
        objective_mask=objective_mask,
        ideal_point=ideal_point,
        tau=tau,
        num_preferences=psl_config.get("num_train_preferences", 64),
        num_steps=psl_config.get("init_steps", 50),
        preference_method=preference_method,
    )

    hv_history = [hv.detach().clone()]
    x_queries, y_queries, selected_preferences = [], [], []
    entropy_history, psl_loss_history = [], [psl_loss]
    max_hv = torch.as_tensor(
        test_function.max_hv, device=device, dtype=hv.dtype
    ).expand_as(hv)
    total_evaluations = num_initial + horizon
    started_at = time.time()
    log(
        "==== PSL-TAMO evaluation budget ====\n"
        f"  initial evaluations:\t{num_initial}\n"
        f"  PSL evaluations:\t{horizon}\n"
        f"  total evaluations:\t{total_evaluations}\n"
        f"  initial HV:\t{_format_batch(hv)}\n"
        f"  max HV:\t{_format_batch(max_hv)}"
    )

    for step in range(1, horizon + 1):
        update_steps = psl_config.get("update_steps", 5)
        if update_steps > 0:
            psl_loss = update_psl_inner_loop(
                psl_models=psl_models,
                psl_optimizers=psl_optimizers,
                objective_model=objective_model,
                x_ctx=x_model,
                y_ctx=y_model,
                x_mask=x_mask,
                objective_mask=objective_mask,
                ideal_point=ideal_point,
                tau=tau,
                num_preferences=psl_config.get("num_train_preferences", 64),
                num_steps=update_steps,
                preference_method=preference_method,
            )

        lambdas = sample_padded_preferences(
            objective_mask,
            psl_config.get("num_policy_preferences", 256),
            preference_method,
            x_model.dtype,
        )
        with torch.no_grad():
            x_candidates = torch.stack(
                [psl(lambdas[b]) for b, psl in enumerate(psl_models)]
            )
            z_candidates = torch.cat((x_candidates, lambdas), dim=-1)
            action = select_next_preference(
                model=model,
                z_ctx=z_context,
                s_ctx=s_context,
                z_candidates=z_candidates,
                z_mask=z_mask,
                t=step,
                T=horizon,
                use_budget=optimization_config.use_time_budget,
                epsilon=optimization_config.epsilon,
            )
            gather_x = action.indices[:, None, None].expand(-1, 1, max_x_dim)
            gather_y = action.indices[:, None, None].expand(-1, 1, max_y_dim)
            x_next_model = torch.gather(x_candidates, 1, gather_x)
            selected_lambda = torch.gather(lambdas, 1, gather_y)

        x_next = x_next_model[..., x_mask_1d]
        x_context, y_context, hv, _ = test_function.step(
            input_bounds=data_config.x_range,
            x_new=x_next,
            x_ctx=x_context,
            y_ctx=y_context,
            compute_hv=True,
            compute_regret=False,
        )
        y_next = y_context[:, -1:]
        y_next_scaled = test_function.transform_outputs(
            outputs=y_next, output_bounds=data_config.y_range
        )
        y_next_model = _pad_for_model(y_next_scaled, y_mask_1d)
        x_model = torch.cat((x_model, x_next_model), dim=1)
        y_model = torch.cat((y_model, y_next_model), dim=1)
        z_context = torch.cat(
            (z_context, torch.cat((x_next_model, selected_lambda), dim=-1)), dim=1
        )
        s_next = smooth_tchebycheff(
            y_next_model,
            selected_lambda,
            ideal_point.unsqueeze(1),
            tau,
            objective_mask,
        )
        s_context = torch.cat((s_context, s_next), dim=1)

        hv_history.append(hv.detach().clone())
        x_queries.append(x_next.detach())
        y_queries.append(y_next.detach())
        selected_preferences.append(selected_lambda.detach())
        entropy_history.append(action.entropy.detach())
        psl_loss_history.append(psl_loss)

        elapsed = time.time() - started_at
        eta = elapsed / step * (horizon - step)
        hv_ratio = hv / max_hv.clamp_min(1e-12)
        if step == 1 or step == horizon or step % psl_config.get("log_every", 1) == 0:
            log(
                f"[PSL step {step:03d}/{horizon}; FE {num_initial + step}/"
                f"{total_evaluations}] HV={_format_batch(hv)} "
                f"HV/maxHV={_format_batch(hv_ratio)} "
                f"PSL-loss={psl_loss:.6f} elapsed={format_duration(elapsed)} "
                f"ETA={format_duration(eta)}"
            )

    result = PSLRolloutResult(
        hv=torch.stack(hv_history, dim=1),
        x_context=x_context,
        y_context=y_context,
        x_queries=torch.cat(x_queries, dim=1),
        y_queries=torch.cat(y_queries, dim=1),
        preferences=torch.cat(selected_preferences, dim=1),
        entropy=torch.stack(entropy_history, dim=1),
        psl_loss=torch.as_tensor(psl_loss_history, dtype=torch.float32),
    )
    final_ratio = result.hv[:, -1] / max_hv.clamp_min(1e-12)
    log(
        "==== PSL-TAMO final HV ====\n"
        f"  evaluations:\t{total_evaluations}\n"
        f"  final HV:\t{_format_batch(result.hv[:, -1])}\n"
        f"  final HV / max HV:\t{_format_batch(final_ratio)}"
    )
    return result


def save_psl_rollout(result: PSLRolloutResult, output_dir: str, log=print) -> str:
    """Save one self-contained rollout artifact."""
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "psl_rollout.pt")
    torch.save(result.to_cpu_dict(), output_path)
    log(f"PSL rollout saved to:\t{output_path}")
    return output_path
