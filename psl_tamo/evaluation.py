"""Deployment-time APSL rollout on a real test function."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Mapping

import torch
from torch import Tensor

from data.base.masking import restore_by_mask
from model import TAMO
from utils.log import format_duration
from .forwards import generate_apsl_solutions, select_next_preference
from .preferences import sample_padded_preferences
from .scalarization import smooth_tchebycheff


@dataclass
class APSLRolloutResult:
    """Tensors produced by an APSL deployment rollout."""

    hv: Tensor
    x_context: Tensor
    y_context: Tensor
    x_queries: Tensor
    y_queries: Tensor
    preferences: Tensor
    entropy: Tensor
    apsl_loss: Tensor

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


def run_apsl_optimization(
    model: TAMO,
    test_function,
    data_config,
    optimization_config,
    psl_config: Mapping[str, Any],
    scalarization_config: Mapping[str, Any],
    device: str,
    seed: int,
    log: callable = print,
) -> APSLRolloutResult:
    """Run ``N_init + T`` function evaluations and report TAMO-compatible HV.

    Hypervolume is always computed from unscaled, true objective observations by
    ``test_function.compute_hv``--the same implementation used by TAMO's
    ``evaluate.run_optimization``. Scaling is limited to surrogate/PSL inputs.
    """
    objective_model = getattr(model, "objective_predictor", None)
    if objective_model is None:
        raise ValueError("APSL checkpoint must contain an objective predictor")
    if getattr(model, "apsl_head", None) is None:
        raise ValueError("APSL checkpoint must contain an amortized Pareto-set head")

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

    model = model.to(device)
    model.eval()

    hv_history = [hv.detach().clone()]
    x_queries, y_queries, selected_preferences = [], [], []
    entropy_history, apsl_loss_history = [], []
    max_hv = torch.as_tensor(
        test_function.max_hv, device=device, dtype=hv.dtype
    ).expand_as(hv)
    total_evaluations = num_initial + horizon
    started_at = time.time()
    log(
        "==== APSL evaluation budget ====\n"
        f"  initial evaluations:\t{num_initial}\n"
        f"  APSL evaluations:\t{horizon}\n"
        f"  total evaluations:\t{total_evaluations}\n"
        f"  initial HV:\t{_format_batch(hv)}\n"
        f"  max HV:\t{_format_batch(max_hv)}"
    )

    for step in range(1, horizon + 1):
        lambdas = sample_padded_preferences(
            objective_mask,
            psl_config.get("num_policy_preferences", 256),
            preference_method,
            x_model.dtype,
        )
        with torch.no_grad():
            x_candidates = generate_apsl_solutions(
                model, x_model, y_model, lambdas, x_mask, objective_mask
            )
            predicted_candidates = objective_model.predictive_mean(
                x_ctx=x_model,
                y_ctx=y_model,
                x_tar=x_candidates,
                x_mask=x_mask,
                y_mask=objective_mask,
            )
            apsl_loss = smooth_tchebycheff(
                predicted_candidates,
                lambdas,
                ideal_point.unsqueeze(1),
                tau,
                objective_mask,
            ).mean()
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
        apsl_loss_history.append(apsl_loss.detach())

        elapsed = time.time() - started_at
        eta = elapsed / step * (horizon - step)
        hv_ratio = hv / max_hv.clamp_min(1e-12)
        if step == 1 or step == horizon or step % psl_config.get("log_every", 1) == 0:
            log(
                f"[APSL step {step:03d}/{horizon}; FE {num_initial + step}/"
                f"{total_evaluations}] HV={_format_batch(hv)} "
                f"HV/maxHV={_format_batch(hv_ratio)} "
                f"APSL-loss={apsl_loss.item():.6f} elapsed={format_duration(elapsed)} "
                f"ETA={format_duration(eta)}"
            )

    result = APSLRolloutResult(
        hv=torch.stack(hv_history, dim=1),
        x_context=x_context,
        y_context=y_context,
        x_queries=torch.cat(x_queries, dim=1),
        y_queries=torch.cat(y_queries, dim=1),
        preferences=torch.cat(selected_preferences, dim=1),
        entropy=torch.stack(entropy_history, dim=1),
        apsl_loss=torch.stack(apsl_loss_history),
    )
    final_ratio = result.hv[:, -1] / max_hv.clamp_min(1e-12)
    log(
        "==== APSL final HV ====\n"
        f"  evaluations:\t{total_evaluations}\n"
        f"  final HV:\t{_format_batch(result.hv[:, -1])}\n"
        f"  final HV / max HV:\t{_format_batch(final_ratio)}"
    )
    return result


def save_apsl_rollout(result: APSLRolloutResult, output_dir: str, log=print) -> str:
    """Save one self-contained rollout artifact."""
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "apsl_rollout.pt")
    torch.save(result.to_cpu_dict(), output_path)
    log(f"APSL rollout saved to:\t{output_path}")
    return output_path


# Compatibility aliases for callers of the former experimental name.
PSLRolloutResult = APSLRolloutResult
run_psl_optimization = run_apsl_optimization
save_psl_rollout = save_apsl_rollout
