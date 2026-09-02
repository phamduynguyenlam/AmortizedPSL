"""Deployment of the REINFORCE-trained Utility-PSL-TAMO policy."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Mapping

import torch
from torch import Tensor

from data.base.masking import restore_by_mask
from model import ParetoSetMLP
from utils.log import format_duration
from .preferences import sample_padded_preferences
from .utility_policy import (
    infer_history_preferences,
    select_preference_with_policy,
    update_preference_to_x_mlp,
)


@dataclass
class UtilityPSLRolloutResult:
    hv: Tensor
    x_context: Tensor
    y_context: Tensor
    x_queries: Tensor
    y_queries: Tensor
    preferences: Tensor
    entropy: Tensor
    psl_loss: Tensor
    inverse_preference_loss: Tensor

    def to_cpu_dict(self) -> dict[str, Tensor]:
        return {key: value.detach().cpu() for key, value in vars(self).items()}


def _active_mask(max_dim: int, active_dim: int, device: str) -> Tensor:
    if active_dim > max_dim:
        raise ValueError(f"Active dimension {active_dim} exceeds limit {max_dim}")
    mask = torch.zeros(max_dim, device=device, dtype=torch.bool)
    mask[:active_dim] = True
    return mask


def _format_batch(values: Tensor) -> str:
    return ", ".join(f"{value:.6f}" for value in values.detach().cpu().tolist())


def _ideal_point(y_ctx: Tensor, objective_mask: Tensor, setting: Any) -> Tensor:
    if isinstance(setting, str):
        if setting != "observed_min":
            raise ValueError("ideal_point must be numeric or 'observed_min'")
        return y_ctx.masked_fill(
            ~objective_mask.unsqueeze(1), float("inf")
        ).amin(dim=1)
    return torch.full(
        (y_ctx.shape[0], y_ctx.shape[-1]), float(setting),
        device=y_ctx.device, dtype=y_ctx.dtype,
    )


def run_utility_psl_optimization(
    model,
    test_function,
    data_config,
    optimization_config,
    psl_config: Mapping[str, Any],
    scalarization_config: Mapping[str, Any],
    utility_config: Mapping[str, Any],
    device: str,
    seed: int,
    log: callable = print,
) -> UtilityPSLRolloutResult:
    """Choose lambda with TAMO's policy, then evaluate h_theta(lambda)."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    batch_size = optimization_config.batch_size
    num_initial = optimization_config.num_initial_points
    horizon = optimization_config.sample_T()
    max_x_dim, max_y_dim = data_config.max_x_dim, data_config.max_y_dim
    x_mask_1d = _active_mask(max_x_dim, test_function.x_dim, device)
    y_mask_1d = _active_mask(max_y_dim, test_function.y_dim, device)
    x_mask = x_mask_1d.unsqueeze(0).expand(batch_size, -1)
    objective_mask = y_mask_1d.unsqueeze(0).expand(batch_size, -1)
    z_mask = torch.cat((x_mask, objective_mask), dim=-1)

    x_context, y_context, hv, _ = test_function.init(
        input_bounds=data_config.x_range, batch_size=batch_size,
        num_initial_points=num_initial,
        regret_type=optimization_config.regret_type,
        compute_hv=True, compute_regret=False, device=device, seed=seed,
    )
    x_model = restore_by_mask(x_context, x_mask_1d, dim=-1)
    y_model = restore_by_mask(
        test_function.transform_outputs(
            outputs=y_context, output_bounds=data_config.y_range
        ), y_mask_1d, dim=-1,
    )

    lower, upper = data_config.x_range
    psl_models = [
        ParetoSetMLP(
            test_function.y_dim, test_function.x_dim,
            psl_config.get("hidden_dim", 256), psl_config.get("depth", 3),
            lower, upper,
        ).to(device)
        for _ in range(batch_size)
    ]
    psl_optimizers = [
        torch.optim.Adam(psl.parameters(), lr=psl_config.get("lr", 1e-3))
        for psl in psl_models
    ]
    preference_method = psl_config.get("preference_method", "dirichlet")
    lambda_star = sample_padded_preferences(
        objective_mask, num_initial, preference_method, y_model.dtype
    )
    inverse_kwargs = dict(
        num_steps=psl_config.get("inverse_steps", 10),
        lr=psl_config.get("inverse_lr", 5e-2),
        reconstruction_weight=psl_config.get("inverse_x_weight", 1.0),
        reference_weight=psl_config.get("inverse_lambda_weight", 1.0),
    )
    model = model.to(device)
    model.eval()
    tau = float(scalarization_config.get("tau", 0.1))
    ideal_setting = scalarization_config.get("ideal_point", -1.0)
    beta = float(utility_config.get("beta", 0.5))
    history_lambda, inverse_loss = infer_history_preferences(
        psl_models, x_model, lambda_star, x_mask, objective_mask,
        **inverse_kwargs,
    )
    psl_loss = update_preference_to_x_mlp(
        psl_models, psl_optimizers, model, x_model, y_model, history_lambda,
        x_mask, objective_mask,
        _ideal_point(y_model, objective_mask, ideal_setting), tau, beta,
        psl_config.get("num_train_preferences", 10),
        psl_config.get("init_steps", 1000), preference_method,
    )

    hv_history = [hv.detach().clone()]
    x_queries, y_queries, chosen_preferences = [], [], []
    entropy_history, psl_losses = [], [psl_loss]
    inverse_losses = [inverse_loss]
    max_hv = torch.as_tensor(
        test_function.max_hv, device=device, dtype=hv.dtype
    ).expand_as(hv)
    started_at = time.time()
    log(
        "==== Utility-PSL-TAMO learned-policy evaluation ====\n"
        f"  budget:\t{num_initial}+{horizon} evaluations\n"
        f"  policy input:\t(x, y, inferred lambda) history\n"
        f"  initial HV:\t{_format_batch(hv)}"
    )

    for step in range(1, horizon + 1):
        history_lambda, inverse_loss = infer_history_preferences(
            psl_models, x_model, lambda_star, x_mask, objective_mask,
            **inverse_kwargs,
        )
        update_steps = psl_config.get("update_steps", 1)
        if update_steps > 0:
            psl_loss = update_preference_to_x_mlp(
                psl_models, psl_optimizers, model, x_model, y_model,
                history_lambda, x_mask, objective_mask,
                _ideal_point(y_model, objective_mask, ideal_setting), tau, beta,
                psl_config.get("num_train_preferences", 10), update_steps,
                preference_method,
            )
            history_lambda, inverse_loss = infer_history_preferences(
                psl_models, x_model, lambda_star, x_mask, objective_mask,
                **inverse_kwargs,
            )

        preferences = sample_padded_preferences(
            objective_mask, psl_config.get("num_policy_preferences", 256),
            preference_method, x_model.dtype,
        )
        with torch.no_grad():
            compact_candidates = torch.stack([
                psl(preferences[b, :, y_mask_1d])
                for b, psl in enumerate(psl_models)
            ])
            x_candidates = restore_by_mask(
                compact_candidates, x_mask_1d, dim=-1
            )
            action = select_preference_with_policy(
                model,
                torch.cat((x_model, history_lambda), dim=-1), y_model,
                torch.cat((x_candidates, preferences), dim=-1),
                z_mask, objective_mask, step, horizon,
                optimization_config.use_time_budget,
                optimization_config.epsilon,
            )
            batch = torch.arange(batch_size, device=device)
            x_next_model = x_candidates[batch, action.indices].unsqueeze(1)
            selected_lambda = preferences[batch, action.indices].unsqueeze(1)

        x_context, y_context, hv, _ = test_function.step(
            input_bounds=data_config.x_range,
            x_new=x_next_model[..., x_mask_1d],
            x_ctx=x_context, y_ctx=y_context,
            compute_hv=True, compute_regret=False,
        )
        y_next = y_context[:, -1:]
        y_next_model = restore_by_mask(
            test_function.transform_outputs(
                outputs=y_next, output_bounds=data_config.y_range
            ), y_mask_1d, dim=-1,
        )
        x_model = torch.cat((x_model, x_next_model), dim=1)
        y_model = torch.cat((y_model, y_next_model), dim=1)
        lambda_star = torch.cat((lambda_star, selected_lambda), dim=1)

        hv_history.append(hv.detach().clone())
        x_queries.append(x_next_model[..., x_mask_1d].detach())
        y_queries.append(y_next.detach())
        chosen_preferences.append(selected_lambda.detach())
        entropy_history.append(action.entropy.detach())
        psl_losses.append(psl_loss)
        inverse_losses.append(inverse_loss)
        elapsed = time.time() - started_at
        eta = elapsed / step * (horizon - step)
        if step == 1 or step == horizon or step % psl_config.get("log_every", 1) == 0:
            log(
                f"[Utility-PSL step {step:03d}/{horizon}] "
                f"HV={_format_batch(hv)} "
                f"HV/maxHV={_format_batch(hv / max_hv.clamp_min(1e-12))} "
                f"PSL-loss={psl_loss:.6f} inverse-loss={inverse_loss:.6f} "
                f"elapsed={format_duration(elapsed)} ETA={format_duration(eta)}"
            )

    result = UtilityPSLRolloutResult(
        hv=torch.stack(hv_history, dim=1), x_context=x_context,
        y_context=y_context, x_queries=torch.cat(x_queries, dim=1),
        y_queries=torch.cat(y_queries, dim=1),
        preferences=torch.cat(chosen_preferences, dim=1),
        entropy=torch.stack(entropy_history, dim=1),
        psl_loss=torch.as_tensor(psl_losses, dtype=torch.float32),
        inverse_preference_loss=torch.as_tensor(
            inverse_losses, dtype=torch.float32
        ),
    )
    log(
        "==== Utility-PSL-TAMO final HV ====\n"
        f"  evaluations:\t{num_initial + horizon}\n"
        f"  final HV:\t{_format_batch(result.hv[:, -1])}\n"
        f"  final HV / max HV:\t"
        f"{_format_batch(result.hv[:, -1] / max_hv.clamp_min(1e-12))}"
    )
    return result


def save_utility_psl_rollout(
    result: UtilityPSLRolloutResult, output_dir: str, log=print
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "utility_psl_rollout.pt")
    torch.save(result.to_cpu_dict(), output_path)
    log(f"Utility PSL rollout saved to:\t{output_path}")
    return output_path
