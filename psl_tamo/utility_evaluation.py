"""Evaluation flow for PSL with a TAMO objective-utility surrogate."""

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
from .psl_inner_loop import update_psl_inner_loop
from .utility_selection import select_preference_by_utility


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

    def to_cpu_dict(self) -> dict[str, Tensor]:
        return {key: value.detach().cpu() for key, value in vars(self).items()}


def _active_mask(max_dim: int, active_dim: int, device: str) -> Tensor:
    if active_dim > max_dim:
        raise ValueError(f"Active dimension {active_dim} exceeds model limit {max_dim}")
    mask = torch.zeros(max_dim, device=device, dtype=torch.bool)
    mask[:active_dim] = True
    return mask


def _format_batch(values: Tensor) -> str:
    return ", ".join(f"{value:.6f}" for value in values.detach().cpu().tolist())


def _ideal_point(y_ctx: Tensor, objective_mask: Tensor, setting: Any) -> Tensor:
    """Resolve either the current observed ideal point or a fixed value."""
    if isinstance(setting, str):
        if setting != "observed_min":
            raise ValueError("ideal_point must be numeric or 'observed_min'")
        masked = y_ctx.masked_fill(~objective_mask.unsqueeze(1), float("inf"))
        return masked.amin(dim=1)
    return torch.full(
        (y_ctx.shape[0], y_ctx.shape[-1]),
        float(setting),
        device=y_ctx.device,
        dtype=y_ctx.dtype,
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
    """Optimize by selecting ``lambda`` and evaluating ``h_phi(lambda)``.

    The TAMO model is trained only as an objective-wise conditional surrogate.
    During each PSL inner loop its parameters are frozen; gradients travel from
    expected smooth Tchebycheff through the utility prediction to ``h_phi``.
    """
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
    x_model = restore_by_mask(x_context, x_mask_1d, dim=-1)
    y_scaled = test_function.transform_outputs(
        outputs=y_context, output_bounds=data_config.y_range
    )
    y_model = restore_by_mask(y_scaled, y_mask_1d, dim=-1)

    # Original PSL-MOBO model.py: m -> 256 -> 256 -> d with ReLU/ReLU/Sigmoid.
    lower, upper = data_config.x_range
    psl_models = [
        ParetoSetMLP(
            preference_dim=test_function.y_dim,
            decision_dim=test_function.x_dim,
            hidden_dim=psl_config.get("hidden_dim", 256),
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
    preference_method = psl_config.get("preference_method", "dirichlet")
    tau = float(scalarization_config.get("tau", 0.1))
    ideal_setting = scalarization_config.get("ideal_point", "observed_min")
    beta = float(utility_config.get("beta", 0.5))
    ideal = _ideal_point(y_model, objective_mask, ideal_setting)
    psl_loss = update_psl_inner_loop(
        psl_models=psl_models,
        psl_optimizers=psl_optimizers,
        objective_model=model,
        x_ctx=x_model,
        y_ctx=y_model,
        x_mask=x_mask,
        objective_mask=objective_mask,
        ideal_point=ideal,
        tau=tau,
        num_preferences=psl_config.get("num_train_preferences", 10),
        num_steps=psl_config.get("init_steps", 1000),
        preference_method=preference_method,
        utility_beta=beta,
        compact_psl_dimensions=True,
    )

    hv_history = [hv.detach().clone()]
    x_queries, y_queries, selected_preferences = [], [], []
    entropy_history, psl_loss_history = [], [psl_loss]
    max_hv = torch.as_tensor(test_function.max_hv, device=device, dtype=hv.dtype)
    max_hv = max_hv.expand_as(hv)
    total_evaluations = num_initial + horizon
    started_at = time.time()
    log(
        "==== Utility PSL-TAMO evaluation budget ====\n"
        f"  initial evaluations:\t{num_initial}\n"
        f"  preference-space evaluations:\t{horizon}\n"
        f"  total evaluations:\t{total_evaluations}\n"
        f"  utility:\tLCB(mean - {beta:g} * std), objective-wise\n"
        f"  scalarization:\tsmooth Tchebycheff (mu={tau:g})\n"
        f"  Pareto-set MLP:\t{test_function.y_dim}->"
        f"{psl_config.get('hidden_dim', 256)}->"
        f"{psl_config.get('hidden_dim', 256)}->{test_function.x_dim}\n"
        f"  initial HV:\t{_format_batch(hv)}"
    )

    for step in range(1, horizon + 1):
        ideal = _ideal_point(y_model, objective_mask, ideal_setting)
        update_steps = psl_config.get("update_steps", 1000)
        if update_steps > 0:
            psl_loss = update_psl_inner_loop(
                psl_models=psl_models,
                psl_optimizers=psl_optimizers,
                objective_model=model,
                x_ctx=x_model,
                y_ctx=y_model,
                x_mask=x_mask,
                objective_mask=objective_mask,
                ideal_point=ideal,
                tau=tau,
                num_preferences=psl_config.get("num_train_preferences", 10),
                num_steps=update_steps,
                preference_method=preference_method,
                utility_beta=beta,
                compact_psl_dimensions=True,
            )

        preferences = sample_padded_preferences(
            objective_mask,
            psl_config.get("num_policy_preferences", 1000),
            preference_method,
            x_model.dtype,
        )
        with torch.no_grad():
            x_candidates_compact = torch.stack(
                [
                    psl(preferences[b, :, y_mask_1d])
                    for b, psl in enumerate(psl_models)
                ]
            )
            x_candidates = restore_by_mask(
                x_candidates_compact, x_mask_1d, dim=-1
            )
            action = select_preference_by_utility(
                objective_model=model,
                x_ctx=x_model,
                y_ctx=y_model,
                x_candidates=x_candidates,
                preferences=preferences,
                x_mask=x_mask,
                objective_mask=objective_mask,
                ideal_point=ideal,
                tau=tau,
                beta=beta,
                epsilon=optimization_config.epsilon,
                temperature=float(utility_config.get("temperature", 1.0)),
            )
            x_next_model = action.decisions
            selected_lambda = action.preferences

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
        y_next_model = restore_by_mask(y_next_scaled, y_mask_1d, dim=-1)
        x_model = torch.cat((x_model, x_next_model), dim=1)
        y_model = torch.cat((y_model, y_next_model), dim=1)

        hv_history.append(hv.detach().clone())
        x_queries.append(x_next.detach())
        y_queries.append(y_next.detach())
        selected_preferences.append(selected_lambda.detach())
        entropy_history.append(action.entropy.detach())
        psl_loss_history.append(psl_loss)

        elapsed = time.time() - started_at
        eta = elapsed / step * (horizon - step)
        if step == 1 or step == horizon or step % psl_config.get("log_every", 1) == 0:
            ratio = hv / max_hv.clamp_min(1e-12)
            log(
                f"[Utility-PSL step {step:03d}/{horizon}; FE "
                f"{num_initial + step}/{total_evaluations}] "
                f"HV={_format_batch(hv)} HV/maxHV={_format_batch(ratio)} "
                f"PSL-loss={psl_loss:.6f} elapsed={format_duration(elapsed)} "
                f"ETA={format_duration(eta)}"
            )

    result = UtilityPSLRolloutResult(
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
        "==== Utility PSL-TAMO final HV ====\n"
        f"  evaluations:\t{total_evaluations}\n"
        f"  final HV:\t{_format_batch(result.hv[:, -1])}\n"
        f"  final HV / max HV:\t{_format_batch(final_ratio)}"
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
