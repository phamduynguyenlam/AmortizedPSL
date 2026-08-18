"""PSL-induced preference action rollout for the unchanged TAMO model."""

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor, nn

from data.base.masking import restore_by_mask
from data.gp_sample_function import GPSampleFunction
from forwards import compute_policy_loss
from model import ParetoSetMLP, TAMO
from .preferences import sample_padded_preferences
from .psl_inner_loop import update_psl_inner_loop
from .scalarization import smooth_tchebycheff


@dataclass
class PreferenceAction:
    indices: Tensor
    log_probs: Tensor
    entropy: Tensor
    logits: Optional[Tensor]


def _full_pool(pool_x: Tensor, x_mask: Tensor) -> Tensor:
    if pool_x.shape[-1] == x_mask.numel():
        return pool_x
    return restore_by_mask(pool_x, x_mask, dim=-1)


def project_psl_to_pool(
    x_cont: Tensor,
    pool_x: Tensor,
    x_mask: Tensor,
    used_indices: Optional[Tensor] = None,
):
    """Greedily project PSL outputs to distinct, unevaluated pool points."""
    if x_cont.ndim == 2:
        x_cont = x_cont.unsqueeze(0)
    if pool_x.ndim == 2:
        pool_x = pool_x.unsqueeze(0).expand(x_cont.shape[0], -1, -1)
    if x_mask.ndim != 1:
        raise ValueError("Synthetic tasks use one shared x mask per rollout")

    pool_full = _full_pool(pool_x, x_mask)
    distances = torch.cdist(x_cont[..., x_mask], pool_full[..., x_mask])
    batch_size, num_candidates, pool_size = distances.shape
    if num_candidates > pool_size:
        raise ValueError("Cannot return unique candidates when Q exceeds pool size")

    selected = torch.empty(
        batch_size, num_candidates, device=x_cont.device, dtype=torch.long
    )
    for b in range(batch_size):
        unavailable = torch.zeros(pool_size, device=x_cont.device, dtype=torch.bool)
        if used_indices is not None:
            valid_used = used_indices[b].reshape(-1)
            valid_used = valid_used[(valid_used >= 0) & (valid_used < pool_size)]
            unavailable[valid_used] = True
        if (~unavailable).sum() < num_candidates:
            raise ValueError("Not enough unevaluated pool points for unique PSL candidates")
        for q in range(num_candidates):
            row = distances[b, q].masked_fill(unavailable, float("inf"))
            index = row.argmin()
            selected[b, q] = index
            unavailable[index] = True

    projected = torch.gather(
        pool_full, 1, selected.unsqueeze(-1).expand(-1, -1, pool_full.shape[-1])
    )
    selected_distances = torch.gather(distances, 2, selected.unsqueeze(-1)).squeeze(-1)
    return selected, projected, selected_distances


def _indices_for_observations(points: Tensor, pool_x: Tensor, x_mask: Tensor) -> Tensor:
    pool_full = _full_pool(pool_x, x_mask)
    distances = torch.cdist(points[..., x_mask], pool_full[..., x_mask])
    return distances.argmin(dim=-1)


def select_next_preference(
    model: TAMO,
    z_ctx: Tensor,
    s_ctx: Tensor,
    z_candidates: Tensor,
    z_mask: Tensor,
    t: int,
    T: int,
    use_budget: bool = True,
    epsilon: float = 1.0,
) -> PreferenceAction:
    """Call the original categorical policy on one augmented candidate space."""
    batch_size, num_candidates, aug_dim = z_candidates.shape
    results = model.action(
        x_ctx=z_ctx,
        y_ctx=s_ctx,
        x_mask=z_mask,
        y_mask=torch.ones((batch_size, 1), device=z_ctx.device, dtype=torch.bool),
        query_chunks=z_candidates.unsqueeze(1),
        query_x_mask=z_mask.unsqueeze(1),
        t=t,
        T=T,
        use_budget=use_budget,
        epsilon=epsilon,
        return_logits=True,
        read_cache=False,
        write_cache=False,
        auto_clear_cache=True,
    )
    indices = results[1].reshape(batch_size)
    return PreferenceAction(
        indices=indices,
        log_probs=results[2].reshape(batch_size),
        entropy=results[3].reshape(batch_size),
        logits=results[4],
    )


def optimization_forward_psl(
    model: TAMO,
    data_cfg,
    opt_config,
    loss_config,
    psl_config,
    scalarization_config,
    T: int,
    device: str,
):
    """Run a PSL-TAMO synthetic trajectory and the original REINFORCE loss."""
    env = GPSampleFunction(
        data_config=data_cfg,
        batch_size=opt_config.batch_size,
        num_samples=opt_config.num_samples,
        d=opt_config.num_query_points,
        use_grid_sampling=opt_config.use_grid_sampling,
        use_factorized_policy=False,
        device=device,
    )
    num_initial = opt_config.num_initial_points
    if num_initial + T > env.num_points:
        raise ValueError(
            f"FE budget ({num_initial}+{T}) exceeds pool size {env.num_points}"
        )
    x_true, y_true, _, _ = env.init(
        num_initial_points=num_initial,
        regret_type=opt_config.regret_type,
        compute_hv=False,
        compute_regret=False,
        device=device,
    )
    batch_size = x_true.shape[0]
    objective_mask = env.y_mask.unsqueeze(0).expand(batch_size, -1)
    x_mask = env.x_mask.unsqueeze(0).expand(batch_size, -1)
    z_mask = torch.cat((x_mask, objective_mask), dim=-1)
    lambdas_ctx = sample_padded_preferences(
        objective_mask, num_initial, psl_config.get("preference_method", "dirichlet"), y_true.dtype
    )
    z_ctx = torch.cat((x_true, lambdas_ctx), dim=-1)
    s_ctx = smooth_tchebycheff(
        y_true, lambdas_ctx, env.y_mins.unsqueeze(1),
        scalarization_config.get("tau", 0.1), objective_mask,
    )

    lower, upper = data_cfg.x_range
    psl_models = [
        ParetoSetMLP(
            data_cfg.max_y_dim, data_cfg.max_x_dim,
            psl_config.get("hidden_dim", 128), psl_config.get("depth", 3),
            lower, upper,
        ).to(device)
        for _ in range(batch_size)
    ]
    psl_optimizers = [
        torch.optim.Adam(psl.parameters(), lr=psl_config.get("lr", 1e-3))
        for psl in psl_models
    ]
    psl_loss = update_psl_inner_loop(
        psl_models, psl_optimizers, model, z_ctx, s_ctx, z_mask,
        objective_mask, psl_config.get("num_train_preferences", 64),
        psl_config.get("init_steps", 50), psl_config.get("preference_method", "dirichlet"),
    )

    used_indices = _indices_for_observations(x_true, env._x, env.x_mask)
    rewards, log_probs, entropies = [], [], []
    projection_distances, unique_ratios = [], []
    num_policy_preferences = psl_config.get("num_policy_preferences", 256)

    for t in range(1, T + 1):
        if psl_config.get("update_steps", 5) > 0:
            psl_loss = update_psl_inner_loop(
                psl_models, psl_optimizers, model, z_ctx, s_ctx, z_mask,
                objective_mask, psl_config.get("num_train_preferences", 64),
                psl_config.get("update_steps", 5), psl_config.get("preference_method", "dirichlet"),
            )
        lambdas = sample_padded_preferences(
            objective_mask, num_policy_preferences,
            psl_config.get("preference_method", "dirichlet"), x_true.dtype,
        )
        x_cont = torch.stack([psl(lambdas[b]) for b, psl in enumerate(psl_models)]).detach()
        pool_full = _full_pool(env._x, env.x_mask)
        raw_nearest = torch.cdist(
            x_cont[..., env.x_mask], pool_full[..., env.x_mask]
        ).argmin(dim=-1)
        raw_unique_ratio = torch.stack(
            [
                torch.tensor(
                    row.unique().numel() / num_policy_preferences,
                    device=device,
                )
                for row in raw_nearest
            ]
        ).mean()
        pool_indices, x_candidates, distances = project_psl_to_pool(
            x_cont, env._x, env.x_mask, used_indices
        )
        z_candidates = torch.cat((x_candidates, lambdas), dim=-1)
        action = select_next_preference(
            model, z_ctx, s_ctx, z_candidates, z_mask, t, T,
            opt_config.use_time_budget, opt_config.epsilon,
        )
        selected_pool = torch.gather(pool_indices, 1, action.indices[:, None]).squeeze(1)
        selected_lambda = torch.gather(
            lambdas, 1,
            action.indices[:, None, None].expand(-1, 1, lambdas.shape[-1]),
        )
        x_true, y_true, _, regret = env.step(
            selected_pool[:, None, None], x_true, y_true,
            compute_hv=False, compute_regret=True, regret_type=opt_config.regret_type,
        )
        used_indices = torch.cat((used_indices, selected_pool[:, None]), dim=1)
        z_new = torch.cat((x_true[:, -1:], selected_lambda), dim=-1)
        s_new = smooth_tchebycheff(
            y_true[:, -1:], selected_lambda, env.y_mins.unsqueeze(1),
            scalarization_config.get("tau", 0.1), objective_mask,
        )
        z_ctx = torch.cat((z_ctx, z_new), dim=1)
        s_ctx = torch.cat((s_ctx, s_new), dim=1)
        rewards.append(-torch.as_tensor(regret, device=device, dtype=torch.float32))
        log_probs.append(action.log_probs)
        entropies.append(action.entropy.detach())
        projection_distances.append(distances.mean().detach())
        unique_ratios.append(raw_unique_ratio)

    reward_tensor = torch.stack(rewards, dim=0)
    log_prob_tensor = torch.stack(log_probs, dim=0)
    loss_acq, step_rewards = compute_policy_loss(
        step_rewards=reward_tensor,
        log_probs=log_prob_tensor,
        use_cumulative_r=loss_config.use_cumulative_rewards,
        discount_factor=loss_config.discount_factor,
        batch_standardize=loss_config.batch_standardize,
        clip_rewards=loss_config.clip_rewards,
        batch_first=False,
    )
    stats = {
        "psl_loss": psl_loss,
        "psl_candidate_unique_ratio": torch.stack(unique_ratios).mean().item(),
        "psl_projection_distance": torch.stack(projection_distances).mean().item(),
        "pref_entropy": torch.stack(entropies).mean().item(),
        "num_unique_pool_candidates": float(num_policy_preferences),
    }
    return (
        loss_acq,
        step_rewards.mean().detach().item(),
        step_rewards[:, -1].mean().detach().item(),
        entropies[-1].mean().item(),
        stats,
    )
