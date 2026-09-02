"""REINFORCE meta-training rollout for MLP-based Utility-PSL-TAMO."""

import torch

from data.base.masking import restore_by_mask
from data.gp_sample_function import GPSampleFunction
from forwards import compute_policy_loss
from model import ParetoSetMLP
from .forwards import _indices_for_observations, _full_pool, project_psl_to_pool
from .preferences import sample_padded_preferences
from .utility_policy import (
    infer_history_preferences,
    select_preference_with_policy,
    update_preference_to_x_mlp,
)


def optimization_forward_utility_psl(
    model,
    data_cfg,
    opt_config,
    loss_config,
    psl_config,
    scalarization_config,
    utility_config,
    T: int,
    device: str,
):
    """Run one policy rollout and return TAMO's REINFORCE loss."""
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
    preference_method = psl_config.get("preference_method", "dirichlet")
    lambda_star = sample_padded_preferences(
        objective_mask, num_initial, preference_method, y_true.dtype
    )

    lower, upper = data_cfg.x_range
    psl_models = [
        ParetoSetMLP(
            int(objective_mask[b].sum()), int(x_mask[b].sum()),
            psl_config.get("hidden_dim", 256), psl_config.get("depth", 3),
            lower, upper,
        ).to(device)
        for b in range(batch_size)
    ]
    psl_optimizers = [
        torch.optim.Adam(psl.parameters(), lr=psl_config.get("lr", 1e-3))
        for psl in psl_models
    ]
    inverse_kwargs = dict(
        num_steps=psl_config.get("inverse_steps", 10),
        lr=psl_config.get("inverse_lr", 5e-2),
        reconstruction_weight=psl_config.get("inverse_x_weight", 1.0),
        reference_weight=psl_config.get("inverse_lambda_weight", 1.0),
    )
    history_lambda, inverse_loss = infer_history_preferences(
        psl_models, x_true, lambda_star, x_mask, objective_mask, **inverse_kwargs
    )
    tau = scalarization_config.get("tau", 0.1)
    beta = utility_config.get("beta", 0.5)
    psl_loss = update_preference_to_x_mlp(
        psl_models, psl_optimizers, model, x_true, y_true, history_lambda,
        x_mask, objective_mask, env.y_mins, tau, beta,
        psl_config.get("num_train_preferences", 10),
        psl_config.get("init_steps", 50), preference_method,
    )

    used_indices = _indices_for_observations(x_true, env._x, env.x_mask)
    rewards, log_probs, entropies = [], [], []
    projection_distances, unique_ratios = [], []
    num_policy_preferences = psl_config.get("num_policy_preferences", 256)

    for t in range(1, T + 1):
        history_lambda, inverse_loss = infer_history_preferences(
            psl_models, x_true, lambda_star, x_mask, objective_mask,
            **inverse_kwargs,
        )
        if psl_config.get("update_steps", 5) > 0:
            psl_loss = update_preference_to_x_mlp(
                psl_models, psl_optimizers, model, x_true, y_true,
                history_lambda, x_mask, objective_mask, env.y_mins, tau, beta,
                psl_config.get("num_train_preferences", 10),
                psl_config.get("update_steps", 5), preference_method,
            )
            history_lambda, inverse_loss = infer_history_preferences(
                psl_models, x_true, lambda_star, x_mask, objective_mask,
                **inverse_kwargs,
            )

        lambdas = sample_padded_preferences(
            objective_mask, num_policy_preferences,
            preference_method, x_true.dtype,
        )
        x_cont = torch.stack([
            restore_by_mask(
                psl(lambdas[b, :, objective_mask[b]]), x_mask[b], dim=-1
            )
            for b, psl in enumerate(psl_models)
        ]).detach()
        pool_full = _full_pool(env._x, env.x_mask)
        raw_nearest = torch.cdist(
            x_cont[..., env.x_mask], pool_full[..., env.x_mask]
        ).argmin(dim=-1)
        unique_ratios.append(torch.stack([
            torch.tensor(row.unique().numel() / num_policy_preferences, device=device)
            for row in raw_nearest
        ]).mean())
        pool_indices, x_candidates, distances = project_psl_to_pool(
            x_cont, env._x, env.x_mask, used_indices
        )
        z_ctx = torch.cat((x_true, history_lambda), dim=-1)
        z_candidates = torch.cat((x_candidates, lambdas), dim=-1)
        action = select_preference_with_policy(
            model, z_ctx, y_true, z_candidates, z_mask, objective_mask,
            t, T, opt_config.use_time_budget, opt_config.epsilon,
        )
        selected_pool = torch.gather(
            pool_indices, 1, action.indices[:, None]
        ).squeeze(1)
        selected_lambda = torch.gather(
            lambdas, 1,
            action.indices[:, None, None].expand(-1, 1, lambdas.shape[-1]),
        )
        x_true, y_true, _, regret = env.step(
            selected_pool[:, None, None], x_true, y_true,
            compute_hv=False, compute_regret=True,
            regret_type=opt_config.regret_type,
        )
        used_indices = torch.cat((used_indices, selected_pool[:, None]), dim=1)
        lambda_star = torch.cat((lambda_star, selected_lambda), dim=1)
        rewards.append(-torch.as_tensor(regret, device=device, dtype=torch.float32))
        log_probs.append(action.log_probs)
        entropies.append(action.entropy.detach())
        projection_distances.append(distances.mean().detach())

    reward_tensor = torch.stack(rewards, dim=0)
    log_prob_tensor = torch.stack(log_probs, dim=0)
    loss_acq, step_rewards = compute_policy_loss(
        step_rewards=reward_tensor, log_probs=log_prob_tensor,
        use_cumulative_r=loss_config.use_cumulative_rewards,
        discount_factor=loss_config.discount_factor,
        batch_standardize=loss_config.batch_standardize,
        clip_rewards=loss_config.clip_rewards, batch_first=False,
    )
    stats = {
        "psl_loss": psl_loss,
        "inverse_preference_loss": inverse_loss,
        "psl_candidate_unique_ratio": torch.stack(unique_ratios).mean().item(),
        "psl_projection_distance": torch.stack(projection_distances).mean().item(),
        "pref_entropy": torch.stack(entropies).mean().item(),
    }
    return (
        loss_acq,
        step_rewards.mean().detach().item(),
        step_rewards[:, -1].mean().detach().item(),
        entropies[-1].mean().item(),
        stats,
    )
