"""Training script."""

import math
import time
import gc
import os.path as osp
from typing import Dict

import torch

from omegaconf import DictConfig, OmegaConf
import hydra
import wandb

from data.gp_sample_function import prepare_prediction_batches
from data.base.preprocessing import has_nan_or_inf
from data.dataset import MultiFileHDF5Dataset, get_datapaths
from utils.paths import get_exp_path
from utils.log import (
    Averager,
    TrainingProgress,
    format_duration,
    get_log_filename,
    get_log_fn,
)
from utils.dataclasses import (
    ExConfig,
    PredictionConfig,
    OptimizationConfig,
    DataConfig,
    LossConfig,
    TrainConfig,
    LogConfig,
)
from utils.config import (
    build_tamo,
    build_optimizer,
    build_scheduler,
    build_dataloader,
    load_checkpoint,
    save_checkpoint,
)
from utils.seed import set_all_seeds
from forwards import optimization_forward, prediction_forward
from model import build_objective_predictor
from psl_tamo.data import prepare_stch_prediction_batches
from psl_tamo.forwards import optimization_forward_psl
from utils.wandb_wrapper import init as wandb_init, save_artifact


@hydra.main(version_base=None, config_path="configs", config_name="train.yaml")
def main(config: DictConfig):
    torch.set_default_dtype(torch.float32)
    torch.set_default_device("cpu")

    # Setup configurations
    exp_cfg = ExConfig(**config.experiment)
    train_cfg = TrainConfig(**config.train)
    pred_cfg = PredictionConfig(**config.prediction)
    opt_cfg = OptimizationConfig(**config.optimization)
    loss_cfg = LossConfig(**config.loss)
    data_cfg = DataConfig(**config.data)
    log_cfg = LogConfig(**config.log)

    # Setup logging
    log_filename = get_log_filename(
        model_name=exp_cfg.model_name, expid=exp_cfg.expid, prefix=exp_cfg.mode
    )
    log = get_log_fn(filename=log_filename)
    log(f"Logs will be saved to:\t{log_filename}")
    log(
        "==== Resolved training configuration ====\n"
        + OmegaConf.to_yaml(config, resolve=True)
        + "========================================="
    )

    if exp_cfg.log_to_wandb:
        log(f"wandb configuration:{config.wandb}\n")
        wandb_init(config=config, **config.wandb)

    # Setup experiment path
    exp_path = get_exp_path(model_name=exp_cfg.model_name, expid=exp_cfg.expid)
    log(f"exp_path:\t{exp_path}")

    train(
        exp_path=exp_path,
        model_kwargs=dict(config.model),
        exp_cfg=exp_cfg,
        opt_cfg=opt_cfg,
        pred_cfg=pred_cfg,
        train_cfg=train_cfg,
        data_cfg=data_cfg,
        loss_cfg=loss_cfg,
        log_cfg=log_cfg,
        method_name=config.get("method", {}).get("name", "tamo"),
        psl_config=config.get("psl", {}),
        scalarization_config=config.get("scalarization", {}),
        objective_prediction_config=config.get("objective_prediction", {}),
        log=log,
    )


def train(
    exp_path: str,
    model_kwargs: Dict,
    exp_cfg: ExConfig,
    opt_cfg: OptimizationConfig,
    pred_cfg: PredictionConfig,
    train_cfg: TrainConfig,
    loss_cfg: LossConfig,
    data_cfg: DataConfig,
    log_cfg: LogConfig = LogConfig(),
    method_name: str = "tamo",
    psl_config=None,
    scalarization_config=None,
    objective_prediction_config=None,
    log: callable = print,
):
    # Set random seed
    set_all_seeds(exp_cfg.seed)
    log(f"seed:\t{exp_cfg.seed}")
    psl_config = psl_config or {}
    scalarization_config = scalarization_config or {}
    objective_prediction_config = objective_prediction_config or {}

    # ===============================================
    # Load checkpoint
    # ===============================================
    ckpt = load_checkpoint(
        exp_path=exp_path, device=exp_cfg.device, resume=exp_cfg.resume
    )

    epoch = ckpt.get("epoch", -1)
    model_state_dict = ckpt.get("model", None)
    optimizer_state_dict = ckpt.get("optimizer", None)
    scheduler_state_dict = ckpt.get("scheduler", None)

    # ===============================================
    # Create dataset
    # ===============================================
    datapaths = get_datapaths(
        mode=exp_cfg.mode,
        data_id=data_cfg.data_id,
        x_dim_list=data_cfg.x_dim_list,
        y_dim_list=data_cfg.y_dim_list,
    )

    log("Creating datasets from datapaths:\n" + "\n".join(f"{dp}" for dp in datapaths))
    dataset = MultiFileHDF5Dataset(
        file_paths=datapaths,
        max_x_dim=data_cfg.max_x_dim,
        max_y_dim=data_cfg.max_y_dim,
        standardize=data_cfg.standardize,
        range_scale=data_cfg.y_range,
    )
    dataset_size = len(dataset)
    log(f"Dataset size:\t{dataset_size}")

    # ===============================================
    # Setup epochs
    # ===============================================
    num_total_epochs = train_cfg.num_total_epochs
    num_burnin_epochs = train_cfg.num_burnin_epochs
    num_after_burnin_epochs = num_total_epochs - num_burnin_epochs
    num_context_size_burnin_epochs = train_cfg.num_nc_burnin_epochs
    planned_prediction_tasks = num_total_epochs * pred_cfg.batch_size
    planned_optimization_tasks = (
        num_after_burnin_epochs * opt_cfg.batch_size * opt_cfg.num_samples
    )

    log(
        f"==== Training workload ====\n"
        f"  last completed epoch index:\t{epoch}\n"
        f"  num_total_epochs:\t{num_total_epochs}\n"
        f"  num_burnin_epochs:\t{num_burnin_epochs}\n"
        f"  num_nc_burnin_epochs:\t{num_context_size_burnin_epochs}\n"
        f"  unique dataset tasks:\t{dataset_size}\n"
        f"  planned prediction task presentations:\t{planned_prediction_tasks}\n"
        f"  planned optimization trajectories:\t{planned_optimization_tasks}"
    )

    # ===============================================
    # Setup model
    # ===============================================
    model = build_tamo(model_kwargs)
    objective_prediction_enabled = method_name == "psl_tamo"
    if objective_prediction_enabled:
        model.objective_predictor = build_objective_predictor(
            scalar_tamo_config=model.config,
            max_x_dim=data_cfg.max_x_dim,
            max_y_dim=data_cfg.max_y_dim,
        )
    model = model.to(exp_cfg.device)
    if model_state_dict:
        missing, unexpected = model.load_state_dict(model_state_dict, strict=False)
        if missing:
            log(
                f"[WARNING] Missing keys after checkpoint load:\n  "
                + "\n  ".join(missing)
            )
        if unexpected:
            log(
                f"[WARNING] Unexpected keys in checkpoint:\n  "
                + "\n  ".join(unexpected)
            )

    log(
        f"==== Model built: TAMO ====\n"
        f"  Config: {model.config}\n"
        f"  Parameters: {sum(p.numel() for p in model.parameters()):,}"
    )
    if objective_prediction_enabled:
        log(
            f"==== Objective predictor enabled ====\n"
            f"  Config: {model.objective_predictor.config}\n"
            f"  Parameters: "
            f"{sum(p.numel() for p in model.objective_predictor.parameters()):,}"
        )

    if exp_cfg.log_to_wandb:
        wandb.watch(model, log="gradients", log_freq=log_cfg.freq_log_grad)

    # ===============================================
    # Setup optimizer and scheduler
    # ===============================================
    log(f"Initializing optimizer...")
    optimizer = build_optimizer(
        model=model,
        optimizer_type=train_cfg.optimizer_type,
        lr=train_cfg.lr1,
        weight_decay=train_cfg.weight_decay,
    )

    if optimizer_state_dict:
        try:
            optimizer.load_state_dict(optimizer_state_dict)
        except ValueError as error:
            log(
                "[WARNING] Optimizer state is incompatible with the objective "
                f"predictor and will be reinitialized: {error}"
            )

    log(f"Initializing scheduler...")
    scheduler = build_scheduler(
        optimizer=optimizer,
        scheduler_type=train_cfg.scheduler_type,
        num_training_steps=(num_burnin_epochs, num_after_burnin_epochs)[
            epoch >= num_burnin_epochs
        ],
        last_epoch=(
            epoch if epoch < num_burnin_epochs else max(epoch - num_burnin_epochs, -1)
        ),
        num_warmup_steps=train_cfg.num_warmup_steps,
    )
    if scheduler_state_dict:
        scheduler.load_state_dict(scheduler_state_dict)

    ravg = Averager()
    completed_at_start = min(max(epoch + 1, 0), num_total_epochs)
    progress = TrainingProgress(
        total_steps=num_total_epochs,
        start_step=completed_at_start,
    )
    prediction_tasks_seen = 0
    prediction_tasks_before_run = min(
        completed_at_start * pred_cfg.batch_size,
        planned_prediction_tasks,
    )

    # Repeat dataset if the requested steps exceed one complete dataset pass.
    batches_per_dataset = max(1, math.ceil(dataset_size / pred_cfg.batch_size))
    repeat_round_start = completed_at_start // batches_per_dataset
    num_repeat_data = math.ceil(num_total_epochs / batches_per_dataset)
    log(f"The loaded datasets would be repeated up to {num_repeat_data} times")
    for repeat_round in range(repeat_round_start, num_repeat_data):
        # ===============================================
        # Create dataloader
        # ===============================================
        dataloader = build_dataloader(
            dataset=dataset,
            batch_size=pred_cfg.batch_size,
            split=exp_cfg.mode,
            device=exp_cfg.device,
            num_workers=train_cfg.num_workers,
            prefetch_factor=train_cfg.prefetch_factor,
        )
        dataloader_iter = iter(dataloader)

        # Start one training epoch
        while epoch < num_total_epochs - 1:
            # Load saved dataset (x, y)
            batch = next(dataloader_iter, None)
            if batch is None:
                log(f"[repeat_round={repeat_round}]: finished.")

                # NOTE delete dataloader instance before reiniting for memory save
                del dataloader, dataloader_iter
                gc.collect()
                torch.cuda.empty_cache()

                break

            x, y, valid_x_counts, valid_y_counts = batch
            if has_nan_or_inf(x, "x", log) or has_nan_or_inf(y, "y", log):
                continue

            epoch += 1
            current_prediction_batch_size = x.shape[0]

            # ===============================================
            # Reinit optimizer and scheduler when starting policy learning
            # ===============================================
            if epoch == num_burnin_epochs:
                log(
                    f"Start policy learning at epoch {epoch}; "
                    f"Re-build optimizer and scheduler with lr2: {train_cfg.lr2}"
                )
                optimizer = build_optimizer(
                    model=model,
                    optimizer_type=train_cfg.optimizer_type,
                    lr=train_cfg.lr2,
                    weight_decay=train_cfg.weight_decay,
                )
                scheduler = build_scheduler(
                    optimizer=optimizer,
                    scheduler_type=train_cfg.scheduler_type,
                    num_training_steps=num_after_burnin_epochs,
                    num_warmup_steps=train_cfg.num_warmup_steps,
                )
                # Joint prediction+RL steps are much slower than burn-in steps;
                # rebase throughput so the ETA quickly reflects the new phase.
                progress = TrainingProgress(
                    total_steps=num_total_epochs,
                    start_step=epoch,
                )

            # Loss curve would change - to avoid confusion!
            if epoch == num_context_size_burnin_epochs:
                log(f"Start training on prediction batches of random context size.")

            t1 = time.time()

            model.train()
            optimizer.zero_grad()

            # ===============================================
            # Prediction batch
            # ===============================================
            x = x.to(exp_cfg.device)  # [B, N, max_x_dim]
            y = y.to(exp_cfg.device)  # [B, N, max_y_dim]
            valid_x_counts = valid_x_counts.to(exp_cfg.device)  # [B]
            valid_y_counts = valid_y_counts.to(exp_cfg.device)  # [B]

            # Prediction batch: (xc, yc, xt, yt)
            if method_name == "psl_tamo":
                (
                    obj_xc,
                    obj_yc,
                    obj_xt,
                    obj_yt,
                    obj_x_mask,
                    obj_y_mask,
                ) = prepare_prediction_batches(
                    x=x,
                    y=y,
                    valid_x_counts=valid_x_counts,
                    valid_y_counts=valid_y_counts,
                    dim_scatter_mode=data_cfg.dim_scatter_mode,
                    min_nc=pred_cfg.min_nc,
                    max_nc=pred_cfg.max_nc,
                    warmup=epoch <= num_context_size_burnin_epochs,
                )
                xc, yc, xt, yt, x_mask, y_mask = prepare_stch_prediction_batches(
                    x=x,
                    y=y,
                    valid_x_counts=valid_x_counts,
                    valid_y_counts=valid_y_counts,
                    dim_scatter_mode=data_cfg.dim_scatter_mode,
                    min_nc=pred_cfg.min_nc,
                    max_nc=pred_cfg.max_nc,
                    warmup=epoch <= num_context_size_burnin_epochs,
                    tau=scalarization_config.get("tau", 0.1),
                    ideal_point=scalarization_config.get("ideal_point", -1.0),
                    preference_method=psl_config.get("preference_method", "dirichlet"),
                )
            else:
                xc, yc, xt, yt, x_mask, y_mask = prepare_prediction_batches(
                    x=x,
                    y=y,
                    valid_x_counts=valid_x_counts,
                    valid_y_counts=valid_y_counts,
                    dim_scatter_mode=data_cfg.dim_scatter_mode,
                    min_nc=pred_cfg.min_nc,
                    max_nc=pred_cfg.max_nc,
                    warmup=epoch <= num_context_size_burnin_epochs,
                )

            # ===============================================
            # Forwards
            # ===============================================
            # Prediction forward (model + loss)
            loss_pre, mse_mean, _ = prediction_forward(
                model=model,
                x_ctx=xc,
                y_ctx=yc,
                x_tar=xt,
                y_tar=yt,
                x_mask=x_mask,
                y_mask=y_mask,
                read_cache=pred_cfg.read_cache,
            )
            loss_stch_pre_val = loss_pre.detach().item()
            mse_mean = mse_mean.detach()

            loss_objective = None
            objective_mse_mean = None
            loss_objective_val = 0.0
            prediction_loss = loss_pre
            if objective_prediction_enabled:
                loss_objective, objective_mse_mean, _ = prediction_forward(
                    model=model.objective_predictor,
                    x_ctx=obj_xc,
                    y_ctx=obj_yc,
                    x_tar=obj_xt,
                    y_tar=obj_yt,
                    x_mask=obj_x_mask,
                    y_mask=obj_y_mask,
                    read_cache=False,
                )
                objective_loss_weight = objective_prediction_config.get(
                    "loss_weight", 1.0
                )
                prediction_loss = (
                    prediction_loss + objective_loss_weight * loss_objective
                )
                loss_objective_val = loss_objective.detach().item()
                objective_mse_mean = objective_mse_mean.detach()

            loss_pre_val = prediction_loss.detach().item()

            # Prediction loss backward and free up graph
            if epoch >= num_burnin_epochs:
                loss_weight = loss_cfg.loss_weight
            else:
                loss_weight = 1.0

            (loss_weight * prediction_loss).backward()

            del loss_pre, prediction_loss
            if loss_objective is not None:
                del loss_objective
            del xc, yc, xt, yt
            del x_mask, y_mask, valid_x_counts, valid_y_counts
            if objective_prediction_enabled:
                del obj_xc, obj_yc, obj_xt, obj_yt
                del obj_x_mask, obj_y_mask

            # Optimization forward (model + loss)
            loss_acq_val = 0.0
            step_reward_mean = 0.0
            final_step_reward_mean = 0.0
            final_step_entropy_mean = 0.0
            psl_stats = {}
            T = 0

            if epoch >= num_burnin_epochs:
                T = opt_cfg.sample_T()
                if method_name == "psl_tamo":
                    (
                        loss_acq,
                        step_reward_mean,
                        final_step_reward_mean,
                        final_step_entropy_mean,
                        psl_stats,
                    ) = optimization_forward_psl(
                        model=model,
                        data_cfg=data_cfg,
                        opt_config=opt_cfg,
                        loss_config=loss_cfg,
                        psl_config=psl_config,
                        scalarization_config=scalarization_config,
                        T=T,
                        device=exp_cfg.device,
                    )
                else:
                    (
                        loss_acq,
                        step_reward_mean,
                        final_step_reward_mean,
                        final_step_entropy_mean,
                    ) = optimization_forward(
                        model=model,
                        data_cfg=data_cfg,
                        opt_config=opt_cfg,
                        loss_config=loss_cfg,
                        T=T,
                        device=exp_cfg.device,
                    )
                loss_acq_val = loss_acq.detach().item()

                # optimization loss backward and free up graph
                loss_acq.backward()
                del loss_acq

            # gradient clipping (must unscale before clipping)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=loss_cfg.max_norm
            )
            optimizer.step()
            scheduler.step()

            # ===============================================
            # Tracking and Logging
            # ===============================================
            epoch_time = time.time() - t1
            prediction_tasks_seen += current_prediction_batch_size
            progress_stats = progress.snapshot(completed_steps=epoch + 1)
            prediction_tasks_completed = min(
                prediction_tasks_before_run + prediction_tasks_seen,
                planned_prediction_tasks,
            )
            task_progress_percent = (
                100.0 * prediction_tasks_completed / planned_prediction_tasks
            )
            mse_dict = {
                f"train/mse_{j}": mse_mean[j].detach().item()
                for j in range(mse_mean.shape[0])
            }
            if method_name == "psl_tamo":
                mse_dict["train/stch_pred_mse"] = mse_mean[0].detach().item()
                mse_dict["train/loss_stch_pred"] = loss_stch_pre_val
                mse_dict["train/loss_objective_pred"] = loss_objective_val
                mse_dict.update(
                    {
                        f"train/objective_mse_{j}": objective_mse_mean[j].item()
                        for j in range(objective_mse_mean.shape[0])
                    }
                )
            log_dict = {
                "train/epoch": epoch,
                "train/loss_pre": loss_pre_val,
                "train/loss_acq": loss_acq_val,
                "train/loss": loss_pre_val + loss_acq_val,
                "train/learning_rate": optimizer.param_groups[0]["lr"],
                "train/step_reward": step_reward_mean,
                "train/step_reward_final": final_step_reward_mean,
                "train/step_entropy_final": final_step_entropy_mean,
                "train/epoch_time": epoch_time,
                "train/progress_percent": progress_stats["percent"],
                "train/task_progress_percent": task_progress_percent,
                "train/eta_seconds": progress_stats["eta_seconds"],
                "train/prediction_tasks_seen": prediction_tasks_seen,
                "train/prediction_tasks_completed": prediction_tasks_completed,
                "train/num_query_points": (
                    opt_cfg.num_query_points if epoch >= num_burnin_epochs else 0
                ),
                "train/opt_batch_size": (
                    opt_cfg.batch_size if epoch >= num_burnin_epochs else 0
                ),
                "train/opt_num_samples": (
                    opt_cfg.num_samples if epoch >= num_burnin_epochs else 0
                ),
                "train/T": T,
                **{f"train/{key}": value for key, value in psl_stats.items()},
                **mse_dict,
            }

            # Tracking
            ravg.batch_update(log_dict)
            
            if exp_cfg.log_to_wandb:
            #    wandb.log(ravg.get_averages())
                wandb.log(log_dict)

            # Logging
            if (epoch > 0 and epoch % log_cfg.freq_log == 0) or (
                epoch == num_total_epochs - 1
            ):
                line = (
                    f"[epoch {epoch + 1} / {num_total_epochs}; "
                    f"{progress_stats['percent']:.2f}%] "
                    f"tasks: {prediction_tasks_completed:,}/"
                    f"{planned_prediction_tasks:,} ({task_progress_percent:.2f}%) "
                    f"tasks_seen_this_run: {prediction_tasks_seen:,} "
                    f"elapsed: {format_duration(progress_stats['elapsed_seconds'])} "
                    f"ETA: {format_duration(progress_stats['eta_seconds'])} "
                    f"lr: {optimizer.param_groups[0]['lr']:.3e} "
                    f"[train] "
                    f"{ravg.info()}"
                )
                log(line)
                ravg.reset()

            # Saving
            if (epoch > 0 and epoch % log_cfg.freq_save == 0) or (
                epoch == num_total_epochs - 1
            ):
                log(f"Saving checkpoint at epoch {epoch} to {exp_path}")
                ckpt, ckpt_filepath = save_checkpoint(
                    exp_path=exp_path,
                    model=model,
                    epoch=epoch,
                    seed=exp_cfg.seed,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    ckpt_name="ckpt.tar",
                )
                torch.save(ckpt, ckpt_filepath)

                # Backup checkpoints
                freq_backup = (
                    log_cfg.freq_save_extra_burnin,
                    log_cfg.freq_save_extra,
                )[epoch >= num_burnin_epochs]
                if epoch % freq_backup == 0:
                    epoch_ckpt_filepath = osp.join(exp_path, f"ckpt_epoch_{epoch}.tar")
                    torch.save(ckpt, epoch_ckpt_filepath)

                    # Save to WandB artifact
                    if exp_cfg.log_to_wandb:
                        save_artifact(
                            run=wandb.run,
                            local_path=ckpt_filepath,
                            name=f"checkpoint_epoch_{epoch}.tar",
                            type="model",
                            log=log,
                        )


if __name__ == "__main__":
    main()
