"""Evaluate a trained PSL-TAMO checkpoint with a real function budget."""

import os.path as osp

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from data.dataset import get_function_environment
from model import build_objective_predictor
from psl_tamo.evaluation import run_psl_optimization, save_psl_rollout
from utils.config import build_tamo, load_checkpoint
from utils.dataclasses import DataConfig, ExConfig, OptimizationConfig
from utils.log import get_log_filename, get_log_fn
from utils.paths import get_exp_path, get_result_data_path, get_filename_base
from utils.seed import set_all_seeds


@hydra.main(version_base=None, config_path="configs", config_name="test_psl.yaml")
def main(config: DictConfig):
    torch.set_default_dtype(torch.float32)
    torch.set_default_device("cpu")

    exp_cfg = ExConfig(**config.experiment)
    data_cfg = DataConfig(**config.data)
    opt_cfg = OptimizationConfig(**config.optimization)
    set_all_seeds(exp_cfg.seed)

    log = get_log_fn(
        get_log_filename(
            model_name=exp_cfg.model_name,
            expid=exp_cfg.expid,
            prefix="test_psl",
        )
    )
    log(
        "==== Resolved PSL evaluation configuration ====\n"
        + OmegaConf.to_yaml(config, resolve=True)
        + "================================================"
    )

    exp_path = get_exp_path(exp_cfg.model_name, exp_cfg.expid)
    checkpoint = load_checkpoint(
        exp_path=exp_path,
        device=exp_cfg.device,
        resume=True,
        ckpt_name=config.extra.ckpt_name,
    )
    model = build_tamo(dict(config.model))
    model.objective_predictor = build_objective_predictor(
        scalar_tamo_config=model.config,
        max_x_dim=data_cfg.max_x_dim,
        max_y_dim=data_cfg.max_y_dim,
    )
    missing, unexpected = model.load_state_dict(checkpoint["model"], strict=False)
    missing_objective = [key for key in missing if key.startswith("objective_predictor.")]
    if missing_objective:
        raise RuntimeError(
            "Checkpoint does not contain the trained objective head:\n  "
            + "\n  ".join(missing_objective)
        )
    if missing:
        log("[WARNING] Missing checkpoint keys:\n  " + "\n  ".join(missing))
    if unexpected:
        log("[WARNING] Unexpected checkpoint keys:\n  " + "\n  ".join(unexpected))
    model = model.to(exp_cfg.device)

    test_function = get_function_environment(
        function_name=data_cfg.function_name,
        mode=exp_cfg.mode,
        seed=exp_cfg.seed,
        device=exp_cfg.device,
        data_id=data_cfg.data_id,
        scene=data_cfg.scene,
    )
    filename_base = get_filename_base(
        function_name=data_cfg.function_name,
        ckpt_name=config.extra.ckpt_name,
        suffix_segment=config.extra.suffix_segment,
    )
    output_dir = osp.join(
        get_result_data_path(
            model_name=exp_cfg.model_name,
            expid=exp_cfg.expid,
            task_type="optimization_psl",
            filename_base=filename_base,
        ),
        str(exp_cfg.seed),
    )

    result = run_psl_optimization(
        model=model,
        test_function=test_function,
        data_config=data_cfg,
        optimization_config=opt_cfg,
        psl_config=config.psl,
        scalarization_config=config.scalarization,
        device=exp_cfg.device,
        seed=exp_cfg.seed,
        log=log,
    )
    save_psl_rollout(result, output_dir, log=log)


if __name__ == "__main__":
    main()
