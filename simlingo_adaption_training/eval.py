import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import hydra
import pytorch_lightning as pl
import torch
from hydra.utils import get_original_cwd
from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
from omegaconf import OmegaConf
from pytorch_lightning import Trainer
from transformers import AutoProcessor

from simlingo_adaption_training.config import TrainConfig
from simlingo_adaption_training.utils.internvl2_utils import resolve_internvl_variant_source
from simlingo_adaption_training.utils.logging_project import setup_logging


DEFAULT_STAGE2_EVAL_CHECKPOINT = (
    "models/stage2Model/"
    "2026_04_29_22_57_57_smart_assigner_stage2_GRPO_seed_9876/"
    "checkpoints/loss_plateau.ckpt"
)
ALLOWED_EVAL_MODES = {"QA", "commentary", "Dreaming"}


def _resolve_path(path):
    if path is None:
        return None
    path = Path(str(path)).expanduser()
    if not path.is_absolute():
        path = Path(get_original_cwd()) / path
    return path.resolve()


def _cfg_or_env(cfg, key, env_name=None, default=None):
    if env_name is not None:
        value = os.getenv(env_name)
        if value is not None and str(value).strip() != "":
            return value
    value = OmegaConf.select(cfg, key)
    if value is not None:
        return value
    return default


def _cfg_or_env_preserve_none(cfg, key, env_name=None, default=None):
    if env_name is not None:
        value = os.getenv(env_name)
        if value is not None and str(value).strip() != "":
            return value

    sentinel = object()
    value = OmegaConf.select(cfg, key, default=sentinel)
    if value is not sentinel:
        return value
    return default


def _parse_optional_float(value):
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in {"", "none", "null"}:
        return None
    return float(value)


def _parse_bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Cannot parse bool value: {value}")


def _format_eval_budget_tag(eval_fixed_budget):
    if eval_fixed_budget is None:
        return ""
    budget_str = f"{float(eval_fixed_budget):.4f}".rstrip("0").rstrip(".")
    budget_str = budget_str.replace("-", "neg").replace(".", "p")
    return f"_budget_{budget_str}"


def _extract_state_dict(checkpoint_obj):
    if (
        isinstance(checkpoint_obj, dict)
        and "state_dict" in checkpoint_obj
        and isinstance(checkpoint_obj["state_dict"], dict)
    ):
        return checkpoint_obj["state_dict"]
    if isinstance(checkpoint_obj, dict):
        return checkpoint_obj
    raise ValueError(f"Unsupported checkpoint object type: {type(checkpoint_obj)}")


def _load_model_weights(model, checkpoint_path):
    checkpoint_path = _resolve_path(checkpoint_path)
    if checkpoint_path is None:
        raise ValueError("checkpoint_path is required")

    if checkpoint_path.is_dir():
        state_dict = get_fp32_state_dict_from_zero_checkpoint(str(checkpoint_path))
    else:
        checkpoint_obj = torch.load(str(checkpoint_path), map_location="cpu")
        state_dict = _extract_state_dict(checkpoint_obj)

    incompatible = model.load_state_dict(state_dict, strict=False)
    missing = list(getattr(incompatible, "missing_keys", []))
    unexpected = list(getattr(incompatible, "unexpected_keys", []))
    print(
        "[eval] loaded checkpoint "
        f"{checkpoint_path} "
        f"(missing_keys={len(missing)}, unexpected_keys={len(unexpected)})"
    )
    if missing:
        print(f"[eval] missing key samples: {missing[:10]}")
    if unexpected:
        print(f"[eval] unexpected key samples: {unexpected[:10]}")

    return checkpoint_path


@hydra.main(config_path=f"config", config_name="config", version_base="1.1")
def main(cfg: TrainConfig):
    
    torch.set_float32_matmul_precision("high")
    pl.seed_everything(42)
    
    eval_mode = str(
        _cfg_or_env(
            cfg,
            "eval_mode",
            env_name="SIMLINGO_LANGUAGE_EVAL_MODE",
            default="QA",
        )
    )
    if eval_mode not in ALLOWED_EVAL_MODES:
        raise ValueError(f"eval_mode must be one of {sorted(ALLOWED_EVAL_MODES)}, got {eval_mode}")

    load_path = _cfg_or_env_preserve_none(
        cfg,
        "load_path",
        env_name="SIMLINGO_LANGUAGE_EVAL_CHECKPOINT",
        default=DEFAULT_STAGE2_EVAL_CHECKPOINT,
    )
    load_path = _resolve_path(load_path)
    eval_gpus = int(_cfg_or_env(cfg, "eval_gpus", env_name="SIMLINGO_LANGUAGE_EVAL_GPUS", default=1))
    eval_num_workers = int(
        _cfg_or_env(cfg, "eval_num_workers", env_name="SIMLINGO_LANGUAGE_EVAL_NUM_WORKERS", default=8)
    )
    eval_batch_size = int(
        _cfg_or_env(cfg, "eval_batch_size", env_name="SIMLINGO_LANGUAGE_EVAL_BATCH_SIZE", default=64)
    )
    eval_strategy = str(_cfg_or_env(cfg, "eval_strategy", env_name="SIMLINGO_LANGUAGE_EVAL_STRATEGY", default="auto"))
    eval_max_samples_raw = _cfg_or_env(
        cfg,
        "eval_max_samples",
        env_name="SIMLINGO_LANGUAGE_EVAL_MAX_SAMPLES",
        default=None,
    )
    eval_max_samples = None if eval_max_samples_raw is None else int(eval_max_samples_raw)
    eval_fixed_budget = _parse_optional_float(
        _cfg_or_env(
            cfg,
            "eval_fixed_budget",
            env_name="SIMLINGO_LANGUAGE_EVAL_FIXED_BUDGET",
            default=None,
        )
    )
    eval_layer_selection_mode = str(
        _cfg_or_env(
            cfg,
            "eval_layer_selection_mode",
            env_name="SIMLINGO_LANGUAGE_EVAL_LAYER_SELECTION_MODE",
            default="stage1",
        )
    )
    eval_limit_language_inference = _parse_bool(
        _cfg_or_env(
            cfg,
            "eval_limit_language_inference",
            env_name="SIMLINGO_LANGUAGE_EVAL_LIMIT_LANGUAGE_INFERENCE",
            default=True,
        )
    )

    qa_dataset = cfg.data_module.qa_dataset
    insteval_dataset = cfg.data_module.insteval_dataset
    if load_path is not None:
        load_path_config = Path(load_path).parent.parent / ".hydra/config.yaml"
        if not load_path_config.exists():
            raise FileNotFoundError(f"Could not find checkpoint hydra config: {load_path_config}")
        cfg = OmegaConf.load(load_path_config)
    
    cfg.data_module.qa_dataset = qa_dataset
    cfg.data_module.insteval_dataset = insteval_dataset
    cfg.checkpoint = str(load_path) if load_path is not None else None
    cfg.gpus = eval_gpus
    cfg.data_module.num_workers = eval_num_workers
    cfg.data_module.batch_size = eval_batch_size
    cfg.data_module.eval_max_samples = eval_max_samples
    cfg.strategy = eval_strategy
    cfg.enable_wandb = False

    print(f'Eval mode: {eval_mode}')
    print(f'Checkpoint: {load_path}')
    print(f"Using {cfg.gpus} GPUs")
    print(f"Eval max samples: {eval_max_samples}")
    print(f"Eval fixed budget: {eval_fixed_budget}")
    print(f"Eval layer selection mode: {eval_layer_selection_mode}")
    print(f"Eval limit language inference: {eval_limit_language_inference}")
    
    if eval_mode == "QA" or eval_mode == "commentary":
        cfg.data_module.dreamer_dataset = None
        cfg.data_module.driving_dataset = None
        cfg.data_module.insteval_dataset = None 
        # Language-only eval does not need stage2 full-budget supervision labels.
        cfg.data_module.full_budget_losses_jsonl = None
        # Match the original SimLingo language evaluation setup: evaluate on the full Town13 validation split.
        cfg.data_module.base_dataset.use_town13 = True
    elif eval_mode == "Dreaming":
        cfg.data_module.dreamer_dataset = None
        cfg.data_module.driving_dataset = None
        cfg.data_module.qa_dataset = None
    
    if eval_mode == "QA":
        cfg.data_module.base_dataset.use_commentary = False
        cfg.data_module.base_dataset.use_qa = True
    elif eval_mode == "commentary":
        cfg.data_module.base_dataset.use_commentary = True
        cfg.data_module.base_dataset.use_qa = False
    elif eval_mode == "Dreaming":
        cfg.data_module.base_dataset.use_safety_flag = True
    
    # disable image augmentation
    cfg.data_module.base_dataset.img_augmentation = False
    
    # disable img_shift_augmentation
    cfg.data_module.base_dataset.img_shift_augmentation = False
    
    internvl_source = resolve_internvl_variant_source(cfg.model.language_model.variant)
    processor = AutoProcessor.from_pretrained(internvl_source, trust_remote_code=True, use_fast=False)
    cache_dir = internvl_source
    
    data_module = hydra.utils.instantiate(
        cfg.data_module, 
        processor=processor,
        encoder_variant=cfg.model.vision_model.variant,
        llm_variant=cfg.model.language_model.variant,
        predict=True,
        _recursive_=False
    )
    
    model = hydra.utils.instantiate(
        cfg.model,
        cfg_data_module=cfg.data_module,
        processor=processor,
        cache_dir=cache_dir,
        _recursive_=False
        )

    prediction_dir_name = f"predictions{_format_eval_budget_tag(eval_fixed_budget)}"
    if cfg.checkpoint is not None:
        loaded_checkpoint_path = _load_model_weights(model, cfg.checkpoint)
        prediction_base_dir = loaded_checkpoint_path.parent.parent
    else:
        prediction_base_dir = Path(get_original_cwd()) / "outputs" / cfg.model.language_model.variant
    model.prediction_save_dir = str((prediction_base_dir / prediction_dir_name).resolve())
    print(f"[eval] prediction output dir: {model.prediction_save_dir}")
    model.set_eval_runtime_profile("smart_assigner")
    model.set_eval_predict_language(True)
    model.set_eval_layer_selection_mode(eval_layer_selection_mode)
    model.set_eval_limit_language_inference(eval_limit_language_inference)
    model.set_eval_fixed_budget(eval_fixed_budget)

        
    print(OmegaConf.to_yaml(cfg))
    os.environ["WANDB_DISABLE_CODE"] = "True"

    
    # setup logging
    setup_logging(cfg)

    # resume training
    resume_path = "./checkpoints/last.ckpt"


    if os.path.exists(resume_path) and cfg.resume:
        resume_path = resume_path
    else:
        resume_path = None
    
    # setup lightning logger
    loggers = []

    strategy = cfg.strategy
    if strategy == "deepspeed_stage_2":
        strategy = pl.strategies.DeepSpeedStrategy(
            stage=2, loss_scale=cfg.fp16_loss_scale, logging_batch_size_per_gpu=cfg.data_module.batch_size
        )
  
    print(f"Number of GPUS: {cfg.gpus}")
    overfit = 0
    
    if cfg.gpus >= 1:
        trainer = Trainer(
            accelerator="gpu",
            benchmark=True,
            devices=cfg.gpus,
            gradient_clip_val=0.3,
            log_every_n_steps=20,
            logger=loggers,
            precision=cfg.precision,
            strategy=strategy,
            sync_batchnorm=True,
            max_epochs=cfg.max_epochs,
            overfit_batches=overfit,
            check_val_every_n_epoch=cfg.val_every_n_epochs,
        )

    trainer.predict(model, data_module)

if __name__ == "__main__":
    main()
