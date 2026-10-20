from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import time
from omegaconf import DictConfig, ListConfig

from hydra.core.config_store import ConfigStore


ALLOWED_TRAIN_SAMPLING_MODES = ("weighted_random", "route_stream")
ALLOWED_SMART_ASSIGNER_TRAIN_STAGES = (1, 2)


def _is_mapping_like(value: Any) -> bool:
    return isinstance(value, (dict, DictConfig))


def _is_list_like(value: Any) -> bool:
    return isinstance(value, (list, ListConfig))

@dataclass
class VLMEncoderConfig:
    variant: str = 'OpenGVLab/InternVL2-1B'
    embed_dim: int = 512
    freeze: bool = False
    _target_: str = "simlingo_adaption_training.models.encoder.vlm.VLMEncoderModel"


@dataclass
class LanguageModelConfig:
    variant: str = 'OpenGVLab/InternVL2-1B'
    lora: bool = True
    lora_alpha: int = 64
    lora_r: int = 32
    lora_dropout: float = 0.1
    num_prefix_layers: Optional[int] = None
    _target_: str = "simlingo_adaption_training.models.language_model.llm.LLM"


@dataclass
class AdaptionConfig:
    # Budget and path decisions occur at the L/2 and L prefix boundaries.
    num_prefix_layers: int = 2
    # Stage2 checkpoints already contain the Stage1 initialization.
    simlingo_checkpoint: Optional[str] = None

@dataclass
class SmartAssignerConv1DConfig:
    in_channels: int = 2
    out_channels: int = 128
    kernel_size: int = 3
    stride: int = 1
    padding: int = 1




@dataclass
class SmartAssignerConfig:
    alpha: float = 0.6
    t_lap: float = 0.2
    feature_dim: int = 896
    scene_cross_attn_num_heads: int = 8
    scene_deviation_mlp_hidden: int = 2048
    route_conv: SmartAssignerConv1DConfig = field(default_factory=lambda: SmartAssignerConv1DConfig(
        in_channels=2,
        out_channels=128,
        kernel_size=3,
        stride=1,
        padding=1,
    ))
    route_mlp_hidden: int = 512
    speed_conv: SmartAssignerConv1DConfig = field(default_factory=lambda: SmartAssignerConv1DConfig(
        in_channels=1,
        out_channels=128,
        kernel_size=3,
        stride=1,
        padding=1,
    ))
    speed_mlp_hidden: int = 512
    budget_scheduler_hidden: int = 512
    path_scheduler_hidden: int = 512
    path_tau: float = 5.0
    path_is_hard: bool = True
    query_init_std: float = 0.02


@dataclass
class SmartAssignerStage1LossConfig:
    # Native driving and language loss weights.
    lambda_wp: float = 1.0
    lambda_lang: float = 1.0
    # Scene diversity and load-balancing regularization.
    div_margin: float = 0.05
    bal_epsilon: float = 0.05
    # The diversity ratio follows a cosine warmup.
    div_ratio_start: float = 0.01
    div_ratio_end: float = 0.05
    div_warmup_steps: int = 1000
    bal_ratio: float = 0.01


@dataclass
class SmartAssignerStage2LossConfig:
    # The driving loss is shared by the full-budget reference, reward, and guard.
    lambda_wp: float = 1.0
    lambda_lang: float = 1.0
    driving_loss_weight: float = 0.0
    driving_loss_warmup_steps: int = 0
    # Actions sampled per state.
    grpo_group_size: int = 3
    # Optimization epochs over one fixed rollout batch.
    grpo_update_epochs: int = 1
    grpo_clip_eps: float = 0.15
    grpo_kl_beta: float = 0.02
    # Entropy warms up to beta, then optionally decays after normalization.
    grpo_entropy_beta: float = 0.002
    grpo_entropy_beta_start: float = 0.0
    grpo_entropy_warmup_steps: int = 1000
    grpo_entropy_beta_final: float = 0.002
    grpo_entropy_decay_steps: int = 0
    grpo_entropy_normalize: bool = True
    # on_policy samples all actions; bucket preserves range-wise comparison.
    grpo_rollout_sampling: str = "on_policy"
    # Allowed degradation = abs_eps + rel_eps * full-budget loss.
    grpo_gap_abs_eps: float = 0.02
    grpo_gap_rel_eps: float = 0.10
    grpo_gate_temp: float = 0.02
    # Reward combines feasibility, savings, margin, and violation terms.
    grpo_reward_hit: float = 0.3
    grpo_reward_save: float = 0.7
    grpo_reward_save_margin: float = 0.03
    grpo_reward_save_temp: float = 0.02
    grpo_reward_save_gamma: float = 1.5
    grpo_reward_margin: float = 0.2
    grpo_reward_margin_scale: float = 0.03
    grpo_reward_violate: float = 1.0
    grpo_reward_violate_scale: float = 0.03
    # The absolute term restores reward level lost by group centering.
    grpo_abs_reward_weight: float = 0.3
    grpo_abs_reward_target: float = 0.0
    grpo_abs_reward_scale: float = 1.0
    grpo_adv_eps: float = 1e-6
    # Training-only budget exploration and annealing range.
    grpo_budget_explore_enable: bool = True
    grpo_budget_explore_temp_start: float = 1.4
    grpo_budget_explore_temp_end: float = 0.7
    grpo_budget_explore_eps_start: float = 0.15
    grpo_budget_explore_eps_end: float = 0.02
    grpo_budget_explore_anneal_ratio: float = 0.4
    # Optionally train LoRA in the first half of the prefix.
    grpo_unfreeze_prefix: bool = False
    # Optionally train the three scene-aware encoders.
    unfreeze_scene_aware: bool = False
    # Prefix parameter anchor weight.
    grpo_prefix_kl_beta: float = 0.0
    # Scene-aware parameter anchor weight.
    grpo_scene_kl_beta: float = 0.0


@dataclass
class SmartAssignerTrainConfig:
    # Stage 1 uses random budgets; Stage 2 trains the policy.
    stage: int = 1
    stage1: SmartAssignerStage1LossConfig = field(default_factory=SmartAssignerStage1LossConfig)
    stage2: SmartAssignerStage2LossConfig = field(default_factory=SmartAssignerStage2LossConfig)




@dataclass
class DrivingModelConfig:
    vision_model: Any
    language_model: Any

    lr: float = 5e-2
    # Gradient clipping for manual optimization path in DrivingModel.
    manual_grad_clip_val: float = 0.3
    # Supported values follow Lightning: "norm" or "value".
    manual_grad_clip_algorithm: str = "norm"

    weight_decay: float = 0.1
    betas: Tuple[float, float] = (0.9, 0.999)
    # Scheduler choices: "one_cycle" keeps historical behavior; "cosine_with_floor"
    # uses linear warmup followed by cosine decay to lr * lr_min_ratio.
    lr_scheduler: str = "one_cycle"
    lr_warmup_ratio: float = 0.005
    lr_min_ratio: float = 0.1
    pct_start: float = 0.05

    adaption: AdaptionConfig = field(default_factory=AdaptionConfig)
    smart_assigner: SmartAssignerConfig = field(default_factory=SmartAssignerConfig)
    smart_assigner_train: SmartAssignerTrainConfig = field(default_factory=SmartAssignerTrainConfig)

    _target_: str = "simlingo_adaption_training.models.driving.DrivingModel"


@dataclass
class DatasetBaseConfig:
    data_path: str = ""
    bucket_path: str = ""
    template_root: Optional[str] = None
    # strict sidecar roots; when omitted, they are derived from data_path:
    # measurement_root = data_path + "/data/simlingo"
    # commentary_root = data_path + "/commentary/simlingo"
    # vqa_root = data_path + "/drivelm/simlingo"
    measurement_root: Optional[str] = None
    commentary_root: Optional[str] = None
    vqa_root: Optional[str] = None

    cut_bottom_quarter: bool = False
    use_1d_wps: bool = False

    use_commentary: bool = False
    use_qa: bool = False
    qa_augmentation: bool = True
    commentary_augmentation: bool = True
    use_old_towns: bool = False
    use_only_old_towns: bool = False
    use_town13: bool = False

    skip_first_n_frames: int = 10
    pred_len: int = 11 # including the current time step
    hist_len: int = 1 # including the current time step
    hist_len_commentary: int = 5 # including the current time step
    
    img_augmentation: bool = True
    img_augmentation_prob: float = 0.5
    img_shift_augmentation: bool = True
    img_shift_augmentation_prob: float = 0.5
    
    use_safety_flag: bool = False
    
    num_route_points: int = 20

    route_as: str = 'target_point_command' # target_point_command, target_point, command
    use_lmdrive_commands: bool = True

@dataclass
class DrivingDatasetConfig:
    _target_: str = "simlingo_adaption_training.dataloader.dataset_driving.Data_Driving"
    
@dataclass
class DreamerDatasetConfig:
    _target_: str = "simlingo_adaption_training.dataloader.dataset_dreamer.Data_Dreamer"
    
@dataclass
class QADatasetConfig:
    _target_: str = "simlingo_adaption_training.dataloader.dataset_eval_qa_comm.Data_Eval"
    
@dataclass
class InstEvalDatasetConfig:
    _target_: str = "simlingo_adaption_training.dataloader.dataset_eval_dreamer.Eval_Dreamer"


@dataclass
class RouteStreamSamplingConfig:
    # Select weighted random or temporal route-stream sampling.
    mode: str = "route_stream"
    # Persistent route-classification cache.
    index_cache_dir: str = ""
    # Rebuild the route cache instead of reusing it.
    rebuild_index_cache: bool = False
    # Shuffle routes between epochs while preserving frame order.
    route_shuffle_each_epoch: bool = True
    # Route-stream seed independent of the global seed.
    route_shuffle_seed: int = 12345
    # Balance initial slots by route length to reduce tail vacancies.
    length_balancing: bool = True
    # Blend weight for bucket probability and quota deficit.
    deficit_alpha: float = 0.6
    deficit_eps: float = 1e-6
    # Drop incomplete tail batches to keep training shapes fixed.
    drop_last: bool = True
    # Require every named bucket to contain a route.
    strict_bucket_nonempty: bool = True
    # Require monotonically increasing frame IDs within each route.
    strict_frame_monotonic: bool = True
    # Tie-break priority for non-all bucket ownership.
    bucket_priority: List[str] = field(default_factory=list)


@dataclass
class DrivingDataModuleConfig:
    
    base_dataset: DatasetBaseConfig
    
    driving_dataset:Optional[ DrivingDatasetConfig] = field(default_factory=DrivingDatasetConfig)
    dreamer_dataset: Optional[DreamerDatasetConfig] = field(default_factory=DreamerDatasetConfig)
    qa_dataset: Optional[QADatasetConfig] = field(default_factory=QADatasetConfig)
    insteval_dataset: Optional[InstEvalDatasetConfig] = field(default_factory=InstEvalDatasetConfig)

    batch_size: int = 16
    num_workers: int = 10
    eval_max_samples: Optional[int] = None
    
    train_partitions: Optional[Dict[str, float]] = None
    train_partitions_dreamer: Optional[Dict[str, float]] = None
    train_sampling: RouteStreamSamplingConfig = field(default_factory=RouteStreamSamplingConfig)
    full_budget_losses_jsonl: Optional[str] = None
    use_global_img: bool = False
    _target_: str = "simlingo_adaption_training.dataloader.datamodule.DataModule"


@dataclass
class TrainConfig:
    model: DrivingModelConfig
    data_module: Any

    seed: int = 42
    gpus: int = 1 # default to 1 GPU

    resume: bool = False
    resume_path: Optional[str] = None

    debug: bool = False
    overfit: int = 0
    fp16_loss_scale: float = 32.0 # 0.0 means dynamic loss scaling, only used with deepspeed

    enable_wandb: bool = True
    wandb_project: Optional[str] = "simlingo"
    max_epochs: int = 20
    precision: str = "16-mixed"
    strategy: str = "deepspeed_stage_2" # deepspeed_stage_2 ddp
    val_every_n_epochs: int = 1

    checkpoint: Optional[str] = None
    # Language evaluation entrypoint options used by simlingo_adaption_training/eval.py.
    eval_mode: str = "QA"  # QA | commentary | Dreaming
    load_path: Optional[str] = None
    eval_gpus: int = 1
    eval_num_workers: int = 8
    eval_batch_size: int = 64
    eval_strategy: str = "auto"
    eval_max_samples: Optional[int] = None
    eval_fixed_budget: Optional[float] = None
    eval_layer_selection_mode: str = "stage1"
    eval_limit_language_inference: bool = True
    # Save the rolling checkpoint after this many unimproved steps.
    loss_plateau_patience_steps: int = 10_000
    # Minimum absolute loss improvement.
    loss_plateau_min_delta: float = 1e-4
    # Fallback metric when outputs['loss'] is unavailable.
    loss_plateau_monitor: str = "train/loss"
    # Fixed filename overwritten by each rolling save.
    loss_plateau_ckpt_name: str = "loss_plateau.ckpt"
    loss_plateau_ckpt_enabled: bool = True
    latest_ckpt_every_n_steps: int = 3000
    latest_ckpt_name: str = "latest.ckpt"
    train_visualize_interval: int = 2000
    val_visualize_interval: int = 2000

    if debug:
        wandb_name: Optional[str] = f"debug"
    else:
        name: Optional[str] = 'test'
        wandb_name: Optional[str] = f"{time.strftime('%Y_%m_%d_%H_%M_%S')}"


def validate_adaption_config(adaption_cfg: Any, context: str = "model.adaption") -> None:
    if adaption_cfg is None:
        raise ValueError(f"{context} is required")

    required_fields = ("num_prefix_layers",)
    missing = [field_name for field_name in required_fields if not hasattr(adaption_cfg, field_name)]
    if missing:
        raise ValueError(f"{context} missing required fields: {missing}")

    num_prefix_layers = int(getattr(adaption_cfg, "num_prefix_layers"))
    if num_prefix_layers < 2:
        raise ValueError(f"{context}.num_prefix_layers must be >= 2, got {num_prefix_layers}")
    if num_prefix_layers % 2 != 0:
        raise ValueError(
            f"{context}.num_prefix_layers must be even for half-split smart forward, got {num_prefix_layers}"
        )

def validate_smart_assigner_train_config(
    smart_train_cfg: Any,
    context: str = "model.smart_assigner_train",
) -> None:
    if smart_train_cfg is None:
        raise ValueError(f"{context} is required for smart_assigner training")

    required_fields = ("stage", "stage1", "stage2")
    missing = [field_name for field_name in required_fields if not hasattr(smart_train_cfg, field_name)]
    if missing:
        raise ValueError(f"{context} missing required fields: {missing}")

    stage = int(getattr(smart_train_cfg, "stage"))
    if stage not in ALLOWED_SMART_ASSIGNER_TRAIN_STAGES:
        raise ValueError(
            f"{context}.stage must be one of {list(ALLOWED_SMART_ASSIGNER_TRAIN_STAGES)}, got {stage}"
        )

    stage1_cfg = getattr(smart_train_cfg, "stage1")
    if stage1_cfg is None:
        raise ValueError(f"{context}.stage1 is required")
    stage1_required = (
        "lambda_wp",
        "lambda_lang",
        "div_margin",
        "bal_epsilon",
        "div_ratio_start",
        "div_ratio_end",
        "div_warmup_steps",
        "bal_ratio",
    )
    stage1_missing = [field_name for field_name in stage1_required if not hasattr(stage1_cfg, field_name)]
    if stage1_missing:
        raise ValueError(f"{context}.stage1 missing required fields: {stage1_missing}")

    lambda_wp_stage1 = float(getattr(stage1_cfg, "lambda_wp"))
    lambda_lang_stage1 = float(getattr(stage1_cfg, "lambda_lang"))
    div_margin = float(getattr(stage1_cfg, "div_margin"))
    bal_epsilon = float(getattr(stage1_cfg, "bal_epsilon"))
    div_ratio_start = float(getattr(stage1_cfg, "div_ratio_start"))
    div_ratio_end = float(getattr(stage1_cfg, "div_ratio_end"))
    div_warmup_steps = int(getattr(stage1_cfg, "div_warmup_steps"))
    bal_ratio = float(getattr(stage1_cfg, "bal_ratio"))

    if lambda_wp_stage1 <= 0.0:
        raise ValueError(f"{context}.stage1.lambda_wp must be > 0, got {lambda_wp_stage1}")
    if lambda_lang_stage1 < 0.0:
        raise ValueError(f"{context}.stage1.lambda_lang must be >= 0, got {lambda_lang_stage1}")
    if div_margin < 0.0:
        raise ValueError(f"{context}.stage1.div_margin must be >= 0, got {div_margin}")
    if bal_epsilon < 0.0:
        raise ValueError(f"{context}.stage1.bal_epsilon must be >= 0, got {bal_epsilon}")
    if div_ratio_start < 0.0 or div_ratio_start > 1.0:
        raise ValueError(f"{context}.stage1.div_ratio_start must be in [0, 1], got {div_ratio_start}")
    if div_ratio_end < 0.0 or div_ratio_end > 1.0:
        raise ValueError(f"{context}.stage1.div_ratio_end must be in [0, 1], got {div_ratio_end}")
    if div_ratio_start > div_ratio_end:
        raise ValueError(
            f"{context}.stage1.div_ratio_start must be <= div_ratio_end, got "
            f"{div_ratio_start} > {div_ratio_end}"
        )
    if div_warmup_steps < 0:
        raise ValueError(f"{context}.stage1.div_warmup_steps must be >= 0, got {div_warmup_steps}")
    if bal_ratio < 0.0 or bal_ratio > 1.0:
        raise ValueError(f"{context}.stage1.bal_ratio must be in [0, 1], got {bal_ratio}")

    stage2_cfg = getattr(smart_train_cfg, "stage2")
    if stage2_cfg is None:
        raise ValueError(f"{context}.stage2 is required")
    stage2_required = (
        "lambda_wp",
        "lambda_lang",
        "driving_loss_weight",
        "driving_loss_warmup_steps",
        "grpo_group_size",
        "grpo_update_epochs",
        "grpo_clip_eps",
        "grpo_kl_beta",
        "grpo_entropy_beta",
        "grpo_rollout_sampling",
        "grpo_gap_abs_eps",
        "grpo_gap_rel_eps",
        "grpo_gate_temp",
        "grpo_reward_hit",
        "grpo_reward_save",
        "grpo_reward_save_gamma",
        "grpo_reward_margin",
        "grpo_reward_margin_scale",
        "grpo_reward_violate",
        "grpo_reward_violate_scale",
        "grpo_abs_reward_weight",
        "grpo_abs_reward_target",
        "grpo_abs_reward_scale",
        "grpo_adv_eps",
        "grpo_budget_explore_enable",
        "grpo_budget_explore_temp_start",
        "grpo_budget_explore_temp_end",
        "grpo_budget_explore_eps_start",
        "grpo_budget_explore_eps_end",
        "grpo_budget_explore_anneal_ratio",
        "grpo_unfreeze_prefix",
        "unfreeze_scene_aware",
        "grpo_prefix_kl_beta",
        "grpo_scene_kl_beta",
    )
    stage2_missing = [field_name for field_name in stage2_required if not hasattr(stage2_cfg, field_name)]
    if stage2_missing:
        raise ValueError(f"{context}.stage2 missing required fields: {stage2_missing}")

    lambda_wp_stage2 = float(getattr(stage2_cfg, "lambda_wp"))
    lambda_lang_stage2 = float(getattr(stage2_cfg, "lambda_lang"))
    grpo_group_size = int(getattr(stage2_cfg, "grpo_group_size"))
    grpo_update_epochs = int(getattr(stage2_cfg, "grpo_update_epochs"))
    grpo_clip_eps = float(getattr(stage2_cfg, "grpo_clip_eps"))
    grpo_kl_beta = float(getattr(stage2_cfg, "grpo_kl_beta"))
    grpo_entropy_beta = float(getattr(stage2_cfg, "grpo_entropy_beta"))
    grpo_entropy_beta_start = float(getattr(stage2_cfg, "grpo_entropy_beta_start", 0.0))
    grpo_entropy_warmup_steps = int(getattr(stage2_cfg, "grpo_entropy_warmup_steps", 0))
    grpo_entropy_beta_final = float(getattr(stage2_cfg, "grpo_entropy_beta_final", grpo_entropy_beta))
    grpo_entropy_decay_steps = int(getattr(stage2_cfg, "grpo_entropy_decay_steps", 0))
    grpo_entropy_normalize = getattr(stage2_cfg, "grpo_entropy_normalize", False)
    grpo_rollout_sampling = str(getattr(stage2_cfg, "grpo_rollout_sampling")).strip().lower()
    grpo_gap_abs_eps = float(getattr(stage2_cfg, "grpo_gap_abs_eps"))
    grpo_gap_rel_eps = float(getattr(stage2_cfg, "grpo_gap_rel_eps"))
    grpo_gate_temp = float(getattr(stage2_cfg, "grpo_gate_temp"))
    grpo_reward_hit = float(getattr(stage2_cfg, "grpo_reward_hit"))
    grpo_reward_save = float(getattr(stage2_cfg, "grpo_reward_save"))
    grpo_reward_save_margin = float(getattr(stage2_cfg, "grpo_reward_save_margin", 0.03))
    grpo_reward_save_temp = float(getattr(stage2_cfg, "grpo_reward_save_temp", grpo_gate_temp))
    grpo_reward_save_gamma = float(getattr(stage2_cfg, "grpo_reward_save_gamma"))
    grpo_reward_margin = float(getattr(stage2_cfg, "grpo_reward_margin"))
    grpo_reward_margin_scale = float(getattr(stage2_cfg, "grpo_reward_margin_scale"))
    grpo_reward_violate = float(getattr(stage2_cfg, "grpo_reward_violate"))
    grpo_reward_violate_scale = float(getattr(stage2_cfg, "grpo_reward_violate_scale"))
    grpo_abs_reward_weight = float(getattr(stage2_cfg, "grpo_abs_reward_weight"))
    grpo_abs_reward_target = float(getattr(stage2_cfg, "grpo_abs_reward_target"))
    grpo_abs_reward_scale = float(getattr(stage2_cfg, "grpo_abs_reward_scale"))
    grpo_adv_eps = float(getattr(stage2_cfg, "grpo_adv_eps"))
    grpo_budget_explore_enable = getattr(stage2_cfg, "grpo_budget_explore_enable")
    grpo_budget_explore_temp_start = float(getattr(stage2_cfg, "grpo_budget_explore_temp_start"))
    grpo_budget_explore_temp_end = float(getattr(stage2_cfg, "grpo_budget_explore_temp_end"))
    grpo_budget_explore_eps_start = float(getattr(stage2_cfg, "grpo_budget_explore_eps_start"))
    grpo_budget_explore_eps_end = float(getattr(stage2_cfg, "grpo_budget_explore_eps_end"))
    grpo_budget_explore_anneal_ratio = float(getattr(stage2_cfg, "grpo_budget_explore_anneal_ratio"))
    grpo_unfreeze_prefix = getattr(stage2_cfg, "grpo_unfreeze_prefix")
    unfreeze_scene_aware = getattr(stage2_cfg, "unfreeze_scene_aware")
    grpo_prefix_kl_beta = float(getattr(stage2_cfg, "grpo_prefix_kl_beta"))
    grpo_scene_kl_beta = float(getattr(stage2_cfg, "grpo_scene_kl_beta"))
    if lambda_wp_stage2 <= 0.0:
        raise ValueError(f"{context}.stage2.lambda_wp must be > 0, got {lambda_wp_stage2}")
    if lambda_lang_stage2 < 0.0:
        raise ValueError(f"{context}.stage2.lambda_lang must be >= 0, got {lambda_lang_stage2}")
    if grpo_group_size < 2:
        raise ValueError(f"{context}.stage2.grpo_group_size must be >= 2, got {grpo_group_size}")
    if grpo_update_epochs < 1:
        raise ValueError(f"{context}.stage2.grpo_update_epochs must be >= 1, got {grpo_update_epochs}")
    if grpo_clip_eps <= 0.0:
        raise ValueError(f"{context}.stage2.grpo_clip_eps must be > 0, got {grpo_clip_eps}")
    if grpo_kl_beta < 0.0:
        raise ValueError(f"{context}.stage2.grpo_kl_beta must be >= 0, got {grpo_kl_beta}")
    if grpo_entropy_beta < 0.0:
        raise ValueError(
            f"{context}.stage2.grpo_entropy_beta must be >= 0, got {grpo_entropy_beta}"
        )
    if grpo_entropy_beta_start < 0.0:
        raise ValueError(
            f"{context}.stage2.grpo_entropy_beta_start must be >= 0, got {grpo_entropy_beta_start}"
        )
    if grpo_entropy_warmup_steps < 0:
        raise ValueError(
            f"{context}.stage2.grpo_entropy_warmup_steps must be >= 0, got {grpo_entropy_warmup_steps}"
        )
    if grpo_entropy_beta_final < 0.0:
        raise ValueError(
            f"{context}.stage2.grpo_entropy_beta_final must be >= 0, got {grpo_entropy_beta_final}"
        )
    if grpo_entropy_decay_steps < 0:
        raise ValueError(
            f"{context}.stage2.grpo_entropy_decay_steps must be >= 0, got {grpo_entropy_decay_steps}"
        )
    if grpo_entropy_beta_final > grpo_entropy_beta:
        raise ValueError(
            f"{context}.stage2.grpo_entropy_beta_final must be <= grpo_entropy_beta, got "
            f"{grpo_entropy_beta_final} > {grpo_entropy_beta}"
        )
    if not isinstance(grpo_entropy_normalize, bool):
        raise ValueError(
            f"{context}.stage2.grpo_entropy_normalize must be a bool, got {type(grpo_entropy_normalize)}"
        )
    if grpo_rollout_sampling not in {"on_policy", "bucket"}:
        raise ValueError(
            f"{context}.stage2.grpo_rollout_sampling must be one of "
            f"{{'on_policy', 'bucket'}}, got {grpo_rollout_sampling!r}"
        )
    if grpo_gap_abs_eps < 0.0:
        raise ValueError(
            f"{context}.stage2.grpo_gap_abs_eps must be >= 0, got {grpo_gap_abs_eps}"
        )
    if grpo_gap_rel_eps < 0.0:
        raise ValueError(
            f"{context}.stage2.grpo_gap_rel_eps must be >= 0, got {grpo_gap_rel_eps}"
        )
    if grpo_gate_temp <= 0.0:
        raise ValueError(f"{context}.stage2.grpo_gate_temp must be > 0, got {grpo_gate_temp}")
    if grpo_reward_hit < 0.0:
        raise ValueError(f"{context}.stage2.grpo_reward_hit must be >= 0, got {grpo_reward_hit}")
    if grpo_reward_save < 0.0:
        raise ValueError(f"{context}.stage2.grpo_reward_save must be >= 0, got {grpo_reward_save}")
    if grpo_reward_save_margin < 0.0:
        raise ValueError(
            f"{context}.stage2.grpo_reward_save_margin must be >= 0, got {grpo_reward_save_margin}"
        )
    if grpo_reward_save_temp <= 0.0:
        raise ValueError(
            f"{context}.stage2.grpo_reward_save_temp must be > 0, got {grpo_reward_save_temp}"
        )
    if grpo_reward_save_gamma < 1.0:
        raise ValueError(
            f"{context}.stage2.grpo_reward_save_gamma must be >= 1, got {grpo_reward_save_gamma}"
        )
    if grpo_reward_margin < 0.0:
        raise ValueError(
            f"{context}.stage2.grpo_reward_margin must be >= 0, got {grpo_reward_margin}"
        )
    if grpo_reward_margin_scale <= 0.0:
        raise ValueError(
            f"{context}.stage2.grpo_reward_margin_scale must be > 0, got {grpo_reward_margin_scale}"
        )
    if grpo_reward_violate < 0.0:
        raise ValueError(
            f"{context}.stage2.grpo_reward_violate must be >= 0, got {grpo_reward_violate}"
        )
    if grpo_reward_violate_scale <= 0.0:
        raise ValueError(
            f"{context}.stage2.grpo_reward_violate_scale must be > 0, got {grpo_reward_violate_scale}"
        )
    if grpo_abs_reward_weight < 0.0:
        raise ValueError(
            f"{context}.stage2.grpo_abs_reward_weight must be >= 0, got {grpo_abs_reward_weight}"
        )
    if grpo_abs_reward_scale <= 0.0:
        raise ValueError(
            f"{context}.stage2.grpo_abs_reward_scale must be > 0, got {grpo_abs_reward_scale}"
        )
    if grpo_adv_eps <= 0.0:
        raise ValueError(f"{context}.stage2.grpo_adv_eps must be > 0, got {grpo_adv_eps}")
    if not isinstance(grpo_budget_explore_enable, bool):
        raise ValueError(
            f"{context}.stage2.grpo_budget_explore_enable must be a bool, "
            f"got {type(grpo_budget_explore_enable)}"
        )
    if grpo_budget_explore_temp_start <= 0.0:
        raise ValueError(
            f"{context}.stage2.grpo_budget_explore_temp_start must be > 0, "
            f"got {grpo_budget_explore_temp_start}"
        )
    if grpo_budget_explore_temp_end <= 0.0:
        raise ValueError(
            f"{context}.stage2.grpo_budget_explore_temp_end must be > 0, "
            f"got {grpo_budget_explore_temp_end}"
        )
    if grpo_budget_explore_temp_start < grpo_budget_explore_temp_end:
        raise ValueError(
            f"{context}.stage2.grpo_budget_explore_temp_start must be >= "
            f"grpo_budget_explore_temp_end, got {grpo_budget_explore_temp_start} < "
            f"{grpo_budget_explore_temp_end}"
        )
    if grpo_budget_explore_eps_start < 0.0 or grpo_budget_explore_eps_start > 1.0:
        raise ValueError(
            f"{context}.stage2.grpo_budget_explore_eps_start must be in [0, 1], "
            f"got {grpo_budget_explore_eps_start}"
        )
    if grpo_budget_explore_eps_end < 0.0 or grpo_budget_explore_eps_end > 1.0:
        raise ValueError(
            f"{context}.stage2.grpo_budget_explore_eps_end must be in [0, 1], "
            f"got {grpo_budget_explore_eps_end}"
        )
    if grpo_budget_explore_eps_start < grpo_budget_explore_eps_end:
        raise ValueError(
            f"{context}.stage2.grpo_budget_explore_eps_start must be >= "
            f"grpo_budget_explore_eps_end, got {grpo_budget_explore_eps_start} < "
            f"{grpo_budget_explore_eps_end}"
        )
    if grpo_budget_explore_anneal_ratio < 0.0 or grpo_budget_explore_anneal_ratio > 1.0:
        raise ValueError(
            f"{context}.stage2.grpo_budget_explore_anneal_ratio must be in [0, 1], "
            f"got {grpo_budget_explore_anneal_ratio}"
        )
    if not isinstance(grpo_unfreeze_prefix, bool):
        raise ValueError(
            f"{context}.stage2.grpo_unfreeze_prefix must be a bool, got {type(grpo_unfreeze_prefix)}"
        )
    if not isinstance(unfreeze_scene_aware, bool):
        raise ValueError(
            f"{context}.stage2.unfreeze_scene_aware must be a bool, got {type(unfreeze_scene_aware)}"
        )
    if grpo_prefix_kl_beta < 0.0:
        raise ValueError(
            f"{context}.stage2.grpo_prefix_kl_beta must be >= 0, got {grpo_prefix_kl_beta}"
        )
    if grpo_scene_kl_beta < 0.0:
        raise ValueError(
            f"{context}.stage2.grpo_scene_kl_beta must be >= 0, got {grpo_scene_kl_beta}"
        )


def validate_train_sampling_config(
    data_module_cfg: Any,
    context: str = "data_module.train_sampling",
) -> None:
    if data_module_cfg is None:
        raise ValueError("data_module config is required")
    if not hasattr(data_module_cfg, "train_sampling") or data_module_cfg.train_sampling is None:
        raise ValueError(f"{context} is required")

    sampling_cfg = data_module_cfg.train_sampling
    mode = str(getattr(sampling_cfg, "mode", "")).strip().lower()
    if mode not in ALLOWED_TRAIN_SAMPLING_MODES:
        raise ValueError(
            f"{context}.mode must be one of {list(ALLOWED_TRAIN_SAMPLING_MODES)}, got {mode}"
        )

    if mode == "weighted_random":
        if getattr(data_module_cfg, "driving_dataset", None) is None and getattr(data_module_cfg, "dreamer_dataset", None) is None:
            raise ValueError("weighted_random mode requires at least one of data_module.driving_dataset/dreamer_dataset")

        if hasattr(data_module_cfg, "train_partitions") and data_module_cfg.train_partitions is not None:
            if not _is_mapping_like(data_module_cfg.train_partitions) or len(data_module_cfg.train_partitions) == 0:
                raise ValueError("data_module.train_partitions must be a non-empty dict when provided")
            for bucket_name, weight in data_module_cfg.train_partitions.items():
                weight_f = float(weight)
                if weight_f <= 0.0:
                    raise ValueError(f"data_module.train_partitions[{bucket_name}] must be > 0, got {weight_f}")

        if hasattr(data_module_cfg, "train_partitions_dreamer") and data_module_cfg.train_partitions_dreamer is not None:
            if not _is_mapping_like(data_module_cfg.train_partitions_dreamer) or len(data_module_cfg.train_partitions_dreamer) == 0:
                raise ValueError("data_module.train_partitions_dreamer must be a non-empty dict when provided")
            for bucket_name, weight in data_module_cfg.train_partitions_dreamer.items():
                weight_f = float(weight)
                if weight_f <= 0.0:
                    raise ValueError(f"data_module.train_partitions_dreamer[{bucket_name}] must be > 0, got {weight_f}")
        return

    cache_dir = str(getattr(sampling_cfg, "index_cache_dir", "")).strip()
    if cache_dir == "":
        raise ValueError(f"{context}.index_cache_dir must be a non-empty path")

    route_shuffle_seed = int(getattr(sampling_cfg, "route_shuffle_seed", -1))
    if route_shuffle_seed < 0:
        raise ValueError(f"{context}.route_shuffle_seed must be >= 0, got {route_shuffle_seed}")

    deficit_alpha = float(getattr(sampling_cfg, "deficit_alpha", -1.0))
    if deficit_alpha < 0.0 or deficit_alpha > 1.0:
        raise ValueError(f"{context}.deficit_alpha must be in [0, 1], got {deficit_alpha}")
    deficit_eps = float(getattr(sampling_cfg, "deficit_eps", 0.0))
    if deficit_eps <= 0.0:
        raise ValueError(f"{context}.deficit_eps must be > 0, got {deficit_eps}")
    drop_last_value = getattr(sampling_cfg, "drop_last", None)
    if not isinstance(drop_last_value, bool):
        raise ValueError(f"{context}.drop_last must be a bool, got {type(drop_last_value)}")
    if drop_last_value is not True:
        raise ValueError(
            f"{context}.drop_last must be true in route_stream mode to keep batch_size constant"
        )
    strict_bucket_nonempty_value = getattr(sampling_cfg, "strict_bucket_nonempty", None)
    if not isinstance(strict_bucket_nonempty_value, bool):
        raise ValueError(
            f"{context}.strict_bucket_nonempty must be a bool, got {type(strict_bucket_nonempty_value)}"
        )
    if not hasattr(data_module_cfg, "train_partitions") or data_module_cfg.train_partitions is None:
        raise ValueError("data_module.train_partitions is required for route_stream mode")
    train_partitions = data_module_cfg.train_partitions
    if not _is_mapping_like(train_partitions) or len(train_partitions) == 0:
        raise ValueError("data_module.train_partitions must be a non-empty dict for route_stream mode")
    if "all" not in train_partitions:
        raise ValueError("data_module.train_partitions must contain 'all' bucket for route_stream scheduling")
    for bucket_name, weight in train_partitions.items():
        weight_f = float(weight)
        if weight_f <= 0.0:
            raise ValueError(
                f"data_module.train_partitions[{bucket_name}] must be > 0, got {weight_f}"
            )

    non_all_buckets = sorted([bucket_name for bucket_name in train_partitions.keys() if bucket_name != "all"])
    bucket_priority = getattr(sampling_cfg, "bucket_priority", None)
    if not _is_list_like(bucket_priority):
        raise ValueError(f"{context}.bucket_priority must be a list")
    if len(set(bucket_priority)) != len(bucket_priority):
        raise ValueError(f"{context}.bucket_priority contains duplicated bucket names: {bucket_priority}")

    if len(non_all_buckets) == 0:
        if len(bucket_priority) != 0:
            raise ValueError(
                f"{context}.bucket_priority must be empty when train_partitions has only 'all', got {bucket_priority}"
            )
    else:
        if len(bucket_priority) == 0:
            raise ValueError(
                f"{context}.bucket_priority must be non-empty when train_partitions includes non-'all' buckets"
            )
        missing = sorted([bucket_name for bucket_name in non_all_buckets if bucket_name not in bucket_priority])
        if missing:
            raise ValueError(
                f"{context}.bucket_priority must include all non-'all' buckets from train_partitions. Missing: {missing}"
            )


def register_configs():
    cs = ConfigStore.instance()
    cs.store(name="train_base", node=TrainConfig)
    cs.store(group="data_module", name="driving", node=DrivingDataModuleConfig)
    cs.store(group="data_module/base_dataset", name="dataset", node=DatasetBaseConfig)
    cs.store(group="model", name="driving", node=DrivingModelConfig)
    cs.store(group="model/vision_model", name="vlm", node=VLMEncoderConfig)
    cs.store(group="model/language_model", name="llm", node=LanguageModelConfig)
    cs.store(group="model/adaption", name="default", node=AdaptionConfig)


register_configs()
