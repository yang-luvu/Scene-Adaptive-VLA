import datetime
import json
import math
import os
from pathlib import Path
from pprint import PrettyPrinter
from typing import Any, Dict, Optional, Tuple, List

from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
import hydra
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.optim import AdamW
from hydra.utils import get_original_cwd

from .adaptors.adaptors import replace_placeholder_tokens
from simlingo_adaption_training.models.adaptors.adaptors import (
    DrivingAdaptor,
    LanguageAdaptor,
    WaypointInputAdaptor,
    BudgetAdaptor,
    SceneAwareAdaptor,
    AssignQueryAdaptor,
    AdaptorList,
)
from simlingo_adaption_training.models.budget_assigner import (
    SmartBudgetAssigner,
    build_budget_assigner,
)
from simlingo_adaption_training.config import (
    validate_adaption_config,
    validate_smart_assigner_train_config,
)
from simlingo_adaption_training.models.losses.smart_aux_loss import SmartAuxLossComputer
from simlingo_adaption_training.models.scheduler.scheduler_utils import build_smart_budget_action_table
from simlingo_adaption_training.models.utils import summarise_losses
from simlingo_adaption_training.utils.custom_types import (DrivingExample, DrivingInput,
                                                DrivingLabel, DrivingOutput,
                                                LanguageLabel,
                                                TrainingOutput)


pprint = PrettyPrinter().pprint

def decode_uint8(encoded: torch.Tensor) -> List[str]:
    return [row.tobytes().decode("utf-8").rstrip("\0") for row in encoded.cpu().numpy()]

class NormZeroOne(nn.Module):
    def __init__(self, min_max: Tuple[float, float]):
        super().__init__()
        self.register_buffer("min_max", torch.tensor(min_max, dtype=torch.float), persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        """Normalise tensor to [0, 1] using values from min_max"""
        return (x - self.min_max[0]) / (self.min_max[1] - self.min_max[0])


class DrivingModel(pl.LightningModule):
    def __init__(
        self,
        cfg_data_module,
        processor,
        cache_dir,
        inference_only: bool = False,
        eval_budget_policy: Optional[str] = None,
        **cfg,
    ):
        super().__init__()
        self.save_hyperparameters()

        if not isinstance(inference_only, bool):
            raise ValueError(
                f"inference_only must be bool, got {type(inference_only)}"
            )
        self.inference_only = inference_only
        self.eval_budget_policy = eval_budget_policy
        
        for key, value in cfg.items():
            setattr(self, key, value)
            
        self.processor = processor
        
        self.prediction = {}
        self.scene_difficulty_metrics = {}

        self.allowed_modes = {"smart_assigner"}
        self._init_adaption_settings()
        self.smart_aux_loss_computer: Optional[SmartAuxLossComputer] = None

        self._assigner_history_alpha = float(getattr(self.smart_assigner, "alpha", 0.6))
        self._assigner_decision_shift_t_lap = float(getattr(self.smart_assigner, "t_lap", 0.2))

        self.budget_assigner = build_budget_assigner(
            mode=self.mode,
            history_alpha=self._assigner_history_alpha,
            decision_shift_t_lap=self._assigner_decision_shift_t_lap,
        )
        self.eval_budget = {}
        # Route-stream slots isolate history resets between routes.
        self._train_slot_route_state: Dict[int, str] = {}
        self._debug_first_train_step_printed = False
        self._debug_first_backward_printed = False

        self.predict_language = False
        self.limit_language_inference = True
        self.eval_fixed_budget: Optional[float] = None
        # Runtime profiles affect evaluation only.
        self.eval_runtime_profile = "smart_assigner"
        
        self.cfg_data_module = cfg_data_module

        self.predict_route_as_wps = True
        self.speed_wps_mode = "2d"
        # model.adaption is the single owner of prefix-layer alignment.
        self.language_model.num_prefix_layers = int(self.adaption_num_prefix_layers)
        
        self.vision_model = hydra.utils.instantiate( 
            self.vision_model,
            cfg_data_module=cfg_data_module,
            processor=self.processor,
            cache_dir=cache_dir,
            _recursive_=False
        )
            
        self.language_model = hydra.utils.instantiate(
            self.language_model,
            cache_dir=cache_dir,
            _recursive_=False
        )
        if not hasattr(self.language_model, "config"):
            raise ValueError("language_model must expose config for adaption prefix control")
        self.language_model.config.num_prefix_layers = int(self.adaption_num_prefix_layers)
         
    
        self.all_predictions = {}
        self.all_losses = {}
        
        driving = None
        driving = DrivingAdaptor(
            self.language_model.hidden_size, 
            speed_wps_mode=self.speed_wps_mode,
            predict_route_as_wps=self.predict_route_as_wps,
        )
        

        self.wp_encoder = WaypointInputAdaptor(
            token_size=self.language_model.hidden_size,
            hidden_size=256,
            hidden_size2=512,
        )
        if 'tokenizer' in self.processor.__dict__:
            self.tokenizer = self.processor.tokenizer
        else:
            self.tokenizer = self.processor

        current_path = Path(__file__).resolve()
        project_path = current_path.parent.parent.parent
        state_dict = None
        if self.simlingo_checkpoint is not None:
            checkpoint_path = os.path.join(project_path, self.simlingo_checkpoint)
            if os.path.isdir(checkpoint_path):
                state_dict = get_fp32_state_dict_from_zero_checkpoint(checkpoint_path)
            else:
                state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            self.load_state_dict(state_dict, strict=False)  # Load base weights before LoRA wrapping.
            
        self.language_model.get_lora_model()
        smart_cfg = getattr(self, "smart_assigner", None)
        query_init_std = float(getattr(smart_cfg, "query_init_std", 0.02))
        self.adaptors = AdaptorList(
            language=LanguageAdaptor(self.language_model),
            driving=driving,
            sceneaware=SceneAwareAdaptor(hidden_size=self.language_model.hidden_size),
            assignquery=AssignQueryAdaptor(hidden_size=self.language_model.hidden_size, query_init_std=query_init_std),
            budget=BudgetAdaptor()
        )
        self._init_budget_assigner_modules()
        self._init_smart_runtime_constants()
        if isinstance(self.budget_assigner, SmartBudgetAssigner):
            self.budget_assigner.set_training_stage(self.smart_train_stage)
            if not self.inference_only:
                if self.smart_assigner_train_cfg is None:
                    raise ValueError(
                        "smart_assigner_train config is required for SmartBudgetAssigner training initialization"
                    )
                self.budget_assigner.configure_stage2_budget_exploration(
                    enabled=bool(self.smart_assigner_train_cfg.stage2.grpo_budget_explore_enable),
                    temp_start=float(self.smart_assigner_train_cfg.stage2.grpo_budget_explore_temp_start),
                    temp_end=float(self.smart_assigner_train_cfg.stage2.grpo_budget_explore_temp_end),
                    eps_start=float(self.smart_assigner_train_cfg.stage2.grpo_budget_explore_eps_start),
                    eps_end=float(self.smart_assigner_train_cfg.stage2.grpo_budget_explore_eps_end),
                    anneal_ratio=float(self.smart_assigner_train_cfg.stage2.grpo_budget_explore_anneal_ratio),
                )
                self.smart_aux_loss_computer = SmartAuxLossComputer(
                    img_context_token_id=int(self._smart_img_context_token_id)
                )

        if state_dict is not None:
            self.load_state_dict(state_dict, strict=False)  # Restore adaptor weights after LoRA wrapping.
        self._configure_parameters_for_runtime()
        self._assert_no_runtime_module_aliases()
        # The old policy rolls forward; parameter anchors stay fixed at Stage2 start.
        self._stage2_old_policy_params: Dict[str, Tensor] = {}
        self._stage2_prefix_anchor_params: Dict[str, Tensor] = {}
        self._stage2_scene_anchor_params: Dict[str, Tensor] = {}
        self._stage2_snapshots_ready: bool = False
        # Stage2 manually repeats updates over one rollout.
        self.automatic_optimization = (
            True if self.inference_only else int(self.smart_train_stage) == 1
        )

    def _init_budget_assigner_modules(self) -> None:
        if isinstance(self.budget_assigner, SmartBudgetAssigner):
            smart_cfg = getattr(self, "smart_assigner", None)
            if smart_cfg is None:
                raise ValueError("smart_assigner config is required")
            self.budget_assigner.init_mode_modules(
                smart_cfg=smart_cfg,
                hidden_size=self.language_model.hidden_size,
                language_model_config=self.language_model.config,
            )
            return
        raise ValueError(f"This version requires SmartBudgetAssigner, got {type(self.budget_assigner)}")

    def _init_smart_runtime_constants(self) -> None:
        if not hasattr(self.tokenizer, "convert_tokens_to_ids"):
            raise ValueError("tokenizer must expose convert_tokens_to_ids for smart_assigner")
        self._smart_img_context_token_id = int(self.tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>"))
        if self._smart_img_context_token_id < 0:
            raise ValueError(f"Invalid <IMG_CONTEXT> token id: {self._smart_img_context_token_id}")
        if not hasattr(self.adaptors, "driving") or self.adaptors.driving is None:
            raise ValueError("driving adaptor is required to infer smart runtime token counts")
        if not hasattr(self.adaptors.driving, "sizes") or not isinstance(self.adaptors.driving.sizes, dict):
            raise ValueError("driving adaptor must expose sizes dict for smart runtime token counts")
        self._smart_route_token_count = int(self.adaptors.driving.sizes.get("route", 0))
        self._smart_speed_token_count = int(self.adaptors.driving.sizes.get("speed_wps", 0))
        if self._smart_route_token_count <= 0:
            raise ValueError(
                f"smart runtime requires route token count > 0, got {self._smart_route_token_count}"
            )
        if self._smart_speed_token_count <= 0:
            raise ValueError(
                f"smart runtime requires speed token count > 0, got {self._smart_speed_token_count}"
            )

    def _assert_no_runtime_module_aliases(self) -> None:
        duplicated_prefixes = []
        state_keys = self.state_dict().keys()
        for prefix in ("budget_assigner.language_model.", "budget_assigner.adaptors."):
            if any(key.startswith(prefix) for key in state_keys):
                duplicated_prefixes.append(prefix)
        if len(duplicated_prefixes) > 0:
            raise ValueError(
                "runtime module aliasing is not allowed in this version; found duplicated state_dict prefixes: "
                f"{duplicated_prefixes}"
            )

    def _init_adaption_settings(self) -> None:
        adaption_cfg = getattr(self, "adaption", None)
        validate_adaption_config(adaption_cfg, context="model.adaption")

        self.mode = "smart_assigner"
        if self.mode not in self.allowed_modes:
            raise ValueError(f"Unsupported mode={self.mode}. Expected one of: {sorted(self.allowed_modes)}")
        self.adaption_num_prefix_layers = int(adaption_cfg.num_prefix_layers)
        if self.adaption_num_prefix_layers < 2:
            raise ValueError(
                f"model.adaption.num_prefix_layers must be >= 2, got {self.adaption_num_prefix_layers}"
            )
        if self.adaption_num_prefix_layers % 2 != 0:
            raise ValueError(
                f"model.adaption.num_prefix_layers must be even for half-split smart forward, got "
                f"{self.adaption_num_prefix_layers}"
            )

        simlingo_checkpoint = getattr(adaption_cfg, "simlingo_checkpoint", None)
        self.simlingo_checkpoint = (
            None
            if simlingo_checkpoint is None or str(simlingo_checkpoint).strip() == ""
            else str(simlingo_checkpoint)
        )

        self.smart_train_stage = 1
        self.smart_assigner_train_cfg = None
        if self.inference_only:
            eval_budget_policy = str(self.eval_budget_policy).strip().lower()
            allowed_eval_budget_policies = {"stage1": 1, "stage2": 2}
            if eval_budget_policy not in allowed_eval_budget_policies:
                raise ValueError(
                    f"eval_budget_policy must be one of {sorted(allowed_eval_budget_policies)}, "
                    f"got {self.eval_budget_policy!r}"
                )
            self.eval_budget_policy = eval_budget_policy
            self.smart_train_stage = allowed_eval_budget_policies[eval_budget_policy]
        else:
            smart_train_cfg = getattr(self, "smart_assigner_train", None)
            validate_smart_assigner_train_config(
                smart_train_cfg,
                context="model.smart_assigner_train",
            )
            self.smart_assigner_train_cfg = smart_train_cfg
            self.smart_train_stage = int(smart_train_cfg.stage)
            if self.smart_train_stage == 1 and self.simlingo_checkpoint is None:
                raise ValueError(
                    "model.adaption.simlingo_checkpoint is required for stage1 initialization"
                )

        self.manual_grad_clip_val = float(getattr(self, "manual_grad_clip_val", 0.3))
        manual_grad_clip_algorithm = str(
            getattr(self, "manual_grad_clip_algorithm", "norm")
        ).strip().lower()
        if manual_grad_clip_algorithm not in {"norm", "value"}:
            raise ValueError(
                "model.manual_grad_clip_algorithm must be one of {'norm', 'value'}, "
                f"got {manual_grad_clip_algorithm!r}"
            )
        self.manual_grad_clip_algorithm = manual_grad_clip_algorithm

    def _configure_parameters_for_runtime(self) -> None:
        if self.inference_only:
            self.requires_grad_(False)
            return
        self._configure_adaption_trainable_parameters()

    def _configure_adaption_trainable_parameters(self) -> None:
        # Freeze everything first so only allowlisted parameters train.
        self.requires_grad_(False)
        self._enable_lora_trainable_parameters()

        if not isinstance(self.budget_assigner, SmartBudgetAssigner):
            raise ValueError(
                f"adaption training now requires SmartBudgetAssigner, got {type(self.budget_assigner)}"
            )
        self._configure_smart_assigner_trainable_parameters()

        trainable_names = [name for name, param in self.named_parameters() if param.requires_grad]
        if len(trainable_names) == 0:
            raise RuntimeError("No trainable parameters found after adaption freeze/unfreeze setup")
        print(f"[AdaptionTrain] trainable parameter tensors: {len(trainable_names)}")

    def _enable_lora_trainable_parameters(self) -> None:
        lora_param_names: List[str] = []
        for name, param in self.language_model.model.named_parameters():
            if "lora_" in name:
                param.requires_grad_(True)
                lora_param_names.append(name)
        if len(lora_param_names) == 0:
            raise ValueError(
                "No LoRA parameters found under language_model.model. "
                "Please check model.language_model.lora and LoRA injection setup."
            )

    @staticmethod
    def _extract_layer_index_from_lora_name(param_name: str) -> Optional[int]:
        marker = "layers."
        marker_pos = param_name.find(marker)
        if marker_pos < 0:
            return None
        after_marker = param_name[marker_pos + len(marker):]
        layer_token = after_marker.split(".", 1)[0]
        if not layer_token.isdigit():
            return None
        return int(layer_token)

    def _enable_prefix_lora_trainable_parameters(self) -> int:
        num_prefix_layers = int(self.adaption_num_prefix_layers)
        if num_prefix_layers < 2:
            raise ValueError(f"num_prefix_layers must be >= 2, got {num_prefix_layers}")
        if num_prefix_layers % 2 != 0:
            raise ValueError(f"num_prefix_layers must be even for half-split smart forward, got {num_prefix_layers}")
        prefix_half_layers = num_prefix_layers // 2
        enabled_count = 0
        for name, param in self.language_model.model.named_parameters():
            if "lora_" not in name:
                continue
            layer_idx = self._extract_layer_index_from_lora_name(name)
            if layer_idx is None:
                continue
            if layer_idx < prefix_half_layers:
                param.requires_grad_(True)
                enabled_count += 1
        return enabled_count

    def _enable_stage2_scene_aware_trainable_parameters(self) -> None:
        self.budget_assigner.smart_scene_cross_attn.requires_grad_(True)
        self.budget_assigner.smart_scene_dev_mlp.requires_grad_(True)
        self.budget_assigner.smart_route_conv.requires_grad_(True)
        self.budget_assigner.smart_route_mlp.requires_grad_(True)
        self.budget_assigner.smart_speed_conv.requires_grad_(True)
        self.budget_assigner.smart_speed_mlp.requires_grad_(True)

    def _configure_smart_assigner_trainable_parameters(self) -> None:
        if self.smart_assigner_train_cfg is None:
            raise ValueError("smart_assigner_train config is required for smart_assigner adaption training")

        if not hasattr(self.adaptors, "assignquery") or self.adaptors.assignquery is None:
            raise ValueError("assignquery adaptor is required in smart_assigner mode")
        required_module_names = (
            "smart_scene_cross_attn",
            "smart_scene_dev_mlp",
            "smart_route_conv",
            "smart_route_mlp",
            "smart_speed_conv",
            "smart_speed_mlp",
            "smart_budget_encoder",
            "smart_budget_scheduler",
            "smart_path_scheduler",
        )
        for module_name in required_module_names:
            if not hasattr(self.budget_assigner, module_name):
                raise ValueError(f"SmartBudgetAssigner missing required module: {module_name}")

        self.wp_encoder.requires_grad_(True)

        self.budget_assigner.smart_scene_cross_attn.requires_grad_(True)
        self.budget_assigner.smart_scene_dev_mlp.requires_grad_(True)
        self.budget_assigner.smart_route_conv.requires_grad_(True)
        self.budget_assigner.smart_route_mlp.requires_grad_(True)
        self.budget_assigner.smart_speed_conv.requires_grad_(True)
        self.budget_assigner.smart_speed_mlp.requires_grad_(True)
        self.budget_assigner.smart_budget_encoder.requires_grad_(True)

        self.budget_assigner.smart_path_scheduler.requires_grad_(True)
        self.adaptors.assignquery.query_path.requires_grad_(True)

        stage = int(self.smart_train_stage)
        if stage == 1:
            self.budget_assigner.smart_budget_scheduler.requires_grad_(False)
            self.adaptors.assignquery.query_budget.requires_grad_(False)
            return
        if stage == 2:
            stage2_cfg = self.smart_assigner_train_cfg.stage2
            # Stage2 trains only the GRPO allowlist.
            for param in self.parameters():
                param.requires_grad_(False)

            self.budget_assigner.smart_budget_scheduler.requires_grad_(True)
            self.adaptors.assignquery.query_budget.requires_grad_(True)

            if bool(stage2_cfg.unfreeze_scene_aware):
                self._enable_stage2_scene_aware_trainable_parameters()

            self.adaptors.assignquery.query_path.requires_grad_(False)
            self.budget_assigner.smart_path_scheduler.requires_grad_(False)

            if bool(stage2_cfg.grpo_unfreeze_prefix):
                self._enable_prefix_lora_trainable_parameters()
            return
        raise ValueError(f"Unsupported smart training stage: {stage}")

    def _is_prefix_lora_param_name(self, param_name: str) -> bool:
        if "lora_" not in param_name:
            return False
        layer_idx = self._extract_layer_index_from_lora_name(param_name)
        if layer_idx is None:
            return False
        return layer_idx < (int(self.adaption_num_prefix_layers) // 2)

    def _collect_stage2_policy_param_names(self) -> List[str]:
        # The dynamic old policy covers every trainable parameter.
        return [name for name, param in self.named_parameters() if param.requires_grad]

    def _collect_stage2_prefix_anchor_param_names(self) -> List[str]:
        return [name for name, _ in self.named_parameters() if self._is_prefix_lora_param_name(name)]

    def _collect_stage2_scene_anchor_param_names(self) -> List[str]:
        scene_prefixes = (
            "budget_assigner.smart_scene_cross_attn.",
            "budget_assigner.smart_scene_dev_mlp.",
            "budget_assigner.smart_route_conv.",
            "budget_assigner.smart_route_mlp.",
            "budget_assigner.smart_speed_conv.",
            "budget_assigner.smart_speed_mlp.",
        )
        return [
            name
            for name, _ in self.named_parameters()
            if any(name.startswith(prefix) for prefix in scene_prefixes)
        ]

    def _clone_named_parameters(self, param_names: List[str]) -> Dict[str, Tensor]:
        snapshots: Dict[str, Tensor] = {}
        for name in param_names:
            snapshots[name] = self.get_parameter(name).detach().clone()
        return snapshots

    def _swap_parameters_with_snapshot(self, snapshot: Dict[str, Tensor]) -> Dict[str, Tensor]:
        backup: Dict[str, Tensor] = {}
        for name, snap_tensor in snapshot.items():
            param = self.get_parameter(name)
            backup[name] = param.detach().clone()
            param.data.copy_(snap_tensor.to(device=param.device, dtype=param.dtype))
        return backup

    def _restore_parameters_from_snapshot(self, snapshot: Dict[str, Tensor]) -> None:
        for name, snap_tensor in snapshot.items():
            param = self.get_parameter(name)
            param.data.copy_(snap_tensor.to(device=param.device, dtype=param.dtype))

    def _refresh_stage2_policy_snapshots(
        self,
        refresh_prefix_anchor: bool,
        refresh_scene_anchor: Optional[bool] = None,
    ) -> None:
        if refresh_scene_anchor is None:
            refresh_scene_anchor = bool(refresh_prefix_anchor)
        if self.mode != "smart_assigner" or int(self.smart_train_stage) != 2:
            self._stage2_old_policy_params = {}
            if refresh_prefix_anchor:
                self._stage2_prefix_anchor_params = {}
            if refresh_scene_anchor:
                self._stage2_scene_anchor_params = {}
            self._stage2_snapshots_ready = False
            return

        policy_param_names = self._collect_stage2_policy_param_names()
        if len(policy_param_names) == 0:
            raise RuntimeError(
                "stage2 snapshot requires dynamic-policy parameters (all trainable stage2 parameters), "
                "but none were found"
            )
        self._stage2_old_policy_params = self._clone_named_parameters(policy_param_names)

        if refresh_prefix_anchor:
            prefix_param_names = self._collect_stage2_prefix_anchor_param_names()
            self._stage2_prefix_anchor_params = self._clone_named_parameters(prefix_param_names)
        if refresh_scene_anchor:
            scene_param_names = self._collect_stage2_scene_anchor_param_names()
            self._stage2_scene_anchor_params = self._clone_named_parameters(scene_param_names)

        self._stage2_snapshots_ready = True

    def _ensure_stage2_policy_snapshots(self) -> None:
        if self.mode != "smart_assigner" or int(self.smart_train_stage) != 2:
            return
        if not self._stage2_snapshots_ready or len(self._stage2_old_policy_params) == 0:
            self._refresh_stage2_policy_snapshots(
                refresh_prefix_anchor=True,
                refresh_scene_anchor=True,
            )

    def _collect_stage2_old_policy_logits_for_batch(
        self,
        batch: DrivingExample,
        route_keys: List[str],
    ) -> Tensor:
        self._ensure_stage2_policy_snapshots()
        live_policy_backup = self._swap_parameters_with_snapshot(self._stage2_old_policy_params)
        cpu_rng_state = torch.random.get_rng_state()
        cuda_rng_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        autocast_cache_enabled = torch.is_autocast_cache_enabled()
        try:
        # Disable caches while swapping snapshots to preserve live Linear gradients.
            torch.set_autocast_cache_enabled(False)
            with torch.no_grad():
                _, _, old_smart_context = self.forward_loss(
                    batch,
                    route_keys=route_keys,
                    return_smart_context=True,
                    update_history=False,
                )
            old_training_signals = old_smart_context["training_signals"]
            if "budget_action_logits" not in old_training_signals:
                raise ValueError("stage2 old-policy forward requires budget_action_logits in training_signals")
            return old_training_signals["budget_action_logits"].float().detach()
        finally:
            torch.set_autocast_cache_enabled(autocast_cache_enabled)
            torch.random.set_rng_state(cpu_rng_state)
            if cuda_rng_states is not None:
                torch.cuda.set_rng_state_all(cuda_rng_states)
            self._restore_parameters_from_snapshot(live_policy_backup)

    def _compute_stage2_parameter_anchor_loss(
        self,
        anchor_params: Dict[str, Tensor],
        loss_name: str,
    ) -> Tensor:
        if len(anchor_params) == 0:
            raise RuntimeError(f"{loss_name} requires a non-empty stage2 anchor snapshot")
        return compute_parameter_anchor_loss([
            (self.get_parameter(name), anchor_tensor)
            for name, anchor_tensor in anchor_params.items()
        ])

    def _compute_stage2_prefix_anchor_loss(self, reference_tensor: Tensor) -> Tensor:
        if self.mode != "smart_assigner" or int(self.smart_train_stage) != 2:
            return reference_tensor.new_zeros(())
        stage2_cfg = self.smart_assigner_train_cfg.stage2
        if not bool(stage2_cfg.grpo_unfreeze_prefix):
            return reference_tensor.new_zeros(())
        return self._compute_stage2_parameter_anchor_loss(
            anchor_params=self._stage2_prefix_anchor_params,
            loss_name="stage2 prefix anchor loss",
        )

    def _compute_stage2_scene_anchor_loss(self, reference_tensor: Tensor) -> Tensor:
        if self.mode != "smart_assigner" or int(self.smart_train_stage) != 2:
            return reference_tensor.new_zeros(())
        stage2_cfg = self.smart_assigner_train_cfg.stage2
        if not bool(stage2_cfg.unfreeze_scene_aware):
            return reference_tensor.new_zeros(())
        return self._compute_stage2_parameter_anchor_loss(
            anchor_params=self._stage2_scene_anchor_params,
            loss_name="stage2 scene anchor loss",
        )

    def set_eval_runtime_profile(self, runtime_profile: str) -> None:
        runtime_profile = str(runtime_profile).strip().lower()
        allowed_profiles = {"smart_assigner", "simlingo_compat"}
        if runtime_profile not in allowed_profiles:
            raise ValueError(
                f"Unsupported eval runtime profile={runtime_profile}, "
                f"expected one of {sorted(allowed_profiles)}"
            )
        self.eval_runtime_profile = runtime_profile

    def set_eval_layer_selection_mode(self, layer_selection_mode: str) -> None:
        self.budget_assigner.set_eval_layer_selection_mode(layer_selection_mode)

    def set_eval_fixed_random_layer_selection(self, enabled: bool, seed: int) -> None:
        self.budget_assigner.set_eval_fixed_random_layer_selection(enabled=enabled, seed=seed)

    def set_eval_budget_sampling(self, enabled: bool, temperature: float) -> None:
        self.budget_assigner.set_eval_budget_sampling(enabled=enabled, temperature=temperature)

    def set_eval_predict_language(self, enabled: bool) -> None:
        if not isinstance(enabled, bool):
            raise ValueError(f"eval predict_language must be bool, got {type(enabled)}")
        self.predict_language = enabled

    def set_eval_limit_language_inference(self, enabled: bool) -> None:
        if not isinstance(enabled, bool):
            raise ValueError(
                f"eval limit_language_inference must be bool, got {type(enabled)}"
            )
        self.limit_language_inference = enabled

    def set_eval_fixed_budget(self, fixed_budget: Optional[float]) -> None:
        if fixed_budget is None:
            self.eval_fixed_budget = None
            return
        fixed_budget = float(fixed_budget)
        if not isinstance(self.budget_assigner, SmartBudgetAssigner):
            raise ValueError(f"smart_assigner mode requires SmartBudgetAssigner, got {type(self.budget_assigner)}")
        budget_min = float(getattr(self.budget_assigner, "smart_budget_min", 0.0))
        if fixed_budget < budget_min - 1e-6 or fixed_budget > 1.0 + 1e-6:
            raise ValueError(
                f"eval fixed_budget must be in [{budget_min:.6f}, 1.0], got {fixed_budget}"
            )
        self.eval_fixed_budget = min(max(fixed_budget, budget_min), 1.0)

    def _resolve_language_inference_budget(
        self,
        driving_budget: Optional[float],
    ) -> Optional[float]:
        if bool(self.limit_language_inference):
            return driving_budget
        return 1.0

    def _get_active_adaptor_keys(self, runtime_profile: str) -> List[str]:
        if runtime_profile == "smart_assigner":
            return ["language", "driving", "sceneaware", "assignquery"]
        if runtime_profile == "simlingo_compat":
            return ["language", "driving"]
        raise ValueError(f"Unsupported runtime_profile={runtime_profile}")

    def _resolve_route_keys_for_mode(
        self,
        example: DrivingExample,
        route_keys: Optional[List[str]],
        batch_size: int,
    ) -> Optional[List[str]]:
        # Smart assignment requires route-scoped history in training and evaluation.
        if route_keys is None:
            route_keys = self._extract_route_keys(example)
        if route_keys is None:
            raise ValueError(
                f"mode={self.mode} requires route_keys, but neither explicit route_keys nor example.run_id is available"
            )
        if len(route_keys) != batch_size:
            raise ValueError(f"route_keys length ({len(route_keys)}) must match batch size ({batch_size})")
        return route_keys

    def _prepare_runtime_budget(
        self,
        driving_input: DrivingInput,
        route_keys: Optional[List[str]],
        external_budget: Optional[float],
        forced_budget_action_ids: Optional[Tensor] = None,
    ) -> Optional[Tensor]:
        # Bind device and dtype before initializing per-forward state.
        if not isinstance(self.budget_assigner, SmartBudgetAssigner):
            raise ValueError(f"smart_assigner mode requires SmartBudgetAssigner, got {type(self.budget_assigner)}")
        self.budget_assigner.bind_runtime_reference(driving_input.camera_images)
        # Budget selection is deferred to the L/2 boundary.
        self.budget_assigner.budget_decide_before_llm(
            batch_size=driving_input.camera_images.size(0),
            route_keys=route_keys,
            external_budget=external_budget,
            forced_budget_action_ids=forced_budget_action_ids,
        )
        return None

    def _resolve_language_eos_token_id(self) -> int:
        variant = str(getattr(self.language_model, "variant", ""))
        if variant == "OpenGVLab/InternVL2-4B":
            if not hasattr(self.tokenizer, "added_tokens_encoder"):
                raise ValueError("tokenizer.added_tokens_encoder is required for InternVL2-4B")
            eos_token_id = self.tokenizer.added_tokens_encoder.get("<|end|>")
            if eos_token_id is None:
                raise ValueError("Failed to resolve eos token '<|end|>' for InternVL2-4B")
            return int(eos_token_id)
        if variant == "OpenGVLab/InternVL2-2B":
            if not hasattr(self.tokenizer, "added_tokens_encoder"):
                raise ValueError("tokenizer.added_tokens_encoder is required for InternVL2-2B")
            eos_token_id = self.tokenizer.added_tokens_encoder.get("<|im_end|>")
            if eos_token_id is None:
                raise ValueError("Failed to resolve eos token '<|im_end|>' for InternVL2-2B")
            return int(eos_token_id)
        eos_token_id = getattr(self.tokenizer, "eos_token_id", None)
        if eos_token_id is None:
            raise ValueError("tokenizer.eos_token_id is required for language generation")
        return int(eos_token_id)

    def _append_driving_predictions(self, predictions: Dict[str, Optional[Tensor]]) -> None:
        for key, value in predictions.items():
            if value is None:
                continue
            previous = getattr(self, key)
            if previous is None:
                setattr(self, key, value)
                continue
            if not isinstance(previous, torch.Tensor):
                raise ValueError(f"prediction buffer for key={key} must be Tensor, got {type(previous)}")
            setattr(self, key, torch.cat((previous, value), dim=0))

    def _slice_language_label_batch(self, label: Optional[LanguageLabel], sample_idx: int) -> Optional[LanguageLabel]:
        if label is None:
            return None
        def _slice_tensor_or_none(value):
            if value is None:
                return None
            if not torch.is_tensor(value):
                return value
            if value.ndim == 0:
                return value
            return value[sample_idx : sample_idx + 1]

        placeholder_values = None
        if label.placeholder_values is not None:
            placeholder_values = [label.placeholder_values[sample_idx]]
        language_string = None
        if label.language_string is not None:
            language_string = [label.language_string[sample_idx]]

        return LanguageLabel(
            phrase_ids=label.phrase_ids[sample_idx : sample_idx + 1],
            phrase_valid=label.phrase_valid[sample_idx : sample_idx + 1],
            phrase_mask=label.phrase_mask[sample_idx : sample_idx + 1],
            placeholder_values=placeholder_values,
            language_string=language_string,
            loss_masking=_slice_tensor_or_none(label.loss_masking),
        )

    def _slice_driving_input_batch(self, driving_input: DrivingInput, sample_idx: int) -> DrivingInput:
        def _slice_tensor_or_none(value):
            if value is None:
                return None
            if not torch.is_tensor(value):
                return value
            if value.ndim == 0:
                return value
            return value[sample_idx : sample_idx + 1]

        return driving_input._replace(
            camera_images=driving_input.camera_images[sample_idx : sample_idx + 1],
            image_sizes=_slice_tensor_or_none(driving_input.image_sizes),
            camera_intrinsics=driving_input.camera_intrinsics[sample_idx : sample_idx + 1],
            camera_extrinsics=driving_input.camera_extrinsics[sample_idx : sample_idx + 1],
            vehicle_speed=driving_input.vehicle_speed[sample_idx : sample_idx + 1],
            target_point=driving_input.target_point[sample_idx : sample_idx + 1],
            prompt=self._slice_language_label_batch(driving_input.prompt, sample_idx),
            prompt_inference=self._slice_language_label_batch(driving_input.prompt_inference, sample_idx),
            ego_xy=_slice_tensor_or_none(driving_input.ego_xy),
            ego_yaw=_slice_tensor_or_none(driving_input.ego_yaw),
            timestamp=_slice_tensor_or_none(driving_input.timestamp),
        )

    @staticmethod
    def _slice_adaptor_dict_batch(
        adaptor_dict: Dict[str, Any],
        sample_idx: int,
        batch_size: int,
    ) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for key, value in adaptor_dict.items():
            if torch.is_tensor(value):
                if value.ndim > 0 and value.size(0) == batch_size:
                    out[key] = value[sample_idx : sample_idx + 1].clone()
                else:
                    out[key] = value.clone()
                continue
            if isinstance(value, list):
                out[key] = list(value)
                continue
            out[key] = value
        return out

    @staticmethod
    def _build_smart_language_sampling_adaptor_dict(
        adaptor_dict: Dict[str, Any],
    ) -> Dict[str, Any]:
        sampling_keys = ["sceneaware", "assignquery", "language"]
        out: Dict[str, Any] = {}
        inputs_list: List[Tensor] = []
        masks_list: List[Tensor] = []
        split_sizes: List[int] = []

        for key in sampling_keys:
            inputs_key = f"{key}_inputs"
            mask_key = f"{key}_inputs_mask"
            if inputs_key not in adaptor_dict or mask_key not in adaptor_dict:
                raise ValueError(
                    f"Missing required smart language sampling segment key={key}"
                )
            inputs_tensor = adaptor_dict[inputs_key]
            mask_tensor = adaptor_dict[mask_key]
            out[inputs_key] = inputs_tensor
            out[mask_key] = mask_tensor
            inputs_list.append(inputs_tensor)
            masks_list.append(mask_tensor)
            split_sizes.append(int(inputs_tensor.size(1)))

        inputs = torch.cat(inputs_list, dim=1)
        masks = torch.cat(masks_list, dim=1)
        total_len = int(inputs.size(1))
        perm = torch.arange(
            total_len,
            device=inputs.device,
            dtype=torch.long,
        ).unsqueeze(0).expand(inputs.size(0), -1)

        out["active_adaptor_keys"] = list(sampling_keys)
        out["inputs"] = inputs
        out["inputs_mask"] = masks
        out["split_sizes"] = torch.as_tensor(split_sizes, device=inputs.device, dtype=torch.long)
        out["perm"] = perm
        return out

    def _set_eval_budget_for_compat(self) -> None:
        self.eval_budget = {
            "mode": self.mode,
            "runtime_profile": "simlingo_compat",
            "phase": "simlingo_compat_fixed",
            "fixed_budget": 1.0,
            "value": 1.0,
            "used_budget": 1.0,
        }

    def _forward_simlingo_compat(
        self,
        driving_input: DrivingInput,
        adaptor_dict: Dict[str, Tensor],
        generate_language: bool,
    ) -> None:
        if "language_inputs" not in adaptor_dict:
            raise ValueError("language_inputs is required for simlingo_compat forward")
        if "language_inputs_mask" not in adaptor_dict:
            raise ValueError("language_inputs_mask is required for simlingo_compat forward")
        if "driving_inputs" not in adaptor_dict:
            raise ValueError("driving_inputs is required for simlingo_compat forward")

        adaptor_dict = replace_placeholder_tokens(
            adaptor_dict=adaptor_dict,
            pixel_values=driving_input.camera_images,
            placeholder_values=driving_input.prompt_inference.placeholder_values,
            image_encoder=self.vision_model.image_encoder,
            wp_encoder=self.wp_encoder,
        )
        language_inputs = adaptor_dict["language_inputs"]
        language_inputs_mask = adaptor_dict["language_inputs_mask"]
        driving_inputs = adaptor_dict["driving_inputs"]
        if language_inputs.ndim != 3 or language_inputs_mask.ndim != 2 or driving_inputs.ndim != 3:
            raise ValueError(
                "simlingo_compat expects language_inputs(B,T,C), language_inputs_mask(B,T), driving_inputs(B,Td,C)"
            )
        if language_inputs.size(0) != driving_inputs.size(0):
            raise ValueError(
                f"Batch mismatch: language_inputs={language_inputs.size(0)} driving_inputs={driving_inputs.size(0)}"
            )

        if generate_language:
            eos_token_id = self._resolve_language_eos_token_id()
            rng_devices = [language_inputs.device] if language_inputs.is_cuda else []
            with torch.random.fork_rng(devices=rng_devices):
                for batch_idx in range(language_inputs.size(0)):
                    input_embed = language_inputs[batch_idx : batch_idx + 1].to(
                        dtype=self.language_model.model.dtype
                    )
                    attention_mask = language_inputs_mask[batch_idx : batch_idx + 1].to(dtype=torch.float32)
                    sampled_tokens, _ = self.language_model.greedy_sample(
                        input_embed,
                        eos_token_id=eos_token_id,
                        max_new_tokens=100,
                        input_embed_matrix=self.adaptors.language.embed_tokens.weight,
                        logit_matrix=self.adaptors.language.lm_head.weight,
                        attention_mask=attention_mask,
                    )

                    decoded_text = self.tokenizer.batch_decode(sampled_tokens, skip_special_tokens=True)
                    if len(decoded_text) != 1:
                        raise ValueError(f"Expected one decoded language output per sample, got {len(decoded_text)}")
                    self.language.append(decoded_text[0])

        adaptor_embeds = adaptor_dict["inputs"].to(dtype=self.language_model.model.dtype)
        attention_mask = adaptor_dict["inputs_mask"]
        features, _ = self.language_model.forward(
            embeddings=adaptor_embeds,
            attention_mask=attention_mask,
        )
        outputs_by_adaptor = self.adaptors.split_outputs_by_adaptor(adaptor_dict, features)
        predictions = self.adaptors.driving.get_predictions(outputs_by_adaptor["driving"])
        self._append_driving_predictions(predictions)

    def _forward_smart_assigner_prepared(self, adaptor_dict: Dict[str, Tensor]) -> Tuple[Tensor, Tensor]:
        position_ids = None
        adaptor_embeds = adaptor_dict["inputs"]
        input_embeds = adaptor_embeds.to(dtype=self.language_model.model.dtype)
        attention_mask = adaptor_dict["inputs_mask"]

        # Refresh query positions and execution state before entering the LLM.
        self.budget_assigner.prepare_llm_runtime(adaptor_dict)

        outputs = self.language_model.model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=input_embeds,
            output_hidden_states=True,
            return_dict=True,
            # Prefix-boundary decisions run through the injected assigner.
            assigner=self.budget_assigner,
        )
        features = outputs.hidden_states[-1]
        logits = outputs.logits

        _, adaptor_features = features.split(
            [features.size(1) - adaptor_embeds.size(1), adaptor_embeds.size(1)], dim=1
        )
        _, adaptor_logits = logits.split(
            [logits.size(1) - adaptor_embeds.size(1), adaptor_embeds.size(1)], dim=1
        )
        return adaptor_features, adaptor_logits

    def _forward_smart_assigner(
        self,
        example: DrivingExample,
        driving_input: DrivingInput,
        route_keys: List[str],
        budget: Optional[float],
        generate_language: bool,
    ) -> None:
        active_adaptor_keys = self._get_active_adaptor_keys(runtime_profile="smart_assigner")
        adaptor_dict = self.adaptors(
            example,
            inference=True,
            budget_assigner=self.budget_assigner,
            active_adaptor_keys=active_adaptor_keys,
        )
        adaptor_dict = replace_placeholder_tokens(
            adaptor_dict=adaptor_dict,
            pixel_values=driving_input.camera_images,
            placeholder_values=driving_input.prompt_inference.placeholder_values,
            image_encoder=self.vision_model.image_encoder,
            wp_encoder=self.wp_encoder,
        )
        self.budget_assigner.build_smart_mode_inputs(
            route_keys=route_keys,
            driving_input=driving_input,
            adaptor_dict=adaptor_dict,
            img_context_token_id=int(self._smart_img_context_token_id),
            route_token_count=int(self._smart_route_token_count),
            speed_token_count=int(self._smart_speed_token_count),
        )

        if generate_language:
            eos_token_id = self._resolve_language_eos_token_id()
            batch_size = int(driving_input.camera_images.size(0))
            rng_devices = [driving_input.camera_images.device] if driving_input.camera_images.is_cuda else []
            with torch.random.fork_rng(devices=rng_devices):
                for batch_idx in range(batch_size):
                    driving_input_single = self._slice_driving_input_batch(driving_input, batch_idx)
                    route_keys_single = [route_keys[batch_idx]]
                    language_budget = self._resolve_language_inference_budget(budget)
                    self._prepare_runtime_budget(
                        driving_input=driving_input_single,
                        route_keys=route_keys_single,
                        external_budget=language_budget,
                        forced_budget_action_ids=None,
                    )

                    adaptor_dict_single = self._slice_adaptor_dict_batch(
                        adaptor_dict=adaptor_dict,
                        sample_idx=batch_idx,
                        batch_size=batch_size,
                    )
                    sampling_adaptor_dict = self._build_smart_language_sampling_adaptor_dict(
                        adaptor_dict_single
                    )
                    self.budget_assigner.prepare_llm_runtime(sampling_adaptor_dict)
                    sampling_inputs = sampling_adaptor_dict["inputs"].to(
                        dtype=self.language_model.model.dtype
                    )
                    sampling_inputs_mask = sampling_adaptor_dict["inputs_mask"].to(dtype=torch.float32)
                    sampled_tokens, _ = self.language_model.greedy_sample(
                        sampling_inputs,
                        eos_token_id=eos_token_id,
                        max_new_tokens=100,
                        input_embed_matrix=self.adaptors.language.embed_tokens.weight,
                        logit_matrix=self.adaptors.language.lm_head.weight,
                        attention_mask=sampling_inputs_mask,
                        assigner=self.budget_assigner,
                    )

                    decoded_text = self.tokenizer.batch_decode(sampled_tokens, skip_special_tokens=True)
                    if len(decoded_text) != 1:
                        raise ValueError(f"Expected one decoded language output per sample, got {len(decoded_text)}")
                    self.language.append(decoded_text[0])

        self._prepare_runtime_budget(
            driving_input=driving_input,
            route_keys=route_keys,
            external_budget=budget,
            forced_budget_action_ids=None,
        )
        features, _ = self._forward_smart_assigner_prepared(adaptor_dict)
        outputs_by_adaptor = self.adaptors.split_outputs_by_adaptor(adaptor_dict, features)
        predictions = self.adaptors.driving.get_predictions(outputs_by_adaptor["driving"])
        self._append_driving_predictions(predictions)

    @torch.no_grad()
    def forward(self,
        example: DrivingExample,
        return_language: Optional[bool] = None,
        budget: Optional[float] = None,
        route_keys: Optional[List[str]] = None,
    ) -> DrivingOutput:
        """Run inference and commit route history after the live forward pass."""
        self.speed_wps, self.route, self.language = None, None, []
        try:
            driving_input = example.driving_input
        except AttributeError:
            driving_input = example

        runtime_profile = str(self.eval_runtime_profile).strip().lower()
        if runtime_profile == "smart_assigner":
            route_keys = self._resolve_route_keys_for_mode(
                example=example,
                route_keys=route_keys,
                batch_size=driving_input.camera_images.size(0),
            )
            generate_language = self.predict_language if return_language is None else bool(return_language)
            self._forward_smart_assigner(
                example=example,
                driving_input=driving_input,
                route_keys=route_keys,
                budget=budget,
                generate_language=generate_language,
            )

            # Commit history only after the live policy completes.
            self.scene_difficulty_metrics = self.budget_assigner.budget_info_update_after_llm(
                route_keys=route_keys,
                driving_input=driving_input,
                current_speed_wps=self.speed_wps,
                current_route=self.route,
            )
            self.eval_budget = self.budget_assigner.get_eval_budget()
            return self.speed_wps, self.route, self.language

        if runtime_profile == "simlingo_compat":
            if budget is not None and not math.isclose(float(budget), 1.0, rel_tol=0.0, abs_tol=1e-6):
                raise ValueError(f"simlingo_compat requires budget=1.0, got {budget}")
            active_adaptor_keys = self._get_active_adaptor_keys(runtime_profile="simlingo_compat")
            adaptor_dict = self.adaptors(
                example,
                inference=True,
                active_adaptor_keys=active_adaptor_keys,
            )
            generate_language = self.predict_language if return_language is None else bool(return_language)
            self._forward_simlingo_compat(
                driving_input=driving_input,
                adaptor_dict=adaptor_dict,
                generate_language=generate_language,
            )
            self.scene_difficulty_metrics = {}
            self._set_eval_budget_for_compat()
            return self.speed_wps, self.route, self.language

        raise ValueError(
            f"Unsupported eval runtime profile={runtime_profile}, expected one of ['simlingo_compat', 'smart_assigner']"
        )


    def forward_model(self, 
                      driving_input: DrivingInput, 
                      adaptor_dict: Dict, 
                      driving_labels: DrivingLabel = None,
                      route_keys: Optional[List[str]] = None,
                      ) -> Tensor:
        """Prepare adaptor inputs and run the configured LLM path."""
        del driving_labels
        adaptor_dict = replace_placeholder_tokens(
                adaptor_dict = adaptor_dict,
                pixel_values = driving_input.camera_images,
                placeholder_values = driving_input.prompt_inference.placeholder_values,
                image_encoder = self.vision_model.image_encoder,
                wp_encoder = self.wp_encoder,
        )
        assigner_mode = str(self.budget_assigner.mode).strip().lower()
        if assigner_mode != "smart_assigner":
            raise ValueError(f"Unsupported budget assigner mode in this version: {assigner_mode}")
        if not isinstance(self.budget_assigner, SmartBudgetAssigner):
            raise ValueError(f"mode=smart_assigner requires SmartBudgetAssigner, got {type(self.budget_assigner)}")
        self.budget_assigner.build_smart_mode_inputs(
            route_keys=route_keys,
            driving_input=driving_input,
            adaptor_dict=adaptor_dict,
            img_context_token_id=int(self._smart_img_context_token_id),
            route_token_count=int(self._smart_route_token_count),
            speed_token_count=int(self._smart_speed_token_count),
        )

        position_ids = None
        adaptor_embeds = adaptor_dict["inputs"]
        input_embeds = adaptor_embeds.to(dtype=self.language_model.model.dtype)
        attention_mask = adaptor_dict["inputs_mask"]

        self.budget_assigner.prepare_llm_runtime(adaptor_dict)

        outputs = self.language_model.model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=input_embeds,
            output_hidden_states=True,
            return_dict=True,
            assigner=self.budget_assigner,
        )
        features = outputs.hidden_states[-1]
        logits = outputs.logits

        _, adaptor_features = features.split(
            [features.size(1) - adaptor_embeds.size(1), adaptor_embeds.size(1)], dim=1
        )
        _, adaptor_logits = logits.split(
            [logits.size(1) - adaptor_embeds.size(1), adaptor_embeds.size(1)], dim=1
        )
        return adaptor_features, adaptor_logits

    def _extract_route_keys(self, example: DrivingExample) -> Optional[List[str]]:
        # Reject missing route keys instead of losing history isolation.
        if not hasattr(example, "run_id") or example.run_id is None:
            return None
        measurement_paths = decode_uint8(example.run_id)
        if len(measurement_paths) == 0:
            raise ValueError("example.run_id was provided but decoded measurement path list is empty")

        def _measurement_to_route_key(measurement_path: str) -> str:
            parts = measurement_path.rsplit("/measurements/", 1)
            if len(parts) == 2:
                route_key = parts[0]
            else:
                route_key = os.path.dirname(measurement_path)
            if route_key is None or str(route_key).strip() == "":
                raise ValueError(f"Failed to parse non-empty route_key from measurement_path={measurement_path}")
            return route_key

        return [_measurement_to_route_key(path) for path in measurement_paths]

    def forward_loss(
        self,
        example: DrivingExample,
        per_sample=False,
        budget=None,
        route_keys: Optional[List[str]] = None,
        return_scene_metrics: bool = False,
        return_smart_context: bool = False,
        forced_budget_action_ids: Optional[Tensor] = None,
        update_history: bool = True,
    ) -> TrainingOutput:
        """Run the training or evaluation forward pass and collect loss context."""
        route_keys = self._resolve_route_keys_for_mode(
            example=example,
            route_keys=route_keys,
            batch_size=example.driving_input.camera_images.size(0),
        )
        self._prepare_runtime_budget(
            driving_input=example.driving_input,
            route_keys=route_keys,
            external_budget=budget,
            forced_budget_action_ids=forced_budget_action_ids,
        )

        active_adaptor_keys = self._get_active_adaptor_keys(runtime_profile="smart_assigner")
        adaptor_dict = self.adaptors(
            example,
            budget_assigner=self.budget_assigner,
            active_adaptor_keys=active_adaptor_keys,
        )

        adaptor_features, adaptor_logits = self.forward_model(
            example.driving_input,
            adaptor_dict,
            driving_labels=example.driving_label,
            route_keys=route_keys,
        )

        outputs_by_adaptor = self.adaptors.split_outputs_by_adaptor(adaptor_dict, adaptor_features)
        predictions = self.adaptors.driving.get_predictions(outputs_by_adaptor["driving"])
        speed_wps_pred = predictions.get("speed_wps")
        route_pred = predictions.get("route")

        if update_history:
            # Commit scene metrics and next-frame state once per live forward.
            self.scene_difficulty_metrics = self.budget_assigner.budget_info_update_after_llm(
                route_keys=route_keys,
                driving_input=example.driving_input,
                current_speed_wps=speed_wps_pred,
                current_route=route_pred,
            )
            self.eval_budget = self.budget_assigner.get_eval_budget()

        loss_dict = self.adaptors.compute_loss(adaptor_features, adaptor_logits, adaptor_dict, example)

        loss_dict_only_losses = {k:v for k, v in loss_dict.items() if k.endswith("loss")}
        loss_logs = {k:v for k, v in loss_dict.items() if k.endswith("log")}

        smart_context: Optional[Dict[str, Any]] = None
        if return_smart_context:
            if self.mode != "smart_assigner":
                raise ValueError("return_smart_context=True is only supported when mode=smart_assigner")
            if not isinstance(self.budget_assigner, SmartBudgetAssigner):
                raise ValueError("smart_assigner mode requires SmartBudgetAssigner")
            if "language_inputs" not in adaptor_dict:
                raise ValueError("language_inputs is required in adaptor_dict for smart loss computation")
            if "language__ids" not in adaptor_dict:
                raise ValueError("language__ids is required in adaptor_dict for smart loss computation")
            if "language_inputs_mask" not in adaptor_dict:
                raise ValueError("language_inputs_mask is required in adaptor_dict for smart loss computation")
            training_signals = self.budget_assigner.get_training_signals()
            smart_context = {
                "training_signals": training_signals,
                "language_inputs": adaptor_dict["language_inputs"],
                "language_ids": adaptor_dict["language__ids"],
                "language_inputs_mask": adaptor_dict["language_inputs_mask"],
                "route_keys": list(route_keys),
                "global_step": int(self.global_step),
                "full_budget_driving_loss": example.full_budget_driving_loss,
            }
        
        pred_labels = {k:v for k, v in loss_dict.items() if not k.endswith("loss") and not k.endswith("log")}
        if per_sample:
            if return_smart_context:
                smart_context["loss_values"] = {k: v for k, (v, _) in loss_dict_only_losses.items()}
                smart_context["loss_counts"] = {k: n for k, (_, n) in loss_dict_only_losses.items()}
                if return_scene_metrics:
                    return loss_dict_only_losses, pred_labels, self.scene_difficulty_metrics, smart_context
                return loss_dict_only_losses, pred_labels, smart_context
            if return_scene_metrics:
                return loss_dict_only_losses, pred_labels, self.scene_difficulty_metrics
            return loss_dict_only_losses, pred_labels

        output = summarise_losses(loss_dict_only_losses)
        if return_smart_context:
            smart_context["loss_values"] = output.loss_values
            smart_context["loss_counts"] = output.loss_counts
            if return_scene_metrics:
                return output, loss_logs, self.scene_difficulty_metrics, smart_context
            return output, loss_logs, smart_context
        if return_scene_metrics:
            return output, loss_logs, self.scene_difficulty_metrics
        return output, loss_logs

    @staticmethod
    def _reduce_to_per_sample_loss(values: Tensor, counts: Tensor, key: str) -> Tensor:
        if values.shape != counts.shape:
            raise ValueError(
                f"loss_values/loss_counts shape mismatch for key={key}: "
                f"{tuple(values.shape)} vs {tuple(counts.shape)}"
            )
        values_f = values.float()
        counts_f = counts.float()
        if values_f.ndim == 0:
            return values_f.reshape(1)
        if values_f.ndim == 1:
            return values_f
        reduce_dims = tuple(range(1, values_f.ndim))
        values_sum = values_f.sum(dim=reduce_dims)
        counts_sum = counts_f.sum(dim=reduce_dims).clamp_min(1.0)
        if values_sum.ndim != 1:
            raise ValueError(
                f"reduced per-sample loss for key={key} must be 1D, got {tuple(values_sum.shape)}"
            )
        return values_sum / counts_sum

    def _extract_primary_losses(self, output: TrainingOutput) -> Tuple[Tensor, Tensor]:
        wp_keys = [
            key for key in output.loss_averages.keys()
            if key.endswith("_loss") and key != "language_loss"
        ]
        if len(wp_keys) == 0:
            raise ValueError(
                "Unable to build smart training objective: no waypoint/driving loss keys found in output.loss_averages"
            )
        if "language_loss" not in output.loss_averages:
            raise ValueError("Unable to build smart training objective: language_loss is missing in output.loss_averages")
        wp_loss = torch.stack([output.loss_averages[key] for key in sorted(wp_keys)]).sum()
        lang_loss = output.loss_averages["language_loss"]
        return wp_loss, lang_loss

    def _extract_primary_losses_per_sample(self, output: TrainingOutput) -> Tuple[Tensor, Tensor]:
        wp_keys = [
            key for key in output.loss_values.keys()
            if key.endswith("_loss") and key != "language_loss"
        ]
        if len(wp_keys) == 0:
            raise ValueError(
                "Unable to build per-sample smart objective: no waypoint/driving loss keys found in output.loss_values"
            )
        if "language_loss" not in output.loss_values:
            raise ValueError("Unable to build per-sample smart objective: language_loss is missing in output.loss_values")
        if "language_loss" not in output.loss_counts:
            raise ValueError("Unable to build per-sample smart objective: language_loss is missing in output.loss_counts")

        wp_per_sample_parts = []
        for key in sorted(wp_keys):
            if key not in output.loss_counts:
                raise ValueError(f"Unable to build per-sample smart objective: {key} is missing in output.loss_counts")
            wp_per_sample_parts.append(
                self._reduce_to_per_sample_loss(
                    values=output.loss_values[key],
                    counts=output.loss_counts[key],
                    key=key,
                )
            )
        wp_per_sample = torch.stack(wp_per_sample_parts, dim=0).sum(dim=0)
        lang_per_sample = self._reduce_to_per_sample_loss(
            values=output.loss_values["language_loss"],
            counts=output.loss_counts["language_loss"],
            key="language_loss",
        )
        if wp_per_sample.shape != lang_per_sample.shape:
            raise ValueError(
                f"per-sample wp/lang shape mismatch: {tuple(wp_per_sample.shape)} vs {tuple(lang_per_sample.shape)}"
            )
        return wp_per_sample, lang_per_sample

    @staticmethod
    def _combine_stage2_driving_loss(stage2_cfg: Any, wp_loss: Tensor, lang_loss: Tensor) -> Tensor:
        return float(stage2_cfg.lambda_wp) * wp_loss + float(stage2_cfg.lambda_lang) * lang_loss

    @staticmethod
    def _compute_stage2_driving_loss_weight(stage2_cfg: Any, global_step: int) -> float:
        base_weight = float(stage2_cfg.driving_loss_weight)
        warmup_steps = int(stage2_cfg.driving_loss_warmup_steps)
        if warmup_steps <= 0:
            return base_weight
        progress = min(max(float(global_step) / float(warmup_steps), 0.0), 1.0)
        return base_weight * progress

    def _build_stage2_direct_driving_loss_items(
        self,
        output: TrainingOutput,
        stage2_cfg: Any,
        global_step: int,
    ) -> Dict[str, Tensor]:
        wp_loss, lang_loss = self._extract_primary_losses(output)
        driving_loss = self._combine_stage2_driving_loss(stage2_cfg, wp_loss, lang_loss)
        driving_loss_weight = driving_loss.new_tensor(
            self._compute_stage2_driving_loss_weight(stage2_cfg, global_step)
        )
        return {
            "stage2/driving_loss": driving_loss,
            "stage2/driving_loss_weight": driving_loss_weight,
            "stage2/driving_weighted_loss": driving_loss_weight * driving_loss,
        }

    @staticmethod
    def _resolve_stage2_full_budget_driving_loss(
        smart_context: Dict[str, Any],
        device: torch.device,
        dtype: torch.dtype,
        batch_size: int,
    ) -> Tensor:
        full_budget_driving_loss = smart_context.get("full_budget_driving_loss", None)
        if full_budget_driving_loss is None:
            raise ValueError(
                "stage2 GRPO requires DrivingExample.full_budget_driving_loss. "
                "Set data_module.full_budget_losses_jsonl and ensure dl_collate_fn injects it."
            )
        if not torch.is_tensor(full_budget_driving_loss):
            raise ValueError("smart_context.full_budget_driving_loss must be a Tensor when provided")
        if full_budget_driving_loss.ndim != 1:
            raise ValueError(
                "smart_context.full_budget_driving_loss must be rank-1, "
                f"got shape={tuple(full_budget_driving_loss.shape)}"
            )
        if int(full_budget_driving_loss.size(0)) != int(batch_size):
            raise ValueError(
                "smart_context.full_budget_driving_loss batch mismatch: "
                f"{int(full_budget_driving_loss.size(0))} vs {int(batch_size)}"
            )
        return full_budget_driving_loss.to(device=device, dtype=dtype)

    def _resolve_stage2_validation_full_budget_driving_loss(
        self,
        batch: DrivingExample,
        route_keys: List[str],
        smart_context: Dict[str, Any],
        stage2_cfg: Any,
        device: torch.device,
        dtype: torch.dtype,
        batch_size: int,
    ) -> Tensor:
        if smart_context.get("full_budget_driving_loss", None) is not None:
            return self._resolve_stage2_full_budget_driving_loss(
                smart_context=smart_context,
                device=device,
                dtype=dtype,
                batch_size=batch_size,
            )

        # Compute missing validation references without committing speculative history.
        with torch.no_grad():
            full_budget_output, _ = self.forward_loss(
                batch,
                budget=1.0,
                route_keys=route_keys,
                return_smart_context=False,
                update_history=False,
            )
        wp_loss, lang_loss = self._extract_primary_losses_per_sample(full_budget_output)
        full_budget_driving_loss = self._combine_stage2_driving_loss(
            stage2_cfg,
            wp_loss,
            lang_loss,
        )
        if full_budget_driving_loss.ndim != 1 or int(full_budget_driving_loss.size(0)) != int(batch_size):
            raise ValueError(
                "online validation full-budget driving loss must be rank-1 with batch size "
                f"{batch_size}, got shape={tuple(full_budget_driving_loss.shape)}"
            )
        return full_budget_driving_loss.to(device=device, dtype=dtype).detach()

    def _collect_stage2_rollout_driving_losses(
        self,
        batch: DrivingExample,
        route_keys: List[str],
        sampled_action_ids: Tensor,
        stage2_cfg: Any,
    ) -> Tensor:
        if sampled_action_ids.ndim != 2:
            raise ValueError(
                f"sampled_action_ids must be rank-2 [B,G], got shape={tuple(sampled_action_ids.shape)}"
            )
        rollout_losses: List[Tensor] = []
        group_size = int(sampled_action_ids.size(1))
        for group_idx in range(group_size):
            forced_actions = sampled_action_ids[:, group_idx]
            with torch.no_grad():
                rollout_output, _ = self.forward_loss(
                    batch,
                    route_keys=route_keys,
                    return_smart_context=False,
                    forced_budget_action_ids=forced_actions,
                    update_history=False,
                )
            wp_per_sample, lang_per_sample = self._extract_primary_losses_per_sample(rollout_output)
            rollout_driving_loss = self._combine_stage2_driving_loss(
                stage2_cfg,
                wp_per_sample,
                lang_per_sample,
            )
            rollout_losses.append(rollout_driving_loss.float())
        return torch.stack(rollout_losses, dim=1)

    def _sample_stage2_grpo_rollout_actions_bucket(
        self,
        budget_action_logits_old: Tensor,
        group_size: int,
    ) -> Tuple[Tensor, Tensor]:
        explore_cfg = self.budget_assigner._resolve_stage2_budget_exploration()
        return sample_budget_actions_by_bucket(
            budget_action_logits_old=budget_action_logits_old,
            group_size=group_size,
            temperature=float(explore_cfg["temperature"]),
            epsilon=float(explore_cfg["epsilon"]),
        )

    def _sample_stage2_grpo_rollout_actions_on_policy(
        self,
        budget_action_logits_old: Tensor,
        group_size: int,
    ) -> Tuple[Tensor, Tensor]:
        # Sample from the behavior distribution used by the ratio correction.
        explore_cfg = self.budget_assigner._resolve_stage2_budget_exploration()
        temperature = float(explore_cfg["temperature"])
        epsilon = float(explore_cfg["epsilon"])
        num_actions = int(budget_action_logits_old.size(1))
        old_probs_temp = F.softmax(budget_action_logits_old / temperature, dim=-1)
        behavior_probs = (1.0 - epsilon) * old_probs_temp + epsilon / float(num_actions)
        sampled_action_ids = torch.distributions.Categorical(probs=behavior_probs).sample(
            (int(group_size),)
        ).transpose(0, 1).contiguous()
        behavior_logp = torch.log(behavior_probs).gather(dim=1, index=sampled_action_ids)
        return sampled_action_ids, behavior_logp

    def _sample_stage2_grpo_rollout_actions(
        self,
        budget_action_logits_old: Tensor,
        stage2_cfg: Any,
    ) -> Tuple[Tensor, Tensor]:
        sampling_mode = str(stage2_cfg.grpo_rollout_sampling).strip().lower()
        if sampling_mode == "bucket":
            return self._sample_stage2_grpo_rollout_actions_bucket(
                budget_action_logits_old=budget_action_logits_old,
                group_size=int(stage2_cfg.grpo_group_size),
            )
        return self._sample_stage2_grpo_rollout_actions_on_policy(
            budget_action_logits_old=budget_action_logits_old,
            group_size=int(stage2_cfg.grpo_group_size),
        )

    def _build_stage2_grpo_rollout_cache(
        self,
        batch: DrivingExample,
        route_keys: List[str],
        stage2_cfg: Any,
        training_signals: Dict[str, Tensor],
        full_budget_driving_loss: Tensor,
        old_budget_action_logits: Tensor,
    ) -> Dict[str, Tensor]:
        # Reuse one rollout across update epochs.
        if "budget_action_logits" not in training_signals:
            raise ValueError("stage2 GRPO requires budget_action_logits in training_signals")
        if "budget_min" not in training_signals:
            raise ValueError("stage2 GRPO requires budget_min in training_signals")

        budget_action_logits_old = old_budget_action_logits.to(
            device=full_budget_driving_loss.device,
            dtype=torch.float32,
        ).detach()
        if budget_action_logits_old.ndim != 2:
            raise ValueError(
                "old_budget_action_logits must be rank-2 [B,K], "
                f"got shape={tuple(budget_action_logits_old.shape)}"
            )
        if int(budget_action_logits_old.size(0)) != int(full_budget_driving_loss.size(0)):
            raise ValueError(
                "old_budget_action_logits/full_budget batch mismatch: "
                f"{int(budget_action_logits_old.size(0))} vs {int(full_budget_driving_loss.size(0))}"
            )

        num_actions = int(training_signals["budget_action_logits"].size(1))
        if int(budget_action_logits_old.size(1)) != num_actions:
            raise ValueError(
                "old_budget_action_logits/training_signals action mismatch: "
                f"{int(budget_action_logits_old.size(1))} vs {num_actions}"
            )

        sampled_action_ids, behavior_logp = self._sample_stage2_grpo_rollout_actions(
            budget_action_logits_old=budget_action_logits_old,
            stage2_cfg=stage2_cfg,
        )

        rollout_driving_losses = self._collect_stage2_rollout_driving_losses(
            batch=batch,
            route_keys=route_keys,
            sampled_action_ids=sampled_action_ids,
            stage2_cfg=stage2_cfg,
        )
        if int(rollout_driving_losses.size(0)) != int(full_budget_driving_loss.size(0)):
            raise ValueError(
                "rollout/full-budget batch mismatch: "
                f"{int(rollout_driving_losses.size(0))} vs {int(full_budget_driving_loss.size(0))}"
            )

        budget_min = float(training_signals["budget_min"].item())
        action_table = build_smart_budget_action_table(
            budget_min=budget_min,
            device=rollout_driving_losses.device,
            dtype=rollout_driving_losses.dtype,
            num_actions=num_actions,
        )
        sampled_budget_values = action_table[sampled_action_ids]

        driving_loss_gap = rollout_driving_losses - full_budget_driving_loss.unsqueeze(1)
        budget_denominator = max(1.0 - budget_min, 1e-6)
        budget_normalized = (sampled_budget_values - budget_min) / budget_denominator
        reward_items = compute_grpo_reward(
            stage2_cfg=stage2_cfg,
            driving_loss_gap=driving_loss_gap,
            full_budget_driving_loss=full_budget_driving_loss,
            budget_normalized=budget_normalized,
        )
        rewards = reward_items["rewards"]
        advantage_items = compute_grpo_advantage(stage2_cfg=stage2_cfg, rewards=rewards)

        return {
            "budget_action_logits_old": budget_action_logits_old,
            "sampled_action_ids": sampled_action_ids,
            "behavior_logp": behavior_logp.detach(),
            **advantage_items,
            "rewards": rewards,
            "driving_loss_gap": driving_loss_gap,
            "allowed_gap": reward_items["allowed_gap"],
            "slack": reward_items["slack"],
            "gate": reward_items["gate"],
            "budget_normalized": budget_normalized,
        }

    def _compute_stage2_grpo_policy_loss(
        self,
        stage2_cfg: Any,
        training_signals: Dict[str, Tensor],
        rollout_cache: Dict[str, Tensor],
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        if "budget_action_logits" not in training_signals:
            raise ValueError("stage2 GRPO requires budget_action_logits in training_signals")
        if "budget_values" not in training_signals:
            raise ValueError("stage2 GRPO requires budget_values in training_signals")

        required_cache_keys = (
            "budget_action_logits_old",
            "sampled_action_ids",
            "behavior_logp",
            "advantages",
            "rewards",
            "driving_loss_gap",
            "allowed_gap",
            "slack",
            "gate",
            "budget_normalized",
        )
        for key in required_cache_keys:
            if key not in rollout_cache:
                raise ValueError(f"stage2 GRPO rollout cache missing key: {key}")

        budget_action_logits_new = training_signals["budget_action_logits"].float()
        if budget_action_logits_new.ndim != 2:
            raise ValueError(
                "budget_action_logits must be rank-2 [B,K], "
                f"got shape={tuple(budget_action_logits_new.shape)}"
            )
        budget_action_logits_old = rollout_cache["budget_action_logits_old"].to(
            device=budget_action_logits_new.device,
            dtype=torch.float32,
        ).detach()
        if budget_action_logits_old.shape != budget_action_logits_new.shape:
            raise ValueError(
                "old/new budget_action_logits shape mismatch: "
                f"{tuple(budget_action_logits_old.shape)} vs {tuple(budget_action_logits_new.shape)}"
            )

        sampled_action_ids = rollout_cache["sampled_action_ids"].to(
            device=budget_action_logits_new.device,
            dtype=torch.long,
        )
        advantages = rollout_cache["advantages"].to(
            device=budget_action_logits_new.device,
            dtype=budget_action_logits_new.dtype,
        ).detach()
        behavior_logp = rollout_cache["behavior_logp"].to(
            device=budget_action_logits_new.device,
            dtype=budget_action_logits_new.dtype,
        ).detach()

        # behavior_logp defines the denominator of the importance ratio.
        policy_loss, metrics = compute_grpo_policy_loss(
            stage2_cfg=stage2_cfg,
            budget_action_logits_new=budget_action_logits_new,
            budget_action_logits_old=budget_action_logits_old,
            sampled_action_ids=sampled_action_ids,
            behavior_logp=behavior_logp,
            advantages=advantages,
            entropy_beta=compute_grpo_entropy_beta(
                stage2_cfg=stage2_cfg,
                global_step=int(self.global_step),
            ),
        )

        budget_values = training_signals["budget_values"].float()
        rewards = rollout_cache["rewards"].to(device=budget_values.device, dtype=budget_values.dtype)
        driving_loss_gap = rollout_cache["driving_loss_gap"].to(
            device=budget_values.device,
            dtype=budget_values.dtype,
        )
        allowed_gap = rollout_cache["allowed_gap"].to(
            device=budget_values.device,
            dtype=budget_values.dtype,
        )
        slack = rollout_cache["slack"].to(device=budget_values.device, dtype=budget_values.dtype)
        gate = rollout_cache["gate"].to(device=budget_values.device, dtype=budget_values.dtype)
        budget_normalized = rollout_cache["budget_normalized"].to(
            device=budget_values.device,
            dtype=budget_values.dtype,
        )
        rel_advantages = rollout_cache["advantages_rel"].to(device=budget_values.device, dtype=budget_values.dtype)
        abs_advantages = rollout_cache["abs_advantage"].to(device=budget_values.device, dtype=budget_values.dtype)
        metrics.update({
            "grpo/reward_mean": rewards.mean(),
            "grpo/reward_std": rewards.std(unbiased=False),
            "grpo/rel_advantage_std": rel_advantages.std(unbiased=False),
            "grpo/abs_advantage_mean": abs_advantages.mean(),
            "grpo/abs_advantage_std": abs_advantages.std(unbiased=False),
            "grpo/advantage_mean": advantages.mean(),
            "grpo/advantage_std": advantages.std(unbiased=False),
            "budget_mean": budget_values.mean(),
            "budget_std": budget_values.std(unbiased=False),
            "driving_loss_gap_mean": driving_loss_gap.mean(),
            "grpo/allowed_gap_mean": allowed_gap.mean(),
            "grpo/slack_mean": slack.mean(),
            "grpo/gate_mean": gate.mean(),
            "grpo/budget_norm_mean": budget_normalized.mean(),
        })
        if "budget_query_feat_std" in training_signals:
            # Track whether budget queries remain state-dependent.
            metrics["grpo/budget_query_feat_std"] = training_signals["budget_query_feat_std"].to(
                device=budget_action_logits_new.device,
                dtype=budget_action_logits_new.dtype,
            )
        return policy_loss, metrics

    def _compute_stage1_lambda_div(self, global_step: int) -> float:
        if self.smart_assigner_train_cfg is None:
            raise ValueError("smart_assigner_train config is required for stage1 lambda_div schedule")
        stage1_cfg = self.smart_assigner_train_cfg.stage1
        ratio_start = float(stage1_cfg.div_ratio_start)
        ratio_end = float(stage1_cfg.div_ratio_end)
        warmup_steps = int(stage1_cfg.div_warmup_steps)
        if warmup_steps <= 0:
            ratio = ratio_end
        else:
            progress = float(max(min(global_step, warmup_steps), 0)) / float(warmup_steps)
            ratio = ratio_start + 0.5 * (1.0 - math.cos(math.pi * progress)) * (ratio_end - ratio_start)
        return float(stage1_cfg.lambda_wp) * ratio

    def _compose_smart_training_loss(
        self,
        output: TrainingOutput,
        smart_context: Dict[str, Any],
    ) -> Dict[str, Tensor]:
        if self.mode != "smart_assigner":
            raise ValueError("smart training loss composition requires mode=smart_assigner")
        if self.smart_assigner_train_cfg is None:
            raise ValueError("smart_assigner_train config is required for smart training loss composition")
        if self.smart_aux_loss_computer is None:
            raise ValueError("smart_aux_loss_computer must be initialized in smart_assigner mode")

        training_signals = smart_context["training_signals"]

        stage = int(self.smart_train_stage)
        if stage == 1:
            wp_loss, lang_loss = self._extract_primary_losses(output)
            stage1_cfg = self.smart_assigner_train_cfg.stage1
            aux_losses = self.smart_aux_loss_computer.compute_stage1_aux_losses(
                language_inputs=smart_context["language_inputs"],
                language_ids=smart_context["language_ids"],
                language_inputs_mask=smart_context["language_inputs_mask"],
                budget_token_features=training_signals["budget_token_features"],
                path_mask_hard=training_signals["path_mask_hard"],
                budget_values=training_signals["budget_values"],
                num_prefix_layers=training_signals["num_prefix_layers"],
                num_hidden_layers=training_signals["num_hidden_layers"],
                div_margin=float(stage1_cfg.div_margin),
                bal_epsilon=float(stage1_cfg.bal_epsilon),
            )
            lambda_div = self._compute_stage1_lambda_div(int(smart_context["global_step"]))
            lambda_bal = float(stage1_cfg.bal_ratio) * float(stage1_cfg.lambda_wp)
            loss_total = (
                float(stage1_cfg.lambda_wp) * wp_loss
                + float(stage1_cfg.lambda_lang) * lang_loss
                + lambda_div * aux_losses["div_loss"]
                + lambda_bal * aux_losses["bal_loss"]
            )
            return {
                "loss_total": loss_total,
                "loss_wp": wp_loss,
                "loss_lang": lang_loss,
                "loss_div": aux_losses["div_loss"],
                "loss_bal": aux_losses["bal_loss"],
                "lambda_div": wp_loss.new_tensor(lambda_div),
                "lambda_bal": wp_loss.new_tensor(lambda_bal),
                "scene_similarity_mean": aux_losses["scene_similarity_mean"],
                "path_similarity_mean": aux_losses["path_similarity_mean"],
                "target_sub_budget": aux_losses["target_sub_budget"],
                "path_mask_hard_mean": aux_losses["path_mask_hard_mean"],
                "budget_mean": training_signals["budget_values"].float().mean(),
                "budget_min": training_signals["budget_min"],
            }
        if stage == 2:
            stage2_cfg = self.smart_assigner_train_cfg.stage2
            if "grpo_rollout_cache" not in smart_context:
                raise ValueError("stage2 GRPO loss composition requires smart_context.grpo_rollout_cache")
            direct_driving_items = self._build_stage2_direct_driving_loss_items(
                output=output,
                stage2_cfg=stage2_cfg,
                global_step=int(smart_context["global_step"]),
            )
            policy_loss, stage2_metrics = self._compute_stage2_grpo_policy_loss(
                stage2_cfg=stage2_cfg,
                training_signals=training_signals,
                rollout_cache=smart_context["grpo_rollout_cache"],
            )
            prefix_anchor_loss = self._compute_stage2_prefix_anchor_loss(reference_tensor=policy_loss)
            scene_anchor_loss = self._compute_stage2_scene_anchor_loss(reference_tensor=policy_loss)
            loss_total = compose_stage2_objective(
                stage2_cfg=stage2_cfg,
                policy_loss=policy_loss,
                prefix_anchor_loss=prefix_anchor_loss,
                scene_anchor_loss=scene_anchor_loss,
                direct_driving_weighted_loss=direct_driving_items["stage2/driving_weighted_loss"],
            )
            out = {
                "loss_total": loss_total,
                "grpo/policy_loss": policy_loss,
                "grpo/prefix_anchor_loss": prefix_anchor_loss,
                "grpo/prefix_anchor_beta": policy_loss.new_tensor(float(stage2_cfg.grpo_prefix_kl_beta)),
                "grpo/scene_anchor_loss": scene_anchor_loss,
                "grpo/scene_anchor_beta": policy_loss.new_tensor(float(stage2_cfg.grpo_scene_kl_beta)),
            }
            out.update(direct_driving_items)
            out.update(stage2_metrics)
            return out
        raise ValueError(f"Unsupported smart training stage: {stage}")

    def _log_smart_loss_items(
        self,
        smart_loss_items: Dict[str, Tensor],
        mode: str,
        on_step: bool,
        on_epoch: bool,
    ) -> None:
        for key, value in sorted(smart_loss_items.items()):
            if key == "loss_total":
                continue
            self.log(
                f"{mode}_smart/{key}",
                value.detach(),
                on_step=on_step,
                on_epoch=on_epoch,
                prog_bar=False,
                logger=True,
                sync_dist=True,
            )

    def _resolve_step_total_loss(
        self,
        output: TrainingOutput,
        smart_context: Optional[Dict[str, Any]],
        mode: str,
        on_step: bool,
        on_epoch: bool,
    ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        if self.mode != "smart_assigner":
            raise ValueError(
                f"training step now requires mode=smart_assigner, got mode={self.mode}"
            )
        if smart_context is None:
            raise ValueError("smart_context is required when mode=smart_assigner")
        smart_loss_items = self._compose_smart_training_loss(
            output=output,
            smart_context=smart_context,
        )
        total_loss = smart_loss_items["loss_total"]
        self._log_smart_loss_items(
            smart_loss_items=smart_loss_items,
            mode=mode,
            on_step=on_step,
            on_epoch=on_epoch,
        )
        return total_loss, smart_loss_items

    def _set_stage2_budget_exploration_progress(self) -> None:
        if not isinstance(self.budget_assigner, SmartBudgetAssigner):
            return
        estimated_total_steps: Optional[int] = None
        trainer = getattr(self, "trainer", None)
        if trainer is not None:
            estimated_steps = getattr(trainer, "estimated_stepping_batches", None)
            if estimated_steps is not None:
                estimated_steps_int = int(estimated_steps)
                if estimated_steps_int > 0:
                    estimated_total_steps = estimated_steps_int
                    if int(self.smart_train_stage) == 2 and self.smart_assigner_train_cfg is not None:
                        estimated_total_steps = (
                            estimated_total_steps * int(self.smart_assigner_train_cfg.stage2.grpo_update_epochs)
                        )
        self.budget_assigner.set_stage2_budget_exploration_progress(
            global_step=int(self.global_step),
            estimated_total_steps=estimated_total_steps,
        )

    def _resolve_single_optimizer(self):
        optimizer = self.optimizers()
        if isinstance(optimizer, (list, tuple)):
            if len(optimizer) != 1:
                raise ValueError(f"Expected exactly one optimizer, got {len(optimizer)}")
            optimizer = optimizer[0]
        return optimizer

    def _resolve_single_lr_scheduler(self):
        lr_scheduler = self.lr_schedulers()
        if isinstance(lr_scheduler, (list, tuple)):
            if len(lr_scheduler) != 1:
                raise ValueError(f"Expected exactly one lr scheduler, got {len(lr_scheduler)}")
            lr_scheduler = lr_scheduler[0]
        if lr_scheduler is None:
            raise ValueError("Manual optimization requires a step-wise lr scheduler")
        return lr_scheduler

    @staticmethod
    def _format_tensor_finite_stats(name: str, tensor: Tensor) -> str:
        tensor_detached = tensor.detach()
        tensor_float = tensor_detached.float()
        finite_values = tensor_float[torch.isfinite(tensor_float)]
        max_abs = float(finite_values.abs().max().item()) if finite_values.numel() > 0 else float("nan")
        return (
            f"{name}: shape={tuple(tensor_detached.shape)}, dtype={tensor_detached.dtype}, "
            f"nan={int(torch.isnan(tensor_float).sum().item())}, "
            f"inf={int(torch.isinf(tensor_float).sum().item())}, "
            f"max_abs={max_abs:.6f}"
        )

    def _first_nonfinite_gradient(self) -> Optional[str]:
        for name, param in self.named_parameters():
            if not param.requires_grad or param.grad is None:
                continue
            if torch.isfinite(param.grad.detach()).all():
                continue
            return self._format_tensor_finite_stats(f"{name}.grad", param.grad)
        return None

    def _raise_if_nonfinite_loss(self, total_loss: Tensor) -> None:
        if torch.isfinite(total_loss.detach()).all():
            return
        print(
            "[finite-check] non-finite total_loss before backward "
            f"rank={int(getattr(self, 'global_rank', 0))} "
            f"global_step={int(self.global_step)} "
            f"{self._format_tensor_finite_stats('total_loss', total_loss)}",
            flush=True,
        )
        raise FloatingPointError("non-finite total_loss before backward")

    def _raise_if_nonfinite_gradients(self, stage: str) -> None:
        issue = self._first_nonfinite_gradient()
        if issue is None:
            return
        print(
            "[finite-check] non-finite gradients detected "
            f"stage={stage} rank={int(getattr(self, 'global_rank', 0))} "
            f"global_step={int(self.global_step)} {issue}",
            flush=True,
        )
        raise FloatingPointError(f"non-finite gradients detected at {stage}")

    def _raise_if_missing_stage2_policy_gradients(self) -> None:
        required_parameters = [
            (
                f"budget_assigner.smart_budget_scheduler.{name}",
                parameter,
            )
            for name, parameter in self.budget_assigner.smart_budget_scheduler.named_parameters()
        ]
        required_parameters.append(
            (
                "adaptors.assignquery.query_budget",
                self.adaptors.assignquery.query_budget,
            )
        )
        missing = [
            name
            for name, parameter in required_parameters
            if parameter.requires_grad and parameter.grad is None
        ]
        if missing:
            raise RuntimeError(
                "stage2 policy parameters received no gradient: " + ", ".join(missing)
            )

    def _manual_backward_step(self, optimizer, lr_scheduler, total_loss: Tensor) -> None:
        self._raise_if_nonfinite_loss(total_loss)
        optimizer.zero_grad()
        self.manual_backward(total_loss)
        self._raise_if_missing_stage2_policy_gradients()
        self._raise_if_nonfinite_gradients(stage="after_backward")
        gradient_clip_val = float(self.manual_grad_clip_val)
        if gradient_clip_val > 0.0:
            self.clip_gradients(
                optimizer,
                gradient_clip_val=gradient_clip_val,
                gradient_clip_algorithm=str(self.manual_grad_clip_algorithm),
            )
            self._raise_if_nonfinite_gradients(stage="after_clip_gradients")
        optimizer.step()
        # The next forward must observe the updated FP32 parameters.
        torch.clear_autocast_cache()
        lr_scheduler.step()

    def training_step(self, batch: DrivingExample, _batch_idx: int = 0):
        if self.mode != "smart_assigner":
            raise ValueError(f"training_step now requires mode=smart_assigner, got mode={self.mode}")
        debug_first_step = bool(getattr(self, "global_rank", 0) == 0 and not self._debug_first_train_step_printed)
        if debug_first_step:
            print(
                f"[train-debug] training_step start step={int(self.global_step)} "
                f"batch_size={int(batch.driving_input.camera_images.size(0))}",
                flush=True,
            )

        route_keys_for_step = self._extract_route_keys(batch)
        if route_keys_for_step is None:
            raise ValueError("smart_assigner training requires route keys from batch.run_id")
        if batch.slot_ids is None:
            raise ValueError("smart_assigner route_stream training requires DrivingExample.slot_ids")
        if batch.slot_ids.ndim != 1:
            raise ValueError(f"slot_ids must be 1D tensor, got shape={tuple(batch.slot_ids.shape)}")
        if len(route_keys_for_step) != int(batch.slot_ids.size(0)):
            raise ValueError(
                f"slot_ids length ({int(batch.slot_ids.size(0))}) must match route_keys length ({len(route_keys_for_step)})"
            )

        reset_route_keys = []
        for idx, route_key in enumerate(route_keys_for_step):
            slot_id = int(batch.slot_ids[idx].item())
            prev_route_key = self._train_slot_route_state.get(slot_id, None)
            if prev_route_key is not None and prev_route_key != route_key:
                reset_route_keys.append(prev_route_key)
            self._train_slot_route_state[slot_id] = route_key
        if len(reset_route_keys) > 0:
            self.budget_assigner.reset(route_keys=sorted(set(reset_route_keys)))

        is_stage2 = int(self.smart_train_stage) == 2
        if not is_stage2:
            output, _, smart_context = self.forward_loss(
                batch,
                route_keys=route_keys_for_step,
                return_smart_context=True,
                update_history=True,
            )
            total_loss, smart_loss_items = self._resolve_step_total_loss(
                output=output,
                smart_context=smart_context,
                mode="train",
                on_step=True,
                on_epoch=True,
            )
            logs = output  # .update(loss_logs)
            self.log_training_output(logs, "train")
            batch_size = int(batch.driving_input.camera_images.size(0))
            self.log(
                "train/loss",
                total_loss,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                batch_size=batch_size,
                sync_dist=True,
                add_dataloader_idx=False,
            )
            if debug_first_step:
                print(
                    f"[train-debug] training_step loss ready={float(total_loss.detach().cpu()):.6f}",
                    flush=True,
                )
                self._debug_first_train_step_printed = True
            return {"loss": total_loss, "outputs": output, "smart_losses": smart_loss_items}

        if self.smart_assigner_train_cfg is None:
            raise ValueError("smart_assigner_train config is required for stage2 training")
        update_epochs = int(self.smart_assigner_train_cfg.stage2.grpo_update_epochs)
        self._set_stage2_budget_exploration_progress()
        stage2_old_policy_logits = self._collect_stage2_old_policy_logits_for_batch(
            batch=batch,
            route_keys=route_keys_for_step,
        )
        optimizer = self._resolve_single_optimizer()
        lr_scheduler = self._resolve_single_lr_scheduler()

        grpo_rollout_cache: Optional[Dict[str, Tensor]] = None
        final_output: Optional[TrainingOutput] = None
        final_loss: Optional[Tensor] = None
        final_smart_losses: Optional[Dict[str, Tensor]] = None

        for update_idx in range(update_epochs):
            self._set_stage2_budget_exploration_progress()
            update_history = update_idx == (update_epochs - 1)
            output, _, smart_context = self.forward_loss(
                batch,
                route_keys=route_keys_for_step,
                return_smart_context=True,
                update_history=update_history,
            )

            smart_context["old_budget_action_logits"] = stage2_old_policy_logits
            if grpo_rollout_cache is None:
                training_signals = smart_context["training_signals"]
                budget_action_logits = training_signals["budget_action_logits"].float()
                full_budget_driving_loss = self._resolve_stage2_full_budget_driving_loss(
                    smart_context=smart_context,
                    device=budget_action_logits.device,
                    dtype=budget_action_logits.dtype,
                    batch_size=int(budget_action_logits.size(0)),
                )
                grpo_rollout_cache = self._build_stage2_grpo_rollout_cache(
                    batch=batch,
                    route_keys=route_keys_for_step,
                    stage2_cfg=self.smart_assigner_train_cfg.stage2,
                    training_signals=training_signals,
                    full_budget_driving_loss=full_budget_driving_loss,
                    old_budget_action_logits=stage2_old_policy_logits,
                )
            smart_context["grpo_rollout_cache"] = grpo_rollout_cache

            total_loss, smart_loss_items = self._resolve_step_total_loss(
                output=output,
                smart_context=smart_context,
                mode="train",
                on_step=True,
                on_epoch=True,
            )
            logs = output  # .update(loss_logs)
            self.log_training_output(logs, "train")
            batch_size = int(batch.driving_input.camera_images.size(0))
            self.log(
                "train/loss",
                total_loss,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                batch_size=batch_size,
                sync_dist=True,
                add_dataloader_idx=False,
            )
            self._manual_backward_step(optimizer=optimizer, lr_scheduler=lr_scheduler, total_loss=total_loss)

            final_output = output
            final_loss = total_loss
            final_smart_losses = smart_loss_items

        # Refresh after all update epochs so the ratio does not collapse to one.
        self._refresh_stage2_policy_snapshots(refresh_prefix_anchor=False)

        if final_output is None or final_loss is None or final_smart_losses is None:
            raise ValueError("training_step produced no optimization result")

        return {"loss": final_loss, "outputs": final_output, "smart_losses": final_smart_losses}

    def on_before_backward(self, loss: Tensor) -> None:
        if bool(getattr(self, "global_rank", 0) == 0 and not self._debug_first_backward_printed):
            print(f"[train-debug] backward start loss={float(loss.detach().cpu()):.6f}", flush=True)

    def on_after_backward(self) -> None:
        if bool(getattr(self, "global_rank", 0) == 0 and not self._debug_first_backward_printed):
            self._debug_first_backward_printed = True
            print("[train-debug] backward done", flush=True)

    def on_fit_start(self) -> None:
        if self.mode != "smart_assigner" or int(self.smart_train_stage) != 2:
            return
        self._disable_stage2_lora_dropout()
        refresh_prefix_anchor = len(self._stage2_prefix_anchor_params) == 0
        refresh_scene_anchor = len(self._stage2_scene_anchor_params) == 0
        self._refresh_stage2_policy_snapshots(
            refresh_prefix_anchor=refresh_prefix_anchor,
            refresh_scene_anchor=refresh_scene_anchor,
        )

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        if self.mode != "smart_assigner" or int(self.smart_train_stage) != 2:
            return
        if not self._stage2_snapshots_ready:
            return
        checkpoint["stage2_old_policy_params"] = {
            key: value.detach().cpu() for key, value in self._stage2_old_policy_params.items()
        }
        checkpoint["stage2_prefix_anchor_params"] = {
            key: value.detach().cpu() for key, value in self._stage2_prefix_anchor_params.items()
        }
        checkpoint["stage2_scene_anchor_params"] = {
            key: value.detach().cpu() for key, value in self._stage2_scene_anchor_params.items()
        }

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        old_policy = checkpoint.get("stage2_old_policy_params", None)
        prefix_anchor = checkpoint.get("stage2_prefix_anchor_params", None)
        scene_anchor = checkpoint.get("stage2_scene_anchor_params", None)

        if isinstance(old_policy, dict):
            self._stage2_old_policy_params = {
                str(key): value.detach().clone()
                for key, value in old_policy.items()
                if torch.is_tensor(value)
            }
        else:
            self._stage2_old_policy_params = {}

        if isinstance(prefix_anchor, dict):
            self._stage2_prefix_anchor_params = {
                str(key): value.detach().clone()
                for key, value in prefix_anchor.items()
                if torch.is_tensor(value)
            }
        else:
            self._stage2_prefix_anchor_params = {}

        if isinstance(scene_anchor, dict):
            self._stage2_scene_anchor_params = {
                str(key): value.detach().clone()
                for key, value in scene_anchor.items()
                if torch.is_tensor(value)
            }
        else:
            self._stage2_scene_anchor_params = {}

        self._stage2_snapshots_ready = len(self._stage2_old_policy_params) > 0

    def _disable_stage2_lora_dropout(self) -> None:
        """Disable Stage2 LoRA dropout for deterministic policy comparisons."""
        if self.mode != "smart_assigner" or int(self.smart_train_stage) != 2:
            return
        for name, module in self.language_model.named_modules():
            if "lora_dropout" in name and isinstance(module, nn.Dropout):
                module.eval()

    def on_train_epoch_start(self) -> None:
        # Prevent route history from leaking across epochs.
        self.budget_assigner.reset()
        self._train_slot_route_state = {}
        self._disable_stage2_lora_dropout()


    def validation_step(self, batch: DrivingExample, _batch_idx: int = 0):
        if self.mode != "smart_assigner":
            raise ValueError(f"validation_step now requires mode=smart_assigner, got mode={self.mode}")

        route_keys = self._extract_route_keys(batch)
        if route_keys is None:
            raise ValueError("validation requires route keys from batch.run_id")

        if int(self.smart_train_stage) == 2:
            if self.smart_assigner_train_cfg is None:
                raise ValueError("smart_assigner_train config is required for stage2 validation")
            stage2_old_policy_logits = self._collect_stage2_old_policy_logits_for_batch(
                batch=batch,
                route_keys=route_keys,
            )
            _, _, provisional_smart_context = self.forward_loss(
                batch,
                route_keys=route_keys,
                return_smart_context=True,
                update_history=False,
            )
            training_signals = provisional_smart_context["training_signals"]
            budget_action_logits = training_signals["budget_action_logits"].float()
            full_budget_driving_loss = self._resolve_stage2_validation_full_budget_driving_loss(
                batch=batch,
                route_keys=route_keys,
                smart_context=provisional_smart_context,
                stage2_cfg=self.smart_assigner_train_cfg.stage2,
                device=budget_action_logits.device,
                dtype=budget_action_logits.dtype,
                batch_size=int(budget_action_logits.size(0)),
            )
            grpo_rollout_cache = self._build_stage2_grpo_rollout_cache(
                batch=batch,
                route_keys=route_keys,
                stage2_cfg=self.smart_assigner_train_cfg.stage2,
                training_signals=training_signals,
                full_budget_driving_loss=full_budget_driving_loss,
                old_budget_action_logits=stage2_old_policy_logits,
            )
            # Compare all policies from one frame state, then commit the live result.
            output, loss_logs, smart_context = self.forward_loss(
                batch,
                route_keys=route_keys,
                return_smart_context=True,
                update_history=True,
            )
            smart_context["old_budget_action_logits"] = stage2_old_policy_logits
            smart_context["grpo_rollout_cache"] = grpo_rollout_cache
        else:
            output, loss_logs, smart_context = self.forward_loss(
                batch,
                route_keys=route_keys,
                return_smart_context=True,
                update_history=True,
            )

        total_loss, smart_loss_items = self._resolve_step_total_loss(
            output=output,
            smart_context=smart_context,
            mode="val",
            on_step=False,
            on_epoch=True,
        )
        logs = output #.update(loss_logs)
        self.log_training_output(logs, "val")

        # log the loss
        batch_size = int(batch.driving_input.camera_images.size(0))
        self.log(
            "val/loss",
            total_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=batch_size,
            sync_dist=True,
            add_dataloader_idx=False,
        )

        return {"loss": total_loss, "outputs": output, "smart_losses": smart_loss_items}

    def predict_step(self, batch: DrivingExample, _batch_idx: int = 0):
        run_ids = decode_uint8(batch.run_id)
        
        speed_wps, route, language = self.forward(
            batch,
            return_language=True,
            budget=self.eval_fixed_budget,
        )

        self.num_route_points = 20
        route_equal = []
        for i in range(len(route)):
            route_equal.append(self.equal_spacing_route(route[i].cpu()))
        route_equal = torch.tensor(route_equal)
        route = route_equal.to(route.device)
        
        
        route_gt = batch.driving_label.path
        speed_wps_gt = batch.driving_label.waypoints
        language_gt = batch.driving_label.answer.language_string
        
        if len(self.prediction) == 0:
            self.prediction = {
                "waypoints": [speed_wps],
                "route": [route],
                "language": language,
                "waypoints_gt": [speed_wps_gt],
                "route_gt": [route_gt],
                "language_gt": language_gt,
                "prompt": batch.driving_input.prompt.language_string,
                "path": run_ids,
                "qa_templates": batch.qa_templates,
                "eval_infos": batch.driving_label.eval_infos,
            }
        else:
            self.prediction["waypoints"].append(speed_wps)
            self.prediction["route"].append(route)
            self.prediction["language"].extend(language)
            self.prediction["waypoints_gt"].append(speed_wps_gt)
            self.prediction["route_gt"].append(route_gt)
            self.prediction["language_gt"].extend(language_gt)
            self.prediction["prompt"].extend(batch.driving_input.prompt.language_string)
            self.prediction["path"].extend(run_ids)
            self.prediction["qa_templates"].extend(batch.qa_templates)
            self.prediction["eval_infos"].extend(batch.driving_label.eval_infos)
            
        
        return speed_wps, route, language, speed_wps_gt, route_gt, language_gt

    def equal_spacing_route(self, points):
        route = np.concatenate((np.zeros_like(points[:1]),  points)) # Add 0 to front
        shift = np.roll(route, 1, axis=0) # Shift by 1
        shift[0] = shift[1] # Set wraparound value to 0

        dists = np.linalg.norm(route-shift, axis=1)
        dists = np.cumsum(dists)
        dists += np.arange(0, len(dists))*1e-4 # Prevents dists not being strictly increasing

        x = np.arange(0, 20, 1)
        interp_points = np.array([np.interp(x, dists, route[:, 0]), np.interp(x, dists, route[:, 1])]).T

        return interp_points

    def on_predict_epoch_end(self) -> None:    

        repo_path = get_original_cwd()

        prediction_save_dir = getattr(self, "prediction_save_dir", None)
        if prediction_save_dir is not None:
            save_prediction_path = Path(prediction_save_dir)
        elif self.trainer.ckpt_path is not None:
            ckpt_path = Path(self.trainer.ckpt_path).parent.parent
            save_prediction_path = ckpt_path / "predictions"
        else:
            ckpt_path = Path(f'{repo_path}/outputs/{self.language_model.variant}')
            save_prediction_path = ckpt_path / "predictions"
        save_prediction_path.mkdir(exist_ok=True, parents=True)
        
        samples_cot = [i for i, l in enumerate(self.prediction["prompt"]) if "What should the ego do next?" in l]
        samples_qa = [i for i, l in enumerate(self.prediction["prompt"]) if "Q:" in l]
        samples_all = [i for i in range(len(self.prediction["prompt"]))]
        language = [(l, l_gt, p) for l, l_gt, p in zip(self.prediction["language"], self.prediction["language_gt"], self.prediction["path"])]
        
        if len(samples_qa) > 0:
            # sort by templates
            sorted_samples = {} # question: {answer: [language, language_gt]}
            for qa_template, language_sample in zip(self.prediction["qa_templates"], language):
                question = qa_template[0]
                answer = qa_template[1]
                if question not in sorted_samples:
                    sorted_samples[question] = {}
                if answer not in sorted_samples[question]:
                    sorted_samples[question][answer] = []
                sorted_samples[question][answer].append(language_sample)
            
            if os.path.exists(f"{str(save_prediction_path)}/sorted_qa_templates_rank_{self.local_rank}.json"):
                time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                with open(f"{str(save_prediction_path)}/sorted_qa_templates_rank_{self.local_rank}_{time}.json", "w") as f:
                    json.dump(sorted_samples, f, indent=4)
            else:
                with open(f"{str(save_prediction_path)}/sorted_qa_templates_rank_{self.local_rank}.json", "w") as f:
                    json.dump(sorted_samples, f, indent=4)
        
        for samples, name in zip([samples_cot, samples_qa, samples_all], ["cot", "qa", "all"]):
            language_samples = [l for i, l in enumerate(language) if i in samples]
        
            # save language predictions
            save_path_tmp = f"{str(save_prediction_path)}/language_preds_{name}_rank_{self.local_rank}.json"
            if os.path.exists(save_path_tmp):
                time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                save_path_tmp = f"{str(save_prediction_path)}/language_preds_{name}_rank_{self.local_rank}_{time}.json"
            with open(save_path_tmp, "w") as f:
                json.dump(language_samples, f, indent=4)
            
        route_preds = self.prediction["route"]
        route_preds = torch.cat(route_preds, dim=0)
        route_gt = self.prediction["route_gt"]
        route_gt = torch.cat(route_gt, dim=0)
        
        waypoints_preds = self.prediction["waypoints"]
        waypoints_preds = torch.cat(waypoints_preds, dim=0)
        waypoints_gt = self.prediction["waypoints_gt"]
        waypoints_gt = torch.cat(waypoints_gt, dim=0)
        
        # calc distance between wps for 1d wps
        waypoints_preds_1d = []
        for i in range(len(waypoints_preds)):
            waypoint_pred = waypoints_preds[i]
            waypoints_preds_1d_tmp = torch.tensor([torch.linalg.norm(waypoint_pred[i+1] - waypoint_pred[i]) for i in range(len(waypoint_pred)-1)])
            # cumsum to get the distance from the start
            waypoints_preds_1d_tmp = torch.cumsum(waypoints_preds_1d_tmp, dim=0)
            waypoints_preds_1d_tmp = [[x, 0] for x in waypoints_preds_1d_tmp]
            waypoints_preds_1d.append(waypoints_preds_1d_tmp)
        waypoints_preds_1d = torch.tensor(waypoints_preds_1d)
        
        waypoints_gt_1d = []
        for i in range(len(waypoints_gt)):
            waypoint_gt = waypoints_gt[i]
            waypoints_gt_1d_tmp = torch.tensor([torch.linalg.norm(waypoint_gt[i+1] - waypoint_gt[i]) for i in range(len(waypoint_gt)-1)])
            # cumsum to get the distance from the start
            waypoints_gt_1d_tmp = torch.cumsum(waypoints_gt_1d_tmp, dim=0)
            waypoints_gt_1d_tmp = [[x, 0] for x in waypoints_gt_1d_tmp]
            waypoints_gt_1d.append(waypoints_gt_1d_tmp)
        waypoints_gt_1d = torch.tensor(waypoints_gt_1d)
        
        # calculate ade and fde for samples seperatly whihc have <SAFETY> in prompt and for <INSTRUCTION_FOLLOWING>
        samples_safety = [i for i, l in enumerate(self.prediction["prompt"]) if "<SAFETY>" in l]
        samples_instruction = [i for i, l in enumerate(self.prediction["prompt"]) if "<INSTRUCTION_FOLLOWING>" in l]
        samples_neither = [i for i, l in enumerate(self.prediction["prompt"]) if "<SAFETY>" not in l and "<INSTRUCTION_FOLLOWING>" not in l]
        samples_all = [i for i in range(len(self.prediction["prompt"]))]
        
        ade_fde = {}
        
        def get_desired_end_speed(wps):
            wp_freq = 5
            carla_fps = 20
            # one WP every 0.25 seconds
            # we want to get speed at the last WP
            last_wp = wps[-1] #.cpu().numpy()
            # we want the WP half second earlier than the last WP
            one_second = int(carla_fps // (wp_freq))
            half_second = one_second // 2
            prev_wp = wps[-1 - half_second] #.cpu().numpy()
            desired_speed = np.linalg.norm(prev_wp - last_wp) * 2.0
            return desired_speed
        def get_desired_speed(wps):
            wp_freq = 5
            carla_fps = 20
            # Waypoints are spaced by 0.25 seconds.
            one_second = int(carla_fps // (wp_freq))
            half_second = one_second // 2
            wp_half_second = wps[half_second] #.cpu().numpy()
            wp_one_second = wps[one_second] #.cpu().numpy()
            desired_speed = np.linalg.norm(wp_half_second - wp_one_second) * 2.0
            return desired_speed
        
        def get_desired_avg_speed(wps):
            wp_freq = 5
            carla_fps = 20
            first_wp = wps[0] #.cpu().numpy()
            last_wp = wps[-1] #.cpu().numpy()
            desired_speed = np.linalg.norm(first_wp - last_wp) / (len(wps) * 0.25)
            return desired_speed
        
        def get_1d_wps(wps):
            waypoints_1d = [np.linalg.norm(wps[i+1] - wps[i]) for i in range(len(wps)-1)]
            # cumsum to get the distance from the start
            waypoints_1d = np.cumsum(waypoints_1d)
            waypoints_1d = [[x, 0] for x in waypoints_1d]
            
            # prepend 0,0
            waypoints_1d = [[0, 0]] + waypoints_1d
            
            return np.array(waypoints_1d).reshape(-1, 2)
        
            
        wp_freq = 5
        carla_fps = 20
            
        
        for samples, name in zip([samples_safety, samples_instruction, samples_neither, samples_all], ["instruction"]):
            if len(samples) == 0:
                continue
            route_preds_sample = route_preds[samples].cpu().numpy()
            route_gt_sample = route_gt[samples].cpu().numpy()
            waypoints_preds_sample = waypoints_preds[samples].cpu().numpy()
            waypoints_gt_sample = waypoints_gt[samples].cpu().numpy()
            eval_infos_sample = [self.prediction["eval_infos"][i] for i in samples]
            waypoints_org_sample = [eval_infos_sample[i]["org_wps"] for i in range(len(samples))]
            route_org_sample = [eval_infos_sample[i]["org_path"] for i in range(len(samples))]
            waypoints_instruction_sample = [np.array(eval_infos_sample[i]["new_wps"]) for i in range(len(samples))]
            route_instruction_sample = [eval_infos_sample[i]["new_path"] for i in range(len(samples))]
            prompts = [self.prediction["prompt"][i].replace("<IMG_CONTEXT>", "") for i in samples]
            pred_language = [self.prediction["language"][i] for i in samples]
            paths = [self.prediction["path"][i] for i in samples]
            
            success_rate_all = []
            success_rate_by_mode = {}
            success_rate_by_allowed = {}
            
            paths_by_mode = {}
            
            for i in range(len(samples)):
                mode = eval_infos_sample[i]["mode"]
                allowed = eval_infos_sample[i]['allowed']
                sample_path = paths[i]
                
                # Initialize mode in dictionary if not present
                if mode not in success_rate_by_mode:
                    success_rate_by_mode[mode] = []
                if mode not in paths_by_mode:
                    paths_by_mode[mode] = []
                
                # Initialize allowed in dictionary if not present
                if allowed not in success_rate_by_allowed:
                    success_rate_by_allowed[allowed] = []
                
                # get desired speed form WPs
                desired_end_speed_pred = get_desired_end_speed(waypoints_preds_sample[i])
                desired_end_speed_gt = get_desired_end_speed(waypoints_gt_sample[i])
                desired_end_speed_org = get_desired_end_speed(waypoints_org_sample[i])
                desired_end_speed_instruction = get_desired_end_speed(waypoints_instruction_sample[i])
                
                desired_speed_pred = get_desired_speed(waypoints_preds_sample[i])
                desired_speed_gt = get_desired_speed(waypoints_gt_sample[i])
                desired_speed_org = get_desired_speed(waypoints_org_sample[i])
                desired_speed_instruction = get_desired_speed(waypoints_instruction_sample[i])
                
                desired_avg_speed_pred = get_desired_avg_speed(waypoints_preds_sample[i])
                desired_avg_speed_gt = get_desired_avg_speed(waypoints_gt_sample[i])
                desired_avg_speed_org = get_desired_avg_speed(waypoints_org_sample[i])
                desired_avg_speed_instruction = get_desired_avg_speed(waypoints_instruction_sample[i])

                
                pred_wps_1d = get_1d_wps(waypoints_preds_sample[i])
                pred_wps_1d_diffs = np.diff(pred_wps_1d[:, 0])
                pred_speeds = pred_wps_1d_diffs / (wp_freq/carla_fps)
                
                org_wps_1d = get_1d_wps(waypoints_org_sample[i])
                org_wps_1d_diffs = np.diff(org_wps_1d[:, 0])
                org_speeds = org_wps_1d_diffs / (wp_freq/carla_fps)
                
                instruction_wps_1d = get_1d_wps(waypoints_instruction_sample[i])
                instruction_wps_1d_diffs = np.diff(instruction_wps_1d[:, 0])
                instruction_speeds = instruction_wps_1d_diffs / (wp_freq/carla_fps)
                
                x = np.arange(len(pred_speeds))*0.25
                
                # linear regression np
                slope_pred, intercept_pred = np.polyfit(x, pred_speeds, 1)
                slope_org, intercept_org = np.polyfit(x, org_speeds, 1)
                slope_instruction, intercept_instruction = np.polyfit(x, instruction_speeds, 1)
                
                current_speed = float(prompts[i].split("Current speed: ")[-1].split(" ")[0])
                
                if mode == 'stop':
                    # route doesnt matter
                    paths_by_mode[mode].append(sample_path)
                    if name == 'instruction' or name == 'neither':
                        if np.min(pred_speeds) < 0.1:
                            success_rate_all.append(1)
                            success_rate_by_mode[mode].append(1)
                            success_rate_by_allowed[allowed].append(1)
                        else:
                            success_rate_all.append(0)
                            success_rate_by_mode[mode].append(0)
                            success_rate_by_allowed[allowed].append(0)
                            
                elif mode == 'slower':
                    paths_by_mode[mode].append(sample_path)

                    if name == 'instruction' or name == 'neither':
                        # forced instruction following
                        if slope_pred < (-0.05 * current_speed):
                            success_rate_all.append(1)
                            success_rate_by_mode[mode].append(1)
                            success_rate_by_allowed[allowed].append(1)
                        else:
                            success_rate_all.append(0)
                            success_rate_by_mode[mode].append(0)
                            success_rate_by_allowed[allowed].append(0)
                        
                elif mode == 'faster':
                    paths_by_mode[mode].append(sample_path)
                    
                    if name == 'instruction' or name == 'neither':
                        # forced instruction following
                        if slope_pred > (0.05 * current_speed):
                            success_rate_all.append(1)
                            success_rate_by_mode[mode].append(1)
                            success_rate_by_allowed[allowed].append(1)
                        else:
                            success_rate_all.append(0)
                            success_rate_by_mode[mode].append(0)
                            success_rate_by_allowed[allowed].append(0)
                elif mode == 'target_speed':
                    paths_by_mode[mode].append(sample_path)
                    
                    try:
                        target_speed = float(prompts[i].split("Target waypoint: ")[-1].split("Command")[-1].split(".<|im_end|>")[0].split(" ")[-2])
                    except:
                        target_speed = float(prompts[i].split("Target waypoint: ")[-1].split("Command")[-1].split(".<|im_end|>")[0].split(" ")[-3])
                    if name == 'instruction' or name == 'neither':
                        if ((desired_end_speed_pred > 0.8 * desired_end_speed_instruction and desired_end_speed_pred < 1.2 * desired_end_speed_instruction) or (desired_end_speed_pred > 0.8 * target_speed and desired_end_speed_pred < 1.2 * target_speed)):
                            success_rate_all.append(1)
                            success_rate_by_mode[mode].append(1)
                            success_rate_by_allowed[allowed].append(1)
                        else:
                            success_rate_all.append(0)
                            success_rate_by_mode[mode].append(0)
                            success_rate_by_allowed[allowed].append(0)
                    
                elif mode == 'lane_change':
                    paths_by_mode[mode].append(sample_path)
                    
                    # on path
                    fde_pred_org = np.linalg.norm(route_preds_sample[i][-1] - route_org_sample[i][-1], axis=-1)
                    fde_pred_instruction = np.linalg.norm(route_preds_sample[i][-1] - route_instruction_sample[i][-1], axis=-1)
                    if name == 'instruction' or name == 'neither':
                        if fde_pred_instruction < fde_pred_org:
                            success_rate_all.append(1)
                            success_rate_by_mode[mode].append(1)
                            success_rate_by_allowed[allowed].append(1)
                        else:
                            success_rate_all.append(0)
                            success_rate_by_mode[mode].append(0)
                            success_rate_by_allowed[allowed].append(0)
                elif mode == 'crash':
                    paths_by_mode[mode].append(sample_path)
                    
                    ade_path_org_instruction = np.mean(np.linalg.norm(route_org_sample[i] - route_instruction_sample[i], axis=-1))
                    ade_path_pred_org = np.mean(np.linalg.norm(route_preds_sample[i] - route_org_sample[i], axis=-1))
                    ade_path_pred_instruction = np.mean(np.linalg.norm(route_preds_sample[i] - route_instruction_sample[i], axis=-1))
                    if ade_path_org_instruction > 1.0:
                        if name == 'instruction' or name == 'neither':
                            if ade_path_pred_instruction < ade_path_pred_org:
                                success_rate_all.append(1)
                                success_rate_by_mode[mode].append(1)
                                success_rate_by_allowed[allowed].append(1)
                            else:
                                success_rate_all.append(0)
                                success_rate_by_mode[mode].append(0)
                                success_rate_by_allowed[allowed].append(0)
                    else:
                        if name == 'instruction' or name == 'neither':
                            if ade_path_pred_instruction < 1.0 and (np.mean(pred_speeds) < 1.3 * np.mean(instruction_speeds) or np.mean(pred_speeds) > 0.7 * np.mean(instruction_speeds)):
                                success_rate_all.append(1)
                                success_rate_by_mode[mode].append(1)
                                success_rate_by_allowed[allowed].append(1)
                            else:
                                success_rate_all.append(0)
                                success_rate_by_mode[mode].append(0)
                                success_rate_by_allowed[allowed].append(0)

                else:
                    print(f"Unknown mode: {mode} in sample {i} with path {sample_path}")
                                
            # save result per sample
            per_sample_results = {
                'paths_by_mode': paths_by_mode,
                'success_rate_by_mode': success_rate_by_mode
            }
            save_path_tmp = f"{str(save_prediction_path)}/results_per_sample_{name}_rank_{self.local_rank}.json"
            if os.path.exists(save_path_tmp):
                time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                save_path_tmp = f"{str(save_prediction_path)}/results_per_sample_{name}_rank_{self.local_rank}_{time}.json"
            with open(save_path_tmp, "w") as f:
                json.dump(per_sample_results, f, indent=4)
            
            if len(success_rate_all) > 0:
                total_success_rate = sum(success_rate_all) / len(success_rate_all)
                ade_fde.update({f"success_rate_total_{name}": total_success_rate})
            else:
                ade_fde.update({f"success_rate_total_{name}": 0})
                
            min_samples_per_mode = min([len(success_rate_by_mode[mode]) for mode in success_rate_by_mode])
            balanced_total_success_rate = 0
            # Calculate success rate for each mode
            for mode in success_rate_by_mode:
                if len(success_rate_by_mode[mode]) > 0:
                    success_rate = sum(success_rate_by_mode[mode]) / len(success_rate_by_mode[mode])
                    ade_fde.update({f"success_rate_{name}_{mode}": success_rate})
                else:
                    ade_fde.update({f"success_rate_{name}_{mode}": 0})
                
            ade_route = np.mean(np.linalg.norm(route_preds_sample - route_gt_sample, axis=-1), axis=-1)
            
            ade_fde.update({
                f"num_samples_{name}": len(ade_route),
            })

        save_path_tmp = f"{str(save_prediction_path)}/dreamer_results_rank_{self.local_rank}.json"
        if os.path.exists(save_path_tmp):
            time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            save_path_tmp = f"{str(save_prediction_path)}/dreamer_results_rank_{self.local_rank}_{time}.json"
        with open(save_path_tmp, "w") as f:
            json.dump(ade_fde, f, indent=4)
        

    def log_training_output(self, training_output: TrainingOutput, mode: str, dataset: Optional[str] = None):
        losses = {k: n.detach() for k, n in training_output.loss_averages.items()}
        counts = {k: n.detach().sum() for k, n in training_output.loss_counts.items()}
        losses["loss"] = training_output.loss.detach()
        counts["loss"] = 1  # loss is already averaged
        for k, v in sorted(losses.items()):
            log_key = f"{mode}_losses/{k}"
            self.log(log_key, v, batch_size=counts[k], sync_dist=True, add_dataloader_idx=False)

    def _resolve_lr_scheduler_total_steps(self) -> int:
        if self.trainer.max_steps == -1:
            max_steps = self.trainer.estimated_stepping_batches
            if int(self.smart_train_stage) == 2:
                # Count every manual GRPO optimizer step in the schedule.
                max_steps = int(max_steps) * int(self.smart_assigner_train_cfg.stage2.grpo_update_epochs)
        else:
            max_steps = self.trainer.max_steps
        max_steps = int(max_steps)
        if max_steps <= 0:
            raise ValueError(f"LR scheduler requires total_steps > 0, got {max_steps}")
        return max_steps

    def lr_scheduler_step(self, scheduler, metric: Optional[Any]) -> None:
        if isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
            target_total_steps = self._resolve_lr_scheduler_total_steps()
            current_total_steps = int(getattr(scheduler, "total_steps", 0))
            if target_total_steps > current_total_steps and int(scheduler.last_epoch) + 1 > current_total_steps:
                scheduler.total_steps = target_total_steps
                scheduler._schedule_phases[-1]["end_step"] = target_total_steps - 1
                print(
                    "[lr_scheduler] extended OneCycleLR total_steps "
                    f"from {current_total_steps} to {target_total_steps}",
                    flush=True,
                )
        if metric is None:
            scheduler.step()
        else:
            scheduler.step(metric)

    def _build_lr_scheduler(self, optimizer, total_steps: int):
        scheduler_name = str(getattr(self, "lr_scheduler", "one_cycle")).strip().lower()
        if scheduler_name in {"one_cycle", "onecycle", "one_cycle_lr"}:
            return torch.optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=self.lr, total_steps=total_steps, pct_start=self.pct_start, verbose=False
            )

        if scheduler_name in {"cosine_with_floor", "cosine_floor", "warmup_cosine_floor"}:
            warmup_ratio = float(getattr(self, "lr_warmup_ratio", 0.0))
            min_ratio = float(getattr(self, "lr_min_ratio", 0.1))
            if warmup_ratio < 0.0 or warmup_ratio >= 1.0:
                raise ValueError(f"lr_warmup_ratio must be in [0, 1), got {warmup_ratio}")
            if min_ratio <= 0.0 or min_ratio > 1.0:
                raise ValueError(f"lr_min_ratio must be in (0, 1], got {min_ratio}")
            warmup_steps = int(round(float(total_steps) * warmup_ratio))
            warmup_steps = max(0, min(warmup_steps, total_steps - 1))

            def lr_lambda(step: int) -> float:
                step_i = max(0, int(step))
                if warmup_steps > 0 and step_i < warmup_steps:
                    return max(min_ratio, float(step_i + 1) / float(warmup_steps))
                decay_steps = max(1, total_steps - warmup_steps)
                decay_progress = float(step_i - warmup_steps) / float(decay_steps)
                decay_progress = min(max(decay_progress, 0.0), 1.0)
                cosine = 0.5 * (1.0 + math.cos(math.pi * decay_progress))
                return min_ratio + (1.0 - min_ratio) * cosine

            return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

        raise ValueError(
            f"Unsupported lr_scheduler={scheduler_name}. "
            "Expected one of: one_cycle, cosine_with_floor"
        )

    def configure_optimizers(self):
        params = [p for p in self.parameters() if p.requires_grad]
        print(f"Optimizer optimization over {len(params)} tensors (filtered from total).")
        
        optimizer = AdamW(
            params,
            lr=self.lr,
            weight_decay=self.weight_decay,
            betas=self.betas,
        )
        max_steps = self._resolve_lr_scheduler_total_steps()
        lr_scheduler = self._build_lr_scheduler(optimizer=optimizer, total_steps=max_steps)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": lr_scheduler, "frequency": 1, "interval": "step"}}
