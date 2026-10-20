from collections import deque
import math
from typing import Any, Dict, List, Optional

import torch.nn.functional as F
import torch
from torch import Tensor, nn

from simlingo_adaption_training.models.adaptors.adaptors import refresh_adaptor_inputs
from simlingo_adaption_training.models.metrics.driving_metrics import DrivingMetricsComputer
from simlingo_adaption_training.models.scheduler.scheduler_utils import (
    SMART_BUDGET_NUM_ACTIONS,
    build_smart_budget_action_table,
    budget_quantizing,
    masked_gumbel_softmax_topk,
    resolve_smart_budget_min,
)
from simlingo_adaption_training.utils.custom_types import DrivingInput


class BaseBudgetAssigner(nn.Module):
    def __init__(
        self,
        mode: str,
        fixed_budget: float = 1.0,
        history_alpha: float = 0.6,
        decision_shift_t_lap: float = 0.2,
    ) -> None:
        """Initialize shared state without selecting a budget policy."""
        super().__init__()
        self.mode = str(mode).strip().lower()
        self.fixed_budget = float(fixed_budget)
        if not (0.0 <= self.fixed_budget <= 1.0):
            raise ValueError(f"fixed_budget must be in [0, 1], got {self.fixed_budget}")
        self.history_alpha = float(history_alpha)
        if not (0.0 <= self.history_alpha <= 1.0):
            raise ValueError(f"history_alpha must be in [0, 1], got {self.history_alpha}")
        self.decision_shift_t_lap = float(decision_shift_t_lap)
        if self.decision_shift_t_lap <= 0.0:
            raise ValueError(f"decision_shift_t_lap must be > 0, got {self.decision_shift_t_lap}")

        # Policy implementations share route-scoped state.
        self.state_by_route: Dict[str, Dict[str, Any]] = {}
        self.prev_decision_state_by_route: Dict[str, Dict[str, Any]] = {}
        self.last_decision_info: List[Dict[str, Any]] = []
        self.last_update_info: List[Dict[str, Any]] = []
        self.last_scene_metrics: Dict[str, Any] = {}

        # BudgetAdaptor and Qwen2 share these per-forward values.
        self.scheduler: Optional[nn.Module] = None
        self._runtime_ref_tensor: Optional[Tensor] = None
        self._runtime_budget_values: Optional[Tensor] = None
        self._runtime_budget_token_position: Optional[torch.LongTensor] = None
        self._runtime_execution_plan: Optional[Tensor] = None

    def reset(self, route_keys: Optional[List[str]] = None) -> None:
        # Reset runtime state without changing module configuration.
        if route_keys is None:
            self.state_by_route = {}
            self.prev_decision_state_by_route = {}
            self.last_decision_info = []
            self.last_update_info = []
            self.last_scene_metrics = {}
            self._runtime_ref_tensor = None
            self._runtime_budget_values = None
            self._runtime_budget_token_position = None
            self._runtime_execution_plan = None
            return

        for route_key in route_keys:
            key = self._normalize_route_key(route_key)
            self.state_by_route.pop(key, None)
            self.prev_decision_state_by_route.pop(key, None)

        self._runtime_ref_tensor = None
        self._runtime_budget_values = None
        self._runtime_budget_token_position = None
        self._runtime_execution_plan = None

    @staticmethod
    def _format_tensor_finite_stats(name: str, tensor: Tensor) -> str:
        tensor_detached = tensor.detach()
        tensor_float = tensor_detached.float()
        finite_mask = torch.isfinite(tensor_float)
        finite_values = tensor_float[finite_mask]
        max_abs = float(finite_values.abs().max().item()) if finite_values.numel() > 0 else float("nan")
        return (
            f"{name}: shape={tuple(tensor_detached.shape)}, dtype={tensor_detached.dtype}, "
            f"nan={int(torch.isnan(tensor_float).sum().item())}, "
            f"inf={int(torch.isinf(tensor_float).sum().item())}, "
            f"max_abs={max_abs:.6f}"
        )

    def bind_runtime_reference(self, ref_tensor: Tensor) -> None:
        self._runtime_ref_tensor = ref_tensor

    def set_runtime_budget_values(self, budget_tensor: Optional[Tensor]) -> None:
        self._runtime_budget_values = None if budget_tensor is None else budget_tensor

    def get_runtime_budget_values(self, batch_size: int) -> Optional[Tensor]:
        if self._runtime_budget_values is None:
            return None
        if self._runtime_ref_tensor is None:
            raise RuntimeError("runtime reference tensor is not bound")
        budget_tensor = self._to_budget_tensor(
            self._runtime_budget_values,
            batch_size,
            ref_tensor=self._runtime_ref_tensor,
        )
        self._runtime_budget_values = budget_tensor
        return budget_tensor

    def encode_budget_token(self, budget_tensor: Tensor) -> Tensor:
        raise NotImplementedError

    def prepare_llm_runtime(self, adaptor_dict: Dict[str, Tensor]) -> None:
        """Prepare per-forward state before Qwen2 runs."""
        raise NotImplementedError

    def build_execution_plan_on_prefix_layer(
        self,
        hidden_states: Tensor,
        layer_idx: int,
        num_prefix_layers: int,
    ) -> Optional[Tensor]:
        """Build the execution plan at the prefix boundary."""
        raise NotImplementedError

    def process_hidden_states_before_layer(
        self,
        hidden_states: Tensor,
        layer_idx: int,
        num_prefix_layers: int,
    ) -> Tensor:
        """Process hidden states before a decoder layer."""
        del layer_idx, num_prefix_layers
        return hidden_states

    def budget_decide_before_llm(
        self,
        batch_size: int,
        route_keys: Optional[List[str]],
        external_budget: Optional[float] = None,
        forced_budget_action_ids: Optional[Tensor] = None,
    ) -> Optional[Tensor]:
        """Choose the budget before the LLM runs."""
        raise NotImplementedError

    def budget_info_update_after_llm(
        self,
        route_keys: Optional[List[str]],
        driving_input: DrivingInput,
        current_speed_wps: Optional[Tensor],
        current_route: Optional[Tensor],
        adaptor_dict: Optional[Dict[str, Tensor]] = None,
        inputs_embeds: Optional[Tensor] = None,
        tokenizer: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Update policy state after the LLM runs."""
        raise NotImplementedError

    def build_mode_inputs(
        self,
        adaptor_dict: Dict[str, Tensor],
        driving_input: DrivingInput,
        route_keys: Optional[List[str]],
        inputs_embeds: Tensor,
        position_ids: Optional[Tensor],
    ) -> None:
        del adaptor_dict, driving_input, route_keys, inputs_embeds, position_ids

    def get_eval_budget(self) -> Dict[str, Any]:
        decision_rows = self.last_decision_info if isinstance(self.last_decision_info, list) else []

        def _extract_from_last_decision(key: str):
            values = []
            for row in decision_rows:
                if isinstance(row, dict):
                    values.append(row.get(key))
            if not values:
                return None
            if len(values) == 1:
                return values[0]
            return values

        return {
            "mode": self.mode,
            "fixed_budget": float(self.fixed_budget),
            "layer_selection_mode": getattr(self, "eval_layer_selection_mode", None),
            "eval_budget_sampling": bool(getattr(self, "eval_budget_sampling", False)),
            "eval_budget_sampling_temperature": float(getattr(self, "eval_budget_sampling_temperature", 1.0)),
            # Flat fields keep logging and leaderboard aggregation stable.
            "phase": _extract_from_last_decision("phase"),
            "used_budget": _extract_from_last_decision("used_budget"),
            "value": _extract_from_last_decision("value"),
            "base_budget": _extract_from_last_decision("base_budget"),
            "budget_action": _extract_from_last_decision("budget_action"),
            "decision_id": _extract_from_last_decision("decision_id"),
            "path_mask_hard": _extract_from_last_decision("path_mask_hard"),
            "active_sub_layer_indices": _extract_from_last_decision("active_sub_layer_indices"),
            "last_decision": self.last_decision_info,
            "last_update": self.last_update_info,
            "last_scene_metrics": self.last_scene_metrics,
        }

    @staticmethod
    def _normalize_route_key(route_key: Optional[str]) -> str:
        if route_key is None or str(route_key).strip() == "":
            raise ValueError("route_key must be a non-empty string")
        return str(route_key)

    @classmethod
    def _resolve_route_keys(cls, route_keys: Optional[List[str]], batch_size: int) -> List[str]:
        if route_keys is None:
            raise ValueError("route_keys is required")
        if len(route_keys) != batch_size:
            raise ValueError(f"route_keys length ({len(route_keys)}) must match batch size ({batch_size})")
        return [cls._normalize_route_key(k) for k in route_keys]

    @staticmethod
    def _to_budget_tensor(
        budget: Any,
        batch_size: int,
        ref_tensor: Tensor,
        out_dtype: Optional[torch.dtype] = None,
    ) -> Tensor:
        target_device = ref_tensor.device
        target_dtype = (
            out_dtype
            if out_dtype is not None
            else (ref_tensor.dtype if torch.is_floating_point(ref_tensor) else torch.float32)
        )
        budget_tensor = torch.as_tensor(budget, device=target_device, dtype=target_dtype)
        return budget_tensor.repeat(batch_size) if budget_tensor.ndim == 0 else budget_tensor.reshape(batch_size)


class SmartBudgetAssigner(BaseBudgetAssigner):
    def __init__(
        self,
        mode: str = "smart_assigner",
        fixed_budget: float = 1.0,
        history_alpha: float = 0.6,
        decision_shift_t_lap: float = 0.2,
    ) -> None:
        """Initialize the two-stage smart budget policy."""
        mode = str(mode).strip().lower()
        if mode != "smart_assigner":
            raise ValueError(f"Unsupported smart assigner mode: {mode}")
        super().__init__(
            mode=mode,
            fixed_budget=fixed_budget,
            history_alpha=history_alpha,
            decision_shift_t_lap=decision_shift_t_lap,
        )
        self.smart_modules_initialized = False
        self.smart_assigner_state_by_route: Dict[str, Dict[str, Any]] = {}
        self.metrics_computer = DrivingMetricsComputer(self)
        self.smart_num_hidden_layers = 0
        self.smart_num_prefix_layers = 0
        self.smart_budget_split_layer = 0
        self.smart_num_attention_heads = 0
        self.smart_num_sub_layers = 0
        self.smart_budget_min = 0.0
        self.smart_budget_num_actions = SMART_BUDGET_NUM_ACTIONS
        self.smart_path_tau = 5.0
        self.smart_path_is_hard = True
        self._runtime_route_keys: List[str] = []
        self._runtime_budget_query_position: Optional[torch.LongTensor] = None
        self._runtime_path_query_position: Optional[torch.LongTensor] = None
        self._runtime_sceneaware_token_positions: Optional[torch.LongTensor] = None
        self._runtime_budget_token_features: Optional[Tensor] = None
        self._runtime_budget_query_feat_std: Optional[Tensor] = None
        self._runtime_budget_decided = False
        self._runtime_path_decided = False
        self._runtime_decision_id = 0
        self._decision_counter = 0
        self.training_stage = 1
        self.stage2_budget_explore_enable = False
        self.stage2_budget_explore_temp_start = 1.0
        self.stage2_budget_explore_temp_end = 1.0
        self.stage2_budget_explore_eps_start = 0.0
        self.stage2_budget_explore_eps_end = 0.0
        self.stage2_budget_explore_anneal_ratio = 0.0
        self._stage2_explore_global_step = 0
        self._stage2_explore_total_steps: Optional[int] = None
        self._runtime_budget_action_logits: Optional[Tensor] = None
        self._runtime_budget_action_hard: Optional[Tensor] = None
        self._runtime_path_logits: Optional[Tensor] = None
        self._runtime_budget_units_hard: Optional[Tensor] = None
        self._runtime_path_mask_hard: Optional[Tensor] = None
        self._runtime_external_budget: Optional[float] = None
        self._runtime_forced_budget_action_ids: Optional[Tensor] = None
        self._runtime_next_g_hist: Optional[Tensor] = None
        self._runtime_next_g_hist_route_keys: List[str] = []
        self.eval_layer_selection_mode = "stage1"
        self.eval_fixed_random_layer_selection = False
        self.eval_random_layer_selection_seed = 0
        self.eval_budget_sampling = False
        self.eval_budget_sampling_temperature = 1.0

    def init_mode_modules(
        self,
        smart_cfg: Optional[Any],
        hidden_size: int,
        language_model_config: Optional[Any],
    ) -> None:
        cfg = smart_cfg
        if cfg is None:
            raise ValueError("smart_assigner config is required")
        if language_model_config is None:
            raise ValueError("language_model_config is required for smart_assigner")

        self.smart_cfg = cfg
        self.smart_feature_dim = int(cfg.feature_dim)
        if self.smart_feature_dim != int(hidden_size):
            raise ValueError(
                f"smart_assigner.feature_dim={self.smart_feature_dim} must equal language hidden_size={hidden_size}"
            )

        self.smart_num_hidden_layers = int(language_model_config.num_hidden_layers)
        self.smart_num_prefix_layers = int(language_model_config.num_prefix_layers)
        self.smart_num_attention_heads = int(language_model_config.num_attention_heads)
        if not (2 <= self.smart_num_prefix_layers < self.smart_num_hidden_layers):
            raise ValueError(
                f"smart_assigner requires 2 <= num_prefix_layers < num_hidden_layers, got "
                f"{self.smart_num_prefix_layers} and {self.smart_num_hidden_layers}"
            )
        if int(self.smart_num_prefix_layers) % 2 != 0:
            raise ValueError(
                f"smart_assigner requires even num_prefix_layers for half-split forward, got "
                f"{self.smart_num_prefix_layers}"
            )
        self.smart_budget_split_layer = self._resolve_budget_split_layer(self.smart_num_prefix_layers)
        self.smart_num_sub_layers = self.smart_num_hidden_layers - self.smart_num_prefix_layers
        self.smart_budget_min = resolve_smart_budget_min(
            self.smart_num_prefix_layers,
            self.smart_num_hidden_layers,
        )

        route_cfg = cfg.route_conv
        speed_cfg = cfg.speed_conv

        self.smart_alpha = float(cfg.alpha)
        self.smart_t_lap = float(cfg.t_lap)
        # Match the time alignment used by the metrics computer.
        self.decision_shift_t_lap = self.smart_t_lap
        self.smart_path_tau = float(cfg.path_tau)
        self.smart_path_is_hard = bool(cfg.path_is_hard)

        self.smart_scene_cross_attn = nn.MultiheadAttention(
            embed_dim=self.smart_feature_dim,
            num_heads=int(cfg.scene_cross_attn_num_heads),
            batch_first=True,
        )
        self.smart_scene_dev_mlp = nn.Sequential(
            nn.Linear(self.smart_feature_dim * 3, int(cfg.scene_deviation_mlp_hidden)),
            nn.SiLU(True),
            nn.Linear(int(cfg.scene_deviation_mlp_hidden), self.smart_feature_dim),
        )

        self.smart_route_conv = nn.Conv1d(
            in_channels=int(route_cfg.in_channels),
            out_channels=int(route_cfg.out_channels),
            kernel_size=int(route_cfg.kernel_size),
            stride=int(route_cfg.stride),
            padding=int(route_cfg.padding),
        )
        self.smart_route_mlp = nn.Sequential(
            nn.Linear(int(route_cfg.out_channels), int(cfg.route_mlp_hidden)),
            nn.SiLU(True),
            nn.Linear(int(cfg.route_mlp_hidden), self.smart_feature_dim),
        )

        self.smart_speed_conv = nn.Conv1d(
            in_channels=int(speed_cfg.in_channels),
            out_channels=int(speed_cfg.out_channels),
            kernel_size=int(speed_cfg.kernel_size),
            stride=int(speed_cfg.stride),
            padding=int(speed_cfg.padding),
        )
        self.smart_speed_mlp = nn.Sequential(
            nn.Linear(int(speed_cfg.out_channels), int(cfg.speed_mlp_hidden)),
            nn.SiLU(True),
            nn.Linear(int(cfg.speed_mlp_hidden), self.smart_feature_dim),
        )

        self.smart_budget_scheduler = nn.Sequential(
            nn.LayerNorm(self.smart_feature_dim * 4),
            nn.Linear(self.smart_feature_dim * 4, self.smart_feature_dim),
            nn.SiLU(True),
            nn.LayerNorm(self.smart_feature_dim),
            nn.Linear(self.smart_feature_dim, int(cfg.budget_scheduler_hidden)),
            nn.SiLU(True),
            nn.Linear(int(cfg.budget_scheduler_hidden), int(self.smart_budget_num_actions)),
        )
        nn.init.xavier_uniform_(self.smart_budget_scheduler[-1].weight)
        nn.init.zeros_(self.smart_budget_scheduler[-1].bias)

        self.smart_budget_encoder = nn.Sequential(
            nn.LayerNorm(256),
            nn.Linear(256, self.smart_feature_dim),
            nn.SiLU(True),
            nn.Linear(self.smart_feature_dim, self.smart_feature_dim),
        )
        self.smart_path_scheduler = nn.Sequential(
            nn.LayerNorm(self.smart_feature_dim * 2),
            nn.Linear(self.smart_feature_dim * 2, int(cfg.path_scheduler_hidden)),
            nn.SiLU(True),
            nn.Linear(int(cfg.path_scheduler_hidden), self.smart_num_sub_layers),
        )

        self.smart_assigner_state_by_route = {}
        self._runtime_budget_query_position = None
        self._runtime_path_query_position = None
        self._runtime_sceneaware_token_positions = None
        self._runtime_budget_token_features = None
        self._runtime_budget_query_feat_std = None
        self._runtime_budget_decided = False
        self._runtime_path_decided = False
        self._runtime_budget_action_logits = None
        self._runtime_budget_action_hard = None
        self._runtime_path_logits = None
        self._runtime_budget_units_hard = None
        self._runtime_path_mask_hard = None
        self.smart_modules_initialized = True

    @staticmethod
    def _resolve_budget_split_layer(num_prefix_layers: int) -> int:
        # The budget decision occurs after the first half of the prefix.
        num_prefix = int(num_prefix_layers)
        if num_prefix < 2:
            raise ValueError(f"num_prefix_layers must be >= 2 for half-split forward, got {num_prefix}")
        if num_prefix % 2 != 0:
            raise ValueError(f"num_prefix_layers must be even for half-split forward, got {num_prefix}")
        return num_prefix // 2

    def set_training_stage(self, stage: int) -> None:
        stage = int(stage)
        if stage not in (1, 2):
            raise ValueError(f"smart training stage must be 1 or 2, got {stage}")
        self.training_stage = stage

    def set_eval_layer_selection_mode(self, layer_selection_mode: str) -> None:
        layer_selection_mode = str(layer_selection_mode).strip().lower()
        allowed_modes = {"stage1", "random"}
        if layer_selection_mode not in allowed_modes:
            raise ValueError(
                f"Unsupported eval layer_selection_mode={layer_selection_mode}, "
                f"expected one of {sorted(allowed_modes)}"
            )
        self.eval_layer_selection_mode = layer_selection_mode

    def set_eval_fixed_random_layer_selection(self, enabled: bool, seed: int) -> None:
        if not isinstance(enabled, bool):
            raise ValueError(
                f"eval fixed_random_layer_selection must be bool, got {type(enabled)}"
            )
        self.eval_fixed_random_layer_selection = enabled
        self.eval_random_layer_selection_seed = int(seed)

    def set_eval_budget_sampling(self, enabled: bool, temperature: float) -> None:
        # Eval-only temperature sampling for the stage2 budget action, LLM-style.
        temperature = float(temperature)
        if enabled and temperature <= 0.0:
            raise ValueError(
                "eval budget sampling temperature must be > 0 when sampling is enabled"
            )
        self.eval_budget_sampling = bool(enabled)
        self.eval_budget_sampling_temperature = temperature

    def configure_stage2_budget_exploration(
        self,
        enabled: bool,
        temp_start: float,
        temp_end: float,
        eps_start: float,
        eps_end: float,
        anneal_ratio: float,
    ) -> None:
        if not isinstance(enabled, bool):
            raise ValueError(f"stage2 budget exploration enabled must be bool, got {type(enabled)}")
        temp_start_f = float(temp_start)
        temp_end_f = float(temp_end)
        eps_start_f = float(eps_start)
        eps_end_f = float(eps_end)
        anneal_ratio_f = float(anneal_ratio)
        if temp_start_f <= 0.0 or temp_end_f <= 0.0:
            raise ValueError(
                f"stage2 budget exploration temperatures must be > 0, got {temp_start_f}, {temp_end_f}"
            )
        if temp_start_f < temp_end_f:
            raise ValueError(
                f"stage2 budget exploration temp_start must be >= temp_end, got {temp_start_f} < {temp_end_f}"
            )
        if eps_start_f < 0.0 or eps_start_f > 1.0:
            raise ValueError(f"stage2 budget exploration eps_start must be in [0, 1], got {eps_start_f}")
        if eps_end_f < 0.0 or eps_end_f > 1.0:
            raise ValueError(f"stage2 budget exploration eps_end must be in [0, 1], got {eps_end_f}")
        if eps_start_f < eps_end_f:
            raise ValueError(
                f"stage2 budget exploration eps_start must be >= eps_end, got {eps_start_f} < {eps_end_f}"
            )
        if anneal_ratio_f < 0.0 or anneal_ratio_f > 1.0:
            raise ValueError(
                f"stage2 budget exploration anneal_ratio must be in [0, 1], got {anneal_ratio_f}"
            )

        self.stage2_budget_explore_enable = enabled
        self.stage2_budget_explore_temp_start = temp_start_f
        self.stage2_budget_explore_temp_end = temp_end_f
        self.stage2_budget_explore_eps_start = eps_start_f
        self.stage2_budget_explore_eps_end = eps_end_f
        self.stage2_budget_explore_anneal_ratio = anneal_ratio_f

    def set_stage2_budget_exploration_progress(
        self,
        global_step: int,
        estimated_total_steps: Optional[int],
    ) -> None:
        global_step_i = int(global_step)
        if global_step_i < 0:
            global_step_i = 0
        self._stage2_explore_global_step = global_step_i

        if estimated_total_steps is None:
            self._stage2_explore_total_steps = None
            return
        total_steps_i = int(estimated_total_steps)
        if total_steps_i <= 0:
            self._stage2_explore_total_steps = None
            return
        self._stage2_explore_total_steps = total_steps_i

    def _resolve_stage2_budget_exploration(self) -> Dict[str, float]:
        enabled = bool(self.stage2_budget_explore_enable) and self.training and int(self.training_stage) == 2
        if not enabled:
            return {
                "enabled": 0.0,
                "temperature": 1.0,
                "epsilon": 0.0,
            }

        total_steps = self._stage2_explore_total_steps
        if total_steps is None:
            progress = 0.0
        else:
            anneal_steps = max(1, int(math.ceil(float(total_steps) * float(self.stage2_budget_explore_anneal_ratio))))
            progress = float(self._stage2_explore_global_step) / float(anneal_steps)
            progress = min(max(progress, 0.0), 1.0)

        temperature = (
            float(self.stage2_budget_explore_temp_start)
            + progress * (float(self.stage2_budget_explore_temp_end) - float(self.stage2_budget_explore_temp_start))
        )
        epsilon = (
            float(self.stage2_budget_explore_eps_start)
            + progress * (float(self.stage2_budget_explore_eps_end) - float(self.stage2_budget_explore_eps_start))
        )
        return {
            "enabled": 1.0,
            "temperature": float(temperature),
            "epsilon": float(epsilon),
        }

    def get_training_signals(self) -> Dict[str, Tensor]:
        if not self._runtime_budget_decided:
            raise RuntimeError("runtime budget is not decided when requesting smart training signals")
        if not self._runtime_path_decided:
            raise RuntimeError("runtime path is not decided when requesting smart training signals")
        if self._runtime_budget_token_features is None:
            raise RuntimeError("runtime budget token features are missing")
        if self._runtime_path_logits is None:
            raise RuntimeError("runtime path logits are missing")
        if self._runtime_budget_units_hard is None:
            raise RuntimeError("runtime hard budget units are missing")
        if self._runtime_path_mask_hard is None:
            raise RuntimeError("runtime hard path mask is missing")
        batch_size = int(self._runtime_path_logits.size(0))
        budget_values = self.get_runtime_budget_values(batch_size=batch_size)
        if budget_values is None:
            raise RuntimeError("runtime budget values are missing")
        out: Dict[str, Tensor] = {
            "budget_values": budget_values,
            "budget_token_features": self._runtime_budget_token_features,
            "budget_units_hard": self._runtime_budget_units_hard,
            "path_logits": self._runtime_path_logits,
            "path_mask_hard": self._runtime_path_mask_hard,
            "num_prefix_layers": torch.tensor(self.smart_num_prefix_layers, device=budget_values.device, dtype=torch.long),
            "num_hidden_layers": torch.tensor(self.smart_num_hidden_layers, device=budget_values.device, dtype=torch.long),
            "budget_min": torch.tensor(float(self.smart_budget_min), device=budget_values.device, dtype=torch.float32),
        }
        if self._runtime_budget_action_logits is not None:
            out["budget_action_logits"] = self._runtime_budget_action_logits
        if self._runtime_budget_action_hard is not None:
            out["budget_action_hard"] = self._runtime_budget_action_hard
        if self._runtime_budget_query_feat_std is not None:
            out["budget_query_feat_std"] = self._runtime_budget_query_feat_std
        return out

    def budget_decide_before_llm(
        self,
        batch_size: int,
        route_keys: Optional[List[str]],
        external_budget: Optional[float] = None,
        forced_budget_action_ids: Optional[Tensor] = None,
    ) -> Optional[Tensor]:
        # The smart budget is chosen later at the L/2 boundary.
        self._runtime_route_keys = self._resolve_route_keys(route_keys, batch_size)
        if external_budget is not None and forced_budget_action_ids is not None:
            raise ValueError("external_budget and forced_budget_action_ids are mutually exclusive")

        if forced_budget_action_ids is not None:
            if not torch.is_tensor(forced_budget_action_ids):
                raise ValueError("forced_budget_action_ids must be a Tensor when provided")
            if forced_budget_action_ids.ndim != 1:
                raise ValueError(
                    "forced_budget_action_ids must be rank-1, "
                    f"got shape={tuple(forced_budget_action_ids.shape)}"
                )
            if int(forced_budget_action_ids.size(0)) != int(batch_size):
                raise ValueError(
                    "forced_budget_action_ids batch mismatch: "
                    f"{int(forced_budget_action_ids.size(0))} vs {int(batch_size)}"
                )
            forced_ids = forced_budget_action_ids.to(dtype=torch.long)
            if torch.any(forced_ids < 0) or torch.any(forced_ids >= int(self.smart_budget_num_actions)):
                raise ValueError(
                    "forced_budget_action_ids out of range: "
                    f"valid=[0, {int(self.smart_budget_num_actions) - 1}]"
                )
            self._runtime_forced_budget_action_ids = forced_ids.detach().clone()
        else:
            self._runtime_forced_budget_action_ids = None

        if external_budget is None:
            self._runtime_external_budget = None
        else:
            try:
                external_budget_value = float(external_budget)
            except Exception as exc:
                raise ValueError(
                    f"external_budget must be float in [{self.smart_budget_min:.6f}, 1.0], got {external_budget}"
                ) from exc
            if external_budget_value < self.smart_budget_min or external_budget_value > 1.0:
                raise ValueError(
                    f"external_budget must be in [{self.smart_budget_min:.6f}, 1.0], got {external_budget_value}"
                )
            self._runtime_external_budget = external_budget_value
        self.set_runtime_budget_values(None)
        self._runtime_budget_decided = False
        self._runtime_path_decided = False
        self._runtime_budget_token_features = None
        self._runtime_execution_plan = None
        self._runtime_decision_id = 0
        self._runtime_budget_action_logits = None
        self._runtime_budget_action_hard = None
        self._runtime_path_logits = None
        self._runtime_budget_units_hard = None
        self._runtime_path_mask_hard = None
        self.last_decision_info = [
            {
                "route_key": route_key,
                "mode": self.mode,
                "phase": "smart_pending",
                "used_budget": None,
                "value": None,
                "decision_id": None,
            }
            for route_key in self._runtime_route_keys
        ]
        return None

    def encode_budget_token(self, budget_tensor: Tensor) -> Tensor:
        if not self.smart_modules_initialized:
            raise ValueError("smart_assigner modules are not initialized")
        budget_tensor = budget_tensor.float().view(-1)
        if torch.any(budget_tensor < self.smart_budget_min) or torch.any(budget_tensor > 1.0):
            raise ValueError(
                f"smart budget must be in [{self.smart_budget_min:.6f}, 1.0], got "
                f"min={float(budget_tensor.min().item()):.6f}, max={float(budget_tensor.max().item()):.6f}"
            )
        # Preserve the legacy 256-D quantized sinusoidal encoding.
        quantized_budget = budget_quantizing(
            budget_tensor,
            self.smart_num_prefix_layers,
            self.smart_num_hidden_layers,
        )[1]
        scaled_values = quantized_budget * (2.0 * torch.pi)  # shape: (B,)
        frequencies = 1.0 / (
            10000
            ** (torch.arange(128, device=quantized_budget.device, dtype=quantized_budget.dtype) / 128)
        )
        sin_values = torch.sin(scaled_values[:, None] * frequencies[None, :])  # shape: (B, 128)
        cos_values = torch.cos(scaled_values[:, None] * frequencies[None, :])  # shape: (B, 128)
        budget_emb = torch.cat((sin_values, cos_values), dim=-1)  # shape: (B, 256)
        target_dtype = self.smart_budget_encoder[1].weight.dtype
        return self.smart_budget_encoder(budget_emb.to(dtype=target_dtype))  # shape: (B, C)

    def prepare_llm_runtime(self, adaptor_dict: Dict[str, Tensor]) -> None:
        budget_pos, path_pos = self._resolve_assignquery_token_positions(adaptor_dict)
        self._runtime_budget_query_position = budget_pos
        self._runtime_path_query_position = path_pos
        self._runtime_sceneaware_token_positions = self._resolve_sceneaware_token_positions(adaptor_dict)
        self._runtime_budget_token_position = budget_pos
        # Forced actions remain valid until this forward makes its decision.
        self._runtime_execution_plan = None
        self._runtime_budget_decided = False
        self._runtime_path_decided = False
        self._runtime_budget_token_features = None
        self._runtime_budget_query_feat_std = None
        self._runtime_budget_action_logits = None
        self._runtime_budget_action_hard = None
        self._runtime_path_logits = None
        self._runtime_budget_units_hard = None
        self._runtime_path_mask_hard = None

    def build_execution_plan_on_prefix_layer(
        self,
        hidden_states: Tensor,
        layer_idx: int,
        num_prefix_layers: int,
    ) -> Optional[Tensor]:
        if int(num_prefix_layers) != self.smart_num_prefix_layers:
            raise ValueError(
                f"num_prefix_layers mismatch in smart_assigner runtime: "
                f"{num_prefix_layers} vs {self.smart_num_prefix_layers}"
            )
        if layer_idx != int(num_prefix_layers):
            return self._runtime_execution_plan
        if self._runtime_execution_plan is not None:
            return self._runtime_execution_plan
        if not self._runtime_budget_decided:
            raise RuntimeError("smart budget must be decided before path scheduling")
        if self._runtime_path_query_position is None:
            raise RuntimeError("path query position is not prepared for smart_assigner")
        if self._runtime_budget_token_features is None:
            raise RuntimeError("budget token feature is missing for smart_assigner")

        batch_size = int(hidden_states.size(0))
        path_pos = self._expand_token_positions(
            self._runtime_path_query_position,
            batch_size,
            hidden_states.device,
        )
        batch_indices = torch.arange(batch_size, device=hidden_states.device)
        path_query_feat = hidden_states[batch_indices, path_pos]  # shape: (B, C)
        budget_feat = self._runtime_budget_token_features.to(
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )  # shape: (B, C)
        if budget_feat.size(0) != batch_size:
            raise ValueError(
                f"budget feature batch mismatch: {budget_feat.size(0)} vs {batch_size}"
            )
        if path_query_feat.shape != budget_feat.shape:
            raise ValueError(
                f"path query and budget feature shape mismatch: {path_query_feat.shape} vs {budget_feat.shape}"
            )

        # The path scheduler conditions on both path query and budget embedding.
        scheduler_in = torch.cat([path_query_feat, budget_feat], dim=-1)  # shape: (B, 2C)
        sched_dtype = self.smart_path_scheduler[1].weight.dtype
        path_logits = self.smart_path_scheduler(scheduler_in.to(dtype=sched_dtype))  # shape: (B, N_sub)

        budget_values = self.get_runtime_budget_values(batch_size)
        if budget_values is None:
            raise RuntimeError("runtime budget is missing when building smart execution plan")
        budget_values = budget_values.to(device=hidden_states.device, dtype=torch.float32)
        budget_units_hard = budget_quantizing(
            budget_values,
            self.smart_num_prefix_layers,
            self.smart_num_hidden_layers,
        )[0]

        sampled_sub_plan: List[Tensor] = []
        if self.eval_layer_selection_mode == "random":
            self._runtime_path_logits = None
            num_sub_layers = int(path_logits.size(1))
            for units_i in budget_units_hard:
                k = int(units_i.item())
                sampled = torch.zeros(
                    (num_sub_layers,),
                    device=hidden_states.device,
                    dtype=hidden_states.dtype,
                )
                if k > 0:
                    if k >= num_sub_layers:
                        sampled.fill_(1.0)
                    else:
                        if self.eval_fixed_random_layer_selection:
                            randperm_gen = torch.Generator(device="cpu")
                            randperm_gen.manual_seed(int(self.eval_random_layer_selection_seed))
                            selected = torch.randperm(
                                num_sub_layers,
                                generator=randperm_gen,
                            )[:k].to(device=hidden_states.device)
                        else:
                            selected = torch.randperm(num_sub_layers, device=hidden_states.device)[:k]
                        sampled[selected] = 1.0
                sampled_sub_plan.append(sampled)
        else:
            self._runtime_path_logits = path_logits
        # Stage 2 uses deterministic top-k after freezing the stage 1 path policy.
            sample_path_with_gumbel = self.training and int(self.training_stage) == 1
            for logits_i, units_i in zip(path_logits, budget_units_hard):
                sampled = masked_gumbel_softmax_topk(
                    logits=logits_i,
                    masks=torch.zeros_like(logits_i),
                    k=int(units_i.item()),
                    tau=self.smart_path_tau,
                    hard=self.smart_path_is_hard,
                    training=sample_path_with_gumbel,
                )
                sampled_sub_plan.append(sampled)
        sampled_sub_plan = torch.stack(sampled_sub_plan, dim=0).to(dtype=hidden_states.dtype)  # shape: (B, N_sub)

        hard_counts = sampled_sub_plan.sum(dim=1).round().to(dtype=torch.long)
        hard_units_int = budget_units_hard.round().to(dtype=torch.long)
        if not torch.equal(hard_counts, hard_units_int):
            raise ValueError(
                "hard path mask count mismatches hard budget units: "
                f"counts={hard_counts.tolist()}, units={hard_units_int.tolist()}"
            )

        self._runtime_budget_units_hard = budget_units_hard.to(dtype=torch.float32)
        self._runtime_path_mask_hard = sampled_sub_plan
        sampled_sub_plan_for_llm = sampled_sub_plan[:, :, None, None].repeat(
            1, 1, 2, self.smart_num_attention_heads
        )  # shape: (B, N_sub, 2, H)

        prefix_plan = torch.ones(
            (batch_size, self.smart_num_prefix_layers, 2, self.smart_num_attention_heads),
            device=hidden_states.device,
            dtype=sampled_sub_plan_for_llm.dtype,
        )  # shape: (B, L, 2, H)
        full_plan = torch.cat([prefix_plan, sampled_sub_plan_for_llm], dim=1)  # shape: (B, N, 2, H)
        if full_plan.size(1) != self.smart_num_hidden_layers:
            raise ValueError(
                f"smart execution plan layer mismatch: expected {self.smart_num_hidden_layers}, got {full_plan.size(1)}"
            )
        self._runtime_execution_plan = full_plan.transpose(0, 1)  # shape: (N, B, 2, H)
        self._runtime_path_decided = True

        if len(self._runtime_route_keys) == batch_size:
            for idx, route_key in enumerate(self._runtime_route_keys):
                prev_row = self.last_decision_info[idx] if idx < len(self.last_decision_info) else {}
                if not isinstance(prev_row, dict):
                    prev_row = {}
                path_mask_hard_row = [
                    int(round(float(value)))
                    for value in sampled_sub_plan[idx].detach().cpu().tolist()
                ]
                active_sub_layer_indices = [
                    layer_idx
                    for layer_idx, enabled in enumerate(path_mask_hard_row)
                    if int(enabled) != 0
                ]
                self.last_decision_info[idx] = {
                    "route_key": route_key,
                    "mode": self.mode,
                    "phase": prev_row.get("phase", "smart_assigner"),
                    "used_budget": float(budget_values[idx].item()),
                    "value": float(budget_values[idx].item()),
                    "budget_units": int(budget_units_hard[idx].item()),
                    "active_sub_layers": int(sampled_sub_plan[idx].sum().item()),
                    "path_mask_hard": path_mask_hard_row,
                    "active_sub_layer_indices": active_sub_layer_indices,
                    "budget_action": prev_row.get("budget_action"),
                    "decision_id": prev_row.get("decision_id"),
                    "layer_selection_mode": self.eval_layer_selection_mode,
                }
        return self._runtime_execution_plan

    def _resolve_budget_action_values(self, device: torch.device, dtype: torch.dtype) -> Tensor:
        return build_smart_budget_action_table(
            budget_min=self.smart_budget_min,
            device=device,
            dtype=dtype,
            num_actions=self.smart_budget_num_actions,
        )

    def _decode_discrete_budget(self, action_logits: Tensor) -> Dict[str, Any]:
        if action_logits.ndim != 2:
            raise ValueError(f"budget action logits must be rank-2, got shape={tuple(action_logits.shape)}")
        if action_logits.size(1) != int(self.smart_budget_num_actions):
            raise ValueError(
                f"budget action logits width mismatch: expected {self.smart_budget_num_actions}, "
                f"got {action_logits.size(1)}"
            )
        action_probs = F.softmax(action_logits.float(), dim=-1)
        explore_cfg = self._resolve_stage2_budget_exploration()
        use_exploration = bool(explore_cfg["enabled"] > 0.0)
        use_eval_sampling = bool(self.eval_budget_sampling) and not self.training
        if use_exploration:
            temperature = max(float(explore_cfg["temperature"]), 1e-6)
            epsilon = min(max(float(explore_cfg["epsilon"]), 0.0), 1.0)
            sample_probs = F.softmax(action_logits.float() / temperature, dim=-1)
            sampled_action_ids = torch.distributions.Categorical(probs=sample_probs).sample()
            if epsilon > 0.0:
                random_mask = torch.rand_like(sampled_action_ids.float()) < epsilon
                random_action_ids = torch.randint(
                    low=0,
                    high=int(self.smart_budget_num_actions),
                    size=sampled_action_ids.shape,
                    device=sampled_action_ids.device,
                    dtype=sampled_action_ids.dtype,
                )
                sampled_action_ids = torch.where(random_mask, random_action_ids, sampled_action_ids)
            action_probs_for_st = sample_probs
        elif use_eval_sampling:
        # Evaluation sampling affects only the budget action.
            temperature = float(self.eval_budget_sampling_temperature)
            epsilon = 0.0
            sample_probs = F.softmax(action_logits.float() / temperature, dim=-1)
            sampled_action_ids = torch.distributions.Categorical(probs=sample_probs).sample()
            action_probs_for_st = sample_probs
        else:
            temperature = 1.0
            epsilon = 0.0
            sampled_action_ids = action_probs.argmax(dim=-1)
            action_probs_for_st = action_probs

        action_hard = F.one_hot(
            sampled_action_ids,
            num_classes=int(self.smart_budget_num_actions),
        ).to(dtype=action_probs.dtype)
        # Use discrete actions forward and softmax gradients backward.
        action_for_value = action_hard - action_probs_for_st.detach() + action_probs_for_st
        action_values = self._resolve_budget_action_values(
            device=action_probs.device,
            dtype=action_probs.dtype,
        )
        budget_values = (action_for_value * action_values.unsqueeze(0)).sum(dim=-1)
        return {
            "budget_values": budget_values,
            "action_logits": action_logits,
            "action_hard": action_hard,
            "action_ids": sampled_action_ids,
            "used_exploration": use_exploration,
            "used_eval_sampling": use_eval_sampling,
            "explore_temperature": action_probs.new_tensor(float(temperature)),
            "explore_epsilon": action_probs.new_tensor(float(epsilon)),
        }

    def process_hidden_states_before_layer(
        self,
        hidden_states: Tensor,
        layer_idx: int,
        num_prefix_layers: int,
    ) -> Tensor:
        if int(num_prefix_layers) != self.smart_num_prefix_layers:
            raise ValueError(
                f"num_prefix_layers mismatch in smart_assigner runtime: "
                f"{num_prefix_layers} vs {self.smart_num_prefix_layers}"
            )
        if not self.smart_modules_initialized:
            raise ValueError("smart_assigner modules are not initialized")
        target_idx = int(self.smart_budget_split_layer)
        if layer_idx != target_idx:
            return hidden_states
        if self._runtime_budget_decided:
            return hidden_states
        if self._runtime_budget_query_position is None:
            raise RuntimeError("budget query position is not prepared for smart_assigner")

        batch_size = int(hidden_states.size(0))
        budget_pos = self._expand_token_positions(
            self._runtime_budget_query_position,
            batch_size,
            hidden_states.device,
        )
        batch_indices = torch.arange(batch_size, device=hidden_states.device)
        budget_action_logits: Optional[Tensor]
        budget_action_hard: Optional[Tensor]
        budget_action_ids: Optional[Tensor]
        if self._runtime_external_budget is not None:
            budget_action_logits = None
            budget_action_hard = None
            budget_action_ids = None
            budget_values = torch.full(
                (batch_size,),
                float(self._runtime_external_budget),
                device=hidden_states.device,
                dtype=torch.float32,
            )
            phase = "smart_budget_eval_fixed"
        elif int(self.training_stage) == 1:
            budget_action_logits = None
            budget_action_hard = None
            budget_action_ids = None
            budget_values = self.smart_budget_min + (1.0 - self.smart_budget_min) * torch.rand(
                (batch_size,),
                device=hidden_states.device,
                dtype=torch.float32,
            )
            phase = "smart_budget_stage1_random"
        elif int(self.training_stage) == 2:
            budget_query_feat = hidden_states[batch_indices, budget_pos]  # shape: (B, C)
            self._runtime_budget_query_feat_std = budget_query_feat.float().std(
                dim=0,
                unbiased=False,
            ).mean()
            if self._runtime_sceneaware_token_positions is None:
                raise RuntimeError("sceneaware token positions are not prepared for smart budget policy")
            sceneaware_pos = self._runtime_sceneaware_token_positions.to(
                device=hidden_states.device,
                dtype=torch.long,
            )
            if sceneaware_pos.ndim != 2 or sceneaware_pos.size(0) != batch_size or sceneaware_pos.size(1) != 3:
                raise ValueError(
                    "sceneaware token positions must have shape [B,3], "
                    f"got shape={tuple(sceneaware_pos.shape)} for batch_size={batch_size}"
                )
            sceneaware_feat = hidden_states[
                batch_indices.unsqueeze(1),
                sceneaware_pos,
            ].reshape(batch_size, self.smart_feature_dim * 3)
            sched_dtype = self.smart_budget_scheduler[1].weight.dtype
        # Explicit scene features avoid bottlenecking state in one budget query.
            budget_query_feat_for_policy = torch.cat([budget_query_feat, sceneaware_feat], dim=-1)
            budget_action_logits = self.smart_budget_scheduler(
                budget_query_feat_for_policy.to(dtype=sched_dtype)
            )  # shape: (B, K)
            if not torch.isfinite(budget_action_logits).all():
                print(
                    "[finite-check] non-finite budget_action_logits after scheduler "
                    f"decision_id={int(self._runtime_decision_id)} "
                    f"route_keys={self._runtime_route_keys[:3]} "
                    f"scheduler_dtype={sched_dtype} "
                    f"{self._format_tensor_finite_stats('budget_query_feat', budget_query_feat)} "
                    f"{self._format_tensor_finite_stats('sceneaware_feat', sceneaware_feat)} "
                    f"{self._format_tensor_finite_stats('budget_action_logits', budget_action_logits)}",
                    flush=True,
                )
                raise FloatingPointError("non-finite budget_action_logits after smart_budget_scheduler")
            if self._runtime_forced_budget_action_ids is not None:
                forced_ids = self._runtime_forced_budget_action_ids.to(
                    device=hidden_states.device,
                    dtype=torch.long,
                )
                if int(forced_ids.size(0)) != batch_size:
                    raise ValueError(
                        "runtime forced action batch mismatch: "
                        f"{int(forced_ids.size(0))} vs {batch_size}"
                    )
                budget_action_hard = F.one_hot(
                    forced_ids,
                    num_classes=int(self.smart_budget_num_actions),
                ).to(dtype=budget_action_logits.dtype)
                action_values = self._resolve_budget_action_values(
                    device=budget_action_logits.device,
                    dtype=budget_action_logits.dtype,
                )
                budget_values = (budget_action_hard * action_values.unsqueeze(0)).sum(dim=-1)
                budget_action_ids = forced_ids
                phase = "smart_budget_stage2_forced"
            else:
                decoded = self._decode_discrete_budget(budget_action_logits)
                budget_values = decoded["budget_values"]
                budget_action_hard = decoded["action_hard"]
                budget_action_ids = decoded["action_ids"]
                if bool(decoded["used_exploration"]):
                    phase = "smart_budget_stage2_explore"
                elif bool(decoded["used_eval_sampling"]):
                    phase = "smart_budget_stage2_eval_sample"
                else:
                    phase = "smart_budget_stage2_discrete"
        else:
            raise ValueError(f"Unsupported smart training stage: {self.training_stage}")
        if torch.any(budget_values < self.smart_budget_min) or torch.any(budget_values > 1.0):
            raise ValueError(
                f"decoded smart budget is out of range [{self.smart_budget_min:.6f}, 1.0], "
                f"min={float(budget_values.min().item()):.6f}, max={float(budget_values.max().item()):.6f}"
            )
        block_driving_to_budget_branch = int(self.training_stage) == 2
        if block_driving_to_budget_branch:
            # Stage2 direct driving loss uses a detached discrete budget value; budget policy logits are trained by GRPO.
            budget_values_for_driving = budget_values.detach()
        else:
            budget_values_for_driving = budget_values
        self.set_runtime_budget_values(budget_values_for_driving)
        self._runtime_budget_action_logits = budget_action_logits
        self._runtime_budget_action_hard = budget_action_hard
        budget_token_feat = self.encode_budget_token(budget_values_for_driving).to(
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )  # shape: (B, C)
        self._runtime_budget_token_features = budget_token_feat
        self._runtime_budget_decided = True
        self._decision_counter += 1
        self._runtime_decision_id = int(self._decision_counter)

        if len(self._runtime_route_keys) == batch_size:
            for idx, route_key in enumerate(self._runtime_route_keys):
                self.last_decision_info[idx] = {
                    "route_key": route_key,
                    "mode": self.mode,
                    "phase": phase,
                    "used_budget": float(budget_values[idx].item()),
                    "value": float(budget_values[idx].item()),
                    "budget_action": None if budget_action_ids is None else int(budget_action_ids[idx].item()),
                    "decision_id": int(self._runtime_decision_id),
                    "layer_selection_mode": self.eval_layer_selection_mode,
                }

        # `index_put` + bf16 autocast can trigger
        # RuntimeError: Unexpected floating ScalarType in at::autocast::prioritize.
        # Do the writeback in fp32 outside autocast, then restore the original dtype.
        with torch.autocast(device_type=hidden_states.device.type, enabled=False):
            updated_hidden_states = hidden_states.float().clone()
            updated_hidden_states[batch_indices, budget_pos] = budget_token_feat.float()
        return updated_hidden_states.to(dtype=hidden_states.dtype)

    def build_smart_mode_inputs(
        self,
        adaptor_dict: Dict[str, Tensor],
        driving_input: DrivingInput,
        route_keys: Optional[List[str]],
        img_context_token_id: int,
        route_token_count: int,
        speed_token_count: int,
    ) -> None:
        del driving_input
        # Input construction creates the scene candidate; frame commit consumes it.
        self._runtime_next_g_hist = None
        self._runtime_next_g_hist_route_keys = []
        # Populate the scene-aware segment with D_s, D_rw, and D_sw.
        self._append_smart_assigner_segments(
            adaptor_dict=adaptor_dict,
            route_keys=route_keys,
            img_context_token_id=img_context_token_id,
            route_token_count=route_token_count,
            speed_token_count=speed_token_count,
        )

    def budget_info_update_after_llm(
        self,
        route_keys: Optional[List[str]],
        driving_input: DrivingInput,
        current_speed_wps: Optional[Tensor],
        current_route: Optional[Tensor],
        adaptor_dict: Optional[Dict[str, Tensor]] = None,
        inputs_embeds: Optional[Tensor] = None,
        tokenizer: Optional[Any] = None,
    ) -> Dict[str, Any]:
        del adaptor_dict, inputs_embeds, tokenizer
        self._update_smart_assigner_history(
            driving_input=driving_input,
            current_speed_wps=current_speed_wps,
            current_route=current_route,
            route_keys=route_keys,
        )
        # Commit scene history once after the full frame update.
        self._commit_smart_scene_history(route_keys=route_keys)
        self.last_scene_metrics = {}
        self.last_update_info = []
        return self.last_scene_metrics

    def reset(self, route_keys: Optional[List[str]] = None) -> None:
        super().reset(route_keys)
        self._runtime_route_keys = []
        self._runtime_budget_query_position = None
        self._runtime_path_query_position = None
        self._runtime_sceneaware_token_positions = None
        self._runtime_budget_token_features = None
        self._runtime_budget_query_feat_std = None
        self._runtime_budget_decided = False
        self._runtime_path_decided = False
        self._runtime_decision_id = 0
        self._runtime_budget_action_logits = None
        self._runtime_budget_action_hard = None
        self._runtime_path_logits = None
        self._runtime_budget_units_hard = None
        self._runtime_path_mask_hard = None
        self._runtime_external_budget = None
        self._runtime_forced_budget_action_ids = None
        self._runtime_next_g_hist = None
        self._runtime_next_g_hist_route_keys = []
        if route_keys is None:
            self.smart_assigner_state_by_route = {}
            self._decision_counter = 0
            return
        for key in route_keys:
            k = str(key)
            self.smart_assigner_state_by_route.pop(k, None)

    def _resolve_assignquery_token_positions(self, adaptor_dict: Dict[str, Tensor]):
        active_keys = adaptor_dict.get("active_adaptor_keys", [])
        if "assignquery" not in active_keys:
            raise ValueError("assignquery segment is required for smart_assigner")
        assign_idx = active_keys.index("assignquery")
        split_sizes = adaptor_dict["split_sizes"]
        assign_len = int(split_sizes[assign_idx].item())
        if assign_len != 2:
            raise ValueError(f"assignquery must contain exactly 2 tokens, got {assign_len}")
        start = int(split_sizes[:assign_idx].sum().item()) if assign_idx > 0 else 0
        perm = adaptor_dict["perm"]
        inv_perm = perm.argsort(-1)
        assign_orig_indices = torch.arange(start, start + 2, device=perm.device, dtype=torch.long)
        assign_positions = inv_perm[:, assign_orig_indices].long()  # shape: (B, 2)
        return assign_positions[:, 0], assign_positions[:, 1]

    def _resolve_sceneaware_token_positions(self, adaptor_dict: Dict[str, Tensor]) -> torch.LongTensor:
        active_keys = adaptor_dict.get("active_adaptor_keys", [])
        if "sceneaware" not in active_keys:
            raise ValueError("sceneaware segment is required for smart_assigner")
        scene_idx = active_keys.index("sceneaware")
        split_sizes = adaptor_dict["split_sizes"]
        scene_len = int(split_sizes[scene_idx].item())
        if scene_len != 3:
            raise ValueError(f"sceneaware must contain exactly 3 tokens, got {scene_len}")
        start = int(split_sizes[:scene_idx].sum().item()) if scene_idx > 0 else 0
        perm = adaptor_dict["perm"]
        inv_perm = perm.argsort(-1)
        scene_orig_indices = torch.arange(start, start + 3, device=perm.device, dtype=torch.long)
        return inv_perm[:, scene_orig_indices].long()  # shape: (B, 3)

    @staticmethod
    def _expand_token_positions(
        token_positions: torch.LongTensor,
        batch_size: int,
        device: torch.device,
    ) -> torch.LongTensor:
        token_positions = token_positions.to(device=device, dtype=torch.long)
        if token_positions.ndim == 0:
            token_positions = token_positions.unsqueeze(0)
        if token_positions.size(0) == 1 and batch_size > 1:
            token_positions = token_positions.expand(batch_size)
        if token_positions.size(0) != batch_size:
            raise ValueError(
                f"token position batch mismatch: {token_positions.size(0)} vs {batch_size}"
            )
        return token_positions

    def _get_or_init_smart_state(
        self,
        route_key: str,
        device: torch.device,
        dtype: torch.dtype,
        route_len: int,
        speed_dev_len: int,
    ) -> Dict[str, Any]:
        """Get or initialize route-scoped smart history."""
        key = self._normalize_route_key(route_key)
        if key not in self.smart_assigner_state_by_route:
            self.smart_assigner_state_by_route[key] = {
                "g_hist": torch.zeros((1, self.smart_feature_dim), device=device, dtype=dtype),
                "cached_route_dev": torch.zeros((route_len, 2), device=device, dtype=dtype),
                "cached_speed_dev": torch.zeros((speed_dev_len, 1), device=device, dtype=dtype),
                "prev_route": None,
                "prev_speed_wps": None,
                "prev_ego_xy": None,
                "prev_ego_yaw": None,
                "prev_timestamp": None,
            }
        return self.smart_assigner_state_by_route[key]

    def _append_smart_assigner_segments(
        self,
        adaptor_dict: Dict[str, Tensor],
        route_keys: Optional[List[str]],
        img_context_token_id: int,
        route_token_count: int,
        speed_token_count: int,
    ) -> None:
        """Populate the scene-aware segment with D_s, D_rw, and D_sw."""
        if not self.smart_modules_initialized:
            raise ValueError("smart_assigner modules are not initialized")

        route_len = int(route_token_count)
        speed_len = int(speed_token_count)
        if route_len <= 0:
            raise ValueError(f"route_token_count must be > 0, got {route_len}")
        if speed_len <= 0:
            raise ValueError(f"speed_token_count must be > 0, got {speed_len}")
        img_context_id = int(img_context_token_id)
        if img_context_id < 0:
            raise ValueError(f"img_context_token_id must be >= 0, got {img_context_id}")
        speed_dev_len = max(speed_len - 1, 1)

        language_ids = adaptor_dict["language__ids"]
        language_inputs = adaptor_dict["language_inputs"]

        scene_tokens_list = []
        token_counts = []
        for b_idx in range(language_ids.size(0)):
            token_pos = torch.nonzero(language_ids[b_idx] == img_context_id, as_tuple=False).squeeze(-1)
            if token_pos.numel() == 0:
                raise ValueError("<IMG_CONTEXT> tokens not found in language ids for smart_assigner mode")
            scene_tokens_list.append(language_inputs[b_idx, token_pos])
            token_counts.append(int(token_pos.numel()))
        if len(set(token_counts)) != 1:
            raise ValueError(f"Inconsistent image token counts in batch: {token_counts}")

        scene_tokens = torch.stack(scene_tokens_list, dim=0)  # (B, N_c, C)
        g_t = scene_tokens.mean(dim=1, keepdim=True)  # (B, 1, C)
        resolved_route_keys = self._resolve_route_keys(route_keys, int(scene_tokens.size(0)))

        state_device = adaptor_dict["inputs"].device
        state_dtype = adaptor_dict["inputs"].dtype

        g_hist_list = []
        route_dev_list = []
        speed_dev_list = []
        for b_idx in range(scene_tokens.size(0)):
            route_key = resolved_route_keys[b_idx]
            state = self._get_or_init_smart_state(
                route_key=route_key,
                device=state_device,
                dtype=state_dtype,
                route_len=route_len,
                speed_dev_len=speed_dev_len,
            )
            g_hist_list.append(state["g_hist"])
            route_dev_list.append(state["cached_route_dev"])
            speed_dev_list.append(state["cached_speed_dev"])

        g_hist_prev = torch.stack(g_hist_list, dim=0)  # (B, 1, C)
        scene_tokens_attn = scene_tokens.to(dtype=self.smart_scene_cross_attn.in_proj_weight.dtype)
        g_hist_attn = g_hist_prev.to(dtype=self.smart_scene_cross_attn.in_proj_weight.dtype)
        c_t, _ = self.smart_scene_cross_attn(query=g_hist_attn, key=scene_tokens_attn, value=scene_tokens_attn)
        g_h_t = self.smart_alpha * c_t + (1.0 - self.smart_alpha) * g_hist_attn

        # Speculative forwards stage history; the live policy commits it.
        self._runtime_next_g_hist = g_h_t.to(device=state_device, dtype=state_dtype).detach()
        self._runtime_next_g_hist_route_keys = list(resolved_route_keys)

        scene_dev_in = torch.cat(
            [
                g_t.to(dtype=self.smart_scene_dev_mlp[0].weight.dtype),
                c_t,
                g_t.to(dtype=self.smart_scene_dev_mlp[0].weight.dtype) - c_t,
            ],
            dim=-1,
        )
        d_s = self.smart_scene_dev_mlp(scene_dev_in)  # (B, 1, C)

        route_dev = torch.stack(route_dev_list, dim=0)
        speed_dev = torch.stack(speed_dev_list, dim=0)

        if route_dev.size(-1) != int(self.smart_route_conv.in_channels):
            raise ValueError(
                f"Route deviation channels={route_dev.size(-1)} mismatches smart_assigner.route_conv.in_channels={self.smart_route_conv.in_channels}"
            )
        route_conv_in = route_dev.transpose(1, 2).to(dtype=self.smart_route_conv.weight.dtype)
        route_feat = self.smart_route_conv(route_conv_in).mean(dim=-1)
        d_rw = self.smart_route_mlp(route_feat).unsqueeze(1)  # (B, 1, C)

        if speed_dev.size(-1) != int(self.smart_speed_conv.in_channels):
            raise ValueError(
                f"Speed deviation channels={speed_dev.size(-1)} mismatches smart_assigner.speed_conv.in_channels={self.smart_speed_conv.in_channels}"
            )
        speed_conv_in = speed_dev.transpose(1, 2).to(dtype=self.smart_speed_conv.weight.dtype)
        speed_feat = self.smart_speed_conv(speed_conv_in).mean(dim=-1)
        d_sw = self.smart_speed_mlp(speed_feat).unsqueeze(1)  # (B, 1, C)

        token_dtype = adaptor_dict["inputs"].dtype
        token_device = adaptor_dict["inputs"].device
        d_s = d_s.to(device=token_device, dtype=token_dtype)
        d_rw = d_rw.to(device=token_device, dtype=token_dtype)
        d_sw = d_sw.to(device=token_device, dtype=token_dtype)

        sceneaware_tokens = torch.cat([d_s, d_rw, d_sw], dim=1)  # (B, 3, C)
        if "sceneaware_inputs" not in adaptor_dict:
            raise ValueError("sceneaware_inputs not found in adaptor_dict for smart_assigner mode")
        if adaptor_dict["sceneaware_inputs"].shape != sceneaware_tokens.shape:
            raise ValueError(
                f"sceneaware_inputs shape mismatch: expected {adaptor_dict['sceneaware_inputs'].shape}, got {sceneaware_tokens.shape}"
            )
        adaptor_dict["sceneaware_inputs"] = sceneaware_tokens
        refresh_adaptor_inputs(adaptor_dict)

    def _commit_smart_scene_history(self, route_keys: Optional[List[str]]) -> None:
        """Commit scene history produced by the current forward pass."""
        if self._runtime_next_g_hist is None:
            raise RuntimeError("smart scene history candidate is missing")
        resolved_route_keys = self._resolve_route_keys(
            route_keys,
            int(self._runtime_next_g_hist.size(0)),
        )
        if resolved_route_keys != self._runtime_next_g_hist_route_keys:
            raise RuntimeError("smart scene history route keys changed before commit")

        for b_idx, route_key in enumerate(resolved_route_keys):
            state = self.smart_assigner_state_by_route[self._normalize_route_key(route_key)]
            state["g_hist"] = self._runtime_next_g_hist[b_idx].detach()
        self._runtime_next_g_hist = None
        self._runtime_next_g_hist_route_keys = []

    def _update_smart_assigner_history(
        self,
        driving_input: DrivingInput,
        current_speed_wps: Optional[Tensor],
        current_route: Optional[Tensor],
        route_keys: Optional[List[str]],
    ) -> None:
        """Cache route and speed deviations for the next frame."""
        if current_speed_wps is None or current_route is None:
            raise ValueError("smart_assigner requires both speed_wps and route predictions")
        if driving_input.ego_xy is None or driving_input.ego_yaw is None or driving_input.timestamp is None:
            raise ValueError("smart_assigner requires ego_xy, ego_yaw, timestamp in DrivingInput")

        def _compute_speed_from_waypoints(waypoints: Tensor) -> Tensor:
            if waypoints.size(0) < 2:
                return torch.zeros((1, 1), device=waypoints.device, dtype=waypoints.dtype)
            deltas = waypoints[1:] - waypoints[:-1]
            return torch.linalg.norm(deltas, dim=-1, keepdim=True) / self.smart_t_lap

        route_len = int(current_route.size(1))
        speed_dev_len = max(int(current_speed_wps.size(1)) - 1, 1)
        resolved_route_keys = self._resolve_route_keys(route_keys, int(current_route.size(0)))

        for b_idx in range(current_route.size(0)):
            route_key = resolved_route_keys[b_idx]
            state = self._get_or_init_smart_state(
                route_key=route_key,
                device=current_route.device,
                dtype=current_route.dtype,
                route_len=route_len,
                speed_dev_len=speed_dev_len,
            )

            route_dev = torch.zeros((route_len, 2), device=current_route.device, dtype=current_route.dtype)
            speed_dev = torch.zeros((speed_dev_len, 1), device=current_speed_wps.device, dtype=current_speed_wps.dtype)

            has_prev = (
                state["prev_route"] is not None
                and state["prev_speed_wps"] is not None
                and state["prev_ego_xy"] is not None
                and state["prev_ego_yaw"] is not None
                and state["prev_timestamp"] is not None
            )

            if has_prev:
                ego_xy = driving_input.ego_xy[b_idx]
                ego_yaw = driving_input.ego_yaw[b_idx]
                timestamp = driving_input.timestamp[b_idx]

                dx_global = ego_xy[0] - state["prev_ego_xy"][0]
                dy_global = ego_xy[1] - state["prev_ego_xy"][1]
                cos_prev = torch.cos(state["prev_ego_yaw"])
                sin_prev = torch.sin(state["prev_ego_yaw"])
                delta_x_prev = cos_prev * dx_global + sin_prev * dy_global
                delta_y_prev = -sin_prev * dx_global + cos_prev * dy_global
                delta_xy_prev_frame = torch.stack([delta_x_prev, delta_y_prev], dim=-1)
                delta_yaw = ego_yaw - state["prev_ego_yaw"]
                delta_tau = timestamp - state["prev_timestamp"]

                prev_route_curr_frame = self.metrics_computer._transform_prev_waypoints_to_curr_frame(
                    state["prev_route"].unsqueeze(0),
                    delta_xy_prev_frame.unsqueeze(0),
                    delta_yaw.unsqueeze(0),
                )
                aligned_prev_route, valid_route = self.metrics_computer._time_align_prev_waypoints(
                    prev_route_curr_frame,
                    delta_tau.unsqueeze(0),
                )
                aligned_prev_route = aligned_prev_route.squeeze(0)
                valid_route = valid_route.squeeze(0)
                route_delta = current_route[b_idx] - aligned_prev_route
                route_dev = torch.where(valid_route.unsqueeze(-1), route_delta, torch.zeros_like(route_delta))

                prev_speed_curr_frame = self.metrics_computer._transform_prev_waypoints_to_curr_frame(
                    state["prev_speed_wps"].unsqueeze(0),
                    delta_xy_prev_frame.unsqueeze(0),
                    delta_yaw.unsqueeze(0),
                )
                aligned_prev_speed, valid_speed = self.metrics_computer._time_align_prev_waypoints(
                    prev_speed_curr_frame,
                    delta_tau.unsqueeze(0),
                )
                aligned_prev_speed = aligned_prev_speed.squeeze(0)
                valid_speed = valid_speed.squeeze(0)
                speed_curr = _compute_speed_from_waypoints(current_speed_wps[b_idx])
                speed_prev = _compute_speed_from_waypoints(aligned_prev_speed)
                valid_len = min(speed_curr.size(0), speed_prev.size(0), speed_dev_len)
                speed_valid = (valid_speed[:-1] & valid_speed[1:]).unsqueeze(-1)
                speed_delta = speed_curr[:valid_len] - speed_prev[:valid_len]
                speed_dev[:valid_len] = torch.where(
                    speed_valid[:valid_len],
                    speed_delta,
                    torch.zeros_like(speed_delta),
                )

            # Cache deviation inputs for the next frame.
            state["cached_route_dev"] = route_dev.detach()
            state["cached_speed_dev"] = speed_dev.detach()
            # Update the reference state.
            state["prev_route"] = current_route[b_idx].detach()
            state["prev_speed_wps"] = current_speed_wps[b_idx].detach()
            state["prev_ego_xy"] = driving_input.ego_xy[b_idx].detach()
            state["prev_ego_yaw"] = driving_input.ego_yaw[b_idx].detach()
            state["prev_timestamp"] = driving_input.timestamp[b_idx].detach()


def build_budget_assigner(
    mode: str,
    history_alpha: float = 0.6,
    decision_shift_t_lap: float = 0.2,
) -> SmartBudgetAssigner:
    mode = str(mode).strip().lower()
    if mode != "smart_assigner":
        raise ValueError(f"Only smart_assigner mode is supported, got {mode}")
    return SmartBudgetAssigner(
        mode=mode,
        history_alpha=history_alpha,
        decision_shift_t_lap=decision_shift_t_lap,
    )
