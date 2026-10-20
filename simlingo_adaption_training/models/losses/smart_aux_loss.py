from typing import Dict

import torch
import torch.nn.functional as F
from torch import Tensor

from simlingo_adaption_training.models.scheduler.scheduler_utils import (
    budget_quantizing,
    resolve_smart_budget_min,
)


class SmartAuxLossComputer:
    def __init__(self, img_context_token_id: int) -> None:
        self.img_context_token_id = int(img_context_token_id)
        if self.img_context_token_id < 0:
            raise ValueError(f"img_context_token_id must be >= 0, got {self.img_context_token_id}")

    @staticmethod
    def _require_rank(tensor: Tensor, expected_rank: int, name: str) -> None:
        if tensor.ndim != expected_rank:
            raise ValueError(f"{name} must be rank-{expected_rank}, got shape={tuple(tensor.shape)}")

    @staticmethod
    def _to_scalar_int(value: Tensor, name: str) -> int:
        if value.numel() != 1:
            raise ValueError(f"{name} must contain exactly one element, got shape={tuple(value.shape)}")
        return int(value.item())

    @staticmethod
    def _masked_argmax_pool(tokens: Tensor, mask: Tensor, name: str) -> Tensor:
        # shape: tokens=(B, T, C), mask=(B, T)
        if tokens.ndim != 3:
            raise ValueError(f"{name}: tokens must be rank-3, got shape={tuple(tokens.shape)}")
        if mask.ndim != 2:
            raise ValueError(f"{name}: mask must be rank-2, got shape={tuple(mask.shape)}")
        if tokens.size(0) != mask.size(0) or tokens.size(1) != mask.size(1):
            raise ValueError(
                f"{name}: token/mask shape mismatch, got tokens={tuple(tokens.shape)}, mask={tuple(mask.shape)}"
            )
        if mask.dtype != torch.bool:
            raise ValueError(f"{name}: mask must be bool, got {mask.dtype}")
        valid_counts = mask.sum(dim=1)
        if torch.any(valid_counts == 0):
            raise ValueError(f"{name}: each sample must have at least one valid token")
        neg_inf = torch.finfo(tokens.dtype).min
        masked_tokens = torch.where(mask.unsqueeze(-1), tokens, torch.full_like(tokens, neg_inf))
        return masked_tokens.max(dim=1).values  # shape: (B, C)

    def compute_stage1_aux_losses(
        self,
        language_inputs: Tensor,
        language_ids: Tensor,
        language_inputs_mask: Tensor,
        budget_token_features: Tensor,
        path_mask_hard: Tensor,
        budget_values: Tensor,
        num_prefix_layers: Tensor,
        num_hidden_layers: Tensor,
        div_margin: float,
        bal_epsilon: float,
    ) -> Dict[str, Tensor]:
        if div_margin < 0.0:
            raise ValueError(f"div_margin must be >= 0, got {div_margin}")
        if bal_epsilon < 0.0:
            raise ValueError(f"bal_epsilon must be >= 0, got {bal_epsilon}")

        self._require_rank(language_inputs, expected_rank=3, name="language_inputs")
        self._require_rank(language_ids, expected_rank=2, name="language_ids")
        self._require_rank(language_inputs_mask, expected_rank=2, name="language_inputs_mask")
        self._require_rank(budget_token_features, expected_rank=2, name="budget_token_features")
        self._require_rank(path_mask_hard, expected_rank=2, name="path_mask_hard")
        self._require_rank(budget_values, expected_rank=1, name="budget_values")

        if language_inputs.size(0) != language_ids.size(0) or language_inputs.size(1) != language_ids.size(1):
            raise ValueError(
                f"language_inputs/language_ids shape mismatch: "
                f"{tuple(language_inputs.shape)} vs {tuple(language_ids.shape)}"
            )
        if language_inputs.size(0) != language_inputs_mask.size(0) or language_inputs.size(1) != language_inputs_mask.size(1):
            raise ValueError(
                f"language_inputs/language_inputs_mask shape mismatch: "
                f"{tuple(language_inputs.shape)} vs {tuple(language_inputs_mask.shape)}"
            )
        if language_inputs_mask.dtype != torch.bool:
            raise ValueError(f"language_inputs_mask must be bool, got {language_inputs_mask.dtype}")

        batch_size, _, hidden_size = language_inputs.shape
        if budget_token_features.size(0) != batch_size or budget_token_features.size(1) != hidden_size:
            raise ValueError(
                f"budget_token_features shape mismatch with language_inputs: "
                f"{tuple(budget_token_features.shape)} vs hidden_size={hidden_size}"
            )
        if path_mask_hard.size(0) != batch_size:
            raise ValueError(
                f"path_mask_hard batch mismatch: {path_mask_hard.size(0)} vs {batch_size}"
            )
        if budget_values.size(0) != batch_size:
            raise ValueError(f"budget_values batch mismatch: {budget_values.size(0)} vs {batch_size}")

        path_mask_float = path_mask_hard.float()
        if torch.any(path_mask_float < -1e-6) or torch.any(path_mask_float > 1.0 + 1e-6):
            raise ValueError(
                f"path_mask_hard must be within [0, 1], got "
                f"min={float(path_mask_float.min().item()):.6f}, max={float(path_mask_float.max().item()):.6f}"
            )

        # Pool prompt, visual, and budget tokens by their strongest response.
        visual_mask = language_inputs_mask & language_ids.eq(self.img_context_token_id)
        prompt_mask = language_inputs_mask & language_ids.ne(self.img_context_token_id)
        prompt_pool = self._masked_argmax_pool(language_inputs.float(), prompt_mask, name="prompt_pool")
        visual_pool = self._masked_argmax_pool(language_inputs.float(), visual_mask, name="visual_pool")
        budget_pool = budget_token_features.float()  # shape: (B, C)
        scene_features = torch.maximum(torch.maximum(prompt_pool, visual_pool), budget_pool)

        scene_norm = F.normalize(scene_features, p=2.0, dim=-1, eps=1e-6)  # shape: (B, C)
        # Use the executed layer mask so the target matches the actual path.
        path_norm = F.normalize(path_mask_float, p=2.0, dim=-1, eps=1e-6)  # shape: (B, N_sub)
        scene_similarity = scene_norm @ scene_norm.transpose(0, 1)  # shape: (B, B)
        path_similarity = path_norm @ path_norm.transpose(0, 1)  # shape: (B, B)
        div_margin_tensor = scene_similarity.new_tensor(float(div_margin))
        div_loss = torch.relu(path_similarity - scene_similarity - div_margin_tensor).pow(2).mean()

        num_prefix = self._to_scalar_int(num_prefix_layers, "num_prefix_layers")
        num_hidden = self._to_scalar_int(num_hidden_layers, "num_hidden_layers")
        if not (2 <= num_prefix < num_hidden):
            raise ValueError(
                f"invalid layer config for balance loss: num_prefix_layers={num_prefix}, num_hidden_layers={num_hidden}"
            )
        if num_prefix % 2 != 0:
            raise ValueError(
                f"num_prefix_layers must be even for half-split smart forward, got {num_prefix}"
            )
        num_sub_layers = num_hidden - num_prefix
        if path_mask_hard.size(1) != num_sub_layers:
            raise ValueError(
                f"path_mask_hard width must equal num_sub_layers={num_sub_layers}, "
                f"got {path_mask_hard.size(1)}"
            )

        budget_min = resolve_smart_budget_min(
            num_prefix_layers=num_prefix,
            num_hidden_layers=num_hidden,
        )
        budget_values_float = budget_values.float()
        # Mixed-precision runtime can introduce tiny numerical drift around the exact lower bound.
        budget_tol = max(1e-4, float(torch.finfo(budget_values_float.dtype).eps) * 8.0)
        if torch.any(budget_values_float < budget_min - budget_tol) or torch.any(
            budget_values_float > 1.0 + budget_tol
        ):
            raise ValueError(
                f"budget_values must be within [{budget_min:.6f}, 1.0], got "
                f"min={float(budget_values_float.min().item()):.6f}, max={float(budget_values_float.max().item()):.6f}"
            )
        budget_values_float = budget_values_float.clamp(min=budget_min, max=1.0)

        # Reuse training quantization so the target matches the top-k constraint.
        sub_layer_budget = budget_quantizing(
            budget_values_float,
            num_prefix_units=num_prefix,
            num_total_units=num_hidden,
        )[0] / float(num_sub_layers)
        sub_layer_budget = sub_layer_budget.clamp(0.0, 1.0)
        target_sub_budget = sub_layer_budget.mean()  # shape: ()

        # Balance long-term layer usage rather than each sample independently.
        mean_path_activation = path_mask_float.mean(dim=0)  # shape: (N_sub,)
        bal_eps_tensor = mean_path_activation.new_tensor(float(bal_epsilon))
        bal_loss = torch.relu((mean_path_activation - target_sub_budget).abs() - bal_eps_tensor).pow(2).mean()

        return {
            "div_loss": div_loss,
            "bal_loss": bal_loss,
            "scene_similarity_mean": scene_similarity.mean(),
            "path_similarity_mean": path_similarity.mean(),
            "target_sub_budget": target_sub_budget,
            "path_mask_hard_mean": path_mask_float.mean(),
        }

    def compute_stage2_aux_losses(
        self,
        driving_loss_per_sample: Tensor,
        full_budget_driving_loss: Tensor,
        distill_tau: float,
    ) -> Dict[str, Tensor]:
        # Stage2 keeps only the per-sample distillation guard.
        # shape: driving/full=(B,)
        self._require_rank(driving_loss_per_sample, expected_rank=1, name="driving_loss_per_sample")
        self._require_rank(full_budget_driving_loss, expected_rank=1, name="full_budget_driving_loss")
        if driving_loss_per_sample.size(0) != full_budget_driving_loss.size(0):
            raise ValueError(
                "driving_loss_per_sample/full_budget_driving_loss batch mismatch: "
                f"{driving_loss_per_sample.size(0)} vs {full_budget_driving_loss.size(0)}"
            )
        if distill_tau < 0.0:
            raise ValueError(f"distill_tau must be >= 0, got {distill_tau}")

        distill_tau_tensor = driving_loss_per_sample.new_tensor(float(distill_tau))
        distill_gap = (
            driving_loss_per_sample.float()
            - full_budget_driving_loss.float()
            - distill_tau_tensor
        )
        loss_distill = torch.relu(distill_gap).pow(2).mean()
        return {
            "loss_distill": loss_distill,
            "distill_gap_mean": distill_gap.mean(),
            "distill_violation_rate": distill_gap.gt(0.0).float().mean(),
            "full_budget_driving_loss_mean": full_budget_driving_loss.float().mean(),
        }
