from typing import Dict, List, Optional

import torch
from torch import Tensor

from simlingo_adaption_training.utils.custom_types import DrivingInput

class DrivingMetricsComputer:
    """Compute novelty and decision-shift metrics for one forward step."""

    def __init__(self, owner):
        self.owner = owner

    # Convert nested tensors to python scalars/lists for logging/state update.
    def to_python_metrics(self, outputs: Dict[str, Tensor]) -> Dict[str, object]:
        if not isinstance(outputs, dict):
            raise TypeError(f"to_python_metrics expects dict input, got {type(outputs).__name__}")

        def _to_python(value):
            if isinstance(value, dict):
                return {k: _to_python(v) for k, v in value.items()}
            if isinstance(value, torch.Tensor):
                tensor = value.detach().cpu()
                return [float(tensor.item())] if tensor.ndim == 0 else tensor.tolist()
            return value

        return _to_python(outputs)

    def compute_novelty_metrics(
        self,
        route_keys: List[str],
        adaptor_dict: Dict[str, torch.Tensor],
        inputs_embeds: torch.Tensor,
        tokenizer,
    ) -> Dict[str, Dict[str, List[Optional[float]]]]:
        """Compute visual-history similarity/novelty and update per-route `v_hist`."""
        sim_in_list: List[Optional[float]] = [None for _ in range(len(route_keys))]
        novelty_list: List[Optional[float]] = [None for _ in range(len(route_keys))]
        visual_positions = self._extract_visual_token_positions(adaptor_dict, tokenizer)
        if visual_positions is None:
            return {
                "novelty": novelty_list,
                "history_similarity": {"sim_in": sim_in_list, "alpha": float(self.owner.history_alpha)},
            }

        for idx, route_key in enumerate(route_keys):
            state = self.owner.state_by_route[route_key]
            vis_idx = visual_positions[idx]
            vis_idx = vis_idx[(vis_idx >= 0) & (vis_idx < inputs_embeds.size(1))]
            if vis_idx.numel() == 0:
                continue

            v_global = inputs_embeds[idx, vis_idx].mean(dim=0).detach().float()
            prev_hist = state.get("v_hist")
            if prev_hist is None:
                sim_in = 1.0
                updated_hist = v_global
            else:
                prev_hist = prev_hist.to(device=v_global.device, dtype=v_global.dtype)
                denom = float((v_global.norm(p=2) * prev_hist.norm(p=2)).item())
                cos = float(torch.dot(v_global, prev_hist).item() / max(denom, 1e-8))
                sim_in = float(torch.clamp(torch.tensor(0.5 * (1.0 + cos)), 0.0, 1.0).item())
                updated_hist = self.owner.history_alpha * v_global + (1.0 - self.owner.history_alpha) * prev_hist

            state["v_hist"] = updated_hist.detach()
            sim_in_list[idx] = sim_in
            novelty_list[idx] = float(torch.clamp(torch.tensor(1.0 - sim_in), 0.0, 1.0).item())

        return {
            "novelty": novelty_list,
            "history_similarity": {"sim_in": sim_in_list, "alpha": float(self.owner.history_alpha)},
        }

    @staticmethod
    def _extract_visual_token_positions(
        adaptor_dict: Dict[str, torch.Tensor],
        tokenizer,
    ) -> Optional[List[torch.Tensor]]:
        """Extract reordered visual token positions (<IMG_CONTEXT>) for each sample."""
        language_ids = adaptor_dict.get("language__ids")
        perm = adaptor_dict.get("perm")
        if language_ids is None or perm is None:
            return None
        inv_perm = perm.argsort(-1)
        img_context_token_id = tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
        positions: List[torch.Tensor] = []
        for b_idx in range(language_ids.size(0)):
            visual_orig = torch.nonzero(language_ids[b_idx] == img_context_token_id, as_tuple=False).squeeze(-1)
            positions.append(inv_perm[b_idx, visual_orig].long())
        return positions

    def _transform_prev_waypoints_to_curr_frame(self, prev_waypoints: Tensor, delta_xy_prev_frame: Tensor, delta_yaw: Tensor):
        centered = prev_waypoints - delta_xy_prev_frame[:, None, :]
        cos_yaw = torch.cos(delta_yaw)[:, None]
        sin_yaw = torch.sin(delta_yaw)[:, None]
        x_prev = centered[..., 0]
        y_prev = centered[..., 1]
        x_curr = cos_yaw * x_prev + sin_yaw * y_prev
        y_curr = -sin_yaw * x_prev + cos_yaw * y_prev
        return torch.stack([x_curr, y_curr], dim=-1)

    def _time_align_prev_waypoints(self, prev_waypoints_curr_frame: Tensor, delta_tau: Tensor):
        batch_size, waypoint_count, _ = prev_waypoints_curr_frame.shape
        base_idx = torch.arange(waypoint_count, device=prev_waypoints_curr_frame.device, dtype=prev_waypoints_curr_frame.dtype)
        j_star = base_idx[None, :] + (delta_tau[:, None] / self.owner.decision_shift_t_lap)
        k = torch.floor(j_star).long()
        alpha = j_star - k.to(j_star.dtype)
        k_next = k + 1
        valid_mask = (k >= 0) & (k_next < waypoint_count)

        k = k.clamp(0, waypoint_count - 1)
        k_next = k_next.clamp(0, waypoint_count - 1)
        k_idx = k.unsqueeze(-1).expand(batch_size, waypoint_count, 2)
        k_next_idx = k_next.unsqueeze(-1).expand(batch_size, waypoint_count, 2)
        wp_k = torch.gather(prev_waypoints_curr_frame, dim=1, index=k_idx)
        wp_k_next = torch.gather(prev_waypoints_curr_frame, dim=1, index=k_next_idx)
        aligned = (1.0 - alpha.unsqueeze(-1)) * wp_k + alpha.unsqueeze(-1) * wp_k_next
        return aligned, valid_mask

    def _compute_decision_shift(self, current_waypoints: Tensor, aligned_prev_waypoints: Tensor, valid_mask: Tensor):
        point_error = torch.linalg.norm(current_waypoints - aligned_prev_waypoints, dim=-1)
        valid_float = valid_mask.to(point_error.dtype)
        valid_count = valid_float.sum(dim=-1)
        error_sum = (point_error * valid_float).sum(dim=-1)
        e_mean = torch.where(
            valid_count > 0,
            error_sum / valid_count,
            torch.full_like(error_sum, float("nan")),
        )
        e_norm = e_mean / (e_mean + 1.0)
        return e_mean, e_norm

    def _decision_shift_for_waypoint_set(
        self,
        current_waypoints: Optional[Tensor],
        prev_waypoints: Optional[Tensor],
        delta_xy_prev_frame: Tensor,
        delta_yaw: Tensor,
        delta_tau: Tensor,
    ):
        if current_waypoints is None or prev_waypoints is None:
            return {"e_mean": None, "e_norm": None}
        prev_curr_frame = self._transform_prev_waypoints_to_curr_frame(prev_waypoints, delta_xy_prev_frame, delta_yaw)
        prev_aligned, valid_mask = self._time_align_prev_waypoints(prev_curr_frame, delta_tau)
        e_mean, e_norm = self._compute_decision_shift(current_waypoints, prev_aligned, valid_mask)
        return {"e_mean": e_mean, "e_norm": e_norm}

    def compute_decision_shift_metrics(
        self,
        driving_input: DrivingInput,
        current_speed_wps: Optional[Tensor],
        current_route: Optional[Tensor],
        route_keys: List[str],
    ):
        """Compute route-wise decision shift and update `prev_decision_state_by_route`."""
        batch_size = len(route_keys)
        ego_xy = driving_input.ego_xy
        ego_yaw = driving_input.ego_yaw
        timestamp = driving_input.timestamp
        device = ego_xy.device

        speed_e_mean = torch.full((batch_size,), float("nan"), device=device)
        speed_e_norm = torch.full((batch_size,), float("nan"), device=device)
        route_e_mean = torch.full((batch_size,), float("nan"), device=device)
        route_e_norm = torch.full((batch_size,), float("nan"), device=device)
        delta_tau_out = torch.full((batch_size,), float("nan"), device=device)

        for b_idx, route_key in enumerate(route_keys):
            ego_xy_i = ego_xy[b_idx]
            ego_yaw_i = ego_yaw[b_idx]
            timestamp_i = timestamp[b_idx]

            prev_state = self.owner.prev_decision_state_by_route.get(route_key, None)
            current_speed_i = None if current_speed_wps is None else current_speed_wps[b_idx]
            current_route_i = None if current_route is None else current_route[b_idx]

            if prev_state:
                prev_ego_xy = prev_state["ego_xy"]
                prev_ego_yaw = prev_state["ego_yaw"]
                prev_timestamp = prev_state["timestamp"]

                dx_global = ego_xy_i[0] - prev_ego_xy[0]
                dy_global = ego_xy_i[1] - prev_ego_xy[1]
                cos_prev = torch.cos(prev_ego_yaw)
                sin_prev = torch.sin(prev_ego_yaw)
                delta_x_prev = cos_prev * dx_global + sin_prev * dy_global
                delta_y_prev = -sin_prev * dx_global + cos_prev * dy_global
                delta_xy_prev_frame = torch.stack([delta_x_prev, delta_y_prev], dim=-1).unsqueeze(0)
                delta_yaw = (ego_yaw_i - prev_ego_yaw).unsqueeze(0)
                delta_tau = (timestamp_i - prev_timestamp).unsqueeze(0)
                delta_tau_out[b_idx] = delta_tau[0]

                speed_metrics = self._decision_shift_for_waypoint_set(
                    current_waypoints=None if current_speed_i is None else current_speed_i.unsqueeze(0),
                    prev_waypoints=None if prev_state.get("speed_wps") is None else prev_state["speed_wps"].unsqueeze(0),
                    delta_xy_prev_frame=delta_xy_prev_frame,
                    delta_yaw=delta_yaw,
                    delta_tau=delta_tau,
                )
                route_metrics = self._decision_shift_for_waypoint_set(
                    current_waypoints=None if current_route_i is None else current_route_i.unsqueeze(0),
                    prev_waypoints=None if prev_state.get("route") is None else prev_state["route"].unsqueeze(0),
                    delta_xy_prev_frame=delta_xy_prev_frame,
                    delta_yaw=delta_yaw,
                    delta_tau=delta_tau,
                )

                if speed_metrics["e_mean"] is not None:
                    speed_e_mean[b_idx] = speed_metrics["e_mean"][0]
                    speed_e_norm[b_idx] = speed_metrics["e_norm"][0]
                if route_metrics["e_mean"] is not None:
                    route_e_mean[b_idx] = route_metrics["e_mean"][0]
                    route_e_norm[b_idx] = route_metrics["e_norm"][0]

            self.owner.prev_decision_state_by_route[route_key] = {
                "speed_wps": None if current_speed_i is None else current_speed_i.detach().clone(),
                "route": None if current_route_i is None else current_route_i.detach().clone(),
                "ego_xy": ego_xy_i.detach().clone(),
                "ego_yaw": ego_yaw_i.detach().clone(),
                "timestamp": timestamp_i.detach().clone(),
            }

        return {
            "speed_wps": {"e_mean": speed_e_mean, "e_norm": speed_e_norm},
            "route": {"e_mean": route_e_mean, "e_norm": route_e_norm},
            "delta_tau": delta_tau_out,
            "t_lap": torch.full((batch_size,), float(self.owner.decision_shift_t_lap), device=device),
        }
