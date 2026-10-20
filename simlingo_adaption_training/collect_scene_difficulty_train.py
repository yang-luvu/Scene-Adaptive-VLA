import datetime
import json
import os
import sys
from collections import defaultdict, deque
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

# Ensure project root is importable when this script is executed as a file path.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import hydra
import numpy as np
import pytorch_lightning as pl
import torch
from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
from hydra.utils import get_original_cwd
from torch.utils.data import ConcatDataset
from tqdm import tqdm
from transformers import AutoProcessor

from simlingo_adaption_training.config import (
    TrainConfig,
    validate_adaption_config,
    validate_train_sampling_config,
)
from simlingo_adaption_training.utils.internvl2_utils import resolve_internvl_variant_source


@dataclass
class SampleRecord:
    index: int
    measurement_path: str
    route_key: str
    frame_id: int


def _decode_path(value) -> str:
    if isinstance(value, (bytes, bytearray, np.bytes_)):
        return value.decode("utf-8")
    return str(value)


def _route_key_from_measurement_path(measurement_path: str) -> str:
    parts = measurement_path.rsplit("/measurements/", 1)
    if len(parts) == 2:
        return parts[0]
    return os.path.dirname(measurement_path)


def _records_from_base_dataset(dataset, global_offset: int = 0) -> List[SampleRecord]:
    records: List[SampleRecord] = []
    for local_idx in range(len(dataset)):
        measurement_root = _decode_path(dataset.measurements[local_idx][0])
        frame_id = int(dataset.sample_start[local_idx]) + int(dataset.hist_len) - 1
        measurement_path = f"{measurement_root}/{frame_id:04}.json.gz"
        route_key = _route_key_from_measurement_path(measurement_path)
        records.append(
            SampleRecord(
                index=global_offset + local_idx,
                measurement_path=measurement_path,
                route_key=route_key,
                frame_id=frame_id,
            )
        )
    return records


def build_sample_records(dataset) -> List[SampleRecord]:
    if isinstance(dataset, ConcatDataset):
        records: List[SampleRecord] = []
        offset = 0
        for sub_dataset in dataset.datasets:
            records.extend(_records_from_base_dataset(sub_dataset, global_offset=offset))
            offset += len(sub_dataset)
        return records
    return _records_from_base_dataset(dataset, global_offset=0)


def build_route_batches(records: Sequence[SampleRecord], route_batch_size: int = 1) -> Iterable[List[SampleRecord]]:
    if route_batch_size <= 0:
        raise ValueError(f"route_batch_size must be >= 1, got {route_batch_size}")

    route_to_records = defaultdict(list)
    for record in records:
        route_to_records[record.route_key].append(record)

    route_order = sorted(route_to_records.keys())
    for route_key in route_order:
        frames = sorted(route_to_records[route_key], key=lambda x: x.frame_id)
        frame_ids = [r.frame_id for r in frames]
        for i in range(1, len(frame_ids)):
            if frame_ids[i] < frame_ids[i - 1]:
                raise RuntimeError(
                    f"Route frame order violation in {route_key}: {frame_ids[i - 1]} -> {frame_ids[i]}"
                )
        route_to_records[route_key] = deque(frames)

    # Fixed-slot multi-route scheduling:
    # - maintain up to `route_batch_size` active routes as processing slots.
    # - each slot keeps the same route until that route is fully consumed.
    # - for each step, emit one frame from every active slot route.
    # This guarantees route-local continuity:
    #   batch i contains frame i of each active route (until a route ends),
    # then a new route is inserted into the freed slot.
    pending_routes = deque(route_order)
    active_routes: List[str] = []
    for _ in range(min(route_batch_size, len(pending_routes))):
        active_routes.append(pending_routes.popleft())

    while active_routes:
        batch_records: List[SampleRecord] = []
        next_active_routes: List[str] = []
        for route_key in active_routes:
            batch_records.append(route_to_records[route_key].popleft())

            if route_to_records[route_key]:
                next_active_routes.append(route_key)
            elif pending_routes:
                next_active_routes.append(pending_routes.popleft())

        # Safety check: batch should never contain duplicate route.
        route_set = {r.route_key for r in batch_records}
        if len(route_set) != len(batch_records):
            raise RuntimeError("Batch contains duplicate routes, which violates per-route sequential semantics.")

        yield batch_records
        active_routes = next_active_routes


def move_to_device(x, device):
    if torch.is_tensor(x):
        return x.to(device)
    if isinstance(x, tuple) and hasattr(x, "_fields"):
        return type(x)(*(move_to_device(v, device) for v in x))
    if isinstance(x, list):
        return [move_to_device(v, device) for v in x]
    if isinstance(x, dict):
        return {k: move_to_device(v, device) for k, v in x.items()}
    return x


def per_sample_loss_scalars(loss_dict: Dict[str, tuple]) -> Dict[str, torch.Tensor]:
    per_sample = {}
    for key, (values, counts) in loss_dict.items():
        values = values.detach().float()
        counts = counts.detach().float()
        reduce_dims = tuple(range(1, values.dim()))
        if len(reduce_dims) > 0:
            values_sum = values.sum(dim=reduce_dims)
            counts_sum = counts.sum(dim=reduce_dims).clamp_min(1.0)
            per_sample[key] = values_sum / counts_sum
        else:
            per_sample[key] = values
    return per_sample


def metric_at(metrics: Dict, path: Sequence[str], idx: int):
    value = metrics
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key, None)
    if value is None:
        return None
    if isinstance(value, list):
        if idx >= len(value):
            return None
        value = value[idx]
    if isinstance(value, (list, tuple)):
        return value[0] if value else None
    return value


def resolve_autocast_settings(cfg_precision: str, device: torch.device):
    """
    Resolve autocast dtype from Lightning-style precision strings.
    Returns (enabled, dtype, reason).
    """
    if device.type != "cuda":
        return False, None, "cpu-no-autocast"

    # Optional runtime override for data collection jobs.
    override = os.getenv("SIMLINGO_COLLECT_PRECISION", "").strip().lower()
    precision = override if override else str(cfg_precision).strip().lower()

    fp16_aliases = {"16", "16-mixed", "16-true", "fp16", "fp16-mixed", "fp16-true", "half"}
    bf16_aliases = {"bf16", "bf16-mixed", "bf16-true"}
    fp32_aliases = {"32", "32-true", "fp32", "full"}

    if precision in bf16_aliases:
        return True, torch.bfloat16, f"{precision}->bf16"
    if precision in fp16_aliases:
        return True, torch.float16, f"{precision}->fp16"
    if precision in fp32_aliases:
        # InternVL flash-attn path requires fp16/bf16. Force fp16 for stability.
        return True, torch.float16, f"{precision}->forced-fp16-for-flash-attn"

    # Unknown precision string: choose fp16 as a safe default for this model path.
    return True, torch.float16, f"unknown({precision})->fp16"


def to_repo_relative(path: str, repo_root: Path) -> str:
    path_obj = Path(path).resolve()
    try:
        return str(path_obj.relative_to(repo_root))
    except ValueError:
        return str(path_obj)


def _route_id_from_route_key(route_key: str) -> str:
    return Path(route_key).name


def load_processed_route_ids(processed_jsonl_path: Path) -> set:
    """
    Load processed route ids from an existing metrics JSONL file.
    A route is considered processed if it appears at least once in that file.
    """
    route_ids = set()
    if not processed_jsonl_path.exists():
        return route_ids

    with open(processed_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue

            route_id = row.get("route_id", None)
            if route_id:
                route_ids.add(str(route_id))
                continue

            route_key = row.get("route_key", None)
            if route_key:
                route_ids.add(_route_id_from_route_key(str(route_key)))
                continue

            measurement_path = row.get("measurement_path", None)
            if measurement_path:
                route_key_from_measurement = _route_key_from_measurement_path(str(measurement_path))
                route_ids.add(_route_id_from_route_key(route_key_from_measurement))

    return route_ids


def resolve_route_batch_size(cfg) -> int:
    cfg_batch_size = int(getattr(cfg.data_module, "batch_size", 1))
    env_override = os.getenv("SIMLINGO_ROUTE_BATCH_SIZE", "").strip()
    if env_override:
        batch_size = int(env_override)
        source = f"env(SIMLINGO_ROUTE_BATCH_SIZE={env_override})"
    else:
        batch_size = cfg_batch_size
        source = f"cfg.data_module.batch_size={cfg_batch_size}"

    if batch_size <= 0:
        raise ValueError(f"route batch size must be >= 1, got {batch_size} from {source}")
    print(f"[collect] route_batch_size={batch_size} (source: {source})")
    return batch_size


@hydra.main(config_path="config", config_name="config", version_base="1.1")
def main(cfg: TrainConfig):
    torch.set_float32_matmul_precision("high")
    pl.seed_everything(cfg.seed, workers=True)
    validate_adaption_config(cfg.model.adaption, context="model.adaption")
    validate_train_sampling_config(cfg.data_module, context="data_module.train_sampling")

    # Keep data deterministic for correlation analysis.
    if hasattr(cfg.data_module.base_dataset, "img_augmentation"):
        cfg.data_module.base_dataset.img_augmentation = False
    if hasattr(cfg.data_module.base_dataset, "img_shift_augmentation"):
        cfg.data_module.base_dataset.img_shift_augmentation = False
    if hasattr(cfg.data_module.base_dataset, "commentary_augmentation"):
        cfg.data_module.base_dataset.commentary_augmentation = False
    if hasattr(cfg.data_module.base_dataset, "qa_augmentation"):
        cfg.data_module.base_dataset.qa_augmentation = False
    if hasattr(cfg.data_module.base_dataset, "use_commentary"):
        cfg.data_module.base_dataset.use_commentary = False
    if hasattr(cfg.data_module.base_dataset, "use_qa"):
        cfg.data_module.base_dataset.use_qa = False

    internvl_source = resolve_internvl_variant_source(cfg.model.vision_model.variant)
    processor = AutoProcessor.from_pretrained(internvl_source, trust_remote_code=True)
    data_module = hydra.utils.instantiate(
        cfg.data_module,
        processor=processor,
        encoder_variant=cfg.model.vision_model.variant,
        llm_variant=cfg.model.language_model.variant,
        _recursive_=False,
    )
    data_module.setup()
    train_dataset = data_module.train_base_dataset if data_module.train_base_dataset is not None else data_module.train_dataset
    if train_dataset is None:
        raise RuntimeError("No training dataset available.")

    model = hydra.utils.instantiate(
        cfg.model,
        cfg_data_module=cfg.data_module,
        processor=processor,
        cache_dir=internvl_source,
        _recursive_=False,
    )

    if cfg.checkpoint is not None:
        project_path = Path(__file__).resolve().parent.parent
        checkpoint = os.path.join(project_path, cfg.checkpoint)
        if os.path.isdir(checkpoint):
            state_dict = get_fp32_state_dict_from_zero_checkpoint(checkpoint)
        else:
            state_dict = torch.load(checkpoint, map_location="cpu")
        model.load_state_dict(state_dict, strict=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    use_autocast, autocast_dtype, autocast_reason = resolve_autocast_settings(cfg.precision, device)
    print(f"[collect] autocast enabled={use_autocast}, dtype={autocast_dtype}, reason={autocast_reason}")

    route_batch_size = resolve_route_batch_size(cfg)
    root = Path(get_original_cwd())
    records = build_sample_records(train_dataset)

    processed_jsonl_env = os.getenv("SIMLINGO_SKIP_ROUTES_FROM_JSONL", "").strip()
    if processed_jsonl_env:
        processed_jsonl_path = Path(processed_jsonl_env)
    else:
        processed_jsonl_path = root / "outputs" / "scene_difficulty" / "part_1.jsonl"

    processed_route_ids = load_processed_route_ids(processed_jsonl_path)
    if processed_route_ids:
        before_records = len(records)
        before_route_ids = {_route_id_from_route_key(record.route_key) for record in records}
        records = [record for record in records if _route_id_from_route_key(record.route_key) not in processed_route_ids]
        after_route_ids = {_route_id_from_route_key(record.route_key) for record in records}
        skipped_routes = len(before_route_ids - after_route_ids)
        skipped_records = before_records - len(records)
        print(
            f"[collect] skip processed routes from {processed_jsonl_path}: "
            f"skipped_routes={skipped_routes}, skipped_frames={skipped_records}"
        )
    else:
        print(f"[collect] no processed-route skip file used: {processed_jsonl_path}")

    if not records:
        raise RuntimeError("No records left after filtering processed routes.")

    num_frames = len(records)
    route_batches = build_route_batches(records, route_batch_size=route_batch_size)
    route_counts = defaultdict(int)
    for record in records:
        route_counts[record.route_key] += 1
    print(
        f"[collect] routes={len(route_counts)}, frames={num_frames}, "
        f"min_frames_per_route={min(route_counts.values()) if route_counts else 0}, "
        f"max_frames_per_route={max(route_counts.values()) if route_counts else 0}, "
        f"route_batch_size={route_batch_size}, scheduler=fixed_slot_parallel(route-continuous)"
    )

    save_dir = root / "outputs" / "scene_difficulty"
    save_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = save_dir / f"train_scene_metrics_{ts}.jsonl"

    with torch.no_grad(), open(save_path, "w", encoding="utf-8") as f:
        pbar = tqdm(total=num_frames, desc="Collecting metrics", unit="frame")
        route_frame_index = defaultdict(int)
        route_last_frame_id = {}
        route_started = set()
        for batch_records in route_batches:
            samples = [train_dataset[record.index] for record in batch_records]
            batch = data_module.dl_collate_fn(samples)
            batch = move_to_device(batch, device)
            route_keys = [record.route_key for record in batch_records]
            if len(set(route_keys)) != len(route_keys):
                raise RuntimeError("route scheduling violated: duplicated route in one batch.")

            autocast_ctx = (
                torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=True)
                if use_autocast
                else nullcontext()
            )
            with autocast_ctx:
                loss_dict, _, scene_metrics = model.forward_loss(
                    batch,
                    per_sample=True,
                    route_keys=route_keys,
                    return_scene_metrics=True,
                )

            per_sample_losses = per_sample_loss_scalars(loss_dict)
            loss_keys = sorted(per_sample_losses.keys())
            total_loss = torch.zeros(len(batch_records), device=device)
            for key in loss_keys:
                total_loss = total_loss + per_sample_losses[key]

            for i, record in enumerate(batch_records):
                if record.route_key not in route_started:
                    route_started.add(record.route_key)
                    print(f"[collect] route_start route={record.route_key} frames={route_counts[record.route_key]}")

                last_frame = route_last_frame_id.get(record.route_key, None)
                if last_frame is not None and record.frame_id <= last_frame:
                    raise RuntimeError(
                        f"Frame order violation in route {record.route_key}: prev={last_frame}, curr={record.frame_id}"
                    )
                route_last_frame_id[record.route_key] = record.frame_id

                frame_idx_in_route = route_frame_index[record.route_key]
                route_frame_index[record.route_key] += 1
                row = {
                    "measurement_path": to_repo_relative(record.measurement_path, root.resolve()),
                    "route_key": to_repo_relative(record.route_key, root.resolve()),
                    "route_id": Path(record.route_key).name,
                    "frame_id": record.frame_id,
                    "route_frame_index": frame_idx_in_route,
                    "route_frame_count": route_counts[record.route_key],
                    "loss_total": float(total_loss[i].detach().cpu()),
                    "novelty": metric_at(scene_metrics, ("novelty",), i),
                    "history_similarity": metric_at(scene_metrics, ("history_similarity", "sim_in"), i),
                    "used_budget": metric_at(scene_metrics, ("used_budget",), i),
                    "decision_shift_speed_e_mean": metric_at(scene_metrics, ("decision_shift", "speed_wps", "e_mean"), i),
                    "decision_shift_speed_e_norm": metric_at(scene_metrics, ("decision_shift", "speed_wps", "e_norm"), i),
                    "decision_shift_route_e_mean": metric_at(scene_metrics, ("decision_shift", "route", "e_mean"), i),
                    "decision_shift_route_e_norm": metric_at(scene_metrics, ("decision_shift", "route", "e_norm"), i),
                    "decision_shift_delta_tau": metric_at(scene_metrics, ("decision_shift", "delta_tau"), i),
                }
                for key in loss_keys:
                    row[key] = float(per_sample_losses[key][i].detach().cpu())
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            pbar.update(len(batch_records))
        pbar.close()

    print(f"Saved scene difficulty records to: {save_path}")


if __name__ == "__main__":
    main()
