import hashlib
import json
import math
import os
import pickle as pkl
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from torch.utils.data import Dataset, Sampler


@dataclass(frozen=True)
class RouteFrameRecord:
    sample_index: int
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


def _records_from_base_dataset(dataset, global_offset: int = 0) -> List[RouteFrameRecord]:
    records: List[RouteFrameRecord] = []
    for local_idx in range(len(dataset)):
        measurement_root = _decode_path(dataset.measurements[local_idx][0])
        frame_id = int(dataset.sample_start[local_idx]) + int(dataset.hist_len) - 1
        measurement_path = f"{measurement_root}/{frame_id:04}.json.gz"
        route_key = _route_key_from_measurement_path(measurement_path)
        records.append(
            RouteFrameRecord(
                sample_index=global_offset + local_idx,
                measurement_path=measurement_path,
                route_key=route_key,
                frame_id=frame_id,
            )
        )
    return records


def build_route_records(dataset, global_offset: int = 0) -> List[RouteFrameRecord]:
    if not hasattr(dataset, "measurements") or not hasattr(dataset, "sample_start") or not hasattr(dataset, "hist_len"):
        raise ValueError("route_stream expects base dataset with measurements/sample_start/hist_len")
    return _records_from_base_dataset(dataset, global_offset=global_offset)


def _compute_route_set_hash(route_to_frame_ids: Dict[str, List[int]]) -> str:
    rows = []
    for route_key in sorted(route_to_frame_ids.keys()):
        frame_ids = route_to_frame_ids[route_key]
        rows.append(f"{route_key}::{len(frame_ids)}::{frame_ids[0]}::{frame_ids[-1]}")
    joined = "\n".join(rows)
    return hashlib.sha1(joined.encode("utf-8")).hexdigest()


def _validate_bucket_priority(bucket_priority: List[str], partitions: Dict[str, float]) -> None:
    non_all = [k for k in partitions.keys() if k != "all"]
    if len(non_all) == 0:
        if len(bucket_priority) != 0:
            raise ValueError(
                f"bucket_priority must be empty when partitions has only 'all', got {bucket_priority}"
            )
        return
    if len(bucket_priority) == 0:
        raise ValueError("bucket_priority must be non-empty when non-'all' partitions exist")
    if len(set(bucket_priority)) != len(bucket_priority):
        raise ValueError(f"bucket_priority has duplicates: {bucket_priority}")
    missing = sorted([k for k in non_all if k not in bucket_priority])
    if missing:
        raise ValueError(f"bucket_priority missing non-'all' buckets: {missing}")


def _resolve_primary_bucket(
    route_len: int,
    bucket_hit_counts: Dict[str, int],
    bucket_priority: List[str],
) -> str:
    if route_len <= 0:
        raise ValueError(f"route_len must be > 0, got {route_len}")

    non_all_candidates = []
    for bucket_name, hit_count in bucket_hit_counts.items():
        if bucket_name == "all":
            continue
        if int(hit_count) > 0:
            non_all_candidates.append(bucket_name)

    if len(non_all_candidates) == 0:
        return "all"

    ratios = {
        bucket_name: float(bucket_hit_counts[bucket_name]) / float(route_len)
        for bucket_name in non_all_candidates
    }
    max_ratio = max(ratios.values())
    tied = [
        bucket_name
        for bucket_name, ratio in ratios.items()
        if math.isclose(ratio, max_ratio, rel_tol=0.0, abs_tol=1e-12)
    ]
    if len(tied) == 1:
        return tied[0]

    for bucket_name in bucket_priority:
        if bucket_name in tied:
            return bucket_name
    raise ValueError(
        f"unable to resolve primary bucket tie: tied={tied}, priority={bucket_priority}"
    )


def _resolve_bucket_run_ids_from_bucket_dict(
    bucket_dict: Dict[str, List[str]],
    bucket_name: str,
) -> List[str]:
    # Keep this mapping identical to BaseDataset bucket selection semantics.
    if bucket_name == "acceleration_negative_5":
        return list(bucket_dict["acceleration_-5"])
    if bucket_name == "acceleration_negative_1":
        return list(bucket_dict["acceleration_-1"])
    if bucket_name == "acceleration_positive_1":
        return list(bucket_dict["acceleration_5"])
    if bucket_name == "acceleration_positive_5":
        return list(bucket_dict["acceleration_20"])
    if bucket_name == "lateral_control_1":
        return list(bucket_dict["lateral_control_1"])
    if bucket_name == "lateral_control_1_2":
        return list(bucket_dict["lateral_control_1"]) + list(bucket_dict["lateral_control_2"])
    if bucket_name == "lateral_control_high":
        return (
            list(bucket_dict["lateral_control_2"])
            + list(bucket_dict["lateral_control_5"])
            + list(bucket_dict["lateral_control_1000000"])
        )
    if bucket_name == "lateral_control_higher_5":
        return list(bucket_dict["lateral_control_5"]) + list(bucket_dict["lateral_control_1000000"])
    if bucket_name == "recovery":
        return list(bucket_dict["recovery_data_small"]) + list(bucket_dict["recovery_data_large"])
    if bucket_name not in bucket_dict:
        raise ValueError(f"Bucket name {bucket_name} not found in buckets_paths.pkl")
    return list(bucket_dict[bucket_name])


def _resolve_measurement_path_from_bucket_run_id(
    run_id: str,
    bucket_path: str,
    repo_root: str,
) -> str:
    run_id_rewritten = str(run_id).replace("database/simlingo_v2_2025_01_10", str(bucket_path))
    run_id_path = Path(run_id_rewritten)
    if not run_id_path.is_absolute():
        run_id_path = Path(repo_root) / run_id_path
    return str(run_id_path.resolve())


def _build_bucket_hit_counts_from_manifest(
    measurement_to_route: Dict[str, str],
    route_to_sample_indices: Dict[str, List[int]],
    route_to_len: Dict[str, int],
    train_partitions: Dict[str, float],
    bucket_path: str,
    repo_root: str,
) -> Dict[str, Dict[str, int]]:
    bucket_root = Path(bucket_path).expanduser()
    if not bucket_root.is_absolute():
        bucket_root = Path(repo_root) / bucket_root
    buckets_pkl = bucket_root.resolve() / "buckets_paths.pkl"
    if not buckets_pkl.exists():
        raise ValueError(f"route_stream bucket manifest not found: {buckets_pkl}")
    if not buckets_pkl.is_file():
        raise ValueError(f"route_stream bucket manifest must be a file: {buckets_pkl}")

    with open(buckets_pkl, "rb") as f:
        bucket_dict = pkl.load(f)
    if not isinstance(bucket_dict, dict):
        raise ValueError(f"Invalid buckets_paths.pkl format, expected dict: {buckets_pkl}")

    route_to_bucket_hit_counts: Dict[str, Dict[str, int]] = {
        route_key: {bucket_name: 0 for bucket_name in train_partitions.keys()}
        for route_key in route_to_sample_indices.keys()
    }
    for route_key, route_len in route_to_len.items():
        route_to_bucket_hit_counts[route_key]["all"] = int(route_len)

    for bucket_name in train_partitions.keys():
        if bucket_name == "all":
            continue
        bucket_run_ids = _resolve_bucket_run_ids_from_bucket_dict(
            bucket_dict=bucket_dict,
            bucket_name=bucket_name,
        )
        hit_count = 0
        miss_count = 0
        for run_id in set(bucket_run_ids):
            measurement_path = _resolve_measurement_path_from_bucket_run_id(
                run_id=run_id,
                bucket_path=bucket_path,
                repo_root=repo_root,
            )
            route_key = measurement_to_route.get(measurement_path, None)
            if route_key is None:
                miss_count += 1
                continue
            route_to_bucket_hit_counts[route_key][bucket_name] += 1
            hit_count += 1
        print(
            f"[route_stream] bucket_manifest bucket={bucket_name} "
            f"hits={hit_count} misses={miss_count}"
        )
    return route_to_bucket_hit_counts


def build_or_load_route_bucket_index(
    all_dataset,
    bucket_datasets: Optional[Dict[str, object]],
    train_partitions: Dict[str, float],
    bucket_priority: List[str],
    index_cache_file: str,
    rebuild_index_cache: bool,
    strict_frame_monotonic: bool,
    bucket_path: Optional[str] = None,
    repo_root: Optional[str] = None,
) -> Tuple[Dict[str, List[int]], Dict[str, str], Dict[str, int]]:
    if not isinstance(train_partitions, dict) or len(train_partitions) == 0:
        raise ValueError("train_partitions must be a non-empty dict")
    if bucket_datasets is not None:
        if not isinstance(bucket_datasets, dict) or len(bucket_datasets) == 0:
            raise ValueError("bucket_datasets must be a non-empty dict when provided")
        if "all" not in bucket_datasets:
            raise ValueError("bucket_datasets must include 'all' when provided")
        missing_buckets = sorted([k for k in train_partitions.keys() if k not in bucket_datasets])
        if missing_buckets:
            raise ValueError(
                f"train_partitions buckets are missing in bucket_datasets: {missing_buckets}"
            )

    _validate_bucket_priority(bucket_priority=bucket_priority, partitions=train_partitions)

    all_records = build_route_records(all_dataset)
    if len(all_records) == 0:
        raise ValueError("route_stream index build found zero training records in all_dataset")

    measurement_to_route: Dict[str, str] = {}
    route_to_pairs: Dict[str, List[Tuple[int, int]]] = {}
    route_to_frame_ids: Dict[str, List[int]] = {}
    for rec in all_records:
        if rec.measurement_path in measurement_to_route:
            raise ValueError(
                f"duplicated measurement_path in all_dataset: {rec.measurement_path}"
            )
        measurement_to_route[rec.measurement_path] = rec.route_key
        route_to_pairs.setdefault(rec.route_key, []).append((rec.frame_id, rec.sample_index))
        route_to_frame_ids.setdefault(rec.route_key, []).append(rec.frame_id)

    route_to_sample_indices: Dict[str, List[int]] = {}
    route_to_len: Dict[str, int] = {}
    for route_key, pairs in route_to_pairs.items():
        pairs_sorted = sorted(pairs, key=lambda x: x[0])
        frame_ids = [frame_id for frame_id, _ in pairs_sorted]
        if strict_frame_monotonic:
            for idx in range(1, len(frame_ids)):
                if frame_ids[idx] <= frame_ids[idx - 1]:
                    raise ValueError(
                        f"route frame order violation in {route_key}: {frame_ids[idx - 1]} -> {frame_ids[idx]}"
                    )
        route_to_sample_indices[route_key] = [sample_index for _, sample_index in pairs_sorted]
        route_to_len[route_key] = len(pairs_sorted)

    route_set_hash = _compute_route_set_hash(route_to_frame_ids)
    cache_path = Path(index_cache_file)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    route_to_primary_bucket: Dict[str, str] = {}
    if cache_path.exists() and (not rebuild_index_cache):
        with open(cache_path, "r", encoding="utf-8") as f:
            cached = json.load(f)
        if not isinstance(cached, dict):
            raise ValueError(f"invalid route index cache format in {cache_path}")
        metadata = cached.get("metadata", None)
        if not isinstance(metadata, dict):
            raise ValueError(f"route index cache missing metadata: {cache_path}")
        cached_hash = metadata.get("route_set_hash", None)
        if cached_hash != route_set_hash:
            raise ValueError(
                f"route index cache route_set_hash mismatch for {cache_path}: "
                f"cached={cached_hash}, current={route_set_hash}. "
                "Set train_sampling.rebuild_index_cache=true to rebuild."
            )
        rows = cached.get("routes", None)
        if not isinstance(rows, list):
            raise ValueError(f"route index cache missing routes list: {cache_path}")
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError(f"invalid route row in cache {cache_path}: {row}")
            route_key = row.get("route_key", None)
            primary_bucket = row.get("primary_bucket", None)
            if route_key is None or primary_bucket is None:
                raise ValueError(f"route row missing route_key/primary_bucket in {cache_path}: {row}")
            route_to_primary_bucket[str(route_key)] = str(primary_bucket)

        missing_routes = sorted([route_key for route_key in route_to_sample_indices.keys() if route_key not in route_to_primary_bucket])
        extra_routes = sorted([route_key for route_key in route_to_primary_bucket.keys() if route_key not in route_to_sample_indices])
        if missing_routes or extra_routes:
            raise ValueError(
                f"route cache route set mismatch in {cache_path}: "
                f"missing_routes={missing_routes[:10]}, extra_routes={extra_routes[:10]}"
            )
    else:
        if bucket_datasets is None:
            if bucket_path is None or str(bucket_path).strip() == "":
                raise ValueError("bucket_path is required when bucket_datasets is None")
            if repo_root is None or str(repo_root).strip() == "":
                raise ValueError("repo_root is required when bucket_datasets is None")
            route_to_bucket_hit_counts = _build_bucket_hit_counts_from_manifest(
                measurement_to_route=measurement_to_route,
                route_to_sample_indices=route_to_sample_indices,
                route_to_len=route_to_len,
                train_partitions=train_partitions,
                bucket_path=str(bucket_path),
                repo_root=str(repo_root),
            )
        else:
            route_to_bucket_hit_counts = {
                route_key: {bucket_name: 0 for bucket_name in train_partitions.keys()}
                for route_key in route_to_sample_indices.keys()
            }
            for route_key, route_len in route_to_len.items():
                route_to_bucket_hit_counts[route_key]["all"] = int(route_len)

            for bucket_name, bucket_dataset in bucket_datasets.items():
                if bucket_name == "all":
                    continue
                bucket_records = build_route_records(bucket_dataset)
                for rec in bucket_records:
                    if rec.measurement_path not in measurement_to_route:
                        raise ValueError(
                            f"bucket dataset '{bucket_name}' contains measurement not in all dataset: {rec.measurement_path}"
                        )
                    route_key = measurement_to_route[rec.measurement_path]
                    route_to_bucket_hit_counts[route_key][bucket_name] += 1

        rows = []
        for route_key in sorted(route_to_sample_indices.keys()):
            route_len = int(route_to_len[route_key])
            bucket_hit_counts = route_to_bucket_hit_counts[route_key]
            primary_bucket = _resolve_primary_bucket(
                route_len=route_len,
                bucket_hit_counts=bucket_hit_counts,
                bucket_priority=bucket_priority,
            )
            route_to_primary_bucket[route_key] = primary_bucket
            bucket_hit_ratios = {
                bucket_name: float(hit_count) / float(route_len)
                for bucket_name, hit_count in bucket_hit_counts.items()
            }
            rows.append(
                {
                    "route_key": route_key,
                    "route_len": route_len,
                    "primary_bucket": primary_bucket,
                    "bucket_hit_counts": bucket_hit_counts,
                    "bucket_hit_ratios": bucket_hit_ratios,
                }
            )

        payload = {
            "metadata": {
                "version": 1,
                "route_set_hash": route_set_hash,
                "num_routes": len(route_to_sample_indices),
                "num_frames": len(all_records),
                "partitions": train_partitions,
                "bucket_priority": bucket_priority,
            },
            "routes": rows,
        }
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    route_primary_set = set(route_to_primary_bucket.values())
    allowed_buckets = set(train_partitions.keys())
    invalid_buckets = sorted([bucket_name for bucket_name in route_primary_set if bucket_name not in allowed_buckets])
    if invalid_buckets:
        raise ValueError(
            f"route primary buckets not found in train_partitions: {invalid_buckets}"
        )
    return route_to_sample_indices, route_to_primary_bucket, route_to_len


def _allocate_bucket_quotas(total_routes: int, bucket_weights: Dict[str, float]) -> Dict[str, int]:
    if total_routes <= 0:
        raise ValueError(f"total_routes must be > 0, got {total_routes}")
    if not isinstance(bucket_weights, dict) or len(bucket_weights) == 0:
        raise ValueError("bucket_weights must be a non-empty dict")

    items = sorted(bucket_weights.items(), key=lambda x: x[0])
    weight_sum = 0.0
    for _, weight in items:
        weight_f = float(weight)
        if weight_f <= 0.0:
            raise ValueError(f"bucket weight must be > 0, got {weight_f}")
        weight_sum += weight_f

    float_targets = {bucket_name: float(weight) / weight_sum * float(total_routes) for bucket_name, weight in items}
    quotas = {bucket_name: int(math.floor(target)) for bucket_name, target in float_targets.items()}
    assigned = sum(quotas.values())
    remaining = total_routes - assigned

    remainders = sorted(
        [(bucket_name, float_targets[bucket_name] - quotas[bucket_name]) for bucket_name, _ in items],
        key=lambda x: (-x[1], x[0]),
    )
    for idx in range(remaining):
        bucket_name = remainders[idx][0]
        quotas[bucket_name] += 1
    if sum(quotas.values()) != total_routes:
        raise RuntimeError("bucket quota allocation failed to preserve total_routes")
    return quotas


class RouteStreamBatchSampler(Sampler[List[Tuple[int, int]]]):
    def __init__(
        self,
        route_to_sample_indices: Dict[str, List[int]],
        route_to_primary_bucket: Dict[str, str],
        route_to_len: Dict[str, int],
        bucket_weights: Dict[str, float],
        local_batch_size: int,
        rank: int,
        world_size: int,
        route_shuffle_each_epoch: bool,
        route_shuffle_seed: int,
        length_balancing: bool,
        deficit_alpha: float,
        deficit_eps: float,
        drop_last: bool,
        strict_bucket_nonempty: bool,
    ) -> None:
        if local_batch_size <= 0:
            raise ValueError(f"local_batch_size must be > 0, got {local_batch_size}")
        if int(world_size) <= 0:
            raise ValueError(f"world_size must be > 0, got {world_size}")
        if int(rank) < 0 or int(rank) >= int(world_size):
            raise ValueError(f"rank must be in [0, {int(world_size)-1}], got rank={rank}")
        if not isinstance(route_to_sample_indices, dict) or len(route_to_sample_indices) == 0:
            raise ValueError("route_to_sample_indices must be a non-empty dict")
        if set(route_to_sample_indices.keys()) != set(route_to_primary_bucket.keys()):
            raise ValueError("route_to_sample_indices and route_to_primary_bucket route sets must match")
        if set(route_to_sample_indices.keys()) != set(route_to_len.keys()):
            raise ValueError("route_to_sample_indices and route_to_len route sets must match")
        if float(deficit_alpha) < 0.0 or float(deficit_alpha) > 1.0:
            raise ValueError(f"deficit_alpha must be in [0, 1], got {deficit_alpha}")
        if float(deficit_eps) <= 0.0:
            raise ValueError(f"deficit_eps must be > 0, got {deficit_eps}")
        if not bool(drop_last):
            raise ValueError("route_stream sampler requires drop_last=true to keep batch size constant")
        if "all" not in bucket_weights:
            raise ValueError("bucket_weights must include 'all'")

        for route_key, sample_indices in route_to_sample_indices.items():
            if len(sample_indices) == 0:
                raise ValueError(f"route has empty frame list: {route_key}")
        for route_key, route_len in route_to_len.items():
            if int(route_len) != len(route_to_sample_indices[route_key]):
                raise ValueError(
                    f"route_to_len mismatch for {route_key}: {route_len} vs {len(route_to_sample_indices[route_key])}"
                )
        for route_key, bucket_name in route_to_primary_bucket.items():
            if bucket_name not in bucket_weights:
                raise ValueError(
                    f"route primary bucket '{bucket_name}' for route '{route_key}' is missing in bucket_weights"
                )

        self.route_to_sample_indices = route_to_sample_indices
        self.route_to_primary_bucket = route_to_primary_bucket
        self.route_to_len = route_to_len
        self.bucket_weights = {k: float(v) for k, v in bucket_weights.items()}
        self.local_batch_size = int(local_batch_size)
        self.rank = int(rank)
        self.world_size = int(world_size)
        # shape policy: global batch = local batch * world size
        self.global_batch_size = self.local_batch_size * self.world_size
        self.route_shuffle_each_epoch = bool(route_shuffle_each_epoch)
        self.route_shuffle_seed = int(route_shuffle_seed)
        self.length_balancing = bool(length_balancing)
        self.deficit_alpha = float(deficit_alpha)
        self.deficit_eps = float(deficit_eps)
        self.strict_bucket_nonempty = bool(strict_bucket_nonempty)
        self._epoch = 0
        self._planned_epoch: Optional[int] = None
        # Each element contains this rank's local sample indices.
        self._planned_rank_batches: Optional[List[List[int]]] = None
        self._planned_stats: Optional[Dict[str, object]] = None

        self._total_frames = sum(len(v) for v in self.route_to_sample_indices.values())
        self._total_routes = len(self.route_to_sample_indices)
        if self._total_routes < self.global_batch_size:
            raise ValueError(
                f"route_stream requires num_routes >= global_batch_size, got num_routes={self._total_routes}, "
                f"global_batch_size={self.global_batch_size} (local_batch_size={self.local_batch_size}, world_size={self.world_size})"
            )
        if self._total_frames < self.global_batch_size:
            raise ValueError(
                f"route_stream requires total_frames >= global_batch_size, got total_frames={self._total_frames}, "
                f"global_batch_size={self.global_batch_size}"
            )

    def __len__(self) -> int:
        # Length and iteration share one epoch plan to keep progress exact.
        rank_batches, _ = self._get_epoch_plan(self._epoch)
        return len(rank_batches)

    def set_epoch(self, epoch: int) -> None:
        if int(epoch) < 0:
            raise ValueError(f"epoch must be >= 0, got {epoch}")
        epoch_int = int(epoch)
        if self._epoch != epoch_int:
            self._epoch = epoch_int
            self._clear_epoch_plan_cache()

    def _clear_epoch_plan_cache(self) -> None:
        self._planned_epoch = None
        self._planned_rank_batches = None
        self._planned_stats = None

    def _epoch_seed(self, epoch: int) -> int:
        return self.route_shuffle_seed + (epoch if self.route_shuffle_each_epoch else 0)

    def _build_epoch_route_list(self, rng: random.Random) -> List[str]:
        bucket_to_routes: Dict[str, List[str]] = {bucket_name: [] for bucket_name in self.bucket_weights.keys()}
        for route_key, bucket_name in self.route_to_primary_bucket.items():
            bucket_to_routes[bucket_name].append(route_key)

        for bucket_name, routes in bucket_to_routes.items():
            if self.route_shuffle_each_epoch:
                rng.shuffle(routes)
            else:
                routes.sort()
            bucket_to_routes[bucket_name] = routes

        bucket_quotas = _allocate_bucket_quotas(total_routes=self._total_routes, bucket_weights=self.bucket_weights)

        selected_routes: List[str] = []
        selected_set = set()
        for bucket_name in sorted(bucket_quotas.keys()):
            target = bucket_quotas[bucket_name]
            routes = bucket_to_routes[bucket_name]
            take = min(target, len(routes))
            for route_key in routes[:take]:
                if route_key in selected_set:
                    continue
                selected_set.add(route_key)
                selected_routes.append(route_key)

        # all_then_global is the only supported quota policy.
        if len(selected_set) < self._total_routes:
            all_routes = bucket_to_routes.get("all", [])
            for route_key in all_routes:
                if route_key in selected_set:
                    continue
                selected_set.add(route_key)
                selected_routes.append(route_key)
                if len(selected_set) == self._total_routes:
                    break

        if len(selected_set) < self._total_routes:
            remaining_routes = sorted([route_key for route_key in self.route_to_sample_indices.keys() if route_key not in selected_set])
            if self.route_shuffle_each_epoch:
                rng.shuffle(remaining_routes)
            for route_key in remaining_routes:
                selected_set.add(route_key)
                selected_routes.append(route_key)

        if len(selected_set) != self._total_routes:
            raise RuntimeError(
                f"route selection failed: selected={len(selected_set)}, total={self._total_routes}"
            )
        return selected_routes

    def _build_remaining_routes_by_bucket(
        self,
        routes: List[str],
        rng: random.Random,
    ) -> Dict[str, List[str]]:
        remaining_routes_by_bucket: Dict[str, List[str]] = {bucket_name: [] for bucket_name in self.bucket_weights.keys()}
        for route_key in routes:
            bucket_name = self.route_to_primary_bucket[route_key]
            remaining_routes_by_bucket[bucket_name].append(route_key)

        for bucket_name, route_keys in remaining_routes_by_bucket.items():
            if self.route_shuffle_each_epoch:
                rng.shuffle(route_keys)
            else:
                route_keys.sort()
            if self.length_balancing:
        # Schedule long routes first to reduce tail imbalance.
                route_keys.sort(key=lambda key: self.route_to_len[key])
            remaining_routes_by_bucket[bucket_name] = route_keys
        return remaining_routes_by_bucket

    @staticmethod
    def _sample_bucket_by_prob(prob_by_bucket: Dict[str, float], rng: random.Random) -> str:
        if not prob_by_bucket:
            raise ValueError("prob_by_bucket must be non-empty")
        ordered = sorted(prob_by_bucket.items(), key=lambda x: x[0])
        r = rng.random()
        cumulative = 0.0
        for bucket_name, prob in ordered:
            cumulative += prob
            if r <= cumulative + 1e-12:
                return bucket_name
        return ordered[-1][0]

    def _draw_next_route_key(
        self,
        remaining_routes_by_bucket: Dict[str, List[str]],
        selected_count_by_bucket: Dict[str, int],
        selected_total_routes: int,
        rng: random.Random,
    ) -> Optional[str]:
        available_buckets = [bucket_name for bucket_name, routes in remaining_routes_by_bucket.items() if len(routes) > 0]
        if len(available_buckets) == 0:
            return None

        weight_sum = sum(float(self.bucket_weights[bucket_name]) for bucket_name in available_buckets)
        if weight_sum <= 0.0:
            raise RuntimeError("available bucket weights sum must be > 0")
        p_weight = {bucket_name: float(self.bucket_weights[bucket_name]) / weight_sum for bucket_name in available_buckets}

        target_total = float(selected_total_routes + 1)
        deficit = {
            bucket_name: max(target_total * p_weight[bucket_name] - float(selected_count_by_bucket[bucket_name]), 0.0)
            for bucket_name in available_buckets
        }
        deficit_sum = sum(deficit.values())
        if deficit_sum <= self.deficit_eps:
            p_deficit = dict(p_weight)
        else:
            p_deficit = {bucket_name: deficit[bucket_name] / deficit_sum for bucket_name in available_buckets}

        p_final = {
            bucket_name: (1.0 - self.deficit_alpha) * p_weight[bucket_name] + self.deficit_alpha * p_deficit[bucket_name]
            for bucket_name in available_buckets
        }
        p_final_sum = sum(p_final.values())
        if p_final_sum <= 0.0:
            raise RuntimeError("final bucket probability sum must be > 0")
        p_final = {bucket_name: p_final[bucket_name] / p_final_sum for bucket_name in available_buckets}

        chosen_bucket = self._sample_bucket_by_prob(p_final, rng=rng)
        chosen_routes = remaining_routes_by_bucket[chosen_bucket]
        if len(chosen_routes) == 0:
            raise RuntimeError(f"chosen bucket has no remaining routes: {chosen_bucket}")
        route_key = chosen_routes.pop()
        selected_count_by_bucket[chosen_bucket] += 1
        return route_key

    def _build_epoch_plan(self, epoch: int) -> Tuple[List[List[int]], Dict[str, object]]:
        rng = random.Random(self._epoch_seed(epoch))

        selected_routes = self._build_epoch_route_list(rng=rng)
        remaining_routes_by_bucket = self._build_remaining_routes_by_bucket(routes=selected_routes, rng=rng)

        if self.strict_bucket_nonempty:
            empty_buckets = sorted(
                [
                    bucket_name for bucket_name, weight in self.bucket_weights.items()
                    if bucket_name != "all" and float(weight) > 0.0 and len(remaining_routes_by_bucket[bucket_name]) == 0
                ]
            )
            if empty_buckets:
                raise ValueError(
                    f"route_stream strict_bucket_nonempty violation: these non-'all' buckets have zero routes: {empty_buckets}"
                )

        active_route_by_slot: List[Optional[str]] = [None for _ in range(self.global_batch_size)]
        frame_cursor_by_slot = [0 for _ in range(self.global_batch_size)]
        selected_count_by_bucket: Dict[str, int] = {bucket_name: 0 for bucket_name in self.bucket_weights.keys()}
        selected_total_routes = 0

        for slot_id in range(self.global_batch_size):
            route_key = self._draw_next_route_key(
                remaining_routes_by_bucket=remaining_routes_by_bucket,
                selected_count_by_bucket=selected_count_by_bucket,
                selected_total_routes=selected_total_routes,
                rng=rng,
            )
            if route_key is None:
                continue
            active_route_by_slot[slot_id] = route_key
            selected_total_routes += 1

        rank_batches: List[List[int]] = []
        yielded_batches = 0
        yielded_frames = 0
        stop_due_to_partial_batch = False
        while True:
            has_active = any(route_key is not None for route_key in active_route_by_slot)
            if not has_active:
                break

            # shape: (global_batch_size,)
            global_batch_sample_indices: List[int] = []
            for slot_id in range(self.global_batch_size):
                route_key = active_route_by_slot[slot_id]
                if route_key is None:
                    continue

                frame_cursor = frame_cursor_by_slot[slot_id]
                route_indices = self.route_to_sample_indices[route_key]
                if frame_cursor >= len(route_indices):
                    raise RuntimeError(
                        f"frame cursor out of range for slot={slot_id}, route={route_key}, cursor={frame_cursor}"
                    )

                sample_index = route_indices[frame_cursor]
                global_batch_sample_indices.append(sample_index)
                frame_cursor += 1

                if frame_cursor < len(route_indices):
                    frame_cursor_by_slot[slot_id] = frame_cursor
                    continue

                next_route_key = self._draw_next_route_key(
                    remaining_routes_by_bucket=remaining_routes_by_bucket,
                    selected_count_by_bucket=selected_count_by_bucket,
                    selected_total_routes=selected_total_routes,
                    rng=rng,
                )
                if next_route_key is None:
                    active_route_by_slot[slot_id] = None
                    frame_cursor_by_slot[slot_id] = 0
                    continue

                active_route_by_slot[slot_id] = next_route_key
                frame_cursor_by_slot[slot_id] = 0
                selected_total_routes += 1

            if len(global_batch_sample_indices) == 0:
                break
            if len(global_batch_sample_indices) < self.global_batch_size:
                stop_due_to_partial_batch = True
                break

            rank_start = self.rank * self.local_batch_size
            rank_end = rank_start + self.local_batch_size
            if len(global_batch_sample_indices) != self.global_batch_size:
                raise RuntimeError(
                    f"global batch size mismatch: expected {self.global_batch_size}, got {len(global_batch_sample_indices)}"
                )
            rank_batch_sample_indices = global_batch_sample_indices[rank_start:rank_end]
            if len(rank_batch_sample_indices) != self.local_batch_size:
                raise RuntimeError(
                    f"rank batch slice mismatch: expected {self.local_batch_size}, got {len(rank_batch_sample_indices)}"
                )

            rank_batches.append(rank_batch_sample_indices)
            yielded_batches += 1
            yielded_frames += len(global_batch_sample_indices)

        dropped_tail_frames = int(self._total_frames - yielded_frames)
        bucket_actual = {
            bucket_name: (
                float(selected_count_by_bucket[bucket_name]) / float(selected_total_routes)
                if selected_total_routes > 0 else 0.0
            )
            for bucket_name in sorted(self.bucket_weights.keys())
        }
        bucket_target = {
            bucket_name: float(self.bucket_weights[bucket_name]) / float(sum(self.bucket_weights.values()))
            for bucket_name in sorted(self.bucket_weights.keys())
        }

        plan_stats: Dict[str, object] = {
            "epoch": epoch,
            "yielded_batches": yielded_batches,
            "yielded_frames": yielded_frames,
            "dropped_tail_frames": dropped_tail_frames,
            "dropped_ratio": float(dropped_tail_frames) / float(self._total_frames),
            "stop_due_to_partial_batch": stop_due_to_partial_batch,
            "target_route_ratio": bucket_target,
            "actual_route_ratio": bucket_actual,
        }
        return rank_batches, plan_stats

    def _get_epoch_plan(self, epoch: int) -> Tuple[List[List[int]], Dict[str, object]]:
        if self._planned_epoch != epoch:
            rank_batches, plan_stats = self._build_epoch_plan(epoch)
            self._planned_epoch = epoch
            self._planned_rank_batches = rank_batches
            self._planned_stats = plan_stats

        if self._planned_rank_batches is None or self._planned_stats is None:
            raise RuntimeError("route_stream epoch plan cache is unexpectedly empty")
        return self._planned_rank_batches, self._planned_stats

    def __iter__(self) -> Iterable[List[Tuple[int, int]]]:
        epoch = self._epoch
        rank_batches, plan_stats = self._get_epoch_plan(epoch)
        for rank_batch_sample_indices in rank_batches:
            # shape: (local_batch_size,)
            rank_batch = [
                (local_slot_id, sample_index)
                for local_slot_id, sample_index in enumerate(rank_batch_sample_indices)
            ]
            yield rank_batch

        if self.rank == 0:
            print(
                f"[route_stream] epoch={int(plan_stats['epoch'])} yielded_batches={int(plan_stats['yielded_batches'])} "
                f"yielded_frames={int(plan_stats['yielded_frames'])} dropped_tail_frames={int(plan_stats['dropped_tail_frames'])} "
                f"dropped_ratio={float(plan_stats['dropped_ratio']):.6f} "
                f"stop_due_to_partial_batch={bool(plan_stats['stop_due_to_partial_batch'])}"
            )
            print(f"[route_stream] target_route_ratio={plan_stats['target_route_ratio']}")
            print(f"[route_stream] actual_route_ratio={plan_stats['actual_route_ratio']}")

        self._epoch = epoch + 1
        self._clear_epoch_plan_cache()


class SlotIndexedDataset(Dataset):
    def __init__(self, base_dataset: Dataset) -> None:
        if base_dataset is None:
            raise ValueError("base_dataset is required")
        self.base_dataset = base_dataset

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index):
        if not isinstance(index, tuple) or len(index) != 2:
            raise ValueError(
                f"route_stream dataset index must be (slot_id, sample_index), got {index}"
            )
        slot_id, sample_index = index
        sample = self.base_dataset[int(sample_index)]
        if not hasattr(sample, "_replace"):
            raise ValueError("route_stream expects DatasetOutput-like namedtuple samples")
        return sample._replace(slot_id=int(slot_id))
