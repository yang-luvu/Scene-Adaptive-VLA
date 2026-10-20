# Standard library imports
import itertools
import hashlib
import json
import os
from pathlib import Path
from typing import Dict, List

# Third-party imports
import hydra
import numpy as np
import torch
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, ListConfig
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, ConcatDataset, Subset
from transformers import AutoProcessor

# Local/project specific imports
from simlingo_adaption_training.dataloader.route_stream import (
    RouteStreamBatchSampler,
    SlotIndexedDataset,
    build_or_load_route_bucket_index,
)
from simlingo_adaption_training.utils.custom_types import DrivingExample, DrivingInput, DrivingLabel, LanguageLabel
from simlingo_adaption_training.utils.internvl2_utils import (
    get_custom_chat_template,
    get_num_image_tokens_per_patch,
    preprocess_image_batch,
    resolve_internvl_variant_source,
)
from simlingo_adaption_training.utils.projection import get_camera_intrinsics, get_camera_extrinsics
from simlingo_adaption_training.utils.stage2_full_budget_losses import (
    load_stage2_full_budget_losses_index_fast,
)


def _is_mapping_like(value) -> bool:
    return isinstance(value, (dict, DictConfig))


def _is_list_like(value) -> bool:
    return isinstance(value, (list, ListConfig))


def _decode_path(value) -> str:
    if isinstance(value, (bytes, bytearray, np.bytes_)):
        return value.decode("utf-8")
    return str(value)

def encode_uint8(strings: List[str], common_length: int) -> torch.Tensor:
    max_len = max(len(s) for s in strings)
    assert max_len <= common_length, f"String is too long: {max_len} > {common_length}"
    padded_strings = [s.ljust(common_length, '\0') for s in strings]
    return torch.tensor([bytearray(s, 'utf-8') for s in padded_strings], dtype=torch.uint8)


class DataModule(LightningDataModule):
    def __init__(
        self,
        base_dataset,
        processor,
        predict=False,
        expected_world_size=1,
        **cfg,
    ):
        super().__init__()
        for key, value in cfg.items():
            setattr(self, key, value)
            
        for key, value in base_dataset.items():
            setattr(self, key, value)
            
        self.cfg = cfg
        self.base_dataset = base_dataset
        self.processor = processor
        self.predict = predict
        self.expected_world_size = int(expected_world_size)
        self.train_base_dataset = None
        self.train_batch_sampler = None
        self.sampler_train = None
        self.train_sampling_mode = ""
        self.val_drop_last = True
        try:
            self.workspace_root = Path(get_original_cwd()).resolve()
        except Exception:
            self.workspace_root = Path.cwd().resolve()

        self._full_budget_loss_index: Dict[str, float] = {}
        self._full_budget_all_jsonl: str = ""
        
        self.printed = False

        self.NUM_IMAGE_PATCHES = 2
        self.IMAGES_TO_CONSIDER = ['image_ff'] # front-forward image, other images are not supported
        # taken from:
        # https://github.com/OpenGVLab/InternVL/blob/9d3a709b16874e73ffdd38b9cf53296fae4589b9/internvl_chat/internvl/train/constants.py#L7
        # https://github.com/OpenGVLab/InternVL/blob/9d3a709b16874e73ffdd38b9cf53296fae4589b9/internvl_chat/internvl/model/internvl_chat/modeling_internvl_chat.py#L294
        self.IMG_START_TOKEN='<img>'
        self.IMG_END_TOKEN='</img>'
        self.IMG_CONTEXT_TOKEN='<IMG_CONTEXT>'

        self.num_image_tokens_per_patch = get_num_image_tokens_per_patch(self.encoder_variant)
        self.num_image_tokens_total = self.num_image_tokens_per_patch * self.NUM_IMAGE_PATCHES
            
        # add <WAYPOINT> token
        if 'tokenizer' in self.processor.__dict__:
            self.tokenizer = self.processor.tokenizer
        else:
            self.tokenizer = self.processor
        # TODO: not needed anymore?
        self.tokenizer.add_special_tokens({'additional_special_tokens': ['<WAYPOINTS>','<WAYPOINTS_DIFF>', '<ORG_WAYPOINTS_DIFF>', '<ORG_WAYPOINTS>', '<WAYPOINT_LAST>', '<ROUTE>', '<ROUTE_DIFF>', '<TARGET_POINT>']})
        self.tokenizer.padding_side = "left"

        self._init_full_budget_loss_index()

    def _resolve_measurement_key(self, measurement_path: str) -> str:
        if measurement_path is None:
            raise ValueError("measurement_path is required for full-budget loss lookup")
        raw_path = str(measurement_path).strip()
        if raw_path == "":
            raise ValueError("measurement_path must be non-empty for full-budget loss lookup")
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            path = self.workspace_root / path
        return str(path.resolve())

    def _init_full_budget_loss_index(self) -> None:
        jsonl_path = getattr(self, "full_budget_losses_jsonl", None)
        if jsonl_path is None or str(jsonl_path).strip() == "":
            self._full_budget_loss_index = {}
            self._full_budget_all_jsonl = ""
            return

        loaded = load_stage2_full_budget_losses_index_fast(
            all_jsonl_path=str(jsonl_path),
            workspace_root=Path(str(self.data_path)).expanduser().resolve(),
        )
        self._full_budget_loss_index = dict(loaded.measurement_to_loss_driving_full)
        self._full_budget_all_jsonl = str(loaded.all_jsonl_path)
        print(
            "[full_budget_losses] "
            f"loaded {loaded.num_samples} rows from {self._full_budget_all_jsonl} "
            f"(mean_loss_driving_full={loaded.mean_loss_driving_full:.6f}, index_only_fast_path=true)"
        )

    def _iter_training_measurement_keys_from_dataset(self, dataset):
        if dataset is None:
            return
        if isinstance(dataset, SlotIndexedDataset):
            yield from self._iter_training_measurement_keys_from_dataset(dataset.base_dataset)
            return
        if isinstance(dataset, ConcatDataset):
            for sub_dataset in dataset.datasets:
                yield from self._iter_training_measurement_keys_from_dataset(sub_dataset)
            return

        required_attrs = ("measurements", "sample_start", "hist_len")
        missing_attrs = [attr for attr in required_attrs if not hasattr(dataset, attr)]
        if missing_attrs:
            raise ValueError(
                "strict full-budget coverage scan requires dataset exposing "
                f"{required_attrs}, but got missing attrs={missing_attrs}, type={type(dataset)}"
            )

        hist_len = int(dataset.hist_len)
        for local_idx in range(len(dataset)):
            measurement_root = _decode_path(dataset.measurements[local_idx][0])
            frame_id = int(dataset.sample_start[local_idx]) + hist_len - 1
            measurement_path = f"{measurement_root}/{frame_id:04}.json.gz"
            yield self._resolve_measurement_key(measurement_path)

    def _validate_full_budget_coverage_for_training(self) -> None:
        if len(self._full_budget_loss_index) == 0:
            return

        source_dataset = self.train_base_dataset if self.train_base_dataset is not None else self.train_dataset
        if source_dataset is None:
            raise ValueError(
                "full-budget coverage scan requires train_base_dataset or train_dataset to be initialized"
            )

        train_total_count = 0
        train_unique_keys = set()
        for key in self._iter_training_measurement_keys_from_dataset(source_dataset):
            train_total_count += 1
            train_unique_keys.add(key)

        train_unique_count = len(train_unique_keys)
        if train_unique_count == 0:
            raise ValueError("training dataset produced zero measurement_path entries during full-budget coverage scan")

        full_budget_keys = set(self._full_budget_loss_index.keys())
        missing_keys = train_unique_keys - full_budget_keys
        extra_keys = full_budget_keys - train_unique_keys

        missing_count = len(missing_keys)
        extra_count = len(extra_keys)
        missing_ratio = float(missing_count) / float(train_unique_count)
        extra_ratio = float(extra_count) / float(train_unique_count)

        print(
            "[full_budget_losses] coverage "
            f"train_total={train_total_count} train_unique={train_unique_count} "
            f"index_unique={len(full_budget_keys)} "
            f"missing={missing_count} ({missing_ratio:.6f}) "
            f"extra={extra_count} ({extra_ratio:.6f})"
        )

        if missing_count > 0 or extra_count > 0:
            missing_examples = list(missing_keys)[:3]
            extra_examples = list(extra_keys)[:3]
            raise ValueError(
                "full-budget all.jsonl is incompatible with current training dataset. "
                f"missing={missing_count}/{train_unique_count} ({missing_ratio:.6f}), "
                f"extra={extra_count}/{train_unique_count} ({extra_ratio:.6f}), "
                f"missing_examples={missing_examples}, extra_examples={extra_examples}, "
                f"jsonl={self._full_budget_all_jsonl}"
            )

    def _get_train_sampling_mode(self) -> str:
        # setup owns the complete training-data split by sampling mode.
        if not hasattr(self, "train_sampling") or self.train_sampling is None:
            raise ValueError("data_module.train_sampling is required")
        mode = str(getattr(self.train_sampling, "mode", "")).strip().lower()
        if mode not in {"weighted_random", "route_stream"}:
            raise ValueError(
                f"Unsupported data_module.train_sampling.mode={mode}. Expected one of ['weighted_random', 'route_stream']"
            )
        return mode

    def _validate_weighted_random_training_setup(self) -> None:
        if self.driving_dataset is None and self.dreamer_dataset is None:
            raise ValueError("weighted_random training requires at least one of driving_dataset/dreamer_dataset")

        if self.train_partitions is not None:
            if not _is_mapping_like(self.train_partitions) or len(self.train_partitions) == 0:
                raise ValueError("data_module.train_partitions must be a non-empty dict when provided")
            for bucket_name, weight in self.train_partitions.items():
                if float(weight) <= 0.0:
                    raise ValueError(f"data_module.train_partitions[{bucket_name}] must be > 0")

        if self.train_partitions_dreamer is not None:
            if not _is_mapping_like(self.train_partitions_dreamer) or len(self.train_partitions_dreamer) == 0:
                raise ValueError("data_module.train_partitions_dreamer must be a non-empty dict when provided")
            for bucket_name, weight in self.train_partitions_dreamer.items():
                if float(weight) <= 0.0:
                    raise ValueError(f"data_module.train_partitions_dreamer[{bucket_name}] must be > 0")

    def _validate_route_stream_training_setup(self) -> None:
        mode = self._get_train_sampling_mode()
        if mode != "route_stream":
            raise ValueError(f"Unsupported data_module.train_sampling.mode={mode}. Expected route_stream")
        required_sampling_fields = (
            "deficit_alpha",
            "deficit_eps",
            "drop_last",
            "strict_bucket_nonempty",
        )
        missing_sampling_fields = [
            field_name for field_name in required_sampling_fields if not hasattr(self.train_sampling, field_name)
        ]
        if missing_sampling_fields:
            raise ValueError(
                f"route_stream training requires train_sampling fields: {missing_sampling_fields}"
            )

        if self.driving_dataset is None:
            raise ValueError("route_stream training requires data_module.driving_dataset")
        if self.dreamer_dataset is not None:
            raise ValueError(
                "route_stream training requires data_module.dreamer_dataset=null"
            )

        if not _is_mapping_like(self.train_partitions) or len(self.train_partitions) == 0:
            raise ValueError("route_stream training requires non-empty data_module.train_partitions")
        if "all" not in self.train_partitions:
            raise ValueError("data_module.train_partitions must include 'all' bucket")

        bucket_priority = getattr(self.train_sampling, "bucket_priority", None)
        if not _is_list_like(bucket_priority):
            raise ValueError("data_module.train_sampling.bucket_priority must be a list")
        drop_last_value = getattr(self.train_sampling, "drop_last", None)
        if not isinstance(drop_last_value, bool):
            raise ValueError(
                f"data_module.train_sampling.drop_last must be a bool, got {type(drop_last_value)}"
            )
        if drop_last_value is not True:
            raise ValueError(
                "route_stream training requires data_module.train_sampling.drop_last=true to keep batch size constant"
            )
        non_all_buckets = [bucket_name for bucket_name in self.train_partitions.keys() if bucket_name != "all"]
        if len(non_all_buckets) == 0:
            if len(bucket_priority) != 0:
                raise ValueError(
                    "data_module.train_sampling.bucket_priority must be empty when train_partitions has only 'all'"
                )
        else:
            missing_priority = sorted([bucket_name for bucket_name in non_all_buckets if bucket_name not in bucket_priority])
            if missing_priority:
                raise ValueError(
                    f"data_module.train_sampling.bucket_priority missing buckets: {missing_priority}"
                )

    def _build_route_index_cache_file(self) -> str:
        fingerprint_payload = {
            "dataset_target": str(getattr(self.driving_dataset, "_target_", "unknown")),
            "data_path": str(self.data_path),
            "bucket_path": str(self.bucket_path),
            "train_partitions": dict(self.train_partitions),
            "bucket_priority": list(self.train_sampling.bucket_priority),
            "strict_frame_monotonic": bool(self.train_sampling.strict_frame_monotonic),
        }
        fingerprint_raw = json.dumps(fingerprint_payload, sort_keys=True)
        fingerprint = hashlib.sha1(fingerprint_raw.encode("utf-8")).hexdigest()[:16]

        cache_dir = Path(str(self.train_sampling.index_cache_dir)).expanduser()
        if not cache_dir.is_absolute():
            cache_dir = self.workspace_root / cache_dir
        cache_dir.mkdir(parents=True, exist_ok=True)
        return str(cache_dir / f"route_bucket_index_{fingerprint}.json")

    def _instantiate_train_bucket_dataset(self, bucket_name: str):
        return hydra.utils.instantiate(
            self.driving_dataset,
            split="train",
            bucket_name=bucket_name,
            **self.cfg,
            **self.base_dataset,
            _recursive_=False,
        )

    def _setup_weighted_random_training(self) -> None:
        # Weighted sampling supports mixed driving and Dreamer datasets.
        self._validate_weighted_random_training_setup()
        self.train_batch_sampler = None
        self.train_base_dataset = None
        self.sampler_train = None
        self.val_drop_last = True

        self.val_datasets = []
        bucket_list = []
        datasets = {}
        sample_weights = []

        if self.driving_dataset is not None or self.dreamer_dataset is not None:
            used_driving_datasets = []
            used_train_partitions = []
            is_dreamer_flags = []
            dataset_pairs = [
                (self.driving_dataset, self.train_partitions, False),
                (self.dreamer_dataset, self.train_partitions_dreamer, True),
            ]
            for dataset, partition, is_dreamer in dataset_pairs:
                if dataset is not None:
                    used_driving_datasets.append(dataset)
                    used_train_partitions.append(partition)
                    is_dreamer_flags.append(is_dreamer)

            mixed_two_datasets = self.driving_dataset is not None and self.dreamer_dataset is not None
            weights_driving = 0.5
            weights_dreamer = 1 - weights_driving
            for used_driving_dataset, used_train_partitions, is_dreamer in zip(
                used_driving_datasets,
                used_train_partitions,
                is_dreamer_flags,
            ):
                if used_train_partitions is not None:
                    bucket_list_tmp = list(used_train_partitions.keys())
                    sample_weights_tmp = list(used_train_partitions.values())
                    sum_sample_weights = sum(sample_weights_tmp)
                    sample_weights_tmp = [w / sum_sample_weights for w in sample_weights_tmp]

                    if mixed_two_datasets:
                        dataset_weight = weights_dreamer if is_dreamer else weights_driving
                        sample_weights_tmp = [w * dataset_weight for w in sample_weights_tmp]
                else:
                    bucket_list_tmp = ['all']
                    sample_weights_tmp = [weights_dreamer if is_dreamer else weights_driving] if mixed_two_datasets else [1.0]

                if is_dreamer:
                    bucket_list_tmp = [f"{b}_dreamer" for b in bucket_list_tmp]

                bucket_list.extend(bucket_list_tmp)
                sample_weights.extend(sample_weights_tmp)

                for bucket in bucket_list_tmp:
                    bucket_name = bucket.replace('_dreamer', '')
                    datasets[bucket] = hydra.utils.instantiate(
                        used_driving_dataset,
                        split="train",
                        bucket_name=bucket_name,
                        **self.cfg,
                        **self.base_dataset,
                        _recursive_=False,
                    )

                self.val_datasets.append(
                    hydra.utils.instantiate(
                        used_driving_dataset,
                        split="val",
                        bucket_name="all",
                        **self.cfg,
                        **self.base_dataset,
                        _recursive_=False,
                    )
                )

        self.train_dataset = None
        if len(datasets) > 0:
            sample_weights = [sample_weights[i] for i, bucket in enumerate(bucket_list) if datasets[bucket].__len__() > 0]
            bucket_list = [bucket for bucket in bucket_list if datasets[bucket].__len__() > 0]
            if len(bucket_list) != len(datasets):
                print(f"\033[91mDatasets with 0 samples: {set(datasets.keys()) - set(bucket_list)}\033[00m")
                print(f"\033[91m \nContinue with these bucket {set(bucket_list)}.\033[00m")
            datasets = {key: value for key, value in datasets.items() if value.__len__() > 0}

            self.train_dataset = ConcatDataset([datasets[bucket] for bucket in bucket_list])
        # Expand bucket quotas into per-sample weights.
            weights_train = [[sample_weights[i]] * datasets[bucket].__len__() for i, bucket in enumerate(bucket_list)]
            weights_train = list(itertools.chain.from_iterable(weights_train))
            num_samples_all = [datasets[bucket].__len__() // sample_weights[i] for i, bucket in enumerate(bucket_list)]
            num_samples = int(min(num_samples_all))
            print(f"Num samples: {num_samples}")
            if self.driving_dataset is not None and "all" in datasets:
                print(f"Num samples all: {datasets['all'].__len__()}")
            self.sampler_train = torch.utils.data.WeightedRandomSampler(
                weights=weights_train,
                num_samples=num_samples,
                replacement=True,
            )

        self.val_dataset = ConcatDataset(self.val_datasets) if len(self.val_datasets) > 0 else None
        self.predict_dataset = None

    def _setup_route_stream_training(self) -> None:
        # Route-stream sampling uses only driving data and preserves frame order.
        self._validate_route_stream_training_setup()
        all_train_dataset = self._instantiate_train_bucket_dataset(bucket_name="all")

        cache_file = self._build_route_index_cache_file()
        # Build the full dataset once; derive bucket statistics from its index.
        route_to_sample_indices, route_to_primary_bucket, route_to_len = build_or_load_route_bucket_index(
            all_dataset=all_train_dataset,
            bucket_datasets=None,
            train_partitions=dict(self.train_partitions),
            bucket_priority=list(self.train_sampling.bucket_priority),
            index_cache_file=cache_file,
            rebuild_index_cache=bool(self.train_sampling.rebuild_index_cache),
            strict_frame_monotonic=bool(self.train_sampling.strict_frame_monotonic),
            bucket_path=str(self.bucket_path),
            repo_root=str(self.workspace_root),
        )
        print(
            f"[route_stream] cache={cache_file} routes={len(route_to_sample_indices)} "
            f"frames={sum(len(v) for v in route_to_sample_indices.values())}"
        )

        # Offline analysis reads the unwrapped dataset index.
        self.train_base_dataset = all_train_dataset
        # Route-stream samples carry (slot_id, sample_index).
        self.train_dataset = SlotIndexedDataset(all_train_dataset)
        trainer = getattr(self, "trainer", None)
        if trainer is None:
            raise RuntimeError(
                "route_stream training requires DataModule.trainer for distributed rank/world_size context"
            )
        world_size = int(getattr(trainer, "world_size", -1))
        global_rank = int(getattr(trainer, "global_rank", -1))
        if world_size <= 0:
            raise ValueError(f"invalid trainer.world_size={world_size}")
        if global_rank < 0 or global_rank >= world_size:
            raise ValueError(
                f"invalid trainer.global_rank={global_rank} for world_size={world_size}"
            )
        if int(self.expected_world_size) != world_size:
            raise ValueError(
                f"distributed world_size mismatch: expected_world_size={self.expected_world_size}, trainer.world_size={world_size}"
            )

        self.train_batch_sampler = RouteStreamBatchSampler(
            route_to_sample_indices=route_to_sample_indices,
            route_to_primary_bucket=route_to_primary_bucket,
            route_to_len=route_to_len,
            bucket_weights={k: float(v) for k, v in self.train_partitions.items()},
        # Build global batches before slicing them by rank.
            local_batch_size=int(self.batch_size),
            rank=global_rank,
            world_size=world_size,
            route_shuffle_each_epoch=bool(self.train_sampling.route_shuffle_each_epoch),
            route_shuffle_seed=int(self.train_sampling.route_shuffle_seed),
            length_balancing=bool(self.train_sampling.length_balancing),
            deficit_alpha=float(self.train_sampling.deficit_alpha),
            deficit_eps=float(self.train_sampling.deficit_eps),
        # Route-stream training requires fixed-size batches.
            drop_last=bool(self.train_sampling.drop_last),
            strict_bucket_nonempty=bool(self.train_sampling.strict_bucket_nonempty),
        )
        self.sampler_train = None
        self.val_drop_last = False

        self.val_dataset = hydra.utils.instantiate(
            self.driving_dataset,
            split="val",
            bucket_name="all",
            **self.cfg,
            **self.base_dataset,
            _recursive_=False,
        )
        self.predict_dataset = None

    def setup(self, stage=None):
        if not self.predict:
            self.train_sampling_mode = self._get_train_sampling_mode()
        # Resolve the training mode once and reuse its loader state.
            if self.train_sampling_mode == "weighted_random":
                self._setup_weighted_random_training()
            elif self.train_sampling_mode == "route_stream":
                self._setup_route_stream_training()
            else:
                raise ValueError(f"Unsupported train_sampling_mode={self.train_sampling_mode}")

            if bool(getattr(self, "enable_full_budget_coverage_scan", False)):
                self._validate_full_budget_coverage_for_training()

        else:
            if self.qa_dataset is not None:
                predict_dataset = self.qa_dataset
                
            elif self.insteval_dataset is not None:
                predict_dataset = self.insteval_dataset

            self.predict_dataset = hydra.utils.instantiate(
                    predict_dataset,
                    split="val",
                    bucket_name="all",
                    **self.cfg,
                    **self.base_dataset,
                    _recursive_=False
                )
            eval_max_samples = getattr(self, "eval_max_samples", None)
            if eval_max_samples is not None:
                eval_max_samples = int(eval_max_samples)
                if eval_max_samples <= 0:
                    raise ValueError(f"eval_max_samples must be > 0 when provided, got {eval_max_samples}")
                original_len = len(self.predict_dataset)
                limited_len = min(eval_max_samples, original_len)
                self.predict_dataset = Subset(self.predict_dataset, range(limited_len))
                print(
                    f"[predict_dataset] limited eval samples to {limited_len}/{original_len} "
                    f"(eval_max_samples={eval_max_samples})"
                )


    def train_dataloader(self):
        if self.train_dataset is None:
            return None
        if self.train_sampling_mode == "route_stream":
            # Route-stream batches include a slot ID for every sample.
            return DataLoader(
                self.train_dataset,
                batch_sampler=self.train_batch_sampler,
                num_workers=self.num_workers,
                collate_fn=self.dl_collate_fn,
                pin_memory=True,
            )
        if self.train_sampling_mode == "weighted_random":
            if self.sampler_train is None:
                raise ValueError("weighted_random mode requires sampler_train")
            return DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                drop_last=True,
                collate_fn=self.dl_collate_fn,
                sampler=self.sampler_train,
                pin_memory=True,
            )
        raise ValueError(f"Unsupported train_sampling_mode={self.train_sampling_mode}")

    def predict_dataloader(self):
        return DataLoader(
            self.predict_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False,
            collate_fn=self.dl_collate_fn,
            pin_memory=True,
        )

    def val_dataloader(self):
        if self.val_dataset is None:
            return None
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=self.val_drop_last,
            collate_fn=self.dl_collate_fn_val,
            pin_memory=True,
        )


    def dl_collate_fn(self, data, require_full_budget_loss=True):
        BS = len(data)
        grid_nums = [self.NUM_IMAGE_PATCHES] # we split the front forward into two patches (1x2)

        image_ff_pixel, image_ff_sizes = None, None
        image_ff_org = torch.tensor(np.asarray([data[i].image_ff_org_size for i in range(BS)]))
            
        for idx, img_to_consider in enumerate(self.IMAGES_TO_CONSIDER):
            img_tmp = getattr(data[0], img_to_consider)
            T, C, H, W = img_tmp.shape
            assert T == 1, "Only one timestep as input supported"
            
            images_batch_tensor = torch.tensor(np.asarray([getattr(data[i], img_to_consider) if getattr(data[i], img_to_consider) is not None else np.zeros_like(img_tmp) for i in range(len(data))])).float()
            images_batch_tensor = images_batch_tensor.view(BS*T, C, H, W)
            images_batch_list = list(images_batch_tensor)

            if 'internvl2' in self.encoder_variant.lower():
                # get image patches
                images_processed = preprocess_image_batch(images_batch_list, input_size=448, use_global_img=self.use_global_img, max_num_grid=grid_nums[idx])    
            else:
                raise ValueError(f"Image preprocessing for {self.encoder_variant} not implemented")
                
            images_pixel = images_processed['pixel_values']
            image_sizes = images_processed['image_sizes']
            
            assert images_pixel.shape[0] == BS * T
            num_patches = images_pixel.shape[1]
            assert images_pixel.shape[2] == C
            new_height = images_pixel.shape[3]
            new_width = images_pixel.shape[4]
            images_pixel = images_pixel.view(BS, T, num_patches, C, new_height, new_width)
            
            if img_to_consider == 'image_ff':
                image_ff_pixel = images_pixel
                image_ff_sizes = image_sizes
            else:
                raise ValueError(f"Image type {img_to_consider} not supported")

        conversations = [data[i].conversation for i in range(BS)]
        conversation_dict, question_dict = get_custom_chat_template(conversations, self.tokenizer, self.encoder_variant, self.num_image_tokens_total)

        placeholder_batch_list = []
        for i in range(BS):
            tmp = {}
            for key, value in data[i].placeholder_values.items():
                token_nr_key = self.tokenizer.convert_tokens_to_ids(key)
                tmp[token_nr_key] = value
            placeholder_batch_list.append(tmp)
                
        prompt_languagelabel = LanguageLabel(
            phrase_ids=conversation_dict['phrase_ids'],
            phrase_valid=conversation_dict['phrase_valid'],
            phrase_mask=conversation_dict['phrase_mask'],
            placeholder_values=placeholder_batch_list,
            language_string=conversation_dict['language_string'],
            loss_masking=conversation_dict['loss_masking'],
        )

        prompt_question_languagelabel = LanguageLabel(
            phrase_ids=question_dict['phrase_ids'],
            phrase_valid=question_dict['phrase_valid'],
            phrase_mask=question_dict['phrase_mask'],
            placeholder_values=placeholder_batch_list,
            language_string=question_dict['language_string'],
            loss_masking=question_dict['loss_masking'],
        )
        answer_string_list = [data[i].answer[0]['content'][0]['text'] for i in range(BS)]
        answer_label =  LanguageLabel(
            phrase_ids=None,
            phrase_valid=None,
            phrase_mask=None,
            placeholder_values=None,
            language_string=answer_string_list,
            loss_masking=None,
        )
        
        if self.base_dataset.use_1d_wps:
            waypoints = torch.tensor(np.asarray([data[i].waypoints_1d for i in range(len(data))])).float() # [B, F, 2] 11 future waypoints 0.2s apart
        else:
            waypoints = torch.tensor(np.asarray([data[i].waypoints for i in range(len(data))])).float() # [B, F, 2] 11 future waypoints 0.2s apart
        
        if self.predict:
            qa_templates = [data[i].qa_templates[0] if data[i].qa_templates is not None else None for i in range(BS) ]
            eval_infos = [data[i].eval_infos if data[i].eval_infos is not None else None for i in range(BS) ]
        else:
            qa_templates = None
            eval_infos = None

        ego_xy = []
        ego_yaw = []
        timestamp = []
        for i in range(BS):
            item = data[i]
            if item.ego_xy is None or item.ego_yaw is None or item.timestamp is None:
                ego_xy.append([np.nan, np.nan])
                ego_yaw.append(np.nan)
                timestamp.append(np.nan)
            else:
                ego_xy.append([float(item.ego_xy[0]), float(item.ego_xy[1])])
                ego_yaw.append(float(item.ego_yaw))
                timestamp.append(float(item.timestamp))
        
        driving_input=DrivingInput(
                camera_images=image_ff_pixel,  # [B, T, N, C, H, W] uint8 [0, 255]
                image_sizes=image_ff_sizes,
                camera_intrinsics = torch.repeat_interleave(get_camera_intrinsics(W, H, 110).unsqueeze(0), BS, dim=0).view(BS, 3, 3).float(),
                camera_extrinsics = torch.repeat_interleave(get_camera_extrinsics().unsqueeze(0), BS, dim=0).view(BS, 4, 4).float(),
                vehicle_speed=torch.tensor(np.asarray([data[i].speed for i in range(len(data))])).float(),  # [B, S] float32
                target_point=torch.tensor(np.asarray([data[i].target_points for i in range(len(data))])).float(),  # [B, 2] float32
                prompt=prompt_languagelabel,
                prompt_inference=prompt_question_languagelabel,
                ego_xy=torch.tensor(np.asarray(ego_xy)).float(),
                ego_yaw=torch.tensor(np.asarray(ego_yaw)).float(),
                timestamp=torch.tensor(np.asarray(timestamp)).float(),
            )

        driving_label=DrivingLabel(
                waypoints=waypoints,
                path=torch.tensor(np.asarray([data[i].path for i in range(len(data))])).float(), # [B, 3, RH, RW] uint8 [0, 255]
                answer=answer_label,
                image_ff_org=image_ff_org,
                eval_infos=eval_infos,
            )

        slot_id_values = [getattr(data[i], "slot_id", None) for i in range(BS)]
        if any(v is not None for v in slot_id_values):
            if not all(v is not None for v in slot_id_values):
                raise ValueError(f"Inconsistent slot_id values in batch: {slot_id_values}")
            # Route-stream slot IDs let the model reset history per slot.
            slot_ids = torch.tensor(np.asarray(slot_id_values), dtype=torch.long)
        else:
            slot_ids = None

        full_budget_driving_loss = None
        if len(self._full_budget_loss_index) > 0:
            resolved_measurements = [self._resolve_measurement_key(data[i].measurement_path) for i in range(BS)]
            missing = [path for path in resolved_measurements if path not in self._full_budget_loss_index]
            if len(missing) > 0:
                if require_full_budget_loss:
                    preview = missing[:3]
                    raise ValueError(
                        "Missing full-budget driving loss for measurement_path(s): "
                        f"count={len(missing)}, examples={preview}. "
                        f"jsonl={self._full_budget_all_jsonl}"
                    )
            else:
                values = [self._full_budget_loss_index[path] for path in resolved_measurements]
                full_budget_driving_loss = torch.tensor(np.asarray(values), dtype=torch.float32)
            
        return DrivingExample(
            driving_input=driving_input,
            driving_label=driving_label,
            run_id=encode_uint8([data[i].measurement_path for i in range(BS)], 1000),  # [B] str
            full_budget_driving_loss=full_budget_driving_loss,
            qa_templates=qa_templates,
            slot_ids=slot_ids,
        )

    def dl_collate_fn_val(self, data):
            # Validation computes references online instead of requiring train-only JSONL.
        return self.dl_collate_fn(data, require_full_budget_loss=False)

    def dl_collate_fn_test(self, data):
        pass


@hydra.main(config_path=f"../config", config_name="config", version_base="1.1")
def test(cfg):
    
    get_waypoint_stats = True
    
        
    internvl_source = resolve_internvl_variant_source(cfg.model.vision_model.variant)
    processor = AutoProcessor.from_pretrained(internvl_source, trust_remote_code=True)
    dm = hydra.utils.instantiate(
        cfg.data_module,
        processor=processor,
        encoder_variant=cfg.model.vision_model.variant,
        llm_variant="llava-hf/llava-v1.6-mistral-7b-hf",
        _recursive_=False
    )


    dm.setup()
    dl = dm.val_dataloader()
    print(dl.dataset.__len__())

    iterations = 0
    
    all_waypoints = []
    all_waypoints_diff = []
    for batch in dl:
        
        iterations += 1
        
        if iterations % 100 == 0:
            print(f"Iteration: {iterations}")
        
        if iterations > 20000:
            break
        
        if get_waypoint_stats:
            # get stats about range of waypoints
            waypoints = batch.driving_label.waypoints
            all_waypoints.append(waypoints)
            
            # get residuals
            residuals = waypoints[:,1:] - waypoints[:,:-1]
            all_waypoints_diff.append(residuals)
            
    # get histogram of waypoints
    if get_waypoint_stats:
        all_waypoints = torch.cat(all_waypoints, dim=0)
        all_waypoints_diff = torch.cat(all_waypoints_diff, dim=0)
        
        all_waypoints = all_waypoints.view(-1, 2)
        all_waypoints_diff = all_waypoints_diff.view(-1, 2)
        
        
        
        import matplotlib.pyplot as plt
        plt.hist(all_waypoints[:,0].numpy(), bins=100)
        plt.savefig('waypoints_x.png')
        max_x = all_waypoints[:,0].max().item()
        min_x = all_waypoints[:,0].min().item()
        print(f"Max x: {max_x}, Min x: {min_x}")
        plt.clf()
        plt.hist(all_waypoints[:,1].numpy(), bins=100)
        plt.savefig('waypoints_y.png')
        max_y = all_waypoints[:,1].max().item()
        min_y = all_waypoints[:,1].min().item()
        print(f"Max y: {max_y}, Min y: {min_y}")
        plt.clf()
        
        plt.hist(all_waypoints_diff[:,0].numpy(), bins=100)
        plt.savefig('waypoints_diff_x.png')
        max_x_diff = all_waypoints_diff[:,0].max().item()
        min_x_diff = all_waypoints_diff[:,0].min().item()
        print(f"Max x diff: {max_x_diff}, Min x diff: {min_x_diff}")
        plt.clf()
        plt.hist(all_waypoints_diff[:,1].numpy(), bins=100)
        plt.savefig('waypoints_diff_y.png')
        max_y_diff = all_waypoints_diff[:,1].max().item()
        min_y_diff = all_waypoints_diff[:,1].min().item()
        print(f"Max y diff: {max_y_diff}, Min y diff: {min_y_diff}")
        plt.clf()
            
            

if __name__ == "__main__":
    test()
