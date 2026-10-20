"""
partially taken from https://github.com/autonomousvision/carla_garage/blob/leaderboard_2/team_code_adaption/sensor_agent.py
(MIT licence)
"""


import copy
import json
import math
import os
import pathlib
import random
import time
import xml.etree.ElementTree as ET
from collections import deque
from pathlib import Path
from typing import Optional


import carla
import cv2
import hydra
import numpy as np
import torch
from filterpy.kalman import MerweScaledSigmaPoints
from filterpy.kalman import UnscentedKalmanFilter as UKF
from leaderboard.autoagents import autonomous_agent
from omegaconf import OmegaConf
from PIL import Image
from scipy.interpolate import PchipInterpolator
from scipy.optimize import fsolve
from transformers import AutoConfig, AutoProcessor

import scenario_logger
import team_code_adaption.transfuser_utils as t_u
from scenario_logger import ScenarioLogger
from simlingo_adaption_training.utils.custom_types import DrivingInput, LanguageLabel
from simlingo_adaption_training.config import validate_adaption_config
from simlingo_adaption_training.inference_checkpoint import (
    load_policy_aware_inference_state_dict,
)
from simlingo_adaption_training.utils.internvl2_utils import (
    build_transform,
    dynamic_preprocess,
    resolve_internvl_variant_source,
)
from team_code_adaption.config_simlingo import GlobalConfig
from team_code_adaption.inference_measurement import NvmlPowerSampler
from team_code_adaption.nav_planner import LateralPIDController, RoutePlanner
from team_code_adaption.simlingo_utils import (
    get_camera_extrinsics,
    get_camera_intrinsics,
)

# Configure pytorch for maximum performance
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.allow_tf32 = True


# Leaderboard function that selects the class used as agent.
def get_entry_point():
    return 'LingoAgent'


USE_UKF = True


def _parse_env_bool(value, env_name: str) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        raise ValueError(f"Missing {env_name}")
    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{env_name} must be bool-like (0/1/true/false), got {value!r}")


class LingoAgent(autonomous_agent.AutonomousAgent):
    """
        Main class that runs the agents with the run_step function
        """
    def _resolve_bootstrap_config_for_checkpoint(self, checkpoint_path: Path) -> Path:
        """
        Resolve bootstrap config path for smart_assigner runtime.
        Supports both 2-level and 3-level parent directory layouts.
        """
        # Support both known experiment directory layouts.
        candidate_paths = [
            checkpoint_path.parent.parent / ".hydra" / "config.yaml",
            checkpoint_path.parent.parent.parent / ".hydra" / "config.yaml",
        ]
        for candidate in candidate_paths:
            if candidate.is_file():
                return candidate

        tried = ", ".join(str(p) for p in candidate_paths)
        raise ValueError(
            f"Model bootstrap config file not found for checkpoint {checkpoint_path}. "
            f"Tried: {tried}"
        )

    def setup(self, path_to_conf_file, route_index=None):
        """Sets up the agent. route_index is for logging purposes"""

        torch.cuda.empty_cache()
        self.track = autonomous_agent.Track.SENSORS
        if '+' in path_to_conf_file:
            print(f"path to conf file: {path_to_conf_file}")
            # Split once from the right because parent paths may contain '+'.
            self.agent_config_path = path_to_conf_file.rsplit('+', 1)[0]
            print(f"Agent config path: {self.agent_config_path}")
            self.save_path_root = path_to_conf_file.rsplit('+', 1)[1]
            print(f"Save path root: {self.save_path_root}")
        else:
            self.agent_config_path = path_to_conf_file
            print(f"Agent config path: {self.agent_config_path}")
            self.save_path_root = route_index
            print(f"Save path root: {self.save_path_root}")
        self.step = -1
        self.initialized = False
        self.device = torch.device('cuda')
        self.DrivingInput = {}
        self.config = GlobalConfig()
        self.eval_budget_mode = os.getenv("SIMLINGO_EVAL_BUDGET_MODE")
        if self.eval_budget_mode is None:
            raise ValueError(f"Missing budget.mode ")
        self.eval_budget_mode = self.eval_budget_mode.strip().lower()
        if self.eval_budget_mode != "smart_assigner":
            raise ValueError(
                f"This version only supports SIMLINGO_EVAL_BUDGET_MODE=smart_assigner, got {self.eval_budget_mode}"
            )
        self.eval_runtime_profile = os.getenv("SIMLINGO_EVAL_RUNTIME_PROFILE")
        if self.eval_runtime_profile is None:
            raise ValueError("Missing SIMLINGO_EVAL_RUNTIME_PROFILE")
        self.eval_runtime_profile = self.eval_runtime_profile.strip().lower()
        allowed_runtime_profiles = {"smart_assigner", "simlingo_compat"}
        if self.eval_runtime_profile not in allowed_runtime_profiles:
            raise ValueError(
                f"Unsupported SIMLINGO_EVAL_RUNTIME_PROFILE={self.eval_runtime_profile}, "
                f"expected one of {sorted(allowed_runtime_profiles)}"
            )
        self.measure_wall_clock_latency = _parse_env_bool(
            os.getenv("SIMLINGO_EVAL_MEASURE_WALL_CLOCK_LATENCY", "0"),
            env_name="SIMLINGO_EVAL_MEASURE_WALL_CLOCK_LATENCY",
        )
        self.measure_power = _parse_env_bool(
            os.getenv("SIMLINGO_EVAL_MEASURE_POWER", "0"),
            env_name="SIMLINGO_EVAL_MEASURE_POWER",
        )
        if (self.measure_wall_clock_latency or self.measure_power) and not torch.cuda.is_available():
            raise RuntimeError("Inference measurement requires CUDA")
        self.latest_forward_wall_clock_latency_ms = None
        self.latest_forward_power = None
        self.power_sampler = None
        if self.measure_power:
            self.power_sampler = NvmlPowerSampler(
                interval_ms=float(os.getenv("SIMLINGO_EVAL_POWER_SAMPLE_INTERVAL_MS", "10"))
            )
        predict_language_raw = os.getenv("SIMLINGO_EVAL_PREDICT_LANGUAGE")
        if predict_language_raw is None:
            self.eval_predict_language = self.eval_runtime_profile == "simlingo_compat"
        else:
            self.eval_predict_language = _parse_env_bool(
                predict_language_raw,
                env_name="SIMLINGO_EVAL_PREDICT_LANGUAGE",
            )
        language_sample_count_raw = os.getenv("SIMLINGO_EVAL_LANGUAGE_SAMPLE_COUNT")
        if language_sample_count_raw is None:
            raise ValueError("Missing SIMLINGO_EVAL_LANGUAGE_SAMPLE_COUNT")
        try:
            self.language_sample_count = int(language_sample_count_raw)
        except Exception as exc:
            raise ValueError(
                f"SIMLINGO_EVAL_LANGUAGE_SAMPLE_COUNT must be integer, got {language_sample_count_raw}"
            ) from exc
        if self.language_sample_count <= 0:
            raise ValueError(
                f"SIMLINGO_EVAL_LANGUAGE_SAMPLE_COUNT must be > 0, got {self.language_sample_count}"
            )
        self._language_sample_seen = 0
        self._language_samples = []
        self._language_sample_rng = random.Random(0)
        decision_trace_raw = os.getenv("SIMLINGO_EVAL_DECISION_TRACE_ENABLED", "0")
        self.eval_decision_trace_enabled = _parse_env_bool(
            decision_trace_raw,
            env_name="SIMLINGO_EVAL_DECISION_TRACE_ENABLED",
        )
        decision_trace_interval_raw = os.getenv("SIMLINGO_EVAL_DECISION_TRACE_INTERVAL", "10")
        try:
            self.eval_decision_trace_interval = int(decision_trace_interval_raw)
        except Exception as exc:
            raise ValueError(
                "SIMLINGO_EVAL_DECISION_TRACE_INTERVAL must be integer, got "
                f"{decision_trace_interval_raw}"
            ) from exc
        if self.eval_decision_trace_interval <= 0:
            raise ValueError(
                "SIMLINGO_EVAL_DECISION_TRACE_INTERVAL must be > 0, got "
                f"{self.eval_decision_trace_interval}"
            )
        self.eval_decision_trace_pic = _parse_env_bool(
            os.getenv("SIMLINGO_EVAL_DECISION_TRACE_PIC", "1"),
            env_name="SIMLINGO_EVAL_DECISION_TRACE_PIC",
        )
        self.eval_decision_trace_replay = _parse_env_bool(
            os.getenv("SIMLINGO_EVAL_DECISION_TRACE_REPLAY", "1"),
            env_name="SIMLINGO_EVAL_DECISION_TRACE_REPLAY",
        )
        self._latest_prompt_token_ids = []
        self._latest_prompt_tokens = []
        self._latest_prompt_text = None
        self._decision_trace_jsonl_path = None
        budget_policy_raw = os.getenv("SIMLINGO_EVAL_BUDGET_POLICY", "stage2")
        self.eval_budget_policy = str(budget_policy_raw).strip().lower()
        allowed_budget_policies = {"stage1", "stage2"}
        if self.eval_budget_policy not in allowed_budget_policies:
            raise ValueError(
                f"Unsupported SIMLINGO_EVAL_BUDGET_POLICY={self.eval_budget_policy}, "
                f"expected one of {sorted(allowed_budget_policies)}"
            )
        layer_selection_mode_raw = os.getenv("SIMLINGO_EVAL_LAYER_SELECTION_MODE", "stage1")
        self.eval_layer_selection_mode = str(layer_selection_mode_raw).strip().lower()
        allowed_layer_selection_modes = {"stage1", "random"}
        if self.eval_layer_selection_mode not in allowed_layer_selection_modes:
            raise ValueError(
                f"Unsupported SIMLINGO_EVAL_LAYER_SELECTION_MODE={self.eval_layer_selection_mode}, "
                f"expected one of {sorted(allowed_layer_selection_modes)}"
            )
        limit_language_inference_raw = os.getenv("SIMLINGO_EVAL_LIMIT_LANGUAGE_INFERENCE", "1")
        self.eval_limit_language_inference = _parse_env_bool(
            limit_language_inference_raw,
            env_name="SIMLINGO_EVAL_LIMIT_LANGUAGE_INFERENCE",
        )
        fixed_random_layer_selection_raw = os.getenv("SIMLINGO_EVAL_FIXED_RANDOM_LAYER_SELECTION", "0")
        self.eval_fixed_random_layer_selection = _parse_env_bool(
            fixed_random_layer_selection_raw,
            env_name="SIMLINGO_EVAL_FIXED_RANDOM_LAYER_SELECTION",
        )
        self.eval_budget_sampling = _parse_env_bool(
            os.getenv("SIMLINGO_EVAL_BUDGET_SAMPLING", "0"),
            env_name="SIMLINGO_EVAL_BUDGET_SAMPLING",
        )
        self.eval_budget_sampling_temperature = float(
            os.getenv("SIMLINGO_EVAL_BUDGET_SAMPLING_TEMPERATURE", "1.0")
        )
        random_layer_selection_seed_raw = os.getenv("SIMLINGO_EVAL_RANDOM_LAYER_SELECTION_SEED", "0")
        try:
            self.eval_random_layer_selection_seed = int(random_layer_selection_seed_raw)
        except Exception as exc:
            raise ValueError(
                "SIMLINGO_EVAL_RANDOM_LAYER_SELECTION_SEED must be integer, got "
                f"{random_layer_selection_seed_raw}"
            ) from exc
        self.eval_fixed_budget = None
        if self.eval_budget_policy == "stage1":
            fixed_budget_raw = os.getenv("SIMLINGO_EVAL_FIXED_BUDGET")
            if fixed_budget_raw is None:
                raise ValueError(
                    "Missing SIMLINGO_EVAL_FIXED_BUDGET when SIMLINGO_EVAL_BUDGET_POLICY=stage1"
                )
            try:
                self.eval_fixed_budget = float(fixed_budget_raw)
            except Exception as exc:
                raise ValueError(
                    f"SIMLINGO_EVAL_FIXED_BUDGET must be float in [0, 1], got {fixed_budget_raw}"
                ) from exc
            if self.eval_fixed_budget < 0.0 or self.eval_fixed_budget > 1.0:
                raise ValueError(
                    f"SIMLINGO_EVAL_FIXED_BUDGET must be in [0, 1], got {self.eval_fixed_budget}"
                )

        if self.config.eval_route_as == -1:
            self.config.eval_route_as = self.model.route_as

        self.last_command = -1
        self.last_command_tmp = -1

        self.route_path = os.environ.get('ROUTES', '')
        route_type = self.route_path.split('data/benchmarks/')[-1].split('/')[0]
        route_number = str(pathlib.Path(self.route_path).stem)
        # Smart assigner history is strictly route-scoped.
        self.eval_route_key = f"{route_type}/{route_number}"
        self.eval_route_id = str(os.getenv("SIMLINGO_EVAL_ROUTE_ID", "")).strip()
        self.eval_route_name = str(os.getenv("SIMLINGO_EVAL_ROUTE_NAME", "")).strip()
        self.eval_route_file = str(os.getenv("SIMLINGO_EVAL_ROUTE_FILE", self.route_path)).strip()
        self.eval_scenario_name = str(os.getenv("SIMLINGO_EVAL_SCENARIO_NAME", "")).strip()
        self.eval_town_name = str(os.getenv("SIMLINGO_EVAL_TOWN_NAME", "")).strip()
        self.speed_controller = t_u.PIDController(k_p=self.config.speed_kp,
                                                                                            k_i=self.config.speed_ki,
                                                                                            k_d=self.config.speed_kd,
                                                                                            n=self.config.speed_n)

        self.turn_controller = LateralPIDController(inference_mode=False)

        image_fps = 5
        image_history_length = 1

        self.image_buffer = deque(maxlen=image_fps * image_history_length)

        # config
        self.carla_frame_rate = 1.0 / 20.0  # CARLA frame rate in milliseconds
        self.data_save_freq = 5
        self.lidar_seq_len = 1
        self.logging_freq = 10  # Log every 10 th frame
        self.logger_region_of_interest = 30.0  # Meters around the car that will be logged.
        self.dense_route_planner_min_distance = 1.0
        self.dense_route_planner_max_distance = 50.0
        self.log_route_planner_min_distance = 4.0
        self.route_planner_max_distance = 50.0
        self.route_planner_min_distance = 7.5

        if self.eval_runtime_profile == "smart_assigner":
            # Checkpoint mode uses the Hydra config from the same experiment.
            raw_checkpoint_path = os.getenv("SIMLINGO_SMART_ASSIGNER_CHECKPOINT")
            if raw_checkpoint_path is None or str(raw_checkpoint_path).strip().lower() in {"", "none", "null"}:
                self.model_checkpoint_path = None
            else:
                self.model_checkpoint_path = str(Path(raw_checkpoint_path).expanduser())
                if not os.path.isfile(self.model_checkpoint_path):
                    raise ValueError(
                        f"SIMLINGO_SMART_ASSIGNER_CHECKPOINT not found: {self.model_checkpoint_path}"
                    )

            agent_config_path = Path(self.agent_config_path).expanduser()
            if agent_config_path.suffix.lower() in {".yaml", ".yml"}:
                self.config_load_path = agent_config_path
            else:
                self.config_load_path = self._resolve_bootstrap_config_for_checkpoint(agent_config_path)
        elif self.eval_runtime_profile == "simlingo_compat":
            # The launcher injects the checkpoint for simlingo_compat.
            self.model_checkpoint_path = os.getenv("SIMLINGO_COMPAT_CHECKPOINT")
            if self.model_checkpoint_path is None:
                raise ValueError("Missing SIMLINGO_COMPAT_CHECKPOINT for simlingo_compat runtime")
            self.model_checkpoint_path = str(Path(self.model_checkpoint_path).expanduser())
            if not os.path.isfile(self.model_checkpoint_path):
                raise ValueError(
                    f"SIMLINGO_COMPAT_CHECKPOINT not found: {self.model_checkpoint_path}"
                )
            self.config_load_path = Path(self.agent_config_path).expanduser()
        else:
            raise ValueError(f"Unsupported runtime_profile={self.eval_runtime_profile}")

        if not self.config_load_path.is_file():
            raise ValueError(f"Model bootstrap config file not found: {self.config_load_path}")

        #load config from .hydra folder
        with open(self.config_load_path, 'r') as file:
            cfg = OmegaConf.load(file)
        self._ensure_eval_model_compat(cfg)
        self.cfg = cfg
        self.cfg.model.vision_model.use_global_img = cfg.data_module.use_global_img

        vision_model_source = resolve_internvl_variant_source(cfg.model.vision_model.variant)
        self.internvl_model_source = vision_model_source
    
        processor = AutoProcessor.from_pretrained(vision_model_source, trust_remote_code=True)
        if 'tokenizer' in processor.__dict__:
                self.tokenizer = processor.tokenizer
        else:
                self.tokenizer = processor
        self.tokenizer.add_special_tokens({'additional_special_tokens': ['<WAYPOINTS>','<WAYPOINTS_DIFF>', '<ORG_WAYPOINTS_DIFF>', '<ORG_WAYPOINTS>', '<WAYPOINT_LAST>', '<ROUTE>', '<ROUTE_DIFF>', '<TARGET_POINT>']})
        self.tokenizer.padding_side = "left"
        cache_dir = vision_model_source
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)
        
        self.model = hydra.utils.instantiate(
                cfg.model,
                cfg_data_module=cfg.data_module,
                processor=processor,
                cache_dir=cache_dir,
                inference_only=True,
                eval_budget_policy=self.eval_budget_policy,
                _recursive_=False
            ).to(self.device)
        self._validate_smart_feature_dim_after_model_init()
        torch.set_default_dtype(default_dtype)

        if self.eval_runtime_profile == "smart_assigner":
            if self.model_checkpoint_path is not None:
                ckpt_state_dict = torch.load(
                    self.model_checkpoint_path,
                    map_location="cpu",
                    weights_only=True,
                )
                if not isinstance(ckpt_state_dict, dict):
                    raise ValueError(
                        f"Inference checkpoint must be a plain state_dict: {self.model_checkpoint_path}"
                    )
                if len(ckpt_state_dict) == 0:
                    raise ValueError(
                        f"Checkpoint state_dict is empty: {self.model_checkpoint_path}"
                    )
                load_policy_aware_inference_state_dict(
                    model=self.model,
                    state_dict=ckpt_state_dict,
                    budget_policy=self.eval_budget_policy,
                )
            else:
                print(
                    "[Checkpoint] smart_assigner runs without a checkpoint; "
                    "using bootstrap config only",
                    flush=True,
                )
        elif self.eval_runtime_profile == "simlingo_compat":
            print(
                f"[Checkpoint] simlingo_compat uses merged driving model only: {self.model_checkpoint_path}",
                flush=True,
            )

        model_mode = str(getattr(self.model, "mode", "")).strip().lower()
        if model_mode != self.eval_budget_mode:
            raise ValueError(
                f"Mode mismatch: agent mode={self.eval_budget_mode}, model mode={model_mode}. "
                "Please align eval budget mode and checkpoint model mode."
            )
        self.model.set_eval_runtime_profile(self.eval_runtime_profile)
        self.model.set_eval_layer_selection_mode(self.eval_layer_selection_mode)
        self.model.set_eval_fixed_random_layer_selection(
            bool(self.eval_fixed_random_layer_selection),
            int(self.eval_random_layer_selection_seed),
        )
        self.model.set_eval_budget_sampling(
            bool(self.eval_budget_sampling),
            float(self.eval_budget_sampling_temperature),
        )
        self.model.set_eval_limit_language_inference(bool(self.eval_limit_language_inference))
        self.model.set_eval_predict_language(bool(self.eval_predict_language))
        # Evaluation must disable training-time exploration and path sampling.
        self.model.eval()
        if self.eval_runtime_profile == "smart_assigner":
            if self.model_checkpoint_path is not None:
                self.iter = self.model_checkpoint_path.split("epoch=")[-1].split("/")[0]
                self.session = self.model_checkpoint_path.split("/")[-4]
            else:
                self.iter = "no_checkpoint"
                self.session = "smart_assigner_no_checkpoint"
        elif self.eval_runtime_profile == "simlingo_compat":
            self.iter = "merged"
            self.session = "simlingo_compat"
        else:
            raise ValueError(f"Unsupported runtime_profile={self.eval_runtime_profile}")
        
        self.T = 1
        self.stuck_detector = 0
        self.force_move = 0

        self.commands = deque(maxlen=2)
        self.commands.append(4)
        self.commands.append(4)
        self.target_point_prev = [1e5, 1e5, 1e5]

        # Filtering
        if USE_UKF:
            self.points = MerweScaledSigmaPoints(n=4, alpha=0.00001, beta=2, kappa=0, subtract=residual_state_x)
            self.ukf = UKF(dim_x=4,
                                        dim_z=4,
                                        fx=bicycle_model_forward,
                                        hx=measurement_function_hx,
                                        dt=self.carla_frame_rate,
                                        points=self.points,
                                        x_mean_fn=state_mean,
                                        z_mean_fn=measurement_mean,
                                        residual_x=residual_state_x,
                                        residual_z=residual_measurement_h)

            # State noise, same as measurement because we
            # initialize with the first measurement later
            self.ukf.P = np.diag([0.5, 0.5, 0.000001, 0.000001])
            # Measurement noise
            self.ukf.R = np.diag([0.5, 0.5, 0.000000000000001, 0.000000000000001])
            self.ukf.Q = np.diag([0.0001, 0.0001, 0.001, 0.001])  # Model noise
            # Used to set the filter state equal the first measurement
            self.filter_initialized = False
        # Stores the last filtered positions of the ego vehicle. Need at least 2 for LiDAR 10 Hz realignment
        self.state_log = deque(maxlen=max((self.lidar_seq_len * self.data_save_freq), 2))

        # Path to where visualizations and other debug output gets stored
        save_root = os.environ.get('SAVE_PATH')
        if save_root is None:
            raise ValueError("Missing SAVE_PATH")
        self.save_path = os.path.join(save_root, self.save_path_root)

        # Logger that generates logs used for infraction replay in the results_parser.
        if self.save_path is not None and route_index is not None:
            self.save_path = os.path.join(self.save_path, route_index)
            pathlib.Path(self.save_path).mkdir(parents=True, exist_ok=True)

            self.lon_logger = ScenarioLogger(
                    save_path=self.save_path,
                    route_index=route_index,
                    logging_freq=self.logging_freq,
                    log_only=True,
                    route_only=False,  # with vehicles
                    roi=self.logger_region_of_interest,
            )
        
        self.debug_save_path = os.path.join(
            self.save_path,
            "debug_viz",
            str(self.session),
            f"iter_{self.iter}",
            route_type,
            f"{route_number}_{time.strftime('%Y_%m_%d_%H_%M_%S')}",
        )
        Path(self.debug_save_path).mkdir(parents=True, exist_ok=True)
        self.save_path_metric = os.path.join(self.debug_save_path, "metric")
        Path(self.save_path_metric).mkdir(parents=True, exist_ok=True)
        self.save_path_decision_trace = None
        self.save_path_decision_trace_images = None
        self.save_path_replay_bundle = None
        if self.eval_decision_trace_enabled:
            self.save_path_decision_trace = os.path.join(self.debug_save_path, "decision_trace")
            Path(self.save_path_decision_trace).mkdir(parents=True, exist_ok=True)
            if self.eval_decision_trace_pic:
                self.save_path_decision_trace_images = os.path.join(
                    self.save_path_decision_trace,
                    "images",
                )
                Path(self.save_path_decision_trace_images).mkdir(parents=True, exist_ok=True)
            if self.eval_decision_trace_replay:
                self.save_path_replay_bundle = os.path.join(
                    self.save_path_decision_trace,
                    "replay_bundle",
                )
                Path(self.save_path_replay_bundle).mkdir(parents=True, exist_ok=True)
            self._decision_trace_jsonl_path = os.path.join(
                self.save_path_decision_trace,
                "decision_trace.jsonl",
            )
            metadata_path = os.path.join(self.save_path_decision_trace, "metadata.json")
            metadata = {
                "route_key": self.eval_route_key,
                "route_id": self.eval_route_id,
                "route_name": self.eval_route_name,
                "route_file": self.eval_route_file,
                "scenario_name": self.eval_scenario_name,
                "town_name": self.eval_town_name,
                "runtime_profile": self.eval_runtime_profile,
                "predict_language": bool(self.eval_predict_language),
                "save_interval_steps": int(self.eval_decision_trace_interval),
                "pic": bool(self.eval_decision_trace_pic),
                "replay": bool(self.eval_decision_trace_replay),
                "image_dir": "images" if self.eval_decision_trace_pic else None,
                "replay_bundle_dir": "replay_bundle" if self.eval_decision_trace_replay else None,
            }
            with open(metadata_path, "w", encoding="utf-8") as outfile:
                json.dump(metadata, outfile, indent=4)

    def _ensure_eval_model_compat(self, cfg):
        if not hasattr(cfg, "model"):
            raise ValueError("Missing model config in model bootstrap config")

        if "adaption" not in cfg.model or cfg.model.adaption is None:
            raise ValueError("Missing model.adaption config in model bootstrap config")

        if self.eval_runtime_profile == "simlingo_compat":
            # simlingo_compat always loads the merged stage 1 model.
            cfg.model.adaption.simlingo_checkpoint = str(self.model_checkpoint_path)
            if hasattr(cfg.model, "simlingo_checkpoint"):
                cfg.model.simlingo_checkpoint = str(self.model_checkpoint_path)
            if hasattr(cfg, "simlingo_checkpoint"):
                cfg.simlingo_checkpoint = str(self.model_checkpoint_path)

        validate_adaption_config(cfg.model.adaption, context="model.adaption")

        has_smart_cfg = "smart_assigner" in cfg.model and cfg.model.smart_assigner is not None
        if not has_smart_cfg:
            raise ValueError(
                "Missing model.smart_assigner config for smart_assigner eval mode. "
                "Please provide full smart_assigner config explicitly."
            )

        smart_cfg = cfg.model.smart_assigner

        required_top_level = [
            "alpha",
            "t_lap",
            "scene_cross_attn_num_heads",
            "scene_deviation_mlp_hidden",
            "route_conv",
            "route_mlp_hidden",
            "speed_conv",
            "speed_mlp_hidden",
            "budget_scheduler_hidden",
            "path_scheduler_hidden",
            "path_tau",
            "path_is_hard",
            "query_init_std",
            "feature_dim",
        ]
        missing_top = [key for key in required_top_level if not hasattr(smart_cfg, key)]
        if missing_top:
            raise ValueError(
                f"smart_assigner config missing required fields: {missing_top}. "
                "Please provide full smart_assigner config explicitly."
            )

        for conv_name in ("route_conv", "speed_conv"):
            conv_cfg = getattr(smart_cfg, conv_name)
            if conv_cfg is None:
                raise ValueError(
                    f"smart_assigner.{conv_name} is missing. "
                    "Please provide full smart_assigner config explicitly."
                )
            missing_conv = [
                key for key in ("in_channels", "out_channels", "kernel_size", "stride", "padding")
                if not hasattr(conv_cfg, key)
            ]
            if missing_conv:
                raise ValueError(
                    f"smart_assigner.{conv_name} missing required fields: {missing_conv}. "
                    "Please provide full smart_assigner config explicitly."
                )

    def _validate_smart_feature_dim_after_model_init(self) -> None:
        if self.eval_budget_mode != "smart_assigner":
            return
        if not hasattr(self, "model") or self.model is None:
            raise ValueError("model must be initialized before smart feature_dim validation")
        runtime_hidden_size = int(getattr(self.model.language_model, "hidden_size", 0))
        if runtime_hidden_size <= 0:
            raise ValueError(
                "Failed to resolve runtime hidden_size from instantiated language model"
            )
        smart_cfg = getattr(self.cfg.model, "smart_assigner", None)
        if smart_cfg is None:
            raise ValueError("Missing model.smart_assigner config for smart_assigner eval mode")
        if int(smart_cfg.feature_dim) != runtime_hidden_size:
            raise ValueError(
                f"smart_assigner.feature_dim={smart_cfg.feature_dim} must equal language hidden_size={runtime_hidden_size}"
            )

    def _init(self):
        # The CARLA leaderboard does not expose the lat lon reference value of the GPS which make it impossible to use the
        # GPS because the scale is not known. In the past this was not an issue since the reference was constant 0.0
        # But town 13 has a different value in CARLA 0.9.15. The following code, adapted from Bench2DriveZoo estimates the
        # lat, lon reference values by abusing the fact that the leaderboard exposes the route plan also in CARLA
        # coordinates. The GPS plan is compared to the CARLA coordinate plan to estimate the reference point / scale
        # of the GPS. It seems to work reasonably well, so we use this workaround for now.
        try:
            locx, locy = self._global_plan_world_coord[0][0].location.x, self._global_plan_world_coord[0][0].location.y
            lon, lat = self._global_plan[0][0]['lon'], self._global_plan[0][0]['lat']
            earth_radius_equa = 6378137.0  # Constant from CARLA leaderboard GPS simulation
            def equations(variables):
                x, y = variables
                eq1 = (lon * math.cos(x * math.pi / 180.0) - (locx * x * 180.0) / (math.pi * earth_radius_equa)
                             - math.cos(x * math.pi / 180.0) * y)
                eq2 = (math.log(math.tan((lat + 90.0) * math.pi / 360.0)) * earth_radius_equa
                             * math.cos(x * math.pi / 180.0) + locy - math.cos(x * math.pi / 180.0) * earth_radius_equa
                             * math.log(math.tan((90.0 + x) * math.pi / 360.0)))
                return [eq1, eq2]
            initial_guess = [0.0, 0.0]
            solution = fsolve(equations, initial_guess)
            self.lat_ref, self.lon_ref = solution[0], solution[1]
        except Exception as e:
            print(e, flush=True)
            self.lat_ref, self.lon_ref = 0.0, 0.0
        self._route_planner = RoutePlanner(self.route_planner_min_distance, self.route_planner_max_distance,
                                                                             self.lat_ref, self.lon_ref)
        self._route_planner.set_route(self._global_plan, True)
        self.initialized = True
        self.metric_info = {}

    def sensors(self):
        sensors = []
        for num_cam in self.config.num_cameras:
            # get from config by name as string
            sensors += [
                    {
                            'type': 'sensor.camera.rgb',
                            'x': self.config.__dict__[f'camera_pos_{num_cam}'][0],
                            'y': self.config.__dict__[f'camera_pos_{num_cam}'][1],
                            'z': self.config.__dict__[f'camera_pos_{num_cam}'][2],
                            'roll': self.config.__dict__[f'camera_rot_{num_cam}'][0],
                            'pitch': self.config.__dict__[f'camera_rot_{num_cam}'][1],
                            'yaw': self.config.__dict__[f'camera_rot_{num_cam}'][2],
                            'width': self.config.__dict__[f'camera_width_{num_cam}'],
                            'height': self.config.__dict__[f'camera_height_{num_cam}'],
                            'fov': self.config.__dict__[f'camera_fov_{num_cam}'],
                            'id': f'rgb_{num_cam}'
                    }
            ]

        sensors += [{
                'type': 'sensor.other.imu',
                'x': 0.0,
                'y': 0.0,
                'z': 0.0,
                'roll': 0.0,
                'pitch': 0.0,
                'yaw': 0.0,
                'sensor_tick': self.config.carla_frame_rate,
                'id': 'imu'
        }, {
                'type': 'sensor.other.gnss',
                'x': 0.0,
                'y': 0.0,
                'z': 0.0,
                'roll': 0.0,
                'pitch': 0.0,
                'yaw': 0.0,
                'sensor_tick': 0.01,
                'id': 'gps'
        }, {
                'type': 'sensor.speedometer',
                'reading_frequency': self.config.carla_fps,
                'id': 'speed'
        }, 
        ]

        return sensors

    @torch.inference_mode()  # Turns off gradient computation
    def tick(self, input_data):
        """Pre-processes sensor data and runs the Unscented Kalman Filter"""
        rgb = []

        for camera_pos in self.config.num_cameras:
            rgb_cam = 'rgb_' + str(camera_pos)
            camera = input_data[rgb_cam][1][:, :, :3]
            if camera_pos == 0:
                self.camera_for_viz = camera.copy()

            # Also add jpg artifacts at test time, because the training data was saved as jpg.
            _, compressed_image_i = cv2.imencode('.jpg', camera)
            camera = cv2.imdecode(compressed_image_i, cv2.IMREAD_UNCHANGED)

            rgb_pos = cv2.cvtColor(camera, cv2.COLOR_BGR2RGB)
            rgb_pos = rgb_pos[:int(rgb_pos.shape[0] - (rgb_pos.shape[0] * 4.8) // 16), :, :] # do this from config to ensure it is the same as in training

            # Switch to pytorch channel first order
            rgb_pos = np.transpose(rgb_pos, (2, 0, 1))
            rgb.append(rgb_pos)

        rgb = np.array(rgb)
        self.image_buffer.append(rgb)

        rgbs = rgb
        image_sizes = None
        
        if 'internvl2' in self.cfg.model.vision_model.variant.lower():
            T, C, H, W = rgbs.shape
            transform = build_transform(input_size=448)
            images_processed_tmp = []
            images_sizes_tmp = []
            
            image = Image.fromarray(rgbs.squeeze(0).transpose(1, 2, 0))
            images = dynamic_preprocess(image, image_size=448, use_thumbnail=self.cfg.model.vision_model.use_global_img, max_num=2)
            pixel_values = [transform(image) for image in images]
            pixel_values = torch.stack(pixel_values)
            images_processed_tmp.append(pixel_values)
            images_sizes_tmp.append([image.size[1], image.size[0]])
            
            images_processed = {
                    'pixel_values': torch.stack(images_processed_tmp), 
                    'image_sizes': torch.tensor(images_sizes_tmp)
                    }  
            processed_image = images_processed['pixel_values']
            num_patches = processed_image.shape[1]
            new_height = processed_image.shape[3]
            new_width = processed_image.shape[4]
            processed_image = processed_image.view(1, self.T, num_patches, C, new_height, new_width)
            
        else:
            raise NotImplementedError(f"Encoder {self.cfg.data_module.encoder} not implemented yet")
        
        gps_pos = self._route_planner.convert_gps_to_carla(input_data['gps'][1])
        
        compass = t_u.preprocess_compass(input_data['imu'][1][-1])

        result = {
                'rgb': rgb,
                'compass': compass,
        }
        speed = input_data['speed'][1]['speed']

        if USE_UKF:
            if not self.filter_initialized:
                self.ukf.x = np.array([gps_pos[0], gps_pos[1], t_u.normalize_angle(compass), speed])
                self.filter_initialized = True

            self.ukf.predict(steer=self.control.steer, throttle=self.control.throttle, brake=self.control.brake)
            self.ukf.update(np.array([gps_pos[0], gps_pos[1], t_u.normalize_angle(compass), speed]))
            filtered_state = self.ukf.x

            self.state_log.append(filtered_state)
            result['gps'] = filtered_state[0:2]
        else:
            result['gps'] = np.array([gps_pos[0], gps_pos[1]])
            
        speed = round(input_data['speed'][1]['speed'], 1)

        waypoint_route = self._route_planner.run_step(np.append(result['gps'], gps_pos[2]))

        if len(waypoint_route) > 2:
            target_point, far_command = waypoint_route[1]
            next_target_point, next_far_command = waypoint_route[2]
        elif len(waypoint_route) > 1:
            target_point, far_command = waypoint_route[1]
            next_target_point, next_far_command = waypoint_route[1]
        else:
            target_point, far_command = waypoint_route[0]
            next_target_point, next_far_command = waypoint_route[0]
            
            
        if self.last_command_tmp != far_command:
            self.last_command = self.last_command_tmp
        
        self.last_command_tmp = far_command
        if (target_point != self.target_point_prev).all():
            self.target_point_prev = target_point
            self.commands.append(far_command.value)

        one_hot_command = t_u.command_to_one_hot(self.commands[-2])
        result['command'] = torch.from_numpy(one_hot_command[np.newaxis]).to(self.device, dtype=torch.float32)

        ego_target_point = t_u.inverse_conversion_2d(target_point[:2], result['gps'], result['compass'])
        ego_target_point_torch = torch.from_numpy(ego_target_point[np.newaxis]).to(self.device, dtype=torch.float32)
        ego_next_target_point = t_u.inverse_conversion_2d(next_target_point[:2], result['gps'], result['compass'])

        result['target_point'] = ego_target_point_torch

        self.target_points = None
        placeholder_batch_list = []

        if self.config.eval_route_as == 'target_point' or self.config.eval_route_as == 'target_point_command':
            target_points = [ego_target_point, ego_next_target_point]
            self.target_points = target_points.copy()
            target_points_np = np.array(target_points)
            target_points = torch.from_numpy(target_points_np).to(self.device, dtype=torch.float32).unsqueeze(0)
            result['route'] = target_points
            
            placeholder_values = {'<TARGET_POINT>': target_points_np}
            tmp = {}
            for key, value in placeholder_values.items():
                    token_nr_key = self.tokenizer.convert_tokens_to_ids(key)
                    tmp[token_nr_key] = value
            placeholder_batch_list.append(tmp)
            
            prompt_tp = "Target waypoint: <TARGET_POINT><TARGET_POINT>."
            
        elif self.config.eval_route_as == 'command':
            # get distance from target_point
            dist_to_command = np.linalg.norm(ego_target_point)
            dist_to_command = int(dist_to_command)
            map_command = {
                    1: 'go left at the next intersection',
                    2: 'go right at the next intersection',
                    3: 'go straight at the next intersection',
                    4: 'follow the road',
                    5: 'do a lane change to the left',
                    6: 'do a lane change to the right',        
            }
            command = map_command[far_command]
            next_command = map_command[next_far_command]
            if self.last_command in [1, 2, 3] and far_command == 4:
                next_command = command
                command = map_command[self.last_command]

            if command != next_command:
                    next_command = f' then {next_command}'
            else:
                    next_command = ''

            if far_command == 4:
                    prompt_tp = f'Command: {command}{next_command}.'
            else:
                    prompt_tp = f'Command: {command} in {dist_to_command} meter{next_command}.'
                
        else:
            result['route'] = route_img

        if self.config.use_cot:
            prompt = f"Current speed: {speed} m/s. {prompt_tp} What should the ego do next?"
        else:
            prompt = f"Current speed: {speed} m/s. {prompt_tp} Predict the waypoints."

        result['speed'] = torch.FloatTensor([speed]).unsqueeze(0).to(self.device, dtype=torch.float32)

        B, T, num_patches, C, H, W = processed_image.shape
        assert B == 1
        assert T == self.T
        assert C == 3

        speed = round(speed, 1)
        
        self.prompt_tp = prompt_tp
        self.prompt = prompt
        
        conversation_all = [
                {
                "role": "user",
                "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image"},
                        ],
                },
                {
                "role": "assistant",
                "content": [
                        {"type": "text", "text": "Waypoints:"},
                        ],
                },
        ]
        conv_batch_list = [conversation_all]
        questions = []
        for conv in conv_batch_list:
                for i in range(len(conv)):
                        questions.append(conv[i]['content'][0]['text'])
                        conv[i]['content'] = conv[i]['content'][0]['text']
                        
        # Prompt templates and model assets share one canonical directory.
        from simlingo_adaption_training.models.language_model.internvl_2_1b.conversation import (
            get_conv_template,
        )
        
        if not hasattr(self, 'tmp_config'):
                self.tmp_config = AutoConfig.from_pretrained(
                    self.internvl_model_source,
                    trust_remote_code=True,
                )
                image_size = self.tmp_config.force_image_size or self.tmp_config.vision_config.image_size
                patch_size = self.tmp_config.vision_config.patch_size
                
                self.num_image_token = int((image_size // patch_size) ** 2 * (self.tmp_config.downsample_ratio ** 2))
                
        prompt_batch_list = []
        for idx, conv in enumerate(conv_batch_list):
                question = questions[idx]
                if '<image>' not in question:
                        question = '<image>\n' + question
                template = get_conv_template('internlm2-chat')
                template_inference = None
                
                template_inference = get_conv_template('internlm2-chat')
                for conv_part_idx, conv_part in enumerate(conv):
                        if conv_part['role'] == 'assistant':
                                template.append_message(template.roles[1], None)
                        elif conv_part['role'] == 'user':
                                if conv_part_idx == 0 and '<image>' not in conv_part['content']:
                                        # add image token
                                        conv_part['content'] = '<image>\n' + conv_part['content']
                                template.append_message(template.roles[0], conv_part['content'])
                        else:
                                raise ValueError(f"Role {conv_part['role']} not supported")
                            
                query = template.get_prompt()
                # remove system prompt
                system_prompt = template.system_template.replace('{system_message}', template.system_message) + template.sep
                query = query.replace(system_prompt, '')
                
                IMG_START_TOKEN='<img>'
                IMG_END_TOKEN='</img>'
                IMG_CONTEXT_TOKEN='<IMG_CONTEXT>'
                num_patches_all = 2

                image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.num_image_token * num_patches_all + IMG_END_TOKEN
                query = query.replace('<image>', image_tokens, 1)
                prompt_batch_list.append(query)
                
        prompt_tokenized = self.tokenizer(prompt_batch_list, padding=True, return_tensors="pt", return_offsets_mapping=True, add_special_tokens=False)
        prompt_tokenized_ids = prompt_tokenized["input_ids"]
        prompt_tokenized_char_offsets = prompt_tokenized["offset_mapping"].view(1, -1, 2)
        prompt_tokenized_valid = prompt_tokenized["input_ids"] != self.tokenizer.pad_token_id
        prompt_tokenized_mask = prompt_tokenized_valid
        prompt_token_ids_list = prompt_tokenized_ids[0].detach().cpu().tolist()
        self._latest_prompt_token_ids = [int(token_id) for token_id in prompt_token_ids_list]
        self._latest_prompt_tokens = list(
            self.tokenizer.convert_ids_to_tokens(self._latest_prompt_token_ids)
        )
        self._latest_prompt_text = str(prompt_batch_list[0])
        
        ll = LanguageLabel(
                phrase_ids=prompt_tokenized_ids.to(self.device),
                phrase_valid=prompt_tokenized_valid.to(self.device),
                phrase_mask=prompt_tokenized_mask.to(self.device),
                placeholder_values=placeholder_batch_list,
                language_string=prompt_batch_list,
                loss_masking=None,
        )

        self.DrivingInput["camera_images"] = processed_image.to(self.device).bfloat16()
        self.DrivingInput["image_sizes"] = image_sizes
        self.DrivingInput["camera_intrinsics"] = torch.repeat_interleave(get_camera_intrinsics(W, H, 110).unsqueeze(0), 1, dim=0).view(1, 3, 3).float().to(self.device),
        self.DrivingInput["camera_extrinsics"] = torch.repeat_interleave(get_camera_extrinsics().unsqueeze(0), 1, dim=0).view(1, 4, 4).float().to(self.device),
        self.DrivingInput["vehicle_speed"] = result['speed']
        self.DrivingInput["target_point"] = result['target_point'].to(self.device)
        self.DrivingInput["prompt"] = ll
        self.DrivingInput["prompt_inference"] = ll

        return result

    def _forward_model(self, model_input, budget_target, model_route_keys):
        self.latest_forward_wall_clock_latency_ms = None
        self.latest_forward_power = None
        if not self.measure_wall_clock_latency and not self.measure_power:
            return self.model(
                model_input,
                budget=budget_target,
                route_keys=model_route_keys,
            )

        if not torch.cuda.is_available():
            raise RuntimeError("Inference measurement requires CUDA")

        # Synchronization measures only the model forward pass.
        torch.cuda.synchronize()
        power_started = False
        try:
            if self.measure_power:
                self.power_sampler.start()
                power_started = True
            wall_start = time.perf_counter() if self.measure_wall_clock_latency else None
            outputs = self.model(
                model_input,
                budget=budget_target,
                route_keys=model_route_keys,
            )
            torch.cuda.synchronize()
            if wall_start is not None:
                self.latest_forward_wall_clock_latency_ms = (
                    time.perf_counter() - wall_start
                ) * 1000.0
        finally:
            if power_started:
                self.latest_forward_power = self.power_sampler.stop()
        return outputs

    @torch.no_grad()
    def run_step(self, input_data, timestamp, sensors=None):  # pylint: disable=locally-disabled, unused-argument
        self.step += 1

        if not self.initialized:
            self._init()
            control = carla.VehicleControl(steer=0.0, throttle=0.0, brake=1.0)
            self.control = control
            tick_data = self.tick(input_data)
            return control

        # Need to run this every step for GPS filtering
        tick_data = self.tick(input_data)
        # Guard against external code restoring training mode.
        if self.model.training:
            self.model.eval()
        self.DrivingInput["ego_xy"] = torch.tensor(tick_data["gps"], dtype=torch.float32, device=self.device).unsqueeze(0)
        self.DrivingInput["ego_yaw"] = torch.tensor([tick_data["compass"]], dtype=torch.float32, device=self.device)
        self.DrivingInput["timestamp"] = torch.tensor([float(timestamp)], dtype=torch.float32, device=self.device)

        # initialize DrivingInput with dict self.DrivingInput
        model_input = DrivingInput(**self.DrivingInput)
        budget_target = self.budget_input_process()
        model_route_keys = [self.eval_route_key]
        pred_speed_wps, pred_route, language = self._forward_model(
            model_input,
            budget_target=budget_target,
            model_route_keys=model_route_keys,
        )
        model_eval_budget = getattr(self.model, "eval_budget", None)
        if not isinstance(model_eval_budget, dict):
            raise ValueError("model.eval_budget must be dict after model forward")
        # Budget diagnostics come from the model forward output.
        self.eval_budget = copy.deepcopy(model_eval_budget)
        self.eval_budget["runtime_profile"] = self.eval_runtime_profile
        self._record_language_sample(language)
        pred_speed_wps = pred_speed_wps.float() if pred_speed_wps is not None else None
        pred_route = pred_route.float() if pred_route is not None else None

        # prepare velocity input
        gt_velocity = tick_data['speed']

        steer, throttle, brake = self.control_pid(pred_route, gt_velocity, pred_speed_wps)

        if gt_velocity < 0.1:
            self.stuck_detector += 1
        else:
            self.stuck_detector = 0

        # Restart mechanism in case the car got stuck. Not used a lot anymore but doesn't hurt to keep it.
        if self.stuck_detector > self.config.stuck_threshold:
            self.force_move = self.config.creep_duration

        if self.force_move > 0:
            throttle = max(self.config.creep_throttle, throttle)
            brake = False
            self.force_move -= 1
            print(f"force_move: {self.force_move}")

        control = carla.VehicleControl(steer=float(steer), throttle=float(throttle), brake=float(brake))

        # CARLA will not let the car drive in the initial frames.
        # We set the action to brake so that the filter does not get confused.
        if self.step < self.config.inital_frames_delay:
            self.control = carla.VehicleControl(0.0, 0.0, 1.0)
        else:
            self.control = control
            
        metric_info = self.get_metric_info()
        metric_info["eval_budget"] = copy.deepcopy(self.eval_budget)
        metric_info["runtime_profile"] = self.eval_runtime_profile
        metric_info["predict_language"] = bool(self.eval_predict_language)
        metric_info["language_sample_seen"] = int(self._language_sample_seen)
        metric_info["language_sample_kept"] = int(len(self._language_samples))
        if self.measure_wall_clock_latency or self.measure_power:
            inference_measurements = {}
            if self.measure_wall_clock_latency:
                inference_measurements["wall_clock_latency_ms"] = (
                    self.latest_forward_wall_clock_latency_ms
                )
            if self.measure_power:
                inference_measurements["power"] = copy.deepcopy(self.latest_forward_power)
            metric_info["inference_measurements"] = inference_measurements
        self.metric_info[self.step] = metric_info
        if self.save_path_metric is not None and self.step % 1 == 0:
            metric_path = os.path.join(self.save_path_metric, "metric_info.json")
            with open(metric_path, "w", encoding="utf-8") as outfile:
                json.dump(self.metric_info, outfile, indent=4)
            self._flush_language_samples()
        self._maybe_save_replay_bundle(
            timestamp=timestamp,
            driving_input=model_input,
        )
        self._maybe_save_decision_trace(
            timestamp=timestamp,
            language=language,
            planned_control=control,
            applied_control=self.control,
            pred_route=pred_route,
            pred_speed_wps=pred_speed_wps,
            gt_velocity=gt_velocity,
        )

        return control

    def control_pid(self, route_waypoints, velocity, speed_waypoints):
        """
        Predicts vehicle control with a PID controller.
        Used for waypoint predictions
        """
        assert route_waypoints.size(0) == 1
        route_waypoints = route_waypoints[0].data.cpu().numpy()
        speed = velocity[0].data.cpu().numpy()
        speed_waypoints = speed_waypoints[0].data.cpu().numpy()

        # m / s required to drive
        one_second = int(self.config.carla_fps // (self.config.wp_dilation * self.config.data_save_freq))
        half_second = one_second // 2
        desired_speed = np.linalg.norm(speed_waypoints[half_second - 2] - speed_waypoints[one_second - 2]) * 2.0

        brake = ((desired_speed < self.config.brake_speed) or ((speed / desired_speed) > self.config.brake_ratio))

        delta = np.clip(desired_speed - speed, 0.0, self.config.clip_delta)
        throttle = self.speed_controller.step(delta)
        throttle = np.clip(throttle, 0.0, self.config.clip_throttle)
        throttle = throttle if not brake else 0.0

        route_interp = self.interpolate_waypoints(route_waypoints.squeeze())

        steer = self.turn_controller.step(route_interp, speed)

        steer = np.clip(steer, -1.0, 1.0)
        steer = round(steer, 3)

        return steer, throttle, brake
    
    # In: Waypoints NxD
    # Out: Waypoints NxD equally spaced 0.1 across D
    def interpolate_waypoints(self, waypoints):
            waypoints = waypoints.copy()
            waypoints = np.concatenate((np.zeros_like(waypoints[:1]), waypoints))
            shift = np.roll(waypoints, 1, axis=0)
            shift[0] = shift[1]

            dists = np.linalg.norm(waypoints-shift, axis=1)
            dists = np.cumsum(dists)
            dists += np.arange(0, len(dists)) * 1e-4 # Prevents dists not being strictly increasing

            interp = PchipInterpolator(dists, waypoints, axis=0)

            x = np.arange(0.1, dists[-1], 0.1)

            interp_points = interp(x)

            # There is a possibility that all points are at 0, meaning there is no point distanced 0.1
            # In this case we output the last (assumed to be furthest) waypoint.
            if interp_points.shape[0] == 0:
                    interp_points = waypoints[None, -1]

            return interp_points

    def _record_language_sample(self, language):
        if language is None:
            return
        if not isinstance(language, list):
            raise ValueError(f"language output must be list[str], got {type(language)}")
        if len(language) == 0:
            return

        for language_item in language:
            if not isinstance(language_item, str):
                raise ValueError(f"language item must be str, got {type(language_item)}")
            self._language_sample_seen += 1
            sample = {
                "step": int(self.step),
                "language": language_item,
                "prompt": str(self.prompt),
                "eval_budget": copy.deepcopy(self.eval_budget),
            }
            if len(self._language_samples) < self.language_sample_count:
                self._language_samples.append(sample)
                continue
            replace_rank = self._language_sample_rng.randint(1, int(self._language_sample_seen))
            if replace_rank <= self.language_sample_count:
                self._language_samples[replace_rank - 1] = sample

    def _flush_language_samples(self):
        if self.save_path_metric is None:
            raise ValueError("save_path_metric is required for language sample logging")
        output_dir = Path(self.save_path_metric)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "language_samples.json"
        language_payload = {
            "route_key": self.eval_route_key,
            "runtime_profile": self.eval_runtime_profile,
            "predict_language": bool(self.eval_predict_language),
            "target_count": int(self.language_sample_count),
            "seen_steps": int(self._language_sample_seen),
            "samples": self._language_samples,
        }
        temp_path = output_path.with_suffix(".json.tmp")
        with open(temp_path, "w", encoding="utf-8") as outfile:
            json.dump(language_payload, outfile, indent=4)
        os.replace(temp_path, output_path)

    def _save_decision_trace_image(self, step: int) -> Optional[str]:
        if not self.eval_decision_trace_pic:
            return None
        if self.save_path_decision_trace_images is None:
            return None
        if not hasattr(self, "camera_for_viz") or self.camera_for_viz is None:
            return None
        image_array = cv2.cvtColor(self.camera_for_viz, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image_array)
        image_path = os.path.join(self.save_path_decision_trace_images, f"{int(step):06d}.png")
        image.save(image_path)
        return image_path

    def _to_serializable_cpu(self, value):
        if value is None:
            return None
        if torch.is_tensor(value):
            return value.detach().cpu()
        if isinstance(value, np.ndarray):
            return value.copy()
        if isinstance(value, dict):
            return {
                key: self._to_serializable_cpu(item)
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [self._to_serializable_cpu(item) for item in value]
        if isinstance(value, tuple):
            return tuple(self._to_serializable_cpu(item) for item in value)
        return value

    def _serialize_language_label(self, label):
        if label is None:
            return None
        return {
            "phrase_ids": self._to_serializable_cpu(label.phrase_ids),
            "phrase_valid": self._to_serializable_cpu(label.phrase_valid),
            "phrase_mask": self._to_serializable_cpu(label.phrase_mask),
            "placeholder_values": self._to_serializable_cpu(label.placeholder_values),
            "language_string": self._to_serializable_cpu(label.language_string),
            "loss_masking": self._to_serializable_cpu(label.loss_masking),
        }

    def _serialize_driving_input(self, driving_input: DrivingInput):
        return {
            "camera_images": self._to_serializable_cpu(driving_input.camera_images),
            "image_sizes": self._to_serializable_cpu(driving_input.image_sizes),
            "camera_intrinsics": self._to_serializable_cpu(driving_input.camera_intrinsics),
            "camera_extrinsics": self._to_serializable_cpu(driving_input.camera_extrinsics),
            "vehicle_speed": self._to_serializable_cpu(driving_input.vehicle_speed),
            "target_point": self._to_serializable_cpu(driving_input.target_point),
            "prompt": self._serialize_language_label(driving_input.prompt),
            "prompt_inference": self._serialize_language_label(driving_input.prompt_inference),
            "ego_xy": self._to_serializable_cpu(driving_input.ego_xy),
            "ego_yaw": self._to_serializable_cpu(driving_input.ego_yaw),
            "timestamp": self._to_serializable_cpu(driving_input.timestamp),
        }

    def _maybe_save_replay_bundle(
        self,
        timestamp,
        driving_input: DrivingInput,
    ) -> None:
        if not self.eval_decision_trace_enabled:
            return
        if not self.eval_decision_trace_replay:
            return
        if self.save_path_replay_bundle is None:
            raise ValueError("replay bundle output path is required when decision trace is enabled")
        if self.step <= 0 or self.step % self.eval_decision_trace_interval != 0:
            return

        bundle_path = os.path.join(self.save_path_replay_bundle, f"{int(self.step):06d}.pt")
        replay_bundle = {
            "step": int(self.step),
            "timestamp": float(timestamp),
            "route_key": self.eval_route_key,
            "runtime_profile": self.eval_runtime_profile,
            "predict_language": bool(self.eval_predict_language),
            "prompt": str(self.prompt),
            "prompt_text": self._latest_prompt_text,
            "input_token_ids": list(self._latest_prompt_token_ids),
            "input_tokens": list(self._latest_prompt_tokens),
            "eval_budget": copy.deepcopy(self.eval_budget),
            "driving_input": self._serialize_driving_input(driving_input),
        }
        torch.save(replay_bundle, bundle_path)

    def _maybe_save_decision_trace(
        self,
        timestamp,
        language,
        planned_control,
        applied_control,
        pred_route,
        pred_speed_wps,
        gt_velocity,
    ) -> None:
        if not self.eval_decision_trace_enabled:
            return
        if self._decision_trace_jsonl_path is None:
            raise ValueError("decision trace output path is required when trace is enabled")
        if self.step <= 0 or self.step % self.eval_decision_trace_interval != 0:
            return

        image_path = self._save_decision_trace_image(step=self.step)
        trace_record = {
            "step": int(self.step),
            "timestamp": float(timestamp),
            "route_key": self.eval_route_key,
            "route_id": self.eval_route_id,
            "route_name": self.eval_route_name,
            "scenario_name": self.eval_scenario_name,
            "town_name": self.eval_town_name,
            "prompt": str(self.prompt),
            "prompt_text": self._latest_prompt_text,
            "input_token_ids": list(self._latest_prompt_token_ids),
            "input_tokens": list(self._latest_prompt_tokens),
            "output_language": list(language) if language is not None else [],
            "planned_control": {
                "steer": float(planned_control.steer),
                "throttle": float(planned_control.throttle),
                "brake": float(planned_control.brake),
            },
            "applied_control": {
                "steer": float(applied_control.steer),
                "throttle": float(applied_control.throttle),
                "brake": float(applied_control.brake),
            },
            "vehicle_speed": float(gt_velocity[0].item()),
            "pred_route": pred_route[0].detach().cpu().tolist() if pred_route is not None else None,
            "pred_speed_wps": (
                pred_speed_wps[0].detach().cpu().tolist()
                if pred_speed_wps is not None
                else None
            ),
            "eval_budget": copy.deepcopy(self.eval_budget),
            "image_path": image_path,
        }
        with open(self._decision_trace_jsonl_path, "a", encoding="utf-8") as outfile:
            outfile.write(json.dumps(trace_record, ensure_ascii=False) + "\n")

    def destroy(self, results=None):  # pylint: disable=locally-disabled, unused-argument
        """
        Gets called after a route finished.
        The leaderboard client doesn't properly clear up the agent after the route finishes so we need to do it here.
        Also writes logging files to disk.
        """
        if hasattr(self, "save_path_metric") and self.save_path_metric is not None:
            self._flush_language_samples()
        if hasattr(self, "power_sampler") and self.power_sampler is not None:
            self.power_sampler.close()
        if hasattr(self, "model") and hasattr(self.model, "budget_assigner"):
            self.model.budget_assigner.reset()
        if hasattr(self, "model"):
            del self.model
        if hasattr(self, "config"):
            del self.config
        if hasattr(self, "cfg") and hasattr(self.cfg.data_module, 'encoder') and self.cfg.data_module.encoder == 'llavanext':
            del self.processor

    def budget_input_process(self):
        if self.eval_budget_policy == "stage1":
            if self.eval_fixed_budget is None:
                raise ValueError("eval_fixed_budget is required when budget_policy=stage1")
            return float(self.eval_fixed_budget)
        if self.eval_budget_policy == "stage2":
            return None
        raise ValueError(
            f"Unsupported budget policy={self.eval_budget_policy}, expected one of ['stage1', 'stage2']"
        )

# Filter Functions
def bicycle_model_forward(x, dt, steer, throttle, brake):
    # Kinematic bicycle model.
    # Numbers are the tuned parameters from World on Rails
    front_wb = -0.090769015
    rear_wb = 1.4178275

    steer_gain = 0.36848336
    brake_accel = -4.952399
    throt_accel = 0.5633837

    locs_0 = x[0]
    locs_1 = x[1]
    yaw = x[2]
    speed = x[3]

    if brake:
        accel = brake_accel
    else:
        accel = throt_accel * throttle

    wheel = steer_gain * steer

    beta = math.atan(rear_wb / (front_wb + rear_wb) * math.tan(wheel))
    next_locs_0 = locs_0.item() + speed * math.cos(yaw + beta) * dt
    next_locs_1 = locs_1.item() + speed * math.sin(yaw + beta) * dt
    next_yaws = yaw + speed / rear_wb * math.sin(beta) * dt
    next_speed = speed + accel * dt
    next_speed = next_speed * (next_speed > 0.0)  # Fast ReLU

    next_state_x = np.array([next_locs_0, next_locs_1, next_yaws, next_speed])

    return next_state_x


def measurement_function_hx(vehicle_state):
    '''
        For now we use the same internal state as the measurement state
        :param vehicle_state: VehicleState vehicle state variable containing
                                                    an internal state of the vehicle from the filter
        :return: np array: describes the vehicle state as numpy array.
                                             0: pos_x, 1: pos_y, 2: rotatoion, 3: speed
        '''
    return vehicle_state


def state_mean(state, wm):
    '''
        We use the arctan of the average of sin and cos of the angle to calculate
        the average of orientations.
        :param state: array of states to be averaged. First index is the timestep.
        :param wm:
        :return:
        '''
    x = np.zeros(4)
    sum_sin = np.sum(np.dot(np.sin(state[:, 2]), wm))
    sum_cos = np.sum(np.dot(np.cos(state[:, 2]), wm))
    x[0] = np.sum(np.dot(state[:, 0], wm))
    x[1] = np.sum(np.dot(state[:, 1], wm))
    x[2] = math.atan2(sum_sin, sum_cos)
    x[3] = np.sum(np.dot(state[:, 3], wm))

    return x


def measurement_mean(state, wm):
    '''
    We use the arctan of the average of sin and cos of the angle to
    calculate the average of orientations.
    :param state: array of states to be averaged. First index is the
    timestep.
    '''
    x = np.zeros(4)
    sum_sin = np.sum(np.dot(np.sin(state[:, 2]), wm))
    sum_cos = np.sum(np.dot(np.cos(state[:, 2]), wm))
    x[0] = np.sum(np.dot(state[:, 0], wm))
    x[1] = np.sum(np.dot(state[:, 1], wm))
    x[2] = math.atan2(sum_sin, sum_cos)
    x[3] = np.sum(np.dot(state[:, 3], wm))

    return x


def residual_state_x(a, b):
    y = a - b
    y[2] = t_u.normalize_angle(y[2])
    return y


def residual_measurement_h(a, b):
    y = a - b
    y[2] = t_u.normalize_angle(y[2])
    return y
