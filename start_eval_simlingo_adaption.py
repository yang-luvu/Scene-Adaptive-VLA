import os
import sys
import subprocess
import time
import shutil
import argparse
import copy
import ujson
import yaml
from collections import deque
from tqdm.autonotebook import tqdm
from typing import Any, Dict, List, Optional

from Bench2Drive.tools.merge_route_json import is_success_record
from simlingo_adaption_training.utils.project_paths import (
    DEFAULT_PATHS_CONFIG,
    ProjectPaths,
)



VULKAN_GPU_ID = {0: 0, 1: 1}  # Host-specific CUDA-to-Vulkan device mapping.
POLL_INTERVAL_S = float(os.getenv("SIMLINGO_POLL_INTERVAL", "5.0"))
SIMLINGO_COMPAT_CHECKPOINT_REL_PATH = "models/simlingoWithoutPEFT/merged_driving_model.pth"
SIMLINGO_SMART_ASSIGNER_BOOTSTRAP_REL_PATH = (
    "models/simlingoWithoutPEFT/simlingo_compat_model_bootstrap.yaml"
)
FAILED_ROUTE_STATUSES = {
    "Failed - Agent couldn't be set up",
    "Failed",
    "Failed - Simulation crashed",
    "Failed - Agent crashed",
}

carla_world_ports = set(range(10000, 20000, 50))
carla_tm_ports = set(range(30000, 40000, 50))


def expand_path(path: str) -> str:
    return os.path.expanduser(path)


def normalize_optional_path(path) -> Optional[str]:
    if path is None:
        return None
    if isinstance(path, str) and path.strip().lower() in {"", "none", "null"}:
        return None
    return os.path.expanduser(str(path))


def is_failed_status(status: Optional[str]) -> bool:
    if not isinstance(status, str):
        return False
    normalized = status.strip()
    return normalized in FAILED_ROUTE_STATUSES or normalized.startswith("Failed")


def needs_resubmit(job) -> bool:
    result_file = job["result_file"]
    if not os.path.exists(result_file):
        return True
    try:
        with open(result_file, "r", encoding="utf-8") as f:
            evaluation_data = ujson.load(f)
    except Exception:
        return True

    checkpoint_data = evaluation_data.get("_checkpoint", {})
    progress = checkpoint_data.get("progress")
    if not isinstance(progress, list):
        return True
    if len(progress) < 2 or progress[0] < progress[1]:
        return True

    global_record = checkpoint_data.get("global_record", {})
    if is_failed_status(global_record.get("status")):
        return True

    records = checkpoint_data.get("records", [])
    for record in records:
        if is_failed_status(record.get("status")):
            return True

    # Success requires route completion and no non-speed infractions.
    if job["cfg"]["retry_unsuccess"]:
        if not records or any(not is_success_record(record) for record in records):
            return True

    return False


def cleanup_viz_dir(path: str) -> None:
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def load_eval_yaml(config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data


def _parse_fixed_budget_value(raw_value, field_name: str) -> float:
    try:
        fixed_budget = float(raw_value)
    except Exception as exc:
        raise ValueError(
            f"{field_name} must be float in [0, 1], got {raw_value}"
        ) from exc
    if fixed_budget < 0.0 or fixed_budget > 1.0:
        raise ValueError(f"{field_name} must be in [0, 1], got {fixed_budget}")
    return fixed_budget


def format_budget_dir_name(policy: str, fixed_budget: Optional[float]) -> str:
    if policy == "stage1":
        if fixed_budget is None:
            raise ValueError("fixed_budget is required for stage1 budget directory")
        budget_label = f"{fixed_budget:.6g}".replace("-", "m").replace(".", "p")
        return f"stage1_fixed_{budget_label}"
    return "stage2"


def format_layer_selection_dir_name(layer_selection_mode: str) -> str:
    layer_selection_mode = str(layer_selection_mode).strip().lower()
    if layer_selection_mode not in {"stage1", "random"}:
        raise ValueError(
            f"Unsupported layer_selection_mode={layer_selection_mode}, expected one of ['random', 'stage1']"
        )
    return f"layer_select_{layer_selection_mode}"


def parse_budget_settings(data) -> Dict[str, Any]:
    eval_cfg = data.get("eval", data)
    budget_cfg = data.get("budget", eval_cfg.get("budget", {})) or {}

    mode = str(budget_cfg.get("mode", "smart_assigner")).strip().lower()
    if mode != "smart_assigner":
        raise ValueError(f"This version only supports budget.mode=smart_assigner, got {mode}")
    if "rule_based" in budget_cfg:
        raise ValueError("budget.rule_based is deprecated and unsupported in smart_assigner-only mode")

    policy_raw = budget_cfg.get("policy", None)
    if policy_raw is None:
        policy = "stage1" if (
            "fixed_budget" in budget_cfg or "fixed_budgets" in budget_cfg
        ) else "stage2"
    else:
        policy = str(policy_raw).strip().lower()
    allowed_policies = {"stage1", "stage2"}
    if policy not in allowed_policies:
        raise ValueError(
            f"Unsupported budget.policy={policy}, expected one of {sorted(allowed_policies)}"
        )

    layer_selection_mode = str(budget_cfg.get("layer_selection_mode", "stage1")).strip().lower()
    allowed_layer_selection_modes = {"stage1", "random"}
    if layer_selection_mode not in allowed_layer_selection_modes:
        raise ValueError(
            "Unsupported budget.layer_selection_mode="
            f"{layer_selection_mode}, expected one of {sorted(allowed_layer_selection_modes)}"
        )

    fixed_budgets = [None]
    if policy == "stage1":
        has_fixed_budget = "fixed_budget" in budget_cfg
        has_fixed_budgets = "fixed_budgets" in budget_cfg
        if has_fixed_budget and has_fixed_budgets:
            raise ValueError(
                "Use only one of budget.fixed_budget or budget.fixed_budgets when budget.policy=stage1"
            )
        if not has_fixed_budget and not has_fixed_budgets:
            raise KeyError(
                "Missing required field: budget.fixed_budget or budget.fixed_budgets "
                "when budget.policy=stage1"
            )

        if has_fixed_budgets:
            raw_fixed_budgets = budget_cfg["fixed_budgets"]
            if not isinstance(raw_fixed_budgets, list) or len(raw_fixed_budgets) == 0:
                raise ValueError("budget.fixed_budgets must be a non-empty list")
            fixed_budgets = [
                _parse_fixed_budget_value(value, "budget.fixed_budgets")
                for value in raw_fixed_budgets
            ]
        else:
            fixed_budgets = [
                _parse_fixed_budget_value(budget_cfg["fixed_budget"], "budget.fixed_budget")
            ]
    elif "fixed_budget" in budget_cfg or "fixed_budgets" in budget_cfg:
        raise ValueError(
            "budget.fixed_budget/fixed_budgets are only allowed when budget.policy=stage1"
        )

    limit_language_inference = _parse_bool_flag(
        budget_cfg.get("limit_language_inference", True),
        field_name="budget.limit_language_inference",
    )
    fixed_random_layer_selection = _parse_bool_flag(
        budget_cfg.get("fixed_random_layer_selection", False),
        field_name="budget.fixed_random_layer_selection",
    )
    eval_budget_sampling = _parse_bool_flag(
        budget_cfg.get("eval_budget_sampling", False),
        field_name="budget.eval_budget_sampling",
    )
    eval_budget_sampling_temperature = float(
        budget_cfg.get("eval_budget_sampling_temperature", 1.0)
    )
    if eval_budget_sampling and eval_budget_sampling_temperature <= 0.0:
        raise ValueError(
            "budget.eval_budget_sampling_temperature must be > 0 when "
            "budget.eval_budget_sampling is enabled"
        )

    return {
        "mode": mode,
        "policy": policy,
        "fixed_budgets": fixed_budgets,
        "layer_selection_mode": layer_selection_mode,
        "limit_language_inference": limit_language_inference,
        "fixed_random_layer_selection": fixed_random_layer_selection,
        "eval_budget_sampling": eval_budget_sampling,
        "eval_budget_sampling_temperature": eval_budget_sampling_temperature,
    }


def parse_runtime_profile(data):
    eval_cfg = data.get("eval", data)
    if "runtime_profile" not in eval_cfg:
        raise KeyError("Missing required field: eval.runtime_profile")

    runtime_profile = str(eval_cfg["runtime_profile"]).strip().lower()
    allowed_profiles = {"smart_assigner", "simlingo_compat"}
    if runtime_profile not in allowed_profiles:
        raise ValueError(
            f"Unsupported eval.runtime_profile={runtime_profile}, "
            f"expected one of {sorted(allowed_profiles)}"
        )
    return runtime_profile


def parse_language_sample_count(data):
    eval_cfg = data.get("eval", data)
    if "language_sample_count" not in eval_cfg:
        raise KeyError("Missing required field: eval.language_sample_count")

    raw_value = eval_cfg["language_sample_count"]
    try:
        language_sample_count = int(raw_value)
    except Exception as exc:
        raise ValueError(
            f"eval.language_sample_count must be integer, got {raw_value}"
        ) from exc
    if language_sample_count <= 0:
        raise ValueError(
            f"eval.language_sample_count must be > 0, got {language_sample_count}"
        )
    return language_sample_count


def _parse_bool_flag(value, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        if value in (0, 1):
            return bool(value)
        raise ValueError(f"{field_name} must be bool/0/1, got {value}")
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    raise ValueError(f"{field_name} must be bool, got {value!r}")


def parse_predict_language(data, runtime_profile: str) -> bool:
    eval_cfg = data.get("eval", data)
    raw_value = eval_cfg.get("predict_language", None)
    if raw_value is None:
        return runtime_profile == "simlingo_compat"
    return _parse_bool_flag(raw_value, field_name="eval.predict_language")


def parse_retry_failed(data) -> bool:
    eval_cfg = data.get("eval", data)
    raw_value = eval_cfg.get("retry_failed", None)
    if raw_value is None:
        return False
    return _parse_bool_flag(raw_value, field_name="eval.retry_failed")


def parse_retry_unsuccess(data) -> bool:
    eval_cfg = data.get("eval", data)
    raw_value = eval_cfg.get("retry_unsuccess", None)
    if raw_value is None:
        return False
    return _parse_bool_flag(raw_value, field_name="eval.retry_unsuccess")


def parse_measurement_settings(data) -> Dict[str, Any]:
    eval_cfg = data.get("eval", data)
    measurement_cfg = eval_cfg.get("measurement", {})
    if not isinstance(measurement_cfg, dict):
        raise ValueError("eval.measurement must be a mapping when provided")

    wall_clock_latency = _parse_bool_flag(
        measurement_cfg.get("wall_clock_latency", False),
        field_name="eval.measurement.wall_clock_latency",
    )
    power = _parse_bool_flag(
        measurement_cfg.get("power", False),
        field_name="eval.measurement.power",
    )
    power_sample_interval_ms = float(measurement_cfg.get("power_sample_interval_ms", 10.0))
    if power and power_sample_interval_ms <= 0.0:
        raise ValueError(
            "eval.measurement.power_sample_interval_ms must be > 0 when power measurement is enabled, "
            f"got {power_sample_interval_ms}"
        )
    return {
        "wall_clock_latency": wall_clock_latency,
        "power": power,
        "power_sample_interval_ms": power_sample_interval_ms,
    }


def parse_decision_trace_settings(data) -> Dict[str, Any]:
    eval_cfg = data.get("eval", data)
    raw_cfg = eval_cfg.get("decision_trace", None)
    if raw_cfg is None:
        return {
            "enabled": False,
            "max_routes": 2,
            "save_interval_steps": 10,
            "pic": True,
            "replay": True,
        }
    if not isinstance(raw_cfg, dict):
        raise ValueError("eval.decision_trace must be a mapping when provided")

    enabled = _parse_bool_flag(
        raw_cfg.get("enabled", False),
        field_name="eval.decision_trace.enabled",
    )
    max_routes = int(raw_cfg.get("max_routes", 2))
    save_interval_steps = int(raw_cfg.get("save_interval_steps", 10))
    save_pic = _parse_bool_flag(
        raw_cfg.get("pic", True),
        field_name="eval.decision_trace.pic",
    )
    save_replay = _parse_bool_flag(
        raw_cfg.get("replay", True),
        field_name="eval.decision_trace.replay",
    )
    if max_routes <= 0:
        raise ValueError(
            f"eval.decision_trace.max_routes must be > 0, got {max_routes}"
        )
    if save_interval_steps <= 0:
        raise ValueError(
            "eval.decision_trace.save_interval_steps must be > 0, "
            f"got {save_interval_steps}"
        )
    return {
        "enabled": enabled,
        "max_routes": max_routes,
        "save_interval_steps": save_interval_steps,
        "pic": save_pic,
        "replay": save_replay,
    }


def parse_route_selection(data) -> Dict[str, Any]:
    eval_cfg = data.get("eval", data)
    route_selection_cfg = eval_cfg["route_selection"]
    return {
        "total": int(route_selection_cfg["total"]),
        "index": int(route_selection_cfg["index"]),
    }


def parse_explicit_routes(data, route_path: str) -> Optional[List[Dict[str, Any]]]:
    eval_cfg = data.get("eval", data)
    raw_routes = eval_cfg.get("explicit_routes", None)
    if raw_routes is None:
        return None
    if not isinstance(raw_routes, list) or len(raw_routes) == 0:
        raise ValueError("eval.explicit_routes must be a non-empty list when provided")

    resolved_routes: List[Dict[str, Any]] = []
    for idx, raw_route in enumerate(raw_routes):
        route_meta: Dict[str, Any]
        if isinstance(raw_route, str):
            route_file_raw = raw_route
            route_meta = {"route_file": raw_route}
        elif isinstance(raw_route, dict):
            if "route_file" not in raw_route:
                raise KeyError(
                    f"eval.explicit_routes[{idx}] must contain 'route_file', got {raw_route}"
                )
            route_file_raw = raw_route["route_file"]
            route_meta = copy.deepcopy(raw_route)
        else:
            raise ValueError(
                "eval.explicit_routes entries must be either string route paths "
                f"or mappings, got {type(raw_route)} at index {idx}"
            )

        route_file_text = str(route_file_raw).strip()
        if route_file_text == "":
            raise ValueError(f"eval.explicit_routes[{idx}].route_file must be non-empty")
        expanded_route_file = expand_path(route_file_text)
        if not os.path.isabs(expanded_route_file):
            expanded_route_file = os.path.join(route_path, expanded_route_file)
        expanded_route_file = os.path.normpath(expanded_route_file)
        if not os.path.isfile(expanded_route_file):
            raise FileNotFoundError(
                f"explicit route file not found at index {idx}: {expanded_route_file}"
            )

        route_name = os.path.basename(expanded_route_file)
        route_meta["route_file"] = expanded_route_file
        route_meta["route_name"] = route_name
        resolved_routes.append(route_meta)

    print(
        "[explicit_routes] "
        f"selected={len(resolved_routes)} "
        f"routes={[route['route_name'] for route in resolved_routes]}"
    )
    return resolved_routes


def list_routes(route_path: str) -> List[str]:
    routes = sorted([x for x in os.listdir(route_path) if x.endswith(".xml")])
    return routes


def select_routes(routes: List[str], route_selection: Dict[str, Any]) -> List[str]:
    total = route_selection["total"]
    index = route_selection["index"]
    selected_routes = [route for i, route in enumerate(routes) if i % total == index]
    print(
        "[route_selection] "
        f"total={len(routes)} selected={len(selected_routes)} "
        f"group_total={total} group_index={index}"
    )
    return selected_routes


def resolve_debug_route_limit(args, cfg: dict) -> Optional[int]:
    debug_limit = args.debug_route_count
    if debug_limit is None:
        debug_limit = cfg.get("debug_route_count", None)
    if debug_limit is None:
        return None
    debug_limit = int(debug_limit)
    if debug_limit <= 0:
        raise ValueError(f"debug_route_count must be > 0 in debug mode, got {debug_limit}")
    return debug_limit


def get_route_status(result_file: str) -> Optional[str]:
    if not os.path.exists(result_file):
        return None
    try:
        with open(result_file, "r", encoding="utf-8") as f:
            res_data = ujson.load(f)
    except Exception:
        return None
    status = None
    if "_checkpoint" in res_data and "global_record" in res_data["_checkpoint"]:
        status = res_data["_checkpoint"]["global_record"].get("status", None)
    if isinstance(status, str):
        status = status.strip()
    return status


def is_route_completed(result_file: str) -> bool:
    status = get_route_status(result_file)
    return status in {"Completed", "Perfect"}


def is_route_success(result_file: str) -> bool:
    if not os.path.exists(result_file):
        return False
    try:
        with open(result_file, "r", encoding="utf-8") as f:
            res_data = ujson.load(f)
    except Exception:
        return False

    records = res_data.get("_checkpoint", {}).get("records", [])
    return bool(records) and all(is_success_record(record) for record in records)


def should_queue_route(
    status: Optional[str],
    success: bool,
    retry_failed: bool,
    retry_unsuccess: bool,
) -> bool:
    if retry_unsuccess:
        return not success
    completed = status in {"Completed", "Perfect"}
    failed = is_failed_status(status)
    return (not completed) and (retry_failed or not failed)


def select_debug_candidates(route_candidates: List[dict], debug_limit: Optional[int]) -> List[dict]:
    pending = [x for x in route_candidates if x["queueable"]]
    if debug_limit is None:
        return pending
    return pending[:debug_limit]


def cleanup_job_artifacts(job: dict) -> None:
    for path in (job["result_file"], job["log_file"], job["err_file"]):
        if os.path.exists(path):
            os.remove(path)


def build_eval_config(args, no_server_launch):
    eval_yaml = load_eval_yaml(args.eval_config)
    cfg = eval_yaml.get("eval", eval_yaml)
    project_paths = ProjectPaths.load(args.paths_config)
    budget_settings = parse_budget_settings(eval_yaml)
    budget_mode = str(budget_settings["mode"])
    budget_policy = str(budget_settings["policy"])
    fixed_budgets = budget_settings["fixed_budgets"]
    layer_selection_mode = str(budget_settings["layer_selection_mode"])
    limit_language_inference = bool(budget_settings["limit_language_inference"])
    fixed_random_layer_selection = bool(budget_settings["fixed_random_layer_selection"])
    eval_budget_sampling = bool(budget_settings["eval_budget_sampling"])
    eval_budget_sampling_temperature = float(budget_settings["eval_budget_sampling_temperature"])
    runtime_profile = parse_runtime_profile(eval_yaml)
    language_sample_count = parse_language_sample_count(eval_yaml)
    predict_language = parse_predict_language(eval_yaml, runtime_profile=runtime_profile)
    retry_failed = parse_retry_failed(eval_yaml)
    retry_unsuccess = parse_retry_unsuccess(eval_yaml)
    measurement = parse_measurement_settings(eval_yaml)
    decision_trace = parse_decision_trace_settings(eval_yaml)
    route_path = str(project_paths.route_path)
    explicit_routes = parse_explicit_routes(eval_yaml, route_path=route_path)
    route_selection = None if explicit_routes is not None else parse_route_selection(eval_yaml)

    required_keys = [
        "agent",
        "benchmark",
        "output_path",
        "runtime_profile",
        "language_sample_count",
    ]
    if runtime_profile == "simlingo_compat":
        required_keys.append("model_bootstrap_config")

    missing = [k for k in required_keys if k not in cfg]
    if missing:
        raise KeyError(f"Missing required keys in eval config: {missing}")
    if retry_unsuccess and cfg["benchmark"] != "bench2drive":
        raise ValueError("eval.retry_unsuccess is only supported for benchmark=bench2drive")

    seeds = cfg.get("seeds", [3])
    if args.seed is not None:
        seeds = [args.seed]

    repo_root = str(project_paths.root_path)
    if runtime_profile == "smart_assigner":
        checkpoint_value = normalize_optional_path(cfg.get("checkpoint"))
        checkpoint_path = (
            None
            if checkpoint_value is None
            else str(project_paths.resolve_root(checkpoint_value))
        )
        if checkpoint_path is None:
            bootstrap_value = normalize_optional_path(cfg.get("model_bootstrap_config"))
            bootstrap_config_path = (
                str(project_paths.resolve_root(SIMLINGO_SMART_ASSIGNER_BOOTSTRAP_REL_PATH))
                if bootstrap_value is None
                else str(project_paths.resolve_root(bootstrap_value))
            )
            agent_config_path = bootstrap_config_path
            if not os.path.isfile(agent_config_path):
                raise FileNotFoundError(
                    "smart_assigner bootstrap config is missing. "
                    f"expected={agent_config_path}"
                )
        else:
            agent_config_path = checkpoint_path
    elif runtime_profile == "simlingo_compat":
        checkpoint_path = str(project_paths.resolve_root(SIMLINGO_COMPAT_CHECKPOINT_REL_PATH))
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(
                "simlingo_compat checkpoint is missing. "
                f"expected={checkpoint_path}"
            )
        agent_config_path = str(project_paths.resolve_root(cfg["model_bootstrap_config"]))
        if not os.path.isfile(agent_config_path):
            raise FileNotFoundError(
                "simlingo_compat model_bootstrap_config is missing. "
                f"expected={agent_config_path}"
            )
    else:
        raise ValueError(f"Unsupported runtime_profile={runtime_profile}")

    eval_cfg = {
        "agent": cfg["agent"],
        "checkpoint": checkpoint_path,
        "agent_config_path": agent_config_path,
        "benchmark": cfg["benchmark"],
        "route_path": route_path,
        "seeds": seeds,
        "tries": int(cfg.get("tries", 0)),
        "out_root": str(project_paths.resolve_root(cfg["output_path"])),
        "carla_root": str(project_paths.carla_path),
        "carla_python_api": str(project_paths.carla_python_api),
        "carla_egg": str(project_paths.carla_egg),
        "repo_root": repo_root,
        "bench_root": str(project_paths.bench2drive_path),
        "leaderboard_root": str(project_paths.leaderboard_path),
        "scenario_runner_root": str(project_paths.scenario_runner_path),
        "agent_file": str(project_paths.agent_file),
        "team_code": cfg.get("team_code", "team_code_adaption"),
        "agent_config": cfg.get("agent_config", "not_used"),
        "username": cfg.get("username", os.getenv("USER", "local_user")),
        "no_server_launch": no_server_launch,
        "budget_mode": budget_mode,
        "budget_policy": budget_policy,
        "fixed_budgets": fixed_budgets,
        "layer_selection_mode": layer_selection_mode,
        "limit_language_inference": limit_language_inference,
        "fixed_random_layer_selection": fixed_random_layer_selection,
        "eval_budget_sampling": eval_budget_sampling,
        "eval_budget_sampling_temperature": eval_budget_sampling_temperature,
        "runtime_profile": runtime_profile,
        "language_sample_count": language_sample_count,
        "predict_language": predict_language,
        "retry_failed": retry_failed,
        "retry_unsuccess": retry_unsuccess,
        "measurement": measurement,
        "decision_trace": decision_trace,
        "explicit_routes": explicit_routes,
        "route_selection": route_selection,
        "debug_route_count": cfg.get("debug_route_count", None),
    }
    return eval_cfg


def launch_job(job, gpu_id, world_port, tm_port):
    cfg = job["cfg"]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    carla_root = expand_path(cfg["carla_root"])
    repo_root = expand_path(cfg["repo_root"])
    bench_root = cfg["bench_root"]  # Avoid import collisions in the child process.

    addPYTHOPATH = ":".join(
            [
            cfg["carla_python_api"],
            cfg["carla_egg"],
            repo_root,
            cfg["leaderboard_root"],
            cfg["scenario_runner_root"],
            ]
    )

    env["PYTHONPATH"] = addPYTHOPATH
    env["CARLA_ROOT"] = carla_root
    env["WORK_DIR"] = repo_root
    env["SCENARIO_RUNNER_ROOT"] = cfg["scenario_runner_root"]
    env["SAVE_PATH"] = job["viz_path"]
    env["LEADERBOARD_ROOT"] = cfg["leaderboard_root"]
    env["SIMLINGO_EVAL_BUDGET_MODE"] = str(job["budget_mode"])
    env["SIMLINGO_EVAL_BUDGET_POLICY"] = str(job["budget_policy"])
    fixed_budget_value = job.get("fixed_budget")
    if fixed_budget_value is None:
        env.pop("SIMLINGO_EVAL_FIXED_BUDGET", None)
    else:
        env["SIMLINGO_EVAL_FIXED_BUDGET"] = str(fixed_budget_value)
    env["SIMLINGO_EVAL_LAYER_SELECTION_MODE"] = str(job["layer_selection_mode"])
    env["SIMLINGO_EVAL_LIMIT_LANGUAGE_INFERENCE"] = (
        "1" if bool(job["limit_language_inference"]) else "0"
    )
    env["SIMLINGO_EVAL_FIXED_RANDOM_LAYER_SELECTION"] = (
        "1" if bool(job["fixed_random_layer_selection"]) else "0"
    )
    env["SIMLINGO_EVAL_BUDGET_SAMPLING"] = "1" if bool(job["eval_budget_sampling"]) else "0"
    env["SIMLINGO_EVAL_BUDGET_SAMPLING_TEMPERATURE"] = str(job["eval_budget_sampling_temperature"])
    env["SIMLINGO_EVAL_RANDOM_LAYER_SELECTION_SEED"] = str(job["seed"])
    env["SIMLINGO_EVAL_RUNTIME_PROFILE"] = str(job["runtime_profile"])
    env["SIMLINGO_EVAL_LANGUAGE_SAMPLE_COUNT"] = str(job["language_sample_count"])
    env["SIMLINGO_EVAL_PREDICT_LANGUAGE"] = "1" if bool(job["predict_language"]) else "0"
    env["SIMLINGO_EVAL_MEASURE_WALL_CLOCK_LATENCY"] = (
        "1" if bool(job["measurement"]["wall_clock_latency"]) else "0"
    )
    env["SIMLINGO_EVAL_MEASURE_POWER"] = "1" if bool(job["measurement"]["power"]) else "0"
    env["SIMLINGO_EVAL_POWER_SAMPLE_INTERVAL_MS"] = str(
        job["measurement"]["power_sample_interval_ms"]
    )
    env["SIMLINGO_EVAL_DECISION_TRACE_ENABLED"] = (
        "1" if bool(job["decision_trace"]["enabled"]) else "0"
    )
    env["SIMLINGO_EVAL_DECISION_TRACE_INTERVAL"] = str(
        job["decision_trace"]["save_interval_steps"]
    )
    env["SIMLINGO_EVAL_DECISION_TRACE_PIC"] = (
        "1" if bool(job["decision_trace"]["pic"]) else "0"
    )
    env["SIMLINGO_EVAL_DECISION_TRACE_REPLAY"] = (
        "1" if bool(job["decision_trace"]["replay"]) else "0"
    )
    route_entry = job.get("route_entry", {}) or {}
    env["SIMLINGO_EVAL_ROUTE_ID"] = str(route_entry.get("route_id", job.get("route_id", "")))
    env["SIMLINGO_EVAL_ROUTE_NAME"] = str(route_entry.get("route_name", ""))
    env["SIMLINGO_EVAL_ROUTE_FILE"] = str(route_entry.get("route_file", job.get("route", "")))
    env["SIMLINGO_EVAL_SCENARIO_NAME"] = str(route_entry.get("scenario_name", ""))
    env["SIMLINGO_EVAL_TOWN_NAME"] = str(route_entry.get("town_name", ""))
    env.pop("SIMLINGO_COMPAT_CHECKPOINT", None)
    env.pop("SIMLINGO_SMART_ASSIGNER_CHECKPOINT", None)
    if cfg["runtime_profile"] == "simlingo_compat":
        # The compatibility agent always loads this checkpoint.
        env["SIMLINGO_COMPAT_CHECKPOINT"] = str(cfg["checkpoint"])
    elif cfg["checkpoint"] is not None:
        env["SIMLINGO_SMART_ASSIGNER_CHECKPOINT"] = str(cfg["checkpoint"])
    
    command = [
        sys.executable,
        "-u",
        f"{cfg['leaderboard_root']}/leaderboard/leaderboard_evaluator.py",
        f"--routes={job['route']}",
        "--repetitions=1",
        "--track=SENSORS",
        f"--checkpoint={job['result_file']}",
        "--timeout=600",
        f"--agent={cfg['agent_file']}",
        f"--agent-config={cfg['agent_config_path']}",
        f"--traffic-manager-seed={job['seed']}",
        f"--port={world_port}",
        f"--traffic-manager-port={tm_port}",
        f"--gpu-rank={VULKAN_GPU_ID[gpu_id]}"
    ]

    if cfg.get("no_server_launch"):
        command.append("--no-server-launch")

    stdout = open(job["log_file"], "w", encoding="utf-8")
    stderr = open(job["err_file"], "w", encoding="utf-8")
    stdout.write(" ".join(command) + "\n")
    stdout.flush()

    process = subprocess.Popen(
        command,
        env=env,
        cwd=bench_root, 
        stderr=stderr,
        stdout=stdout
    )

    job["process"] = process
    job["gpu_id"] = gpu_id
    job["ports"] = {world_port, tm_port,world_port+1,world_port+2} # CARLA reserves two adjacent streaming ports.
    job["_stdout_handle"] = stdout
    job["_stderr_handle"] = stderr


def finalize_job(job):
    if not no_server_launch:
        ports = job.get("ports", set())
        for port in ports:
            try:
                subprocess.run(
                    ["fuser", "-k", "-9", f"{port}/tcp"], 
                    stdout=subprocess.DEVNULL, 
                    stderr=subprocess.DEVNULL
                )
            except Exception:
                pass
        
    for handle_key in ("_stdout_handle", "_stderr_handle"):
        handle = job.pop(handle_key, None)
        if handle:
            handle.flush()
            handle.close()
    job.pop("process", None)
    job.pop("ports", None)
    job.pop("gpu_id", None)

def check_and_kill_dead_job(job):
    """Terminate jobs matching known dead-job log signatures."""
    process = job.get("process")
    if not process or process.poll() is not None:
        return

    log_file = job.get("log_file")
    if not log_file or not os.path.exists(log_file):
        print(f"Warning: there is no {log_file}")
        return

    lines = []
    try:
        # Flush owned handles before reading their files.
        stdout_handle = job.get("_stdout_handle")
        if stdout_handle and not stdout_handle.closed:
            stdout_handle.flush()
        
        stderr_handle = job.get("_stderr_handle")
        if stderr_handle and not stderr_handle.closed:
            stderr_handle.flush()

        with open(log_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            
    except Exception as e:
        print(f"Warning: Could not read log {log_file} for dead job check: {e}")
        return

    if not lines:
        return

    CRASH_MARKERS_SUBSTR = (
        "Watchdog exception",
        "Engine crash handling finished; re-raising signal 11",
        "Stopping the route, the agent has crashed",
    )

    def has_crash_marker(lines):
        for line in lines:
            for m in CRASH_MARKERS_SUBSTR:
                if m in line:
                    return True
        return False


    if has_crash_marker(lines):
        print(
            f"Detected dead job {job['route_id']} (PID: {process.pid}) from log. "
            f"Terminating process..."
        )
        process.terminate()



def err_log_has_error(err_file: str) -> bool:
    error_marks = (
        "Traceback (most recent call last)",
        "Segmentation fault",
        "Signal 11",
        "FatalError",
        "LowLevelFatalError",
        "Watchdog exception",
        "Engine crash handling finished",
        "Address already in use",
        "CUDA out of memory",
        "Failed - Agent couldn't be set up",
        "Stopping the route, the agent has crashed",
    )
    if not os.path.exists(err_file):
        return False
    try:
        with open(err_file, "r", encoding="utf-8") as f:
            for line in f:
                for m in error_marks:
                    if m in line:
                        return True
    except Exception:
        return False
    return False


def main(args):
    global no_server_launch
    no_server_launch = False
    if args.remote_carla_port is not None:
        global carla_world_ports, carla_tm_ports
        carla_world_ports = {args.remote_carla_port}
        if args.remote_tm_port is not None:
             carla_tm_ports = {args.remote_tm_port}
        else:
             carla_tm_ports = {args.remote_carla_port + 8000}
        no_server_launch = True
        print(f"Using remote CARLA at port {args.remote_carla_port}")
    
    global GPU_IDS
    GPU_IDS = 0
    if args.gpu is not None:
        if isinstance(args.gpu, int):
            GPU_IDS = [args.gpu]
        else:
            GPU_IDS = args.gpu
    else:
        GPU_IDS = [0]
    global MAX_PARALLEL_JOBS
    MAX_PARALLEL_JOBS = len(GPU_IDS)
    print(f"GPU_IDS: {GPU_IDS}")
    configs = [build_eval_config(args, no_server_launch)]

    job_queue = []
    for cfg in configs:
        route_path = cfg["route_path"]
        explicit_routes = cfg.get("explicit_routes")
        if explicit_routes is not None:
            route_entries = copy.deepcopy(explicit_routes)
        else:
            # Group routes before applying completion and retry filters.
            all_routes = list_routes(route_path)
            selected_routes = select_routes(routes=all_routes, route_selection=cfg["route_selection"])
            route_entries = [
                {
                    "route_name": route_name,
                    "route_file": os.path.join(route_path, route_name),
                }
                for route_name in selected_routes
            ]
        decision_trace_cfg = cfg["decision_trace"]
        trace_enabled = bool(decision_trace_cfg["enabled"])
        trace_max_routes = int(decision_trace_cfg["max_routes"])
        trace_jobs_added = 0
        debug_enabled = bool(getattr(args, "debug", False))
        debug_limit = resolve_debug_route_limit(args, cfg) if debug_enabled else None
        retry_failed = bool(cfg.get("retry_failed", False))
        retry_unsuccess = bool(cfg.get("retry_unsuccess", False))

        fill_zeros = 3 if cfg["benchmark"] == "bench2drive" else 2

        for seed in cfg["seeds"]:
            seed = str(seed)
            for fixed_budget in cfg["fixed_budgets"]:
                base_dir_parts = [
                    cfg["out_root"],
                    cfg["agent"],
                    cfg["benchmark"],
                    seed,
                    cfg["runtime_profile"],
                ]
                if cfg["checkpoint"] is None:
                    base_dir_parts.append("no_checkpoint")
                if cfg["budget_policy"] == "stage1":
                    # Isolate fixed-budget results to prevent cross-budget overwrites.
                    base_dir_parts.append(format_budget_dir_name(cfg["budget_policy"], fixed_budget))
                    base_dir_parts.append(format_layer_selection_dir_name(cfg["layer_selection_mode"]))
                base_dir = os.path.join(*base_dir_parts)
                os.makedirs(os.path.join(base_dir, "run"), exist_ok=True)
                os.makedirs(os.path.join(base_dir, "res"), exist_ok=True)
                os.makedirs(os.path.join(base_dir, "out"), exist_ok=True)
                os.makedirs(os.path.join(base_dir, "err"), exist_ok=True)

                route_candidates = []
                for route_entry in route_entries:
                    route_name = str(route_entry["route_name"])
                    route_file = str(route_entry["route_file"])
                    route_id = route_name.split("_")[-1][:-4].zfill(fill_zeros)

                    viz_path = os.path.join(base_dir, "viz", route_id)
                    os.makedirs(viz_path, exist_ok=True)

                    log_file = os.path.join(base_dir, "out", f"{route_id}_out.log")
                    err_file = os.path.join(base_dir, "err", f"{route_id}_err.log")
                    result_file = os.path.join(base_dir, "res", f"{route_id}_res.json")
                    status = get_route_status(result_file)
                    completed = status in {"Completed", "Perfect"}
                    failed = is_failed_status(status)
                    success = is_route_success(result_file)
                    queueable = should_queue_route(
                        status=status,
                        success=success,
                        retry_failed=retry_failed,
                        retry_unsuccess=retry_unsuccess,
                    )

                    if completed and not queueable:
                        pass
                    elif failed and not retry_failed and not retry_unsuccess:
                        print(
                            f"[skip] route {route_id} mode {cfg['budget_mode']} "
                            f"policy {cfg['budget_policy']} "
                            f"fixed {fixed_budget} "
                            f"layer_select {cfg['layer_selection_mode']} "
                            f"profile {cfg['runtime_profile']} status is {status} "
                            f"and retry_failed is False -> skip"
                        )
                    else:
                        print(
                            f"[queue] route {route_id} mode {cfg['budget_mode']} "
                            f"policy {cfg['budget_policy']} "
                            f"fixed {fixed_budget} "
                            f"layer_select {cfg['layer_selection_mode']} "
                            f"profile {cfg['runtime_profile']} status is {status or 'missing'} -> queue"
                        )

                    route_candidates.append({
                        "cfg": cfg,
                        "route_entry": copy.deepcopy(route_entry),
                        "route": route_file,
                        "route_id": route_id,
                        "seed": seed,
                        "budget_mode": cfg["budget_mode"],
                        "budget_policy": cfg["budget_policy"],
                        "fixed_budget": fixed_budget,
                        "layer_selection_mode": cfg["layer_selection_mode"],
                        "limit_language_inference": cfg["limit_language_inference"],
                        "fixed_random_layer_selection": cfg["fixed_random_layer_selection"],
                        "eval_budget_sampling": cfg["eval_budget_sampling"],
                        "eval_budget_sampling_temperature": cfg["eval_budget_sampling_temperature"],
                        "runtime_profile": cfg["runtime_profile"],
                        "language_sample_count": cfg["language_sample_count"],
                        "predict_language": cfg["predict_language"],
                        "measurement": copy.deepcopy(cfg["measurement"]),
                        "decision_trace": copy.deepcopy(cfg["decision_trace"]),
                        "viz_path": viz_path,
                        "result_file": result_file,
                        "log_file": log_file,
                        "err_file": err_file,
                        "tries_initial": cfg["tries"],
                        "tries_remaining": cfg["tries"],
                        "status": "pending",
                        "completed": completed,
                        "failed": failed,
                        "success": success,
                        "queueable": queueable,
                    })

                selected_candidates = [x for x in route_candidates if x["queueable"]]
                if debug_enabled:
                    pending_total = len(selected_candidates)
                    selected_candidates = select_debug_candidates(route_candidates, debug_limit)
                    print(
                        f"[debug] route selection enabled: total={len(route_candidates)}, "
                        f"pending={pending_total}, selected={len(selected_candidates)}, limit={debug_limit}, "
                        f"fixed_budget={fixed_budget}"
                    )

                if trace_enabled:
                    remaining_slots = trace_max_routes - trace_jobs_added
                    if remaining_slots <= 0:
                        print(
                            "[decision_trace] "
                            f"max_routes={trace_max_routes} reached, skip remaining queued routes"
                        )
                        selected_candidates = []
                    elif len(selected_candidates) > remaining_slots:
                        print(
                            "[decision_trace] "
                            f"limiting queued routes from {len(selected_candidates)} to {remaining_slots}"
                        )
                        selected_candidates = selected_candidates[:remaining_slots]
                    trace_jobs_added += len(selected_candidates)

                for job in selected_candidates:
                    job_queue.append(job)

    pending_jobs = deque(job_queue)
    running_jobs = []
    available_gpus = deque(GPU_IDS)
    progress = tqdm(total=len(job_queue))

    
    while pending_jobs or running_jobs:
        for job in list(running_jobs):
            process = job["process"]
            return_code = process.poll()
            if return_code is None:
                check_and_kill_dead_job(job)
                continue

            # Release the GPU slot and owned log handles.
            available_gpus.append(job["gpu_id"])
            running_jobs.remove(job)
            finalize_job(job)
            

            if not needs_resubmit(job) and return_code == 0: 
                job["status"] = "completed"
                progress.update(1)
                continue

            if job["tries_remaining"] > 0:
                job["status"] = "pending"
                pending_jobs.append(job)
                print(f"Resubmitting job {job['route_id']} (tries left: {job['tries_remaining']}).")
                continue

            job["status"] = "failed"
            progress.update(1)
            if return_code != 0:
                    infos = [
                        f"Job {job['route_id']} exited with return code {return_code}.",
                        f"route: {job.get('route')}",
                        f"log_file: {job.get('log_file')}",
                        f"err_file: {job.get('err_file')}",
                    ]
                    print(infos)
        if len(running_jobs) >= MAX_PARALLEL_JOBS or not available_gpus:
            time.sleep(POLL_INTERVAL_S)
            continue

        job_started = False
        for _ in range(len(pending_jobs)):
            job = pending_jobs.popleft()

            if job["status"] == "completed":
                continue

            if not needs_resubmit(job):
                job["status"] = "completed"
                progress.update(1)
                continue

            used_ports = set()
            for running in running_jobs:
                used_ports.update(running.get("ports", set()))

            try:
                world_port = next(iter(carla_world_ports - used_ports))
                tm_port = next(iter(carla_tm_ports - used_ports))
            except StopIteration:
                pending_jobs.appendleft(job)
                time.sleep(POLL_INTERVAL_S)
                break

            cleanup_job_artifacts(job)
            cleanup_viz_dir(job["viz_path"])
            job["tries_remaining"] -= 1
            job["status"] = "running"

            gpu_id = available_gpus.popleft()
            launch_job(job, gpu_id, world_port, tm_port)
            running_jobs.append(job)

            print(
                f"Started job {job['route_id']} on GPU {gpu_id} "
                f"with mode={job['budget_mode']} "
                f"policy={job['budget_policy']} "
                f"fixed={job['fixed_budget']} "
                f"layer_select={job['layer_selection_mode']} "
                f"profile={job['runtime_profile']} "
                f"predict_language={job['predict_language']} "
                f"(tries left after launch: {job['tries_remaining']})."
            )
            job_started = True
            break

        if not job_started:
            time.sleep(POLL_INTERVAL_S)

    progress.close()
    print("All jobs processed.")
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="一个接收 seed 参数的脚本。")
    default_eval_config = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "configs",
        "stage2_eval.yaml",
    )
    parser.add_argument("--eval-config", type=str, default=default_eval_config, help="评估配置 YAML 路径")
    parser.add_argument(
        "--paths-config",
        type=str,
        default=str(DEFAULT_PATHS_CONFIG),
        help="本机 root/CARLA/data 路径配置",
    )
    parser.add_argument("--seed", type=int, help="用于脚本的随机种子；不传则使用 eval-config 中的 seeds", default=None)
    parser.add_argument("--remote-carla-port", type=int, default=None, help="External CARLA world port")
    parser.add_argument("--remote-tm-port", type=int, default=None, help="External CARLA TM port")
    parser.add_argument("--gpu", type=int, nargs='+', default=0, help="gpu to use")
    parser.add_argument("--debug", action="store_true", help="Debug mode: enable debug route selection logs")
    parser.add_argument(
        "--debug-route-count",
        type=int,
        default=None,
        help="Only effective with --debug. Limit number of queued (non-completed) routes. Overrides eval.debug_route_count.",
    )

    args = parser.parse_args()

    main(args)
