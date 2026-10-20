import argparse
import glob
import json
import math
import os
import re

import numpy as np
from scipy.signal import savgol_filter
from merge_route_json import DEFAULT_TOTAL_ROUTES, collect_budget_metrics, is_success_record, load_route_records


DEFAULT_EVAL_ROOT = "eval_results/opensource_stage2/gp=4_woLanguage/simlingo/bench2drive/5/smart_assigner"

MAX_ABS_MAG_JERK = 8.37
MAX_ABS_LAT_ACCEL = 4.89
MAX_LON_ACCEL = 2.40
MIN_LON_ACCEL = -4.05
MAX_ABS_YAW_ACCEL = 1.93
MAX_ABS_LON_JERK = 4.13
MAX_ABS_YAW_RATE = 0.95


def _linear_quantile(sorted_values, quantile):
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return sorted_values[0]

    position = (len(sorted_values) - 1) * quantile
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return sorted_values[lower]

    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def summarize_numeric(values):
    if not values:
        return {
            "mean": None,
            "p50": None,
            "p90": None,
            "p95": None,
            "min": None,
            "max": None,
        }

    sorted_values = sorted(values)
    return {
        "mean": round(sum(sorted_values) / len(sorted_values), 6),
        "p50": round(_linear_quantile(sorted_values, 0.50), 6),
        "p90": round(_linear_quantile(sorted_values, 0.90), 6),
        "p95": round(_linear_quantile(sorted_values, 0.95), 6),
        "min": round(sorted_values[0], 6),
        "max": round(sorted_values[-1], 6),
    }


def infer_eval_root(path):
    path = os.path.abspath(path)
    if os.path.basename(path) in {"res", "viz", "run", "out", "err"}:
        return os.path.dirname(path)
    return path


def resolve_res_dir(eval_root):
    candidates = [
        os.path.join(eval_root, "res"),
    ]
    for candidate in candidates:
        if os.path.isdir(candidate):
            return candidate
    raise FileNotFoundError(f"Could not find a res directory under: {eval_root}")


def resolve_viz_dir(eval_root):
    candidates = [
        os.path.join(eval_root, "viz"),
        os.path.join(eval_root, "bench2drive", "3", "smart_assigner", "viz"),
    ]
    for candidate in candidates:
        if os.path.isdir(candidate):
            return candidate
    raise FileNotFoundError(f"Could not find a viz directory under: {eval_root}")


def infer_save_name(metric_info_path):
    parts = os.path.normpath(metric_info_path).split(os.sep)
    try:
        viz_index = parts.index("viz")
    except ValueError:
        return None

    if viz_index + 1 >= len(parts):
        return None

    # New layout:
    #   .../viz/<idx>/<save_name>/.../metric/metric_info.json
    # Old layout:
    #   .../viz/<idx><save_name>/.../metric/metric_info.json
    tail_parts = parts[viz_index + 1 :]
    for part in tail_parts:
        if part.startswith("RouteScenario_"):
            return part

    folder_name = parts[viz_index + 1]
    match = re.match(r"^\d{3}(RouteScenario_.*)$", folder_name)
    if match:
        return match.group(1)
    return folder_name


def discover_metric_files(viz_dir):
    metric_paths = glob.glob(os.path.join(viz_dir, "**", "metric", "metric_info.json"), recursive=True)
    metric_map = {}

    for metric_path in metric_paths:
        save_name = infer_save_name(metric_path)
        if not save_name:
            continue
        existing = metric_map.get(save_name)
        if existing and existing != metric_path:
            print(f"Warning: duplicate metric path for {save_name}, keeping {existing}")
            continue
        metric_map[save_name] = metric_path

    return metric_map


def chunk_array(arr, chunk_size):
    return [arr[i * chunk_size:(i + 1) * chunk_size] for i in range((len(arr) + chunk_size - 1) // chunk_size)]


def chunk_arrays(arrays, chunk_size):
    return [chunk_array(arr, chunk_size) for arr in arrays]


def _phase_unwrap(headings):
    two_pi = 2.0 * np.pi
    adjustments = np.zeros_like(headings)
    adjustments[..., 1:] = np.cumsum(np.round(np.diff(headings, axis=-1) / two_pi), axis=-1)
    return headings - two_pi * adjustments


def _within_bound(metric, min_bound=None, max_bound=None):
    min_bound = min_bound if min_bound is not None else float("-inf")
    max_bound = max_bound if max_bound is not None else float("inf")
    metric_values = np.array(metric)
    return np.all((metric_values > min_bound) & (metric_values < max_bound), axis=-1)


def compute_comfort_metric(
    acceleration,
    angular_velocity,
    forward_vector,
    right_vector,
    location,
    rotation,
    window_size=7,
    poly_order=2,
    deriv_order=1,
    time_interval=0.1,
):
    window_size = min(window_size, len(acceleration))
    if window_size < 3 or not (poly_order < window_size):
        return False

    acceleration = np.array(acceleration)
    angular_velocity = np.array(angular_velocity)
    forward_vector = np.array(forward_vector)
    right_vector = np.array(right_vector)

    _2d_acceleration = np.array([[v[0], v[1]] for v in acceleration])
    _2d_forward_vector = np.array([[vec[0], vec[1]] for vec in forward_vector])
    _2d_right_vector = np.array([[vec[0], vec[1]] for vec in right_vector])
    _z_angular_rate = np.array([a[2] for a in angular_velocity])
    _z_yaw_rate = _phase_unwrap(_z_angular_rate)

    lon_acc = np.einsum("ij,ij->i", _2d_acceleration, _2d_forward_vector)
    lat_acc = np.einsum("ij,ij->i", _2d_acceleration, _2d_right_vector)
    magnitude_acc = np.hypot(_2d_acceleration[:, 0], _2d_acceleration[:, 1])

    _z_yaw_acc = savgol_filter(_z_yaw_rate, polyorder=poly_order, window_length=window_size, axis=-1)
    _z_yaw_rate = savgol_filter(_z_yaw_rate, polyorder=poly_order, window_length=window_size, axis=-1)
    lon_acc = savgol_filter(lon_acc, polyorder=poly_order, window_length=window_size, axis=-1)
    lat_acc = savgol_filter(lat_acc, polyorder=poly_order, window_length=window_size, axis=-1)
    magnitude_acc = savgol_filter(magnitude_acc, polyorder=poly_order, window_length=window_size, axis=-1)

    magnitude_jerk = savgol_filter(
        magnitude_acc,
        polyorder=poly_order,
        window_length=window_size,
        deriv=deriv_order,
        delta=time_interval,
        axis=-1,
    )

    lon_jerk = savgol_filter(
        lon_acc,
        polyorder=poly_order,
        window_length=window_size,
        deriv=deriv_order,
        delta=time_interval,
        axis=-1,
    )

    checks = np.array([
        _within_bound(lon_acc, min_bound=MIN_LON_ACCEL, max_bound=MAX_LON_ACCEL),
        _within_bound(lat_acc, min_bound=-MAX_ABS_LAT_ACCEL, max_bound=MAX_ABS_LAT_ACCEL),
        _within_bound(magnitude_jerk, min_bound=-MAX_ABS_MAG_JERK, max_bound=MAX_ABS_MAG_JERK),
        _within_bound(lon_jerk, min_bound=-MAX_ABS_LON_JERK, max_bound=MAX_ABS_LON_JERK),
        _within_bound(_z_yaw_acc, min_bound=-MAX_ABS_YAW_ACCEL, max_bound=MAX_ABS_YAW_ACCEL),
        _within_bound(_z_yaw_rate, min_bound=-MAX_ABS_YAW_RATE, max_bound=MAX_ABS_YAW_RATE),
    ])
    return bool(np.all(checks))


def seg_compute_comfort_metric(
    acceleration,
    angular_velocity,
    forward_vector,
    right_vector,
    location,
    rotation,
    per_step=20,
):
    episode_len = len(angular_velocity)
    if episode_len <= per_step:
        return 1.0 if compute_comfort_metric(
            acceleration,
            angular_velocity,
            forward_vector,
            right_vector,
            location,
            rotation,
        ) else 0.0

    seg_data = chunk_arrays(
        [acceleration, angular_velocity, forward_vector, right_vector, location, rotation],
        per_step,
    )
    results = []
    for index in range(len(seg_data[0])):
        if len(seg_data[0][index]) < per_step:
            continue
        results.append(
            compute_comfort_metric(
                seg_data[0][index],
                seg_data[1][index],
                seg_data[2][index],
                seg_data[3][index],
                seg_data[4][index],
                seg_data[5][index],
            )
        )

    if not results:
        return 0.0
    return results.count(True) / len(results)


def load_metric_episode(metric_path):
    with open(metric_path, encoding="utf-8") as file:
        metric_json = json.load(file)

    ordered_keys = sorted(metric_json.keys(), key=lambda key: int(key) if str(key).isdigit() else str(key))
    episode = {
        "acceleration": [],
        "angular_velocity": [],
        "forward_vector": [],
        "right_vector": [],
        "location": [],
        "rotation": [],
    }

    for key in ordered_keys:
        frame = metric_json[key]
        for field in episode:
            episode[field].append(frame[field])

    for field, values in episode.items():
        episode[field] = np.array(values)

    return episode


def extract_driving_efficiency(record):
    infractions = record.get("infractions", {})
    if not isinstance(infractions, dict):
        return None

    speed_infraction_texts = infractions.get("min_speed_infractions", [])
    if not speed_infraction_texts:
        return None

    extracted = []
    for text in speed_infraction_texts:
        match = re.search(r"\b\d+\.?\d*%", text)
        if not match:
            continue
        value = float(match.group().rstrip("%"))
        if value <= 1000:
            extracted.append(value)

    if not extracted:
        return None
    return sum(extracted) / len(extracted)


def build_merged_summary(records, include_budget=True, budget_keys=None, total_routes=DEFAULT_TOTAL_ROUTES):
    driving_scores = []
    success_num = 0

    for record in records:
        driving_scores.append(record["scores"]["score_composed"])
        if is_success_record(record):
            success_num += 1

    actual_eval_num = len(driving_scores)
    eval_num = total_routes if total_routes else actual_eval_num
    driving_score_mean = (sum(driving_scores) / eval_num) if eval_num else 0.0
    success_rate = (success_num / eval_num) if eval_num else 0.0

    merged_records = sorted(records, key=lambda record: record["route_id"], reverse=True)
    merged_data = {
        "_checkpoint": {
            "records": merged_records,
        },
        "driving score": driving_score_mean,
        "success rate": success_rate,
        "eval num": eval_num,
        "actual eval num": actual_eval_num,
        "missing eval num": max(eval_num - actual_eval_num, 0),
    }

    if include_budget:
        merged_data["budget_metrics"] = collect_budget_metrics(
            merged_records,
            budget_keys=budget_keys,
            require_success=False,
        )

    return merged_data


def to_float_or_none(value):
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def collect_inference_measurements(records):
    route_count = 0
    wall_clock_latency_values = []
    forward_mean_power_values = []
    weighted_power_sum = 0.0
    power_sample_count = 0
    min_power_values = []
    max_power_values = []
    power_sample_intervals = []
    has_wall_clock = False
    has_power = False

    for record in records:
        measurements = record.get("inference_measurements")
        if not isinstance(measurements, dict):
            continue

        route_count += 1
        wall_clock = measurements.get("wall_clock_latency")
        if isinstance(wall_clock, dict):
            has_wall_clock = True
            wall_clock_latency_values.extend(
                value
                for value in (
                    to_float_or_none(item) for item in wall_clock.get("values_ms", [])
                )
                if value is not None
            )

        power = measurements.get("power")
        if not isinstance(power, dict):
            continue
        has_power = True
        forward_mean_power_values.extend(
            value
            for value in (
                to_float_or_none(item) for item in power.get("forward_mean_w_values", [])
            )
            if value is not None
        )

        mean_w = to_float_or_none(power.get("mean_w"))
        sample_count = to_float_or_none(power.get("sample_count"))
        if mean_w is not None and sample_count is not None and sample_count > 0:
            weighted_power_sum += mean_w * int(sample_count)
            power_sample_count += int(sample_count)
        min_w = to_float_or_none(power.get("min_w"))
        max_w = to_float_or_none(power.get("max_w"))
        sample_interval_ms = to_float_or_none(power.get("sample_interval_ms"))
        if min_w is not None:
            min_power_values.append(min_w)
        if max_w is not None:
            max_power_values.append(max_w)
        if sample_interval_ms is not None:
            power_sample_intervals.append(sample_interval_ms)

    summary = {"route_count": route_count}
    if has_wall_clock:
        summary["wall_clock_latency_ms"] = summarize_numeric(wall_clock_latency_values)
    if has_power:
        summary["power"] = {
            "forward_mean_w": summarize_numeric(forward_mean_power_values),
            "mean_w": (
                round(weighted_power_sum / power_sample_count, 6)
                if power_sample_count > 0
                else None
            ),
            "min_w": round(min(min_power_values), 6) if min_power_values else None,
            "max_w": round(max(max_power_values), 6) if max_power_values else None,
            "sample_count": power_sample_count,
            "sample_interval_ms": (
                power_sample_intervals[0] if power_sample_intervals else None
            ),
        }
    return summary


def compute_efficiency_and_smoothness(merged_data, metric_map):
    records = merged_data.get("_checkpoint", {}).get("records", [])
    driving_efficiency = []
    comfort_res = []
    missing_metric_files = []

    for record in records:
        save_name = record.get("save_name")
        metric_path = metric_map.get(save_name)
        if not metric_path:
            missing_metric_files.append(save_name)
            continue

        episode = load_metric_episode(metric_path)
        comfort_res.append(seg_compute_comfort_metric(**episode))

        efficiency = extract_driving_efficiency(record)
        if efficiency is not None:
            driving_efficiency.append(efficiency)

    efficiency_mean = (sum(driving_efficiency) / len(driving_efficiency)) if driving_efficiency else 0.0
    smoothness_mean = (sum(comfort_res) / len(comfort_res)) if comfort_res else 0.0

    return {
        "driving_efficiency": efficiency_mean,
        "driving_smoothness": smoothness_mean,
        "driving_efficiency_count": len(driving_efficiency),
        "driving_smoothness_count": len(comfort_res),
        "missing_metric_count": len(missing_metric_files),
        "missing_metric_samples": missing_metric_files[:10],
    }


def main():
    parser = argparse.ArgumentParser(description="Compute Bench2Drive closed-loop metrics from an eval result folder.")
    parser.add_argument(
        "-e",
        "--eval-root",
        default=DEFAULT_EVAL_ROOT,
        help="Eval result root that contains res/ and viz/.",
    )
    parser.add_argument(
        "--no-budget",
        action="store_true",
        help="Disable budget metric aggregation in merged.json.",
    )
    parser.add_argument(
        "--budget-keys",
        nargs="+",
        default=None,
        help="Optional explicit budget keys to aggregate.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON path for the combined summary. Defaults to <eval-root>/closed_loop_metrics.json.",
    )
    parser.add_argument(
        "--total-routes",
        type=int,
        default=DEFAULT_TOTAL_ROUTES,
        help="Expected total route count. Missing/crashed routes are zero-filled for DS/SR. Use 0 to average over available records.",
    )
    args = parser.parse_args()

    eval_root = infer_eval_root(args.eval_root)
    res_dir = resolve_res_dir(eval_root)
    viz_dir = resolve_viz_dir(eval_root)
    output_path = args.output or os.path.join(eval_root, "closed_loop_metrics.json")

    records = load_route_records(res_dir)
    if args.total_routes and len(records) != args.total_routes:
        print(
            "Warning: there are "
            f"{len(records)} non-crashed routes in {res_dir}, which does not equal {args.total_routes}. "
            "Missing or crashed routes are counted as zero for Driving Score and Success Rate."
        )

    merged_data = build_merged_summary(
        records,
        include_budget=not args.no_budget,
        budget_keys=args.budget_keys,
        total_routes=args.total_routes,
    )

    merged_path = os.path.join(res_dir, "merged.json")
    with open(merged_path, "w", encoding="utf-8") as file:
        json.dump(merged_data, file, indent=4)

    metric_map = discover_metric_files(viz_dir)
    efficiency_summary = compute_efficiency_and_smoothness(merged_data, metric_map)

    combined_summary = {
        "eval_root": eval_root,
        "res_dir": res_dir,
        "viz_dir": viz_dir,
        "merged_json": merged_path,
        "driving_score": merged_data["driving score"],
        "success_rate": merged_data["success rate"],
        "eval_num": merged_data["eval num"],
        "actual_eval_num": merged_data["actual eval num"],
        "missing_eval_num": merged_data["missing eval num"],
        **efficiency_summary,
    }
    if not args.no_budget:
        combined_summary["budget_metrics"] = merged_data.get("budget_metrics")
    combined_summary["inference_measurements"] = collect_inference_measurements(
        merged_data.get("_checkpoint", {}).get("records", [])
    )

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(combined_summary, file, indent=4)

    print(json.dumps(combined_summary, indent=4))


if __name__ == "__main__":
    main()
