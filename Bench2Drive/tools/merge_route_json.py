import argparse
import glob
import json
import math
import os


ROUND_DIGITS = 6
DEFAULT_BUDGET_KEYS = (
    "average_budget",
    "min_budget",
    "max_budget",
    "std_budget",
    "initial_base_budget",
    "final_base_budget",
)
DEFAULT_TOTAL_ROUTES = 220


def to_float_or_none(value):
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def load_route_records(folder_path):
    file_paths = glob.glob(os.path.join(folder_path, "*.json"))
    merged_records = []

    for file_path in file_paths:
        if os.path.basename(file_path) == "merged.json":
            continue
        with open(file_path, encoding="utf-8") as file:
            data = json.load(file)
        records = data.get("_checkpoint", {}).get("records", [])
        for record in records:
            if record.get("status") == "Failed - Agent crashed":
                continue
            clean_record = dict(record)
            clean_record.pop("index", None)
            merged_records.append(clean_record)
    return merged_records


def is_success_record(record):
    if record.get("status") not in {"Completed", "Perfect"}:
        return False

    infractions = record.get("infractions", {})
    if not isinstance(infractions, dict):
        return False

    for key, value in infractions.items():
        if key == "min_speed_infractions":
            continue
        if isinstance(value, list):
            if len(value) > 0:
                return False
        elif value:
            return False
    return True


def extract_budget_values(record, budget_keys=None):
    budget_data = record.get("budget")
    if not isinstance(budget_data, dict):
        return {}

    keys = list(budget_keys) if budget_keys is not None else list(budget_data.keys())
    return {key: to_float_or_none(budget_data.get(key)) for key in keys}


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
        "mean": round(sum(sorted_values) / len(sorted_values), ROUND_DIGITS),
        "p50": round(_linear_quantile(sorted_values, 0.50), ROUND_DIGITS),
        "p90": round(_linear_quantile(sorted_values, 0.90), ROUND_DIGITS),
        "p95": round(_linear_quantile(sorted_values, 0.95), ROUND_DIGITS),
        "min": round(sorted_values[0], ROUND_DIGITS),
        "max": round(sorted_values[-1], ROUND_DIGITS),
    }


def _infer_budget_keys(records):
    discovered = set()
    for record in records:
        budget_data = record.get("budget")
        if isinstance(budget_data, dict):
            discovered.update(budget_data.keys())

    keys = [key for key in DEFAULT_BUDGET_KEYS if key in discovered]
    keys.extend(sorted(key for key in discovered if key not in DEFAULT_BUDGET_KEYS))
    return keys


def collect_budget_metrics(records, budget_keys=None, require_success=False):
    records_for_budget = records
    if require_success:
        records_for_budget = [record for record in records if is_success_record(record)]

    record_count = len(records_for_budget)
    keys = list(budget_keys) if budget_keys is not None else _infer_budget_keys(records_for_budget)

    values_by_key = {key: [] for key in keys}
    budget_record_count = 0

    for record in records_for_budget:
        budget_data = record.get("budget")
        if isinstance(budget_data, dict):
            budget_record_count += 1
        parsed_values = extract_budget_values(record, keys)
        for key in keys:
            value = parsed_values.get(key)
            if value is not None:
                values_by_key[key].append(value)

    fields = {}
    for key in keys:
        valid_count = len(values_by_key[key])
        coverage = round(valid_count / record_count, ROUND_DIGITS) if record_count else 0.0
        fields[key] = {
            "valid_count": valid_count,
            "missing_count": record_count - valid_count,
            "coverage": coverage,
            **summarize_numeric(values_by_key[key]),
        }

    return {
        "record_count": record_count,
        "budget_record_count": budget_record_count,
        "fields": fields,
    }


def _collect_driving_metrics(records, total_routes=DEFAULT_TOTAL_ROUTES):
    driving_scores = []
    success_num = 0

    for record in records:
        driving_scores.append(record["scores"]["score_composed"])
        if is_success_record(record):
            success_num += 1
            print(record.get("route_id"))

    actual_eval_num = len(driving_scores)
    eval_num = total_routes if total_routes else actual_eval_num
    driving_score_mean = (sum(driving_scores) / eval_num) if eval_num else 0.0
    success_rate = (success_num / eval_num) if eval_num else 0.0
    return driving_score_mean, success_rate, eval_num, actual_eval_num


def merge_route_json(folder_path, include_budget=True, budget_keys=None, total_routes=DEFAULT_TOTAL_ROUTES):
    merged_records = load_route_records(folder_path)
    if total_routes and len(merged_records) != total_routes:
        print(
            "-----------------------Warning: there are "
            f"{len(merged_records)} non-crashed routes in your json, which does not equal to {total_routes}. "
            "Missing or crashed routes are counted as zero for Driving Score and Success Rate."
        )

    driving_score_mean, success_rate, eval_num, actual_eval_num = _collect_driving_metrics(
        merged_records,
        total_routes=total_routes,
    )
    merged_records = sorted(merged_records, key=lambda record: record["route_id"], reverse=True)

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

    with open(os.path.join(folder_path, "merged.json"), "w", encoding="utf-8") as file:
        json.dump(merged_data, file, indent=4)

    return merged_data

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-f',
        '--folder',
        help='old foo help',
        default='eval_results/Bench2Drive/reproCVPTsimlingo_2025_05_02_04_08_01_simlingo_withaugmentation_seed2/bench2drive/1/res',
    )
    parser.add_argument(
        '--no-budget',
        action='store_true',
        help='Disable budget metric aggregation in merged.json.',
    )
    parser.add_argument(
        '--budget-keys',
        nargs='+',
        default=None,
        help='Optional explicit budget keys to aggregate.',
    )
    parser.add_argument(
        '--total-routes',
        type=int,
        default=DEFAULT_TOTAL_ROUTES,
        help='Expected total route count. Missing/crashed routes are zero-filled for DS/SR. Use 0 to average over available records.',
    )
    args = parser.parse_args()

    if os.path.isdir(args.folder):
        merge_route_json(
            args.folder,
            include_budget=not args.no_budget,
            budget_keys=args.budget_keys,
            total_routes=args.total_routes,
        )
