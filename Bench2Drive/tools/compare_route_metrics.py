import argparse
import json
import math
from pathlib import Path

from merge_route_json import is_success_record, load_route_records


ROUND_DIGITS = 6
REPO_ROOT = Path(__file__).resolve().parents[2]
REFERENCE_EVAL_ROOT = (
    REPO_ROOT
    / "eval_results"
    / "Bench2Drive_smart_assigner_stage2"
    / "2026_06_12_v2"
    / "simlingo"
    / "bench2drive"
    / "12"
    / "smart_assigner"
)
OUTPUT_NAME = "route_overlap_metrics.json"
CRASH_STATUSES = {
    "Failed - Agent crashed",
    "Failed - Simulation crashed",
}


def load_valid_records(eval_root):
    """Load evaluated routes and validate comparison fields."""
    eval_root = Path(eval_root).expanduser().resolve()
    res_dir = eval_root / "res"
    if not res_dir.is_dir():
        raise FileNotFoundError(f"Could not find res directory: {res_dir}")

    records_by_route = {}
    for record in load_route_records(str(res_dir)):
        if record.get("status") in CRASH_STATUSES:
            continue

        route_id = record.get("route_id")
        if not isinstance(route_id, str) or not route_id:
            raise ValueError(f"Record in {res_dir} has an invalid route_id")
        if route_id in records_by_route:
            raise ValueError(f"Duplicate route_id in {res_dir}: {route_id}")

        scores = record.get("scores")
        if not isinstance(scores, dict) or "score_composed" not in scores:
            raise ValueError(f"Route {route_id} in {res_dir} is missing scores.score_composed")
        try:
            driving_score = float(scores["score_composed"])
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Route {route_id} in {res_dir} has an invalid scores.score_composed"
            ) from error
        if not math.isfinite(driving_score):
            raise ValueError(
                f"Route {route_id} in {res_dir} has a non-finite scores.score_composed"
            )

        records_by_route[route_id] = record

    return eval_root, records_by_route


def compute_metrics(records_by_route, route_ids):
    """Compute DS and strict SR over an identical route set."""
    records = [records_by_route[route_id] for route_id in route_ids]
    success_count = sum(is_success_record(record) for record in records)
    route_count = len(records)
    return {
        "avg_ds": sum(float(record["scores"]["score_composed"]) for record in records)
        / route_count,
        "avg_sr": success_count / route_count,
        "success_count": success_count,
    }


def build_comparison(input_eval_root, reference_eval_root=REFERENCE_EVAL_ROOT):
    """Compare both runs over their common evaluated routes."""
    input_eval_root, input_records = load_valid_records(input_eval_root)
    reference_eval_root, reference_records = load_valid_records(reference_eval_root)

    # Only route_id is stable across evaluation runs.
    overlap_route_ids = sorted(input_records.keys() & reference_records.keys())
    if not overlap_route_ids:
        raise ValueError("No overlapping valid routes were found")

    input_metrics = compute_metrics(input_records, overlap_route_ids)
    reference_metrics = compute_metrics(reference_records, overlap_route_ids)
    gap_avg_ds = input_metrics["avg_ds"] - reference_metrics["avg_ds"]
    gap_avg_sr = input_metrics["avg_sr"] - reference_metrics["avg_sr"]

    return {
        "input_eval_root": str(input_eval_root),
        "reference_eval_root": str(reference_eval_root),
        "input_valid_route_count": len(input_records),
        "reference_valid_route_count": len(reference_records),
        "overlap_route_count": len(overlap_route_ids),
        "input": {
            "avg_ds": round(input_metrics["avg_ds"], ROUND_DIGITS),
            "avg_sr": round(input_metrics["avg_sr"], ROUND_DIGITS),
            "success_count": input_metrics["success_count"],
        },
        "reference": {
            "avg_ds": round(reference_metrics["avg_ds"], ROUND_DIGITS),
            "avg_sr": round(reference_metrics["avg_sr"], ROUND_DIGITS),
            "success_count": reference_metrics["success_count"],
        },
        "gap_input_minus_reference": {
            "avg_ds": round(gap_avg_ds, ROUND_DIGITS),
            "avg_sr": round(gap_avg_sr, ROUND_DIGITS),
            "sr_percentage_points": round(gap_avg_sr * 100.0, ROUND_DIGITS),
        },
    }


def write_comparison(input_eval_root, reference_eval_root=REFERENCE_EVAL_ROOT):
    """Write the report to the input evaluation directory."""
    summary = build_comparison(input_eval_root, reference_eval_root)
    output_path = Path(summary["input_eval_root"]) / OUTPUT_NAME
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4)
        file.write("\n")
    return output_path, summary


def main():
    parser = argparse.ArgumentParser(
        description="Compare overlapping valid Bench2Drive routes with the fixed reference evaluation."
    )
    parser.add_argument(
        "-e",
        "--eval-root",
        required=True,
        help="Evaluation root containing res/.",
    )
    args = parser.parse_args()

    output_path, summary = write_comparison(args.eval_root)
    print(json.dumps(summary, indent=4))
    print(f"Saved comparison to: {output_path}")


if __name__ == "__main__":
    main()
