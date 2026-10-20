import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


MAIN_FIELDS = [
    "spatial_entropy",
    "history_similarity",
    "decision_shift_speed_e_norm",
    "decision_shift_route_e_norm",
    "loss_total",
    "speed_wps_loss",
    "route_loss",
]

LOSS_FIELDS = ["loss_total", "speed_wps_loss", "route_loss"]


def to_float(value):
    if value is None:
        return float("nan")
    try:
        return float(value)
    except Exception:
        return float("nan")


def quantiles(values):
    if not values:
        return {}
    vals = sorted(values)

    def pick(p):
        idx = int((len(vals) - 1) * p)
        return vals[idx]

    return {
        "count": len(vals),
        "min": vals[0],
        "p25": pick(0.25),
        "p50": pick(0.50),
        "p75": pick(0.75),
        "p90": pick(0.90),
        "p99": pick(0.99),
        "max": vals[-1],
    }


def find_input_files(input_glob):
    files = sorted(Path().glob(input_glob))
    filtered = []
    for p in files:
        name = p.name
        if not name.endswith(".jsonl"):
            continue
        if name.startswith("cleaned_"):
            continue
        filtered.append(p)
    return filtered


def load_and_dedup(files):
    rows_by_key = {}
    source_idx_by_key = {}

    total_rows = 0
    duplicate_rows = 0
    overwritten_rows = 0

    for file_idx, path in enumerate(files):
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                total_rows += 1
                row = json.loads(line)

                route_id = row["route_id"]
                frame_id = int(row["frame_id"])
                key = (route_id, frame_id)

                if key in rows_by_key:
                    duplicate_rows += 1
                    overwritten_rows += 1

                rows_by_key[key] = row
                source_idx_by_key[key] = file_idx

    return rows_by_key, source_idx_by_key, {
        "input_row_count": total_rows,
        "unique_row_count": len(rows_by_key),
        "duplicate_row_count": duplicate_rows,
        "overwritten_row_count": overwritten_rows,
    }


def build_clean_rows(rows_by_key, min_route_len):
    route_to_frames = defaultdict(list)
    route_to_min_frame = {}

    for (route_id, frame_id), row in rows_by_key.items():
        route_to_frames[route_id].append(frame_id)
        if route_id not in route_to_min_frame or frame_id < route_to_min_frame[route_id]:
            route_to_min_frame[route_id] = frame_id

    route_lens = {route_id: len(frames) for route_id, frames in route_to_frames.items()}

    clean_rows = []
    for route_id in sorted(route_to_frames.keys()):
        min_frame = route_to_min_frame[route_id]
        route_len = route_lens[route_id]

        for frame_id in sorted(route_to_frames[route_id]):
            src = rows_by_key[(route_id, frame_id)]

            history_similarity = to_float(src.get("history_similarity"))
            novelty = 1.0 - history_similarity if math.isfinite(history_similarity) else float("nan")

            d_speed = to_float(src.get("decision_shift_speed_e_norm"))
            d_route = to_float(src.get("decision_shift_route_e_norm"))
            valid_decision_shift = math.isfinite(d_speed) and math.isfinite(d_route)

            out = {
                "route_id": route_id,
                "frame_id": frame_id,
                "spatial_entropy": to_float(src.get("spatial_entropy")),
                "history_similarity": history_similarity,
                "novelty": novelty,
                "decision_shift_speed_e_norm": d_speed,
                "decision_shift_route_e_norm": d_route,
                "loss_total": to_float(src.get("loss_total")),
                "speed_wps_loss": to_float(src.get("speed_wps_loss")),
                "route_loss": to_float(src.get("route_loss")),
                "is_first_frame_in_route": frame_id == min_frame,
                "valid_decision_shift": valid_decision_shift,
                "route_len": route_len,
                "route_len_ok": route_len >= min_route_len,
            }

            for loss_key in LOSS_FIELDS:
                v = out[loss_key]
                out[f"{loss_key}_log1p"] = math.log1p(v) if math.isfinite(v) and v > -1.0 else float("nan")

            clean_rows.append(out)

    return clean_rows, route_lens


def summarize(clean_rows, route_lens, stats, files, min_route_len):
    nan_counts = defaultdict(int)
    for row in clean_rows:
        for k in MAIN_FIELDS + ["novelty"]:
            v = row.get(k)
            if isinstance(v, float) and math.isnan(v):
                nan_counts[k] += 1

    first_frame_count = 0
    first_frame_decision_shift_nan_count = 0
    non_first_frame_decision_shift_nan_count = 0
    novelty_close_count = 0

    loss_raw = {k: [] for k in LOSS_FIELDS}
    loss_log = {f"{k}_log1p": [] for k in LOSS_FIELDS}

    for row in clean_rows:
        if row["is_first_frame_in_route"]:
            first_frame_count += 1
            if not row["valid_decision_shift"]:
                first_frame_decision_shift_nan_count += 1
        else:
            if not row["valid_decision_shift"]:
                non_first_frame_decision_shift_nan_count += 1

        hs = row["history_similarity"]
        nv = row["novelty"]
        if math.isfinite(hs) and math.isfinite(nv):
            if abs((hs + nv) - 1.0) < 1e-9:
                novelty_close_count += 1

        for k in LOSS_FIELDS:
            rv = row[k]
            lv = row[f"{k}_log1p"]
            if math.isfinite(rv):
                loss_raw[k].append(rv)
            if math.isfinite(lv):
                loss_log[f"{k}_log1p"].append(lv)

    route_len_values = sorted(route_lens.values())
    routes_ge_threshold = sum(1 for v in route_len_values if v >= min_route_len)

    summary = {
        "input_files": [str(p) for p in files],
        "dedup_policy": "last-write-wins-by-file-order",
        "key_fields": ["route_id", "frame_id"],
        "main_analysis_fields": MAIN_FIELDS,
        "added_fields": [
            "novelty",
            "is_first_frame_in_route",
            "valid_decision_shift",
            "route_len",
            "route_len_ok",
            "loss_total_log1p",
            "speed_wps_loss_log1p",
            "route_loss_log1p",
        ],
        "min_route_len": min_route_len,
        **stats,
        "valid_row_count": len(clean_rows),
        "nan_counts": dict(nan_counts),
        "decision_shift_note": (
            "decision_shift_* first-frame NaN is expected by metric definition "
            "(no previous frame), not abnormal data."
        ),
        "decision_shift_alignment": {
            "first_frame_count": first_frame_count,
            "first_frame_invalid_decision_shift_count": first_frame_decision_shift_nan_count,
            "non_first_frame_invalid_decision_shift_count": non_first_frame_decision_shift_nan_count,
        },
        "novelty_check": {
            "rows_with_finite_history_similarity_and_novelty": novelty_close_count,
            "description": "count where abs((history_similarity + novelty) - 1) < 1e-9",
        },
        "route_length_stats": {
            "route_count": len(route_len_values),
            "min": route_len_values[0] if route_len_values else None,
            "p25": route_len_values[int((len(route_len_values) - 1) * 0.25)] if route_len_values else None,
            "p50": route_len_values[int((len(route_len_values) - 1) * 0.50)] if route_len_values else None,
            "p75": route_len_values[int((len(route_len_values) - 1) * 0.75)] if route_len_values else None,
            "p90": route_len_values[int((len(route_len_values) - 1) * 0.90)] if route_len_values else None,
            "max": route_len_values[-1] if route_len_values else None,
            "routes_ge_min_route_len": routes_ge_threshold,
            "routes_lt_min_route_len": len(route_len_values) - routes_ge_threshold,
        },
        "loss_distributions": {
            "raw": {k: quantiles(v) for k, v in loss_raw.items()},
            "log1p": {k: quantiles(v) for k, v in loss_log.items()},
        },
    }
    return summary


def write_jsonl(rows, output_jsonl):
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(data, output_json):
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess scene difficulty JSONL data")
    parser.add_argument(
        "--input-glob",
        default="outputs/scene_difficulty/*.jsonl",
        help="Glob for input JSONL files. cleaned_*.jsonl is excluded automatically.",
    )
    parser.add_argument(
        "--output-jsonl",
        default="outputs/scene_difficulty/cleaned_scene_metrics.jsonl",
        help="Path to cleaned output JSONL",
    )
    parser.add_argument(
        "--output-summary",
        default="outputs/scene_difficulty/preprocess_summary.json",
        help="Path to summary JSON",
    )
    parser.add_argument(
        "--min-route-len",
        type=int,
        default=30,
        help="Minimum route length threshold for route_len_ok",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    files = find_input_files(args.input_glob)
    if not files:
        raise RuntimeError(f"No input JSONL files found for glob: {args.input_glob}")

    rows_by_key, _, stats = load_and_dedup(files)
    clean_rows, route_lens = build_clean_rows(rows_by_key, args.min_route_len)
    summary = summarize(clean_rows, route_lens, stats, files, args.min_route_len)

    output_jsonl = Path(args.output_jsonl)
    output_summary = Path(args.output_summary)

    write_jsonl(clean_rows, output_jsonl)
    write_json(summary, output_summary)

    print(f"[preprocess] input_files={len(files)}")
    print(f"[preprocess] input_rows={summary['input_row_count']}")
    print(f"[preprocess] unique_rows={summary['unique_row_count']}")
    print(f"[preprocess] duplicate_rows={summary['duplicate_row_count']}")
    print(f"[preprocess] output_jsonl={output_jsonl}")
    print(f"[preprocess] output_summary={output_summary}")


if __name__ == "__main__":
    main()
