import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr


METRICS = [
    "spatial_entropy",
    "novelty",
    "decision_shift_speed_e_norm",
    "decision_shift_route_e_norm",
]

LOSS_TARGETS = [
    "loss_total_mean",
    "loss_total_p90",
    "loss_total_hard_frame_ratio",
    "speed_wps_loss_mean",
    "route_loss_mean",
]


def to_float(v):
    try:
        return float(v)
    except Exception:
        return float("nan")


def quantile(values, q):
    vals = sorted(values)
    if not vals:
        return float("nan")
    idx = int((len(vals) - 1) * q)
    return vals[idx]


def basic_stats(values):
    if not values:
        return {}
    vals = sorted(values)
    return {
        "count": len(vals),
        "min": vals[0],
        "p25": quantile(vals, 0.25),
        "p50": quantile(vals, 0.50),
        "p75": quantile(vals, 0.75),
        "p90": quantile(vals, 0.90),
        "max": vals[-1],
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals)),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Across-route correlation analysis")
    parser.add_argument(
        "--input-jsonl",
        default="outputs/scene_difficulty/cleaned_scene_metrics.jsonl",
        help="Preprocessed cleaned JSONL path",
    )
    parser.add_argument(
        "--output-route-jsonl",
        default="outputs/scene_difficulty/route_level_aggregates.jsonl",
        help="Per-route aggregated feature JSONL",
    )
    parser.add_argument(
        "--output-summary-json",
        default="outputs/scene_difficulty/across_route_correlation_summary.json",
        help="Across-route summary JSON path",
    )
    parser.add_argument(
        "--min-route-len",
        type=int,
        default=30,
        help="Only include routes with at least this many frames",
    )
    parser.add_argument(
        "--high-quantile",
        type=float,
        default=0.90,
        help="High quantile for metric/loss high-threshold statistics (e.g., 0.90 or 0.95)",
    )
    return parser.parse_args()


def load_routes(input_jsonl):
    route_rows = defaultdict(list)
    with open(input_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            route_rows[row["route_id"]].append(row)
    for rid in route_rows:
        route_rows[rid].sort(key=lambda r: int(r["frame_id"]))
    return route_rows


def build_global_thresholds(route_rows, high_q):
    all_metric_values = {m: [] for m in METRICS}
    all_loss_total = []

    for rows in route_rows.values():
        for row in rows:
            for m in METRICS:
                if m.startswith("decision_shift_") and (not row.get("valid_decision_shift", False)):
                    continue
                v = to_float(row.get(m))
                if math.isfinite(v):
                    all_metric_values[m].append(v)

            loss_total = to_float(row.get("loss_total"))
            if math.isfinite(loss_total):
                all_loss_total.append(loss_total)

    metric_thresholds = {m: quantile(vals, high_q) for m, vals in all_metric_values.items()}
    loss_total_threshold = quantile(all_loss_total, high_q)
    return metric_thresholds, loss_total_threshold


def aggregate_route(rows, metric_thresholds, loss_total_threshold, high_q):
    out = {
        "route_id": rows[0]["route_id"],
        "route_len": len(rows),
    }

    loss_total_vals = []
    speed_loss_vals = []
    route_loss_vals = []
    for row in rows:
        lv = to_float(row.get("loss_total"))
        sv = to_float(row.get("speed_wps_loss"))
        rv = to_float(row.get("route_loss"))
        if math.isfinite(lv):
            loss_total_vals.append(lv)
        if math.isfinite(sv):
            speed_loss_vals.append(sv)
        if math.isfinite(rv):
            route_loss_vals.append(rv)

    hard_loss_count = sum(1 for v in loss_total_vals if v > loss_total_threshold)
    out["loss_total_mean"] = float(np.mean(loss_total_vals)) if loss_total_vals else float("nan")
    out["loss_total_p90"] = quantile(loss_total_vals, 0.90) if loss_total_vals else float("nan")
    out["loss_total_hard_frame_ratio"] = hard_loss_count / len(loss_total_vals) if loss_total_vals else float("nan")
    out["speed_wps_loss_mean"] = float(np.mean(speed_loss_vals)) if speed_loss_vals else float("nan")
    out["route_loss_mean"] = float(np.mean(route_loss_vals)) if route_loss_vals else float("nan")

    for m in METRICS:
        vals = []
        for row in rows:
            if m.startswith("decision_shift_") and (not row.get("valid_decision_shift", False)):
                continue
            v = to_float(row.get(m))
            if math.isfinite(v):
                vals.append(v)

        out[f"{m}_mean"] = float(np.mean(vals)) if vals else float("nan")
        out[f"{m}_std"] = float(np.std(vals)) if vals else float("nan")
        out[f"{m}_high_q"] = quantile(vals, high_q) if vals else float("nan")

        thr = metric_thresholds[m]
        hard_count = sum(1 for v in vals if v > thr)
        out[f"{m}_high_frame_ratio"] = hard_count / len(vals) if vals else float("nan")
        out[f"{m}_valid_count"] = len(vals)

    return out


def collect_feature_names():
    names = []
    for m in METRICS:
        names.extend(
            [
                f"{m}_mean",
                f"{m}_high_q",
                f"{m}_std",
                f"{m}_high_frame_ratio",
            ]
        )
    return names


def pair_values(route_aggs, x_key, y_key):
    xs, ys = [], []
    for row in route_aggs:
        x = to_float(row.get(x_key))
        y = to_float(row.get(y_key))
        if math.isfinite(x) and math.isfinite(y):
            xs.append(x)
            ys.append(y)
    return xs, ys


def spearman_item(route_aggs, x_key, y_key):
    xs, ys = pair_values(route_aggs, x_key, y_key)
    if len(xs) < 3:
        return {
            "n_routes": len(xs),
            "spearman": None,
            "direction": "no_data",
        }
    corr = spearmanr(xs, ys).statistic
    if not math.isfinite(corr):
        return {
            "n_routes": len(xs),
            "spearman": None,
            "direction": "no_data",
        }
    if corr > 0.10:
        direction = "positive"
    elif corr < -0.10:
        direction = "negative"
    else:
        direction = "near_zero"
    return {
        "n_routes": len(xs),
        "spearman": float(corr),
        "direction": direction,
    }


def main():
    args = parse_args()

    route_rows = load_routes(args.input_jsonl)
    all_route_ids = sorted(route_rows.keys())
    eligible_route_ids = [rid for rid in all_route_ids if len(route_rows[rid]) >= args.min_route_len]
    eligible_routes = {rid: route_rows[rid] for rid in eligible_route_ids}

    metric_thresholds, loss_total_threshold = build_global_thresholds(eligible_routes, args.high_quantile)

    route_aggs = []
    for rid in eligible_route_ids:
        route_aggs.append(
            aggregate_route(
                eligible_routes[rid],
                metric_thresholds=metric_thresholds,
                loss_total_threshold=loss_total_threshold,
                high_q=args.high_quantile,
            )
        )

    out_route = Path(args.output_route_jsonl)
    out_route.parent.mkdir(parents=True, exist_ok=True)
    with out_route.open("w", encoding="utf-8") as f:
        for row in route_aggs:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    feature_names = collect_feature_names()
    corr_matrix = {}
    for feature in feature_names:
        corr_matrix[feature] = {}
        for loss_target in LOSS_TARGETS:
            corr_matrix[feature][loss_target] = spearman_item(route_aggs, feature, loss_target)

    feature_stats = {}
    for feature in feature_names:
        vals = [to_float(r.get(feature)) for r in route_aggs]
        vals = [v for v in vals if math.isfinite(v)]
        feature_stats[feature] = basic_stats(vals)

    loss_stats = {}
    for loss_target in LOSS_TARGETS:
        vals = [to_float(r.get(loss_target)) for r in route_aggs]
        vals = [v for v in vals if math.isfinite(v)]
        loss_stats[loss_target] = basic_stats(vals)

    summary = {
        "analysis_scope": "across-route (route-level aggregated samples)",
        "main_statistic": "spearman",
        "input_jsonl": args.input_jsonl,
        "route_counts": {
            "all_routes": len(all_route_ids),
            "eligible_routes": len(eligible_route_ids),
            "min_route_len": args.min_route_len,
        },
        "aggregation": {
            "metric_features": {
                "central": "mean",
                "high_quantile": args.high_quantile,
                "volatility": "std",
                "high_frame_ratio_threshold": f"global p{int(args.high_quantile*100)}",
            },
            "loss_features": [
                "loss_total_mean",
                "loss_total_p90",
                "loss_total_hard_frame_ratio",
                "speed_wps_loss_mean",
                "route_loss_mean",
            ],
            "decision_shift_note": "decision_shift metrics exclude rows where valid_decision_shift=False (usually route first frame by definition).",
        },
        "global_thresholds": {
            "metric_thresholds": metric_thresholds,
            "loss_total_threshold": loss_total_threshold,
        },
        "route_feature_stats": feature_stats,
        "route_loss_stats": loss_stats,
        "spearman_correlations": corr_matrix,
    }

    out_summary = Path(args.output_summary_json)
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    with out_summary.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(
        f"[across-route] all_routes={len(all_route_ids)}, eligible_routes={len(eligible_route_ids)}, "
        f"min_route_len={args.min_route_len}, high_quantile={args.high_quantile}"
    )
    print(f"[across-route] route aggregates: {out_route}")
    print(f"[across-route] summary: {out_summary}")


if __name__ == "__main__":
    main()
