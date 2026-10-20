import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr


METRICS = [
    "spatial_entropy",
    "novelty",
    "decision_shift_speed_e_norm",
    "decision_shift_route_e_norm",
]

DEFAULT_ROUTE_OUTPUT = "outputs/scene_difficulty/within_route_correlations.jsonl"
DEFAULT_SUMMARY_OUTPUT = "outputs/scene_difficulty/within_route_correlation_summary.json"


def to_float(v):
    try:
        return float(v)
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
        "max": vals[-1],
        "mean": float(np.mean(vals)),
    }


def direction_label(pos_ratio, neg_ratio, median_corr):
    if pos_ratio > 0.55 and pos_ratio > neg_ratio:
        return "overall_positive"
    if neg_ratio > 0.55 and neg_ratio > pos_ratio:
        return "overall_negative"
    if abs(median_corr) <= 0.10 and abs(pos_ratio - neg_ratio) <= 0.10:
        return "overall_near_zero"
    return "mixed_or_weak"


def parse_args():
    parser = argparse.ArgumentParser(description="Within-route frame-level correlation analysis")
    parser.add_argument(
        "--input-jsonl",
        default="outputs/scene_difficulty/cleaned_scene_metrics.jsonl",
        help="Preprocessed cleaned JSONL path",
    )
    parser.add_argument(
        "--output-route-jsonl",
        default=DEFAULT_ROUTE_OUTPUT,
        help="Per-route correlation result JSONL",
    )
    parser.add_argument(
        "--output-summary-json",
        default=DEFAULT_SUMMARY_OUTPUT,
        help="Overall summary JSON path",
    )
    parser.add_argument(
        "--min-route-len",
        type=int,
        default=30,
        help="Only analyze routes with at least this many frames",
    )
    parser.add_argument(
        "--min-pairs",
        type=int,
        default=3,
        help="Minimum valid frame pairs for one route-level correlation",
    )
    parser.add_argument(
        "--use-detrend",
        action="store_true",
        help="Apply within-route detrending: z-score then regress out relative time t=i/route_len.",
    )
    parser.add_argument(
        "--detrend-output-suffix",
        default="detrended",
        help="Suffix for output files when --use-detrend is enabled and default output paths are used.",
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
    for route_id in route_rows:
        route_rows[route_id].sort(key=lambda r: int(r["frame_id"]))
    return route_rows


def paired_values(rows, metric):
    xs, ys, ts = [], [], []
    route_len = len(rows)
    for i, row in enumerate(rows):
        if metric.startswith("decision_shift_") and (not row.get("valid_decision_shift", False)):
            continue
        x = to_float(row.get(metric))
        y = to_float(row.get("speed_wps_loss")) + to_float(row.get("route_loss"))
        if math.isfinite(x) and math.isfinite(y):
            xs.append(x)
            ys.append(y)
            ts.append(i / route_len)
    return xs, ys, ts


def corr_for_route(xs, ys):
    if len(xs) < 3:
        return float("nan"), float("nan")
    try:
        sp = spearmanr(xs, ys).statistic
        pe = pearsonr(xs, ys).statistic
    except Exception:
        return float("nan"), float("nan")
    return float(sp), float(pe)


def zscore(values):
    arr = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(arr))
    std = float(np.std(arr))
    if std <= 0.0:
        return None
    return (arr - mean) / std


def regress_out_time(values, ts):
    arr = np.asarray(values, dtype=np.float64)
    t = np.asarray(ts, dtype=np.float64)
    if arr.size != t.size or arr.size < 2:
        return None
    t_mean = float(np.mean(t))
    t_var = float(np.sum((t - t_mean) ** 2))
    if t_var <= 0.0:
        return None
    slope = float(np.sum((t - t_mean) * (arr - np.mean(arr))) / t_var)
    intercept = float(np.mean(arr) - slope * t_mean)
    fitted = slope * t + intercept
    residual = arr - fitted
    return residual


def resolve_output_paths(args):
    if not args.use_detrend:
        return args.output_route_jsonl, args.output_summary_json

    if args.output_route_jsonl == DEFAULT_ROUTE_OUTPUT:
        route_output = DEFAULT_ROUTE_OUTPUT.replace(".jsonl", f"_{args.detrend_output_suffix}.jsonl")
    else:
        route_output = args.output_route_jsonl

    if args.output_summary_json == DEFAULT_SUMMARY_OUTPUT:
        summary_output = DEFAULT_SUMMARY_OUTPUT.replace(".json", f"_{args.detrend_output_suffix}.json")
    else:
        summary_output = args.output_summary_json

    return route_output, summary_output


def main():
    args = parse_args()
    output_route_jsonl, output_summary_json = resolve_output_paths(args)

    route_rows = load_routes(args.input_jsonl)
    route_ids = sorted(route_rows.keys())

    eligible_routes = [r for r in route_ids if len(route_rows[r]) >= args.min_route_len]
    per_route_records = []

    metric_spearman = {m: [] for m in METRICS}
    metric_pearson = {m: [] for m in METRICS}
    metric_route_used = {m: 0 for m in METRICS}
    metric_invalid_reason_counts = {m: defaultdict(int) for m in METRICS}

    for route_id in eligible_routes:
        rows = route_rows[route_id]
        route_len = len(rows)
        record = {
            "route_id": route_id,
            "route_len": route_len,
            "analysis_mode": "detrended_residual" if args.use_detrend else "raw",
            "wps_loss_definition": "speed_wps_loss + route_loss",
            "correlations": {},
        }

        for metric in METRICS:
            xs, ys, ts = paired_values(rows, metric)
            n_pairs_before = len(xs)
            n_pairs_after = n_pairs_before

            if n_pairs_before < args.min_pairs:
                metric_invalid_reason_counts[metric]["too_few_pairs_before"] += 1
                record["correlations"][metric] = {
                    "n_pairs_before": n_pairs_before,
                    "n_pairs_after": n_pairs_after,
                    "spearman": None,
                    "pearson": None,
                    "invalid_reason": "too_few_pairs_before",
                }
                continue

            corr_x = xs
            corr_y = ys
            invalid_reason = None

            if args.use_detrend:
                zx = zscore(xs)
                zy = zscore(ys)
                if zx is None:
                    invalid_reason = "zero_std_x_after_zscore"
                elif zy is None:
                    invalid_reason = "zero_std_y_after_zscore"
                else:
                    rx = regress_out_time(zx, ts)
                    ry = regress_out_time(zy, ts)
                    if rx is None or ry is None:
                        invalid_reason = "time_regression_failed"
                    else:
                        mask = np.isfinite(rx) & np.isfinite(ry)
                        corr_x = rx[mask].tolist()
                        corr_y = ry[mask].tolist()
                        n_pairs_after = len(corr_x)
                        if n_pairs_after < args.min_pairs:
                            invalid_reason = "too_few_pairs_after"

            if invalid_reason is not None:
                metric_invalid_reason_counts[metric][invalid_reason] += 1
                record["correlations"][metric] = {
                    "n_pairs_before": n_pairs_before,
                    "n_pairs_after": n_pairs_after,
                    "spearman": None,
                    "pearson": None,
                    "invalid_reason": invalid_reason,
                }
                continue

            sp, pe = corr_for_route(corr_x, corr_y)
            if math.isfinite(sp):
                metric_spearman[metric].append(sp)
                metric_pearson[metric].append(pe)
                metric_route_used[metric] += 1
            else:
                metric_invalid_reason_counts[metric]["non_finite_correlation"] += 1

            record["correlations"][metric] = {
                "n_pairs_before": n_pairs_before,
                "n_pairs_after": n_pairs_after,
                "spearman": sp if math.isfinite(sp) else None,
                "pearson": pe if math.isfinite(pe) else None,
            }

        per_route_records.append(record)

    out_route = Path(output_route_jsonl)
    out_route.parent.mkdir(parents=True, exist_ok=True)
    with out_route.open("w", encoding="utf-8") as f:
        for row in per_route_records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    metric_summary = {}
    for metric in METRICS:
        sp_vals = metric_spearman[metric]
        pe_vals = metric_pearson[metric]

        pos = [v for v in sp_vals if v > 0]
        neg = [v for v in sp_vals if v < 0]
        zero = [v for v in sp_vals if v == 0]
        n = len(sp_vals)

        pos_ratio = len(pos) / n if n else 0.0
        neg_ratio = len(neg) / n if n else 0.0
        zero_ratio = len(zero) / n if n else 0.0
        med = float(np.median(sp_vals)) if n else float("nan")

        metric_summary[metric] = {
            "route_count_used": metric_route_used[metric],
            "spearman_distribution": quantiles(sp_vals),
            "pearson_distribution": quantiles(pe_vals),
            "positive_route_ratio": pos_ratio,
            "negative_route_ratio": neg_ratio,
            "zero_route_ratio": zero_ratio,
            "direction_interpretation": direction_label(pos_ratio, neg_ratio, med) if n else "no_data",
            "invalid_reason_counts": dict(metric_invalid_reason_counts[metric]),
        }

    summary = {
        "analysis_scope": (
            "within-route frame-level consistency with detrended residuals (not global mixed trend)"
            if args.use_detrend
            else "within-route frame-level consistency (not global mixed trend)"
        ),
        "analysis_mode": "detrended_residual" if args.use_detrend else "raw",
        "main_statistic": "spearman",
        "aux_statistic": "pearson",
        "loss_target": "wps_loss = speed_wps_loss + route_loss",
        "metrics": METRICS,
        "input_jsonl": args.input_jsonl,
        "output_route_jsonl": output_route_jsonl,
        "min_route_len": args.min_route_len,
        "min_pairs": args.min_pairs,
        "detrend": {
            "enabled": args.use_detrend,
            "method": "zscore_then_residualize_by_relative_time",
            "relative_time_definition": "t = i / route_len, i from frame order by frame_id within route",
            "note": "Apply detrend per metric-route pair on x(metric) and y(wps_loss) separately.",
        },
        "route_counts": {
            "all_routes": len(route_ids),
            "eligible_routes": len(eligible_routes),
        },
        "decision_shift_note": (
            "decision_shift_* at route first frame is typically NaN by definition; "
            "those frames are excluded for decision_shift metrics only."
        ),
        "metric_summaries": metric_summary,
    }

    out_summary = Path(output_summary_json)
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    with out_summary.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(
        f"[within-route] mode={'detrended_residual' if args.use_detrend else 'raw'}, "
        f"all_routes={len(route_ids)}, eligible_routes={len(eligible_routes)}"
    )
    for metric in METRICS:
        ms = metric_summary[metric]
        print(
            f"[within-route] {metric}: used={ms['route_count_used']}, "
            f"pos={ms['positive_route_ratio']:.3f}, neg={ms['negative_route_ratio']:.3f}, "
            f"direction={ms['direction_interpretation']}"
        )
    print(f"[within-route] per-route output: {out_route}")
    print(f"[within-route] summary output: {out_summary}")


if __name__ == "__main__":
    main()
