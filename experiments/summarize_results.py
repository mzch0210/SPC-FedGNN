import argparse
import csv
import math
from pathlib import Path


def read_csv(path):
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def to_float(row, key, default=math.nan):
    try:
        return float(row[key])
    except Exception:
        return default


def percentile(values, q):
    clean = []
    for value in values:
        try:
            value = float(value)
        except Exception:
            continue
        if not math.isnan(value):
            clean.append(value)
    values = sorted(clean)
    if not values:
        return math.nan
    if len(values) == 1:
        return values[0]
    pos = (len(values) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return values[lo]
    frac = pos - lo
    return values[lo] * (1.0 - frac) + values[hi] * frac


def mean(values):
    clean = []
    for value in values:
        try:
            value = float(value)
        except Exception:
            continue
        if not math.isnan(value):
            clean.append(value)
    return sum(clean) / len(clean) if clean else math.nan


def summarize_client_rows(rows):
    if not rows:
        return {}
    accs = [to_float(r, "accuracy") for r in rows]
    f1s = [to_float(r, "macro_f1") for r in rows]
    f1s_sorted = sorted(f1s)
    bottom_count = max(1, math.ceil(0.10 * len(f1s_sorted)))

    coverage_rows = [
        r
        for r in rows
        if to_float(r, "train_num_classes", 999) >= 2
        and to_float(r, "test_num_nodes", 999) >= 5
        and to_float(r, "test_num_classes", 999) >= 2
    ]
    coverage_f1s = [to_float(r, "macro_f1") for r in coverage_rows]
    coverage_accs = [to_float(r, "accuracy") for r in coverage_rows]

    out = {
        "method": rows[0].get("method", ""),
        "dataset": rows[0].get("dataset", ""),
        "partition": rows[0].get("partition", ""),
        "clients": rows[0].get("clients", ""),
        "seed": rows[0].get("seed", ""),
        "run_tag": rows[0].get("run_tag", "base") or "base",
        "mean_accuracy": mean(accs),
        "worst_accuracy": min(accs),
        "p05_accuracy": percentile(accs, 0.05),
        "bottom10_accuracy": mean(sorted(accs)[:bottom_count]),
        "mean_macro_f1": mean(f1s),
        "worst_macro_f1": min(f1s),
        "p05_macro_f1": percentile(f1s, 0.05),
        "bottom10_macro_f1": mean(f1s_sorted[:bottom_count]),
        "coverage_clients": len(coverage_rows),
        "coverage_mean_accuracy": mean(coverage_accs),
        "coverage_worst_accuracy": min(coverage_accs) if coverage_accs else math.nan,
        "coverage_p05_accuracy": percentile(coverage_accs, 0.05),
        "coverage_bottom10_accuracy": mean(sorted(coverage_accs)[: max(1, math.ceil(0.10 * len(coverage_accs)))])
        if coverage_accs
        else math.nan,
        "coverage_mean_macro_f1": mean(coverage_f1s),
        "coverage_worst_macro_f1": min(coverage_f1s) if coverage_f1s else math.nan,
        "coverage_p05_macro_f1": percentile(coverage_f1s, 0.05),
        "coverage_bottom10_macro_f1": mean(sorted(coverage_f1s)[: max(1, math.ceil(0.10 * len(coverage_f1s)))])
        if coverage_f1s
        else math.nan,
        "train_seconds": to_float(rows[0], "train_seconds"),
        "model_parameter_count": to_float(rows[0], "model_parameter_count"),
        "personal_head_parameter_count": to_float(rows[0], "personal_head_parameter_count"),
        "descriptor_dim": to_float(rows[0], "descriptor_dim"),
        "estimated_upload_bytes": to_float(rows[0], "estimated_upload_bytes"),
        "estimated_download_bytes": to_float(rows[0], "estimated_download_bytes"),
        "estimated_model_upload_bytes": to_float(rows[0], "estimated_model_upload_bytes"),
        "estimated_model_download_bytes": to_float(rows[0], "estimated_model_download_bytes"),
        "estimated_descriptor_upload_bytes": to_float(rows[0], "estimated_descriptor_upload_bytes"),
        "estimated_candidate_download_bytes": to_float(rows[0], "estimated_candidate_download_bytes"),
        "fedpub_proxy_nodes": to_float(rows[0], "fedpub_proxy_nodes"),
        "guard_validation_ratio": to_float(rows[0], "guard_validation_ratio"),
        "guard_validation_drop_classes": to_float(rows[0], "guard_validation_drop_classes"),
        "candidate_label_conflict_rate": mean([to_float(r, "candidate_label_conflict_rate") for r in rows]),
        "candidate_label_js_mean": mean([to_float(r, "candidate_label_js_mean") for r in rows]),
        "guard_accept_rate": mean([to_float(r, "guard_accept_rate") for r in rows]),
        "guard_pp_class_accept_rate": mean([to_float(r, "guard_pp_class_accept_rate") for r in rows]),
        "guard_mean_delta": mean([to_float(r, "guard_mean_delta") for r in rows]),
        "guard_mean_uncertainty": mean([to_float(r, "guard_mean_uncertainty") for r in rows]),
    }
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--client-summary",
        action="store_true",
        help="Treat inputs as per-client result CSV files and compute robust/coverage-aware metrics.",
    )
    args = parser.parse_args()

    rows = []
    for item in args.inputs:
        path = Path(item)
        if not path.exists():
            continue
        if args.client_summary:
            summary = summarize_client_rows(read_csv(path))
            if summary:
                summary["source_file"] = str(path)
                rows.append(summary)
        else:
            for row in read_csv(path):
                row = dict(row)
                row["source_file"] = str(path)
                rows.append(row)

    if not rows:
        raise SystemExit("No input rows found.")

    keys = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
