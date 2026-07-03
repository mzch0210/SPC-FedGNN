import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path


MAIN_METRICS = [
    "mean_accuracy",
    "mean_macro_f1",
    "bottom10_macro_f1",
    "p05_macro_f1",
    "coverage_mean_macro_f1",
    "coverage_bottom10_macro_f1",
    "train_seconds",
    "estimated_upload_bytes",
    "estimated_download_bytes",
    "estimated_model_upload_bytes",
    "estimated_model_download_bytes",
    "estimated_descriptor_upload_bytes",
    "estimated_candidate_download_bytes",
    "fedpub_proxy_nodes",
    "guard_validation_ratio",
    "guard_validation_drop_classes",
    "candidate_label_conflict_rate",
    "candidate_label_js_mean",
    "guard_accept_rate",
    "guard_pp_class_accept_rate",
    "guard_mean_delta",
    "guard_mean_uncertainty",
]


def to_float(value):
    try:
        return float(value)
    except Exception:
        return math.nan


def mean(values):
    values = [v for v in values if not math.isnan(v)]
    return sum(values) / len(values) if values else math.nan


def std(values):
    values = [v for v in values if not math.isnan(v)]
    if len(values) <= 1:
        return 0.0 if values else math.nan
    mu = mean(values)
    return math.sqrt(sum((v - mu) ** 2 for v in values) / len(values))


def sign_test_p_value(wins, losses):
    trials = wins + losses
    if trials <= 0:
        return math.nan
    tail = min(wins, losses)
    prob = sum(math.comb(trials, k) for k in range(tail + 1)) / (2 ** trials)
    return min(1.0, 2.0 * prob)


def read_rows(path):
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def stage_table(rows, reference_methods):
    grouped = defaultdict(list)
    for row in rows:
        key = (
            row.get("dataset", ""),
            row.get("partition", ""),
            row.get("clients", ""),
            row.get("method", ""),
            row.get("run_tag", "base") or "base",
        )
        grouped[key].append(row)

    out = []
    ref_lookup = {}
    for key, group in grouped.items():
        dataset, partition, clients, method, run_tag = key
        seeds = sorted({g.get("seed", "") for g in group}, key=lambda x: int(float(x)) if str(x).replace(".", "", 1).isdigit() else str(x))
        item = {
            "dataset": dataset,
            "partition": partition,
            "clients": clients,
            "method": method,
            "run_tag": run_tag,
            "seeds": ",".join(seeds),
            "num_seeds": len(seeds),
        }
        for metric in MAIN_METRICS:
            vals = [to_float(g.get(metric, "")) for g in group]
            item[f"{metric}_mean"] = mean(vals)
            item[f"{metric}_std"] = std(vals)
        out.append(item)
        if method in reference_methods:
            ref_lookup[(dataset, partition, clients, method, run_tag)] = item

    for item in out:
        for ref in reference_methods:
            ref_item = ref_lookup.get((item["dataset"], item["partition"], item["clients"], ref, item["run_tag"]))
            if not ref_item:
                item[f"delta_mean_macro_f1_vs_{ref}"] = math.nan
                item[f"win_count_vs_{ref}"] = ""
                item[f"paired_delta_mean_macro_f1_vs_{ref}"] = math.nan
                item[f"paired_delta_std_macro_f1_vs_{ref}"] = math.nan
                item[f"sign_test_p_vs_{ref}"] = math.nan
                continue
            item[f"delta_mean_macro_f1_vs_{ref}"] = (
                item["mean_macro_f1_mean"] - ref_item["mean_macro_f1_mean"]
            )
            wins = 0
            losses = 0
            ties = 0
            total = 0
            paired_deltas = []
            for seed in item["seeds"].split(","):
                if seed == "":
                    continue
                a = next(
                    (
                        r
                        for r in rows
                        if r.get("dataset") == item["dataset"]
                        and r.get("partition") == item["partition"]
                        and r.get("clients") == item["clients"]
                        and r.get("method") == item["method"]
                        and (r.get("run_tag", "base") or "base") == item["run_tag"]
                        and r.get("seed") == seed
                    ),
                    None,
                )
                b = next(
                    (
                        r
                        for r in rows
                        if r.get("dataset") == item["dataset"]
                        and r.get("partition") == item["partition"]
                        and r.get("clients") == item["clients"]
                        and r.get("method") == ref
                        and (r.get("run_tag", "base") or "base") == item["run_tag"]
                        and r.get("seed") == seed
                    ),
                    None,
                )
                if not a or not b:
                    continue
                av = to_float(a.get("mean_macro_f1", ""))
                bv = to_float(b.get("mean_macro_f1", ""))
                if math.isnan(av) or math.isnan(bv):
                    continue
                total += 1
                paired_deltas.append(av - bv)
                if av > bv:
                    wins += 1
                elif av == bv:
                    ties += 1
                else:
                    losses += 1
            item[f"win_count_vs_{ref}"] = f"{wins}/{total}" + (f" ties={ties}" if ties else "")
            item[f"paired_delta_mean_macro_f1_vs_{ref}"] = mean(paired_deltas)
            item[f"paired_delta_std_macro_f1_vs_{ref}"] = std(paired_deltas)
            item[f"sign_test_p_vs_{ref}"] = sign_test_p_value(wins, losses)

    out.sort(key=lambda r: (r["dataset"], r["partition"], int(r["clients"]), r["run_tag"], -r["mean_macro_f1_mean"]))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="enhanced_overview.csv from colab_full_experiment.py")
    parser.add_argument("--output", required=True)
    parser.add_argument("--reference-methods", default="fedath,fedssp,fedpub_gcn,spc_guard")
    args = parser.parse_args()

    rows = read_rows(Path(args.input))
    refs = [x.strip() for x in args.reference_methods.split(",") if x.strip()]
    table = stage_table(rows, refs)
    if not table:
        raise SystemExit("No rows found.")
    keys = []
    for row in table:
        for key in row:
            if key not in keys:
                keys.append(key)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(table)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
