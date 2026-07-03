import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


def read_rows(paths):
    rows = []
    for path in paths:
        with open(path, newline="", encoding="utf-8") as f:
            rows.extend(csv.DictReader(f))
    return rows


def to_float(value):
    try:
        return float(value)
    except Exception:
        return math.nan


def bootstrap_ci(values, rng, reps=10000, alpha=0.05):
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return math.nan, math.nan
    if values.size == 1:
        return float(values[0]), float(values[0])
    samples = rng.choice(values, size=(reps, values.size), replace=True).mean(axis=1)
    return float(np.quantile(samples, alpha / 2)), float(np.quantile(samples, 1 - alpha / 2))


def cohens_d(values):
    values = np.asarray(values, dtype=float)
    if values.size <= 1:
        return math.nan
    sd = values.std(ddof=1)
    return float(values.mean() / sd) if sd > 1e-12 else math.inf


def sign_test_p(wins, losses):
    n = wins + losses
    if n == 0:
        return math.nan
    tail = min(wins, losses)
    prob = sum(math.comb(n, k) for k in range(tail + 1)) / (2**n)
    return min(1.0, 2.0 * prob)


def main():
    parser = argparse.ArgumentParser(description="Compute paired CI/effect-size summaries from existing seed-level CSVs.")
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--target-method", default="spc_guard_pp")
    parser.add_argument("--reference-methods", default="fedath,fedssp,fedpub_gcn")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    rows = read_rows([Path(p) for p in args.inputs])
    refs = [x.strip() for x in args.reference_methods.split(",") if x.strip()]
    lookup = {}
    for row in rows:
        key = (
            row.get("dataset", ""),
            row.get("partition", ""),
            row.get("clients", ""),
            row.get("seed", ""),
            row.get("method", ""),
        )
        lookup[key] = row

    grouped = defaultdict(list)
    for row in rows:
        if row.get("method") == args.target_method:
            grouped[(row.get("dataset", ""), row.get("partition", ""), row.get("clients", ""))].append(row)

    rng = np.random.default_rng(20260702)
    out = []
    for (dataset, partition, clients), target_rows in sorted(grouped.items()):
        seeds = sorted({r.get("seed", "") for r in target_rows}, key=lambda x: int(float(x)))
        for ref in refs:
            deltas = []
            wins = losses = ties = 0
            used = []
            for seed in seeds:
                a = lookup.get((dataset, partition, clients, seed, args.target_method))
                b = lookup.get((dataset, partition, clients, seed, ref))
                if not a or not b:
                    continue
                av = to_float(a.get("mean_macro_f1", ""))
                bv = to_float(b.get("mean_macro_f1", ""))
                if math.isnan(av) or math.isnan(bv):
                    continue
                delta = av - bv
                deltas.append(delta)
                used.append(seed)
                if delta > 0:
                    wins += 1
                elif delta < 0:
                    losses += 1
                else:
                    ties += 1
            if not deltas:
                continue
            lo, hi = bootstrap_ci(deltas, rng)
            vals = np.asarray(deltas, dtype=float)
            out.append(
                {
                    "dataset": dataset,
                    "partition": partition,
                    "clients": clients,
                    "target_method": args.target_method,
                    "reference_method": ref,
                    "seeds": " ".join(used),
                    "n": len(deltas),
                    "paired_delta_mean_macro_f1": f"{vals.mean():.4f}",
                    "paired_delta_std": f"{vals.std(ddof=1):.4f}" if len(vals) > 1 else "0.0000",
                    "bootstrap95_ci_low": f"{lo:.4f}",
                    "bootstrap95_ci_high": f"{hi:.4f}",
                    "paired_cohens_d": f"{cohens_d(vals):.4f}" if not math.isinf(cohens_d(vals)) else "inf",
                    "wins_losses_ties": f"{wins}/{losses}/{ties}",
                    "sign_test_p": f"{sign_test_p(wins, losses):.4f}" if not math.isnan(sign_test_p(wins, losses)) else "",
                }
            )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        writer.writeheader()
        writer.writerows(out)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
