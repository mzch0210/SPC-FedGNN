import argparse
import csv
import glob
from collections import defaultdict
from pathlib import Path

import numpy as np


def read_rows(inputs):
    rows = []
    for pattern in inputs:
        for path_str in sorted(glob.glob(pattern, recursive=True)):
            path = Path(path_str)
            if path.name == "oracle_bounds_summary.csv":
                continue
            with open(path, newline="", encoding="utf-8") as f:
                rows.extend(csv.DictReader(f))
    return rows


def main():
    parser = argparse.ArgumentParser(description="Summarize oracle-bound aggregated CSV files.")
    parser.add_argument("--inputs", nargs="+", required=True, help="Glob patterns for *_aggregated.csv files.")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    rows = read_rows(args.inputs)
    if not rows:
        raise SystemExit("No aggregated CSV rows found. Check --inputs patterns and downloaded result folders.")
    groups = defaultdict(list)
    for row in rows:
        key = (row["dataset"], row["partition"], row["clients"], row["method"])
        groups[key].append(row)

    out_rows = []
    for (dataset, partition, clients, method), items in sorted(groups.items()):
        f1 = np.array([float(r["mean_macro_f1"]) for r in items], dtype=float)
        acc = np.array([float(r["mean_accuracy"]) for r in items], dtype=float)
        worst = np.array([float(r["worst_macro_f1"]) for r in items], dtype=float)
        seeds = sorted(int(r["seed"]) for r in items)
        out_rows.append(
            {
                "dataset": dataset,
                "partition": partition,
                "clients": clients,
                "method": method,
                "seeds": " ".join(str(s) for s in seeds),
                "n_seeds": len(seeds),
                "mean_macro_f1": f"{f1.mean():.4f}",
                "std_macro_f1": f"{f1.std(ddof=1):.4f}" if len(f1) > 1 else "0.0000",
                "mean_accuracy": f"{acc.mean():.4f}",
                "std_accuracy": f"{acc.std(ddof=1):.4f}" if len(acc) > 1 else "0.0000",
                "mean_worst_macro_f1": f"{worst.mean():.4f}",
                "std_worst_macro_f1": f"{worst.std(ddof=1):.4f}" if len(worst) > 1 else "0.0000",
            }
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
