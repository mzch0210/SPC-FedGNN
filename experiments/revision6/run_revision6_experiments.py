import argparse
import csv
import subprocess
import sys
import time
from pathlib import Path


DEFAULT_DATASETS = ["cora", "pubmed", "amazon_photo"]
DEFAULT_SEEDS = [0, 1, 2]


STAGES = {
    "smoke": {
        "datasets": ["cora"],
        "seeds": [0],
        "methods": ["fedavg_gcn_cb", "spc_guard", "spc_guard_pp"],
        "rounds": 8,
    },
    "fair_cb": {
        "datasets": DEFAULT_DATASETS,
        "seeds": DEFAULT_SEEDS,
        "methods": [
            "fedavg_gcn",
            "fedavg_gcn_cb",
            "fedprox_gcn",
            "fedprox_gcn_cb",
            "fedpub_gcn",
            "fedpub_gcn_cb",
            "fedath",
            "fedath_cb",
            "spc_guard_pp",
        ],
        "rounds": 60,
    },
    "contribution": {
        "datasets": DEFAULT_DATASETS,
        "seeds": DEFAULT_SEEDS,
        "methods": [
            "spc_guard",
            "spc_guard_pp",
            "spc_guard_pp_no_coverage",
            "spc_guard_pp_no_class_balance",
            "spc_guard_random_candidates",
            "spc_no_personalization",
            "spc_protoagg",
        ],
        "rounds": 60,
    },
    "gate_independent": {
        "datasets": DEFAULT_DATASETS,
        "seeds": DEFAULT_SEEDS,
        "methods": ["spc_guard", "spc_guard_pp"],
        "rounds": 60,
    },
}


def output_name(row):
    tag = row["tag"]
    return (
        f"{row['dataset']}_{row['partition']}_k{row['clients']}_"
        f"{row['method']}_s{row['seed']}_{tag}.csv"
    )


def build_rows(stage_name, output_dir, data_root, clients, partition, local_epochs, hidden_dim):
    spec = STAGES[stage_name]
    rows = []
    for dataset in spec["datasets"]:
        for seed in spec["seeds"]:
            for method in spec["methods"]:
                tag = stage_name
                row = {
                    "dataset": dataset,
                    "partition": partition,
                    "clients": str(clients),
                    "seed": str(seed),
                    "method": method,
                    "rounds": str(spec["rounds"]),
                    "local_epochs": str(local_epochs),
                    "hidden_dim": str(hidden_dim),
                    "tag": tag,
                    "done": "0",
                    "output": str(output_dir / output_name(
                        {
                            "dataset": dataset,
                            "partition": partition,
                            "clients": clients,
                            "seed": seed,
                            "method": method,
                            "tag": tag,
                        }
                    )),
                    "data_root": str(data_root),
                }
                rows.append(row)
    return rows


def write_plan(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "partition",
        "clients",
        "seed",
        "method",
        "rounds",
        "local_epochs",
        "hidden_dim",
        "tag",
        "done",
        "output",
        "data_root",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_plan(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def command_for(row, cpu=False):
    cmd = [
        sys.executable,
        "experiments/run_benchmark.py",
        "--dataset",
        row["dataset"],
        "--partition",
        row["partition"],
        "--clients",
        row["clients"],
        "--seed",
        row["seed"],
        "--method",
        row["method"],
        "--rounds",
        row["rounds"],
        "--local-epochs",
        row["local_epochs"],
        "--hidden-dim",
        row["hidden_dim"],
        "--data-root",
        row["data_root"],
        "--output",
        row["output"],
        "--run-tag",
        row["tag"],
    ]
    if row["tag"] == "gate_independent":
        cmd.extend(["--guard-evidence-mode", "independent", "--guard-independent-ratio", "0.2"])
    if cpu:
        cmd.append("--cpu")
    return cmd


def aggregated_path(output):
    path = Path(output)
    return path.with_name(path.stem + "_aggregated.csv")


def run_plan(plan_path, cpu=False):
    rows = read_plan(plan_path)
    for idx, row in enumerate(rows, start=1):
        if row.get("done") == "1" and aggregated_path(row["output"]).exists():
            continue
        output = Path(row["output"])
        output.parent.mkdir(parents=True, exist_ok=True)
        log_path = output.with_suffix(".log")
        cmd = command_for(row, cpu=cpu)
        start = time.time()
        with open(log_path, "w", encoding="utf-8") as log:
            log.write(" ".join(cmd) + "\n")
            log.flush()
            proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
            log.write(f"completed_seconds={time.time() - start:.2f}\n")
            log.write(f"returncode={proc.returncode}\n")
        if proc.returncode == 0 and aggregated_path(row["output"]).exists():
            row["done"] = "1"
        else:
            row["done"] = "0"
        rows[idx - 1] = row
        write_plan(plan_path, rows)
        print(f"[{idx}/{len(rows)}] {row['method']} {row['dataset']} seed={row['seed']} done={row['done']}")


def summarize(inputs, output):
    grouped = {}
    for path in inputs:
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                key = (row["dataset"], row["method"], row.get("run_tag", ""))
                grouped.setdefault(key, []).append(float(row["mean_macro_f1"]))
    rows = []
    for (dataset, method, tag), values in sorted(grouped.items()):
        mean = sum(values) / len(values)
        if len(values) > 1:
            var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
        else:
            var = 0.0
        rows.append(
            {
                "dataset": dataset,
                "method": method,
                "tag": tag,
                "seeds": len(values),
                "mean_macro_f1": f"{mean:.4f}",
                "std_macro_f1": f"{var ** 0.5:.4f}",
            }
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["dataset", "method"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {output}")


def main():
    parser = argparse.ArgumentParser(description="Run revision-6 experiments for SPC-FedGNN.")
    parser.add_argument("--stage", choices=sorted(STAGES), required=True)
    parser.add_argument("--root", default="results/revision6")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--clients", type=int, default=20)
    parser.add_argument("--partition", default="dirichlet_0.1")
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    root = Path(args.root) / args.stage
    plan_path = root / "task_plan.csv"
    if not plan_path.exists():
        rows = build_rows(
            args.stage,
            root,
            Path(args.data_root),
            args.clients,
            args.partition,
            args.local_epochs,
            args.hidden_dim,
        )
        write_plan(plan_path, rows)
        print(f"Wrote {plan_path} with {len(rows)} tasks")
    if args.prepare_only:
        return
    if not args.summarize_only:
        run_plan(plan_path, cpu=args.cpu)
    inputs = sorted(root.glob("*_aggregated.csv"))
    if inputs:
        summarize(inputs, root / "summary.csv")


if __name__ == "__main__":
    main()
