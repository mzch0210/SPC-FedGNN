import argparse
import csv
import json
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


STAGE_CONFIGS = {
    "smoke": {
        "datasets": ["cora"],
        "partitions": ["dirichlet_0.1"],
        "clients": [20],
        "seeds": [0],
        "methods": ["spc_guard_pp", "spc_guard", "fedath", "fedssp"],
        "rounds": 10,
    },
    "p0_main": {
        "datasets": ["cora", "citeseer", "pubmed", "amazon_photo", "roman_empire"],
        "partitions": ["dirichlet_0.1"],
        "clients": [20],
        "seeds": [0, 1, 2],
        "methods": ["spc_guard_pp", "fedath", "fedssp", "fedavg_gcn", "fedprox_gcn"],
        "rounds": 60,
    },
    "p0_ablation": {
        "datasets": ["cora", "citeseer", "pubmed"],
        "partitions": ["dirichlet_0.1"],
        "clients": [20],
        "seeds": [0, 1, 2],
        "methods": [
            "spc_guard_pp",
            "spc_guard",
            "spc_no_protoagg",
            "spc_protoagg",
            "spc_guard_no_gate",
            "spc_guard_no_distill",
            "spc_guard_pp_no_coverage",
            "spc_guard_pp_no_class_balance",
            "spc_guard_random_candidates",
        ],
        "rounds": 60,
    },
    "p0_heterogeneity": {
        "datasets": ["cora", "citeseer", "pubmed"],
        "partitions": ["dirichlet_0.5", "dirichlet_0.1"],
        "clients": [10, 20],
        "seeds": [0, 1, 2],
        "methods": ["spc_guard_pp", "spc_guard", "fedath", "fedssp", "spc_no_protoagg"],
        "rounds": 60,
    },
    "p1_sensitivity": {
        "datasets": ["cora", "citeseer"],
        "partitions": ["dirichlet_0.1"],
        "clients": [20],
        "seeds": [0, 1, 2],
        "methods": ["spc_guard_pp"],
        "rounds": 60,
        "sweeps": [
            {"name": "proto2", "extra": ["--prototype-count", "2"]},
            {"name": "proto5", "extra": ["--prototype-count", "5"]},
            {"name": "cand1", "extra": ["--guard-candidate-count", "1"]},
            {"name": "cand5", "extra": ["--guard-candidate-count", "5"]},
            {"name": "gate025", "extra": ["--guard-threshold", "0.25"]},
            {"name": "gate045", "extra": ["--guard-threshold", "0.45"]},
            {"name": "distill007", "extra": ["--guard-distill-weight", "0.07"]},
            {"name": "distill030", "extra": ["--guard-distill-weight", "0.30"]},
            {"name": "cov015", "extra": ["--guard-coverage-weight", "0.15"]},
            {"name": "cov060", "extra": ["--guard-coverage-weight", "0.60"]},
        ],
    },
    "free_t4_main": {
        "datasets": ["cora", "citeseer"],
        "partitions": ["dirichlet_0.1"],
        "clients": [20],
        "seeds": [0, 1, 2],
        "methods": ["spc_guard_pp", "fedath", "fedssp", "fedavg_gcn", "fedprox_gcn"],
        "rounds": 60,
    },
    "free_t4_ablation": {
        "datasets": ["cora", "citeseer"],
        "partitions": ["dirichlet_0.1"],
        "clients": [20],
        "seeds": [0, 1, 2],
        "methods": [
            "spc_guard_pp",
            "spc_guard",
            "spc_no_protoagg",
            "spc_protoagg",
            "spc_guard_no_gate",
            "spc_guard_no_distill",
            "spc_guard_pp_no_coverage",
            "spc_guard_pp_no_class_balance",
        ],
        "rounds": 60,
    },
    "free_t4_sensitivity": {
        "datasets": ["cora"],
        "partitions": ["dirichlet_0.1"],
        "clients": [20],
        "seeds": [0, 1, 2],
        "methods": ["spc_guard_pp"],
        "rounds": 60,
        "sweeps": [
            {"name": "proto2", "extra": ["--prototype-count", "2"]},
            {"name": "proto5", "extra": ["--prototype-count", "5"]},
            {"name": "cand1", "extra": ["--guard-candidate-count", "1"]},
            {"name": "cand5", "extra": ["--guard-candidate-count", "5"]},
            {"name": "gate025", "extra": ["--guard-threshold", "0.25"]},
            {"name": "gate045", "extra": ["--guard-threshold", "0.45"]},
        ],
    },
    "pro_main_heavy": {
        "datasets": ["pubmed", "amazon_photo", "roman_empire"],
        "partitions": ["dirichlet_0.1"],
        "clients": [20],
        "seeds": [0, 1, 2],
        "methods": ["spc_guard_pp", "fedath", "fedssp", "fedavg_gcn", "fedprox_gcn"],
        "rounds": 60,
    },
    "pro_ablation_heavy": {
        "datasets": ["pubmed"],
        "partitions": ["dirichlet_0.1"],
        "clients": [20],
        "seeds": [0, 1, 2],
        "methods": [
            "spc_guard_pp",
            "spc_guard",
            "spc_no_protoagg",
            "spc_protoagg",
            "spc_guard_no_gate",
            "spc_guard_no_distill",
            "spc_guard_pp_no_coverage",
            "spc_guard_pp_no_class_balance",
            "spc_guard_random_candidates",
        ],
        "rounds": 60,
    },
    "pro_heterogeneity": {
        "datasets": ["cora", "citeseer", "pubmed"],
        "partitions": ["dirichlet_0.5", "dirichlet_0.1"],
        "clients": [10, 20],
        "seeds": [0, 1, 2],
        "methods": ["spc_guard_pp", "spc_guard", "fedath", "fedssp", "spc_no_protoagg"],
        "rounds": 60,
    },
    "t4_extension_datasets": {
        "datasets": ["amazon_computers", "tolokers"],
        "partitions": ["dirichlet_0.1"],
        "clients": [20],
        "seeds": [0, 1, 2],
        "methods": ["spc_guard_pp", "fedath", "fedssp", "fedavg_gcn", "fedprox_gcn"],
        "rounds": 60,
    },
    "t4_5seed_core": {
        "datasets": ["cora", "citeseer", "amazon_photo"],
        "partitions": ["dirichlet_0.1"],
        "clients": [20],
        "seeds": [3, 4],
        "methods": ["spc_guard_pp", "fedath", "fedssp"],
        "rounds": 60,
    },
    "t4_diagnostics": {
        "datasets": ["cora", "citeseer", "pubmed"],
        "partitions": ["dirichlet_0.1"],
        "clients": [20],
        "seeds": [0, 1, 2],
        "methods": ["spc_guard_pp", "spc_guard", "spc_no_protoagg", "spc_protoagg", "fedath", "fedssp"],
        "rounds": 60,
    },
    "t4_efficiency": {
        "datasets": ["cora", "citeseer", "pubmed"],
        "partitions": ["dirichlet_0.1"],
        "clients": [20],
        "seeds": [0],
        "methods": ["fedavg_gcn", "fedprox_gcn"],
        "rounds": 60,
    },
    "t4a_extension_amazon": {
        "datasets": ["amazon_computers"],
        "partitions": ["dirichlet_0.1"],
        "clients": [20],
        "seeds": [0, 1, 2],
        "methods": ["spc_guard_pp", "fedath", "fedssp", "fedavg_gcn", "fedprox_gcn"],
        "rounds": 60,
    },
    "t4b_extension_tolokers": {
        "datasets": ["tolokers"],
        "partitions": ["dirichlet_0.1"],
        "clients": [20],
        "seeds": [0, 1, 2],
        "methods": ["spc_guard_pp", "fedath", "fedssp", "fedavg_gcn", "fedprox_gcn"],
        "rounds": 60,
    },
    "t4a_5seed_core": {
        "datasets": ["cora", "citeseer"],
        "partitions": ["dirichlet_0.1"],
        "clients": [20],
        "seeds": [3, 4],
        "methods": ["spc_guard_pp", "fedath", "fedssp"],
        "rounds": 60,
    },
    "t4b_5seed_amazon_photo": {
        "datasets": ["amazon_photo"],
        "partitions": ["dirichlet_0.1"],
        "clients": [20],
        "seeds": [3, 4],
        "methods": ["spc_guard_pp", "fedath", "fedssp"],
        "rounds": 60,
    },
    "t4a_diagnostics_light": {
        "datasets": ["cora", "citeseer"],
        "partitions": ["dirichlet_0.1"],
        "clients": [20],
        "seeds": [0, 1, 2],
        "methods": ["spc_guard_pp", "spc_guard", "spc_protoagg", "fedath", "fedssp"],
        "rounds": 60,
    },
    "t4b_diagnostics_pubmed": {
        "datasets": ["pubmed"],
        "partitions": ["dirichlet_0.1"],
        "clients": [20],
        "seeds": [0, 1, 2],
        "methods": ["spc_guard_pp", "spc_guard", "spc_protoagg", "fedath", "fedssp"],
        "rounds": 60,
    },
    "t4a_efficiency_light": {
        "datasets": ["cora", "citeseer"],
        "partitions": ["dirichlet_0.1"],
        "clients": [20],
        "seeds": [0],
        "methods": ["fedavg_gcn", "fedprox_gcn"],
        "rounds": 60,
    },
    "t4b_efficiency_pubmed": {
        "datasets": ["pubmed"],
        "partitions": ["dirichlet_0.1"],
        "clients": [20],
        "seeds": [0],
        "methods": ["fedavg_gcn", "fedprox_gcn"],
        "rounds": 60,
    },
    "t4a_fedpub_citation": {
        "datasets": ["cora", "citeseer", "pubmed"],
        "partitions": ["dirichlet_0.1"],
        "clients": [20],
        "seeds": [0, 1, 2],
        "methods": ["fedpub_gcn"],
        "rounds": 60,
    },
    "t4b_fedpub_extended": {
        "datasets": ["amazon_photo", "amazon_computers", "roman_empire", "tolokers"],
        "partitions": ["dirichlet_0.1"],
        "clients": [20],
        "seeds": [0, 1, 2],
        "methods": ["fedpub_gcn"],
        "rounds": 60,
    },
    "rev_a_5seed_core": {
        "datasets": ["cora", "citeseer", "pubmed"],
        "partitions": ["dirichlet_0.1"],
        "clients": [20],
        "seeds": [3, 4],
        "methods": ["spc_guard_pp", "fedath", "fedssp", "fedpub_gcn"],
        "rounds": 60,
    },
    "rev_b_5seed_extension": {
        "datasets": ["amazon_computers", "roman_empire"],
        "partitions": ["dirichlet_0.1"],
        "clients": [20],
        "seeds": [3, 4],
        "methods": ["spc_guard_pp", "fedath", "fedssp", "fedpub_gcn"],
        "rounds": 60,
    },
    "rev_a_sanity_core": {
        "datasets": ["cora", "citeseer", "pubmed"],
        "partitions": ["dirichlet_0.1"],
        "clients": [20],
        "seeds": [0, 1, 2],
        "methods": [
            "spc_guard_pp",
            "spc_guard",
            "spc_no_personalization",
            "spc_guard_random_candidates",
            "spc_protoagg",
            "fedavg_gcn",
        ],
        "rounds": 60,
    },
    "rev_b_sanity_extension": {
        "datasets": ["amazon_computers", "roman_empire"],
        "partitions": ["dirichlet_0.1"],
        "clients": [20],
        "seeds": [0, 1, 2],
        "methods": [
            "spc_guard_pp",
            "spc_guard",
            "spc_no_personalization",
            "spc_guard_random_candidates",
            "spc_protoagg",
            "fedavg_gcn",
        ],
        "rounds": 60,
    },
    "rev_a_comm_core": {
        "datasets": ["cora", "citeseer", "pubmed"],
        "partitions": ["dirichlet_0.1"],
        "clients": [20],
        "seeds": [0],
        "methods": ["spc_guard_pp", "fedpub_gcn", "fedavg_gcn", "fedprox_gcn", "fedssp"],
        "rounds": 60,
    },
    "rev_b_tolokers_audit": {
        "datasets": ["tolokers"],
        "partitions": ["dirichlet_0.1"],
        "clients": [20],
        "seeds": [0, 1, 2, 3, 4],
        "methods": ["spc_guard_pp", "fedath", "fedssp", "fedpub_gcn", "fedavg_gcn", "fedprox_gcn"],
        "rounds": 60,
    },
    "rev_sparse_validation": {
        "datasets": ["cora", "pubmed", "amazon_photo"],
        "partitions": ["dirichlet_0.1"],
        "clients": [20],
        "seeds": [0, 1, 2],
        "methods": ["spc_guard_pp", "spc_guard"],
        "rounds": 60,
        "sweeps": [
            {"name": "val100", "extra": ["--guard-validation-ratio", "1.0", "--guard-validation-drop-classes", "0"]},
            {"name": "val20", "extra": ["--guard-validation-ratio", "0.2", "--guard-validation-drop-classes", "0"]},
            {"name": "val10", "extra": ["--guard-validation-ratio", "0.1", "--guard-validation-drop-classes", "0"]},
            {"name": "val05", "extra": ["--guard-validation-ratio", "0.05", "--guard-validation-drop-classes", "0"]},
            {"name": "drop1cls", "extra": ["--guard-validation-ratio", "0.2", "--guard-validation-drop-classes", "1"]},
        ],
    },
    "rev_comm_runtime": {
        "datasets": ["cora", "citeseer", "pubmed", "amazon_photo", "amazon_computers", "roman_empire", "tolokers"],
        "partitions": ["dirichlet_0.1"],
        "clients": [20],
        "seeds": [0],
        "methods": ["spc_guard_pp", "fedath", "fedssp", "fedpub_gcn", "fedavg_gcn", "fedprox_gcn"],
        "rounds": 60,
    },
}

RAW_RESULT_RE = re.compile(r".+_s\d+(?:_[A-Za-z0-9]+)?\.csv$")


@dataclass(frozen=True)
class Task:
    dataset: str
    partition: str
    clients: int
    seed: int
    method: str
    rounds: int
    tag: str
    extra: tuple


def parse_csv_arg(value, cast=str):
    if value is None:
        return None
    if value.strip() == "":
        return []
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def output_prefix(args, task):
    name = f"{task.dataset}_{task.partition}_k{task.clients}_{task.method}_s{task.seed}"
    if task.tag != "base":
        name += f"_{task.tag}"
    return Path(args.output_dir) / args.stage / name


def result_csv_path(args, task):
    return Path(f"{output_prefix(args, task)}.csv")


def aggregated_csv_path(args, task):
    return Path(f"{output_prefix(args, task)}_aggregated.csv")


def is_done(args, task):
    if getattr(args, "force", False):
        return False
    raw = result_csv_path(args, task)
    agg = aggregated_csv_path(args, task)
    return raw.exists() and agg.exists() and raw.stat().st_size > 0 and agg.stat().st_size > 0


def build_tasks(args):
    if args.stage not in STAGE_CONFIGS:
        raise SystemExit(f"Unknown stage {args.stage}. Choose from: {', '.join(STAGE_CONFIGS)}")
    cfg = dict(STAGE_CONFIGS[args.stage])
    datasets = parse_csv_arg(args.datasets) or cfg["datasets"]
    partitions = parse_csv_arg(args.partitions) or cfg["partitions"]
    clients = parse_csv_arg(args.clients, int) or cfg["clients"]
    seeds = parse_csv_arg(args.seeds, int) or cfg["seeds"]
    methods = parse_csv_arg(args.methods) or cfg["methods"]
    rounds = args.rounds or cfg["rounds"]

    sweeps = cfg.get("sweeps")
    if args.sweep_base:
        sweeps = [{"name": "base", "extra": []}, *(sweeps or [])]
    if not sweeps:
        sweeps = [{"name": "base", "extra": []}]

    tasks = []
    for dataset in datasets:
        for partition in partitions:
            for client_count in clients:
                for seed in seeds:
                    for method in methods:
                        for sweep in sweeps:
                            tasks.append(
                                Task(
                                    dataset=dataset,
                                    partition=partition,
                                    clients=client_count,
                                    seed=seed,
                                    method=method,
                                    rounds=rounds,
                                    tag=sweep["name"],
                                    extra=tuple(sweep.get("extra", [])),
                                )
                            )
    return tasks


def write_plan(args, tasks):
    out = Path(args.output_dir) / args.stage / "task_plan.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dataset",
                "partition",
                "clients",
                "seed",
                "method",
                "rounds",
                "tag",
                "done",
                "output",
                "extra",
            ],
        )
        writer.writeheader()
        for task in tasks:
            writer.writerow(
                {
                    "dataset": task.dataset,
                    "partition": task.partition,
                    "clients": task.clients,
                    "seed": task.seed,
                    "method": task.method,
                    "rounds": task.rounds,
                    "tag": task.tag,
                    "done": int(is_done(args, task)),
                    "output": str(result_csv_path(args, task)),
                    "extra": " ".join(task.extra),
                }
            )
    return out


def run_task(args, task):
    prefix = output_prefix(args, task)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "experiments/run_benchmark.py",
        "--dataset",
        task.dataset,
        "--partition",
        task.partition,
        "--clients",
        str(task.clients),
        "--seed",
        str(task.seed),
        "--method",
        task.method,
        "--rounds",
        str(task.rounds),
        "--local-epochs",
        str(args.local_epochs),
        "--hidden-dim",
        str(args.hidden_dim),
        "--data-root",
        args.data_root,
        "--output",
        str(result_csv_path(args, task)),
        "--run-tag",
        task.tag,
        *task.extra,
    ]
    if args.cpu:
        cmd.append("--cpu")
    log_path = Path(f"{prefix}.log")
    print("\n[run]", " ".join(shlex.quote(x) for x in cmd), flush=True)
    start = time.time()
    with open(log_path, "w", encoding="utf-8") as log:
        log.write(" ".join(shlex.quote(x) for x in cmd) + "\n")
        log.flush()
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
    elapsed = time.time() - start
    if proc.returncode != 0:
        print(f"[failed] {task.dataset}/{task.method}/seed={task.seed} tag={task.tag}; see {log_path}")
        if not args.keep_going:
            raise SystemExit(proc.returncode)
        return False, elapsed
    print(f"[done] {task.dataset}/{task.method}/seed={task.seed} tag={task.tag} in {elapsed/60:.1f} min")
    return True, elapsed


def summarize(args):
    stage_dir = Path(args.output_dir) / args.stage
    raw_files = sorted(
        p
        for p in stage_dir.glob("*.csv")
        if "_aggregated" not in p.name and RAW_RESULT_RE.match(p.name)
    )
    agg_files = sorted(stage_dir.glob("*_aggregated.csv"))
    if raw_files:
        subprocess.run(
            [
                sys.executable,
                "experiments/summarize_results.py",
                "--client-summary",
                "--inputs",
                *[str(p) for p in raw_files],
                "--output",
                str(stage_dir / "enhanced_overview.csv"),
            ],
            check=True,
        )
        subprocess.run(
            [
                sys.executable,
                "experiments/aggregate_stage_tables.py",
                "--input",
                str(stage_dir / "enhanced_overview.csv"),
                "--output",
                str(stage_dir / "stage_table.csv"),
                "--reference-methods",
                "fedath,fedssp,fedpub_gcn,spc_guard",
            ],
            check=True,
        )
    if agg_files:
        subprocess.run(
            [
                sys.executable,
                "experiments/summarize_results.py",
                "--inputs",
                *[str(p) for p in agg_files],
                "--output",
                str(stage_dir / "aggregated_overview.csv"),
            ],
            check=True,
        )


def main():
    parser = argparse.ArgumentParser(description="Colab Pro experiment runner for SPC-FedGNN.")
    parser.add_argument("--stage", required=True, choices=sorted(STAGE_CONFIGS))
    parser.add_argument("--output-dir", default="results/colab_full")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--datasets", help="Comma-separated override, e.g. cora,citeseer")
    parser.add_argument("--partitions", help="Comma-separated override, e.g. dirichlet_0.1")
    parser.add_argument("--clients", help="Comma-separated override, e.g. 10,20")
    parser.add_argument("--seeds", help="Comma-separated override, e.g. 0,1,2")
    parser.add_argument("--methods", help="Comma-separated override")
    parser.add_argument("--rounds", type=int)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--max-tasks", type=int, help="Run at most this many pending tasks.")
    parser.add_argument("--num-shards", type=int, default=1, help="Split this stage into N deterministic shards.")
    parser.add_argument("--shard-index", type=int, default=0, help="Run shard index in [0, num_shards).")
    parser.add_argument("--force", action="store_true", help="Rerun tasks even when result CSVs already exist.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--keep-going", action="store_true")
    parser.add_argument("--no-summarize", action="store_true")
    parser.add_argument("--sweep-base", action="store_true", help="Include base setting in p1_sensitivity.")
    args = parser.parse_args()

    tasks = build_tasks(args)
    if args.num_shards < 1:
        raise SystemExit("--num-shards must be >= 1")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise SystemExit("--shard-index must be in [0, num_shards)")
    if args.num_shards > 1:
        tasks = [task for idx, task in enumerate(tasks) if idx % args.num_shards == args.shard_index]
    plan_path = write_plan(args, tasks)
    pending = [task for task in tasks if not is_done(args, task)]
    if args.max_tasks is not None:
        pending = pending[: args.max_tasks]

    print(
        json.dumps(
            {
                "stage": args.stage,
                "tasks": len(tasks),
                "pending": len(pending),
                "plan": str(plan_path),
                "num_shards": args.num_shards,
                "shard_index": args.shard_index,
            },
            indent=2,
        )
    )
    if args.dry_run:
        for task in pending[:20]:
            print(task)
        if len(pending) > 20:
            print(f"... {len(pending) - 20} more pending tasks")
        return

    ok = 0
    total_time = 0.0
    for task in pending:
        success, elapsed = run_task(args, task)
        ok += int(success)
        total_time += elapsed
        write_plan(args, tasks)
    print(json.dumps({"completed_now": ok, "elapsed_minutes": round(total_time / 60.0, 2)}, indent=2))
    if not args.no_summarize:
        summarize(args)
        write_plan(args, tasks)


if __name__ == "__main__":
    main()
