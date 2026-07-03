import argparse
import csv
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "experiments") not in sys.path:
    sys.path.insert(0, str(ROOT / "experiments"))

rb = None
original_method_flags = None


STAGE_CONFIGS = {
    "smoke": {
        "datasets": ["cora"],
        "partitions": ["dirichlet_0.1"],
        "clients": [20],
        "seeds": [0],
        "methods": ["fedgta_matched", "adafgl_matched", "fediih_matched"],
        "rounds": 10,
    },
    "t4a_strong_core": {
        "datasets": ["cora", "citeseer", "pubmed"],
        "partitions": ["dirichlet_0.1"],
        "clients": [20],
        "seeds": [0, 1, 2],
        "methods": ["fedgta_matched", "adafgl_matched", "fediih_matched"],
        "rounds": 60,
    },
    "t4b_strong_extension": {
        "datasets": ["amazon_computers", "roman_empire"],
        "partitions": ["dirichlet_0.1"],
        "clients": [20],
        "seeds": [0, 1, 2],
        "methods": ["fedgta_matched", "adafgl_matched", "fediih_matched"],
        "rounds": 60,
    },
    "t4_photo_strong": {
        "datasets": ["amazon_photo"],
        "partitions": ["dirichlet_0.1"],
        "clients": [20],
        "seeds": [0, 1, 2],
        "methods": ["fedgta_matched", "adafgl_matched", "fediih_matched"],
        "rounds": 60,
    },
    "t4b_strong_tolokers": {
        "datasets": ["tolokers"],
        "partitions": ["dirichlet_0.1"],
        "clients": [20],
        "seeds": [0, 1, 2],
        "methods": ["fedgta_matched", "adafgl_matched", "fediih_matched"],
        "rounds": 60,
    },
}


BASE_METHOD = {
    "fedgta_matched": "spc_protoagg",
    "adafgl_matched": "fedavg_gcn",
    "fediih_matched": "fedpub_gcn",
}


def matched_method_flags(method):
    base_flags = original_method_flags or load_benchmark().method_flags
    if method == "fedgta_matched":
        flags = base_flags("spc_protoagg")
        flags.update(
            {
                "edge_evaluator": False,
                "hsic": False,
                "counterfactual": False,
                "alignment": False,
            }
        )
        return flags
    if method == "adafgl_matched":
        flags = base_flags("fedavg_gcn")
        flags.update(
            {
                "personalized": True,
                "class_balanced_loss": True,
                "worst_reweight": False,
            }
        )
        return flags
    if method == "fediih_matched":
        flags = base_flags("fedpub_gcn")
        flags.update(
            {
                "personalized": True,
                "class_balanced_loss": True,
            }
        )
        return flags
    return base_flags(method)


def load_benchmark():
    global rb, original_method_flags
    if rb is None:
        import run_benchmark as benchmark  # noqa: WPS433

        rb = benchmark
        original_method_flags = benchmark.method_flags
    return rb


@dataclass(frozen=True)
class Task:
    dataset: str
    partition: str
    clients: int
    seed: int
    method: str
    rounds: int


def parse_csv(value, cast=str):
    if value is None:
        return None
    if value.strip() == "":
        return []
    return [cast(x.strip()) for x in value.split(",") if x.strip()]


def output_prefix(args, task):
    return Path(args.output_dir) / args.stage / (
        f"{task.dataset}_{task.partition}_k{task.clients}_{task.method}_s{task.seed}"
    )


def result_path(args, task):
    return Path(f"{output_prefix(args, task)}.csv")


def agg_path(args, task):
    return Path(f"{output_prefix(args, task)}_aggregated.csv")


def is_done(args, task):
    if args.force:
        return False
    return result_path(args, task).exists() and agg_path(args, task).exists()


def build_tasks(args):
    cfg = STAGE_CONFIGS[args.stage]
    datasets = parse_csv(args.datasets) or cfg["datasets"]
    partitions = parse_csv(args.partitions) or cfg["partitions"]
    clients = parse_csv(args.clients, int) or cfg["clients"]
    seeds = parse_csv(args.seeds, int) or cfg["seeds"]
    methods = parse_csv(args.methods) or cfg["methods"]
    rounds = args.rounds or cfg["rounds"]
    return [
        Task(dataset, partition, client_count, seed, method, rounds)
        for dataset in datasets
        for partition in partitions
        for client_count in clients
        for seed in seeds
        for method in methods
    ]


def write_plan(args, tasks):
    path = Path(args.output_dir) / args.stage / "task_plan.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["dataset", "partition", "clients", "seed", "method", "rounds", "done", "output"],
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
                    "done": int(is_done(args, task)),
                    "output": str(result_path(args, task)),
                }
            )
    return path


def make_rb_args(args, task):
    ns = argparse.Namespace()
    ns.dataset = task.dataset
    ns.partition = task.partition
    ns.clients = task.clients
    ns.seed = task.seed
    ns.method = task.method
    ns.output = str(result_path(args, task))
    ns.run_tag = "strong_matched"
    ns.data_root = args.data_root
    ns.rounds = task.rounds
    ns.local_epochs = args.local_epochs
    ns.hidden_dim = args.hidden_dim
    ns.dropout = 0.5
    ns.lr = 0.01
    ns.weight_decay = 5e-4
    ns.prototype_count = 3
    ns.descriptor_mode = "structure"
    ns.dp_noise = 0.0
    ns.alignment_weight = 0.02
    ns.counterfactual_weight = 0.01
    ns.personalization_weight = 0.25
    ns.prox_mu = 0.001
    ns.mask_keep_ratio = 0.55
    ns.hsic_weight = 0.02
    ns.biased_entropy_weight = 0.02
    ns.edge_sparsity_weight = 0.02
    ns.edge_supervision_weight = 0.05
    ns.prototype_prior_weight = 0.05
    ns.residual_weight = 0.35
    ns.prototype_agg_temperature = 8.0
    ns.worst_client_weight = 0.6
    ns.guard_candidate_count = 3
    ns.guard_threshold = 0.35
    ns.guard_beta = 4.0
    ns.guard_uncertainty_weight = 0.05
    ns.guard_coverage_weight = 0.35
    ns.guard_distill_weight = 0.15
    ns.guard_loss_protect = 0.35
    ns.guard_loss_threshold_weight = 0.05
    ns.class_balance_strength = 0.5
    ns.fedpub_proxy_nodes = 512
    ns.fedpub_temperature = 8.0
    ns.cpu = args.cpu
    return ns


def run_task(args, task):
    benchmark = load_benchmark()
    prefix = output_prefix(args, task)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    log_path = Path(f"{prefix}.log")
    global original_method_flags
    old_method_flags = benchmark.method_flags
    original_method_flags = old_method_flags
    benchmark.method_flags = matched_method_flags
    start = time.time()
    try:
        with open(log_path, "w", encoding="utf-8") as log:
            log.write(
                f"matched_method={task.method}\n"
                f"base_behavior={BASE_METHOD.get(task.method, task.method)}\n"
                "This is a carefully matched unified-protocol baseline, not an official-code result.\n"
            )
        rb_args = make_rb_args(args, task)
        rows = benchmark.train(rb_args)
        for row in rows:
            row["baseline_note"] = "carefully_matched_not_official"
            row["base_behavior"] = BASE_METHOD.get(task.method, task.method)
        benchmark.write_rows(result_path(args, task), rows)
        with open(log_path, "a", encoding="utf-8") as log:
            log.write(f"completed_seconds={time.time() - start:.2f}\n")
        print(f"[done] {task.dataset}/{task.method}/seed={task.seed} in {(time.time() - start)/60:.1f} min")
        return True
    except Exception as exc:
        with open(log_path, "a", encoding="utf-8") as log:
            log.write(f"ERROR: {exc}\n")
        print(f"[failed] {task.dataset}/{task.method}/seed={task.seed}; see {log_path}")
        if not args.keep_going:
            raise
        return False
    finally:
        benchmark.method_flags = old_method_flags


def summarize_stage(args):
    stage_dir = Path(args.output_dir) / args.stage
    agg_files = sorted(stage_dir.glob("*_aggregated.csv"))
    if not agg_files:
        return
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "experiments" / "summarize_results.py"),
            "--inputs",
            *[str(p) for p in agg_files],
            "--output",
            str(stage_dir / "aggregated_overview.csv"),
        ],
        check=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=sorted(STAGE_CONFIGS))
    parser.add_argument("--output-dir", default="results/strong_baselines")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--datasets")
    parser.add_argument("--partitions")
    parser.add_argument("--clients")
    parser.add_argument("--seeds")
    parser.add_argument("--methods")
    parser.add_argument("--rounds", type=int)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--max-tasks", type=int)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--keep-going", action="store_true")
    parser.add_argument("--no-summarize", action="store_true")
    args = parser.parse_args()

    tasks = build_tasks(args)
    plan = write_plan(args, tasks)
    pending = [task for task in tasks if not is_done(args, task)]
    if args.max_tasks is not None:
        pending = pending[: args.max_tasks]
    print(json.dumps({"stage": args.stage, "tasks": len(tasks), "pending": len(pending), "plan": str(plan)}, indent=2))
    if args.dry_run:
        for task in pending[:20]:
            print(task)
        if len(pending) > 20:
            print(f"... {len(pending) - 20} more pending tasks")
        return
    ok = 0
    for task in pending:
        ok += int(run_task(args, task))
        write_plan(args, tasks)
    if not args.no_summarize:
        summarize_stage(args)
        write_plan(args, tasks)
    print(json.dumps({"completed_now": ok}, indent=2))


if __name__ == "__main__":
    main()
