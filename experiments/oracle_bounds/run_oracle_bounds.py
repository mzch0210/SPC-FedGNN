import argparse
import csv
import json
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_benchmark import GCN, load_dataset, macro_f1_torch, make_clients  # noqa: E402


STAGE_CONFIGS = {
    "smoke": {
        "datasets": ["cora"],
        "partitions": ["dirichlet_0.1"],
        "clients": [20],
        "seeds": [0],
        "methods": ["local_only", "centralized", "label_aware_oracle"],
        "rounds": 8,
    },
    "t4a_bounds_core": {
        "datasets": ["cora", "citeseer", "pubmed"],
        "partitions": ["dirichlet_0.1"],
        "clients": [20],
        "seeds": [0, 1, 2],
        "methods": ["local_only", "centralized", "label_aware_oracle"],
        "rounds": 60,
    },
    "t4b_bounds_extension": {
        "datasets": ["amazon_photo", "amazon_computers", "roman_empire", "tolokers"],
        "partitions": ["dirichlet_0.1"],
        "clients": [20],
        "seeds": [0, 1, 2],
        "methods": ["local_only", "centralized", "label_aware_oracle"],
        "rounds": 60,
    },
    "t4a_bounds_5seed_core": {
        "datasets": ["cora", "citeseer", "pubmed"],
        "partitions": ["dirichlet_0.1"],
        "clients": [20],
        "seeds": [3, 4],
        "methods": ["local_only", "centralized", "label_aware_oracle"],
        "rounds": 60,
    },
    "t4b_bounds_5seed_extension": {
        "datasets": ["amazon_photo", "amazon_computers", "roman_empire"],
        "partitions": ["dirichlet_0.1"],
        "clients": [20],
        "seeds": [3, 4],
        "methods": ["local_only", "centralized", "label_aware_oracle"],
        "rounds": 60,
    },
}


@dataclass(frozen=True)
class Task:
    dataset: str
    partition: str
    clients: int
    seed: int
    method: str
    rounds: int


def parse_csv_arg(value, cast=str):
    if value is None:
        return None
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def label_coverage_stats(data, nodes):
    labels = data.y[nodes].detach().cpu()
    counts = torch.bincount(labels, minlength=int(data.num_classes)).numpy()
    present = counts[counts > 0]
    return {
        "num_nodes": int(nodes.numel()),
        "num_classes": int((counts > 0).sum()),
        "min_class_count": int(present.min()) if present.size else 0,
        "max_class_count": int(present.max()) if present.size else 0,
    }


def output_prefix(args, task):
    name = f"{task.dataset}_{task.partition}_k{task.clients}_{task.method}_s{task.seed}"
    return Path(args.output_dir) / args.stage / name


def result_csv_path(args, task):
    return Path(f"{output_prefix(args, task)}.csv")


def aggregated_csv_path(args, task):
    return Path(f"{output_prefix(args, task)}_aggregated.csv")


def is_done(args, task):
    if args.force:
        return False
    raw = result_csv_path(args, task)
    agg = aggregated_csv_path(args, task)
    return raw.exists() and agg.exists() and raw.stat().st_size > 0 and agg.stat().st_size > 0


def build_tasks(args):
    cfg = dict(STAGE_CONFIGS[args.stage])
    datasets = parse_csv_arg(args.datasets) or cfg["datasets"]
    partitions = parse_csv_arg(args.partitions) or cfg["partitions"]
    clients = parse_csv_arg(args.clients, int) or cfg["clients"]
    seeds = parse_csv_arg(args.seeds, int) or cfg["seeds"]
    methods = parse_csv_arg(args.methods) or cfg["methods"]
    rounds = args.rounds or cfg["rounds"]
    return [
        Task(dataset, partition, client_count, seed, method, rounds)
        for dataset in datasets
        for partition in partitions
        for client_count in clients
        for seed in seeds
        for method in methods
    ]


def local_edge_weight(data, nodes):
    node_mask = torch.zeros(data.num_nodes, dtype=torch.bool, device=data.x.device)
    node_mask[nodes.to(data.x.device)] = True
    src, dst = data.edge_index
    return (node_mask[src] & node_mask[dst]).float()


def train_one_model(data, train_nodes, edge_weight, total_steps, args, device):
    model = GCN(data.num_features, args.hidden_dim, data.num_classes, args.dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_nodes = train_nodes.to(device)
    for _ in range(total_steps):
        model.train()
        opt.zero_grad()
        logits = model(data.x, data.edge_index, edge_weight=edge_weight)
        loss = F.cross_entropy(logits[train_nodes], data.y[train_nodes])
        loss.backward()
        opt.step()
    model.eval()
    return model


def train_local_or_centralized(args, task):
    torch.manual_seed(task.seed)
    np.random.seed(task.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    data = load_dataset(task.dataset, Path(args.data_root)).to(device)
    clients = make_clients(data, task.clients, task.partition, task.seed)
    start = time.time()

    rows = []
    total_steps = max(1, int(task.rounds) * int(args.local_epochs))
    if task.method == "centralized":
        all_train = torch.cat([c["train_nodes"] for c in clients]).unique().to(device)
        model = train_one_model(data, all_train, None, total_steps, args, device)
        models = [model for _ in clients]
        edge_weights = [None for _ in clients]
    else:
        edge_weights = [local_edge_weight(data, c["nodes"].to(device)) for c in clients]
        models = [
            train_one_model(data, c["train_nodes"].to(device), edge_weight, total_steps, args, device)
            for c, edge_weight in zip(clients, edge_weights)
        ]

    train_seconds = time.time() - start
    model_parameter_count = sum(p.numel() for p in models[0].parameters())
    with torch.no_grad():
        for c, model, edge_weight in zip(clients, models, edge_weights):
            nodes = c["test_nodes"].to(device)
            logits = model(data.x, data.edge_index, edge_weight=edge_weight)
            pred = logits[nodes].argmax(dim=1)
            y = data.y[nodes]
            rows.append(
                {
                    **{f"train_{k}": v for k, v in label_coverage_stats(data, c["train_nodes"].to(device)).items()},
                    **{f"test_{k}": v for k, v in label_coverage_stats(data, c["test_nodes"].to(device)).items()},
                    "dataset": task.dataset,
                    "partition": task.partition,
                    "clients": task.clients,
                    "seed": task.seed,
                    "method": task.method,
                    "run_tag": "oracle_bounds",
                    "client_id": c["client_id"],
                    "accuracy": float((pred == y).float().mean().item()),
                    "macro_f1": macro_f1_torch(y, pred, data.num_classes),
                    "train_seconds": float(train_seconds),
                    "model_parameter_count": int(model_parameter_count),
                    "personal_head_parameter_count": 0,
                    "descriptor_dim": 0,
                    "estimated_upload_bytes": 0,
                    "estimated_download_bytes": 0,
                    "estimated_model_upload_bytes": 0,
                    "estimated_model_download_bytes": 0,
                    "estimated_descriptor_upload_bytes": 0,
                    "estimated_candidate_download_bytes": 0,
                    "fedpub_proxy_nodes": 0,
                    "candidate_label_conflict_rate": np.nan,
                    "candidate_label_js_mean": np.nan,
                    "candidate_count": 0,
                }
            )
    return rows


def write_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    f1s = np.array([r["macro_f1"] for r in rows], dtype=float)
    accs = np.array([r["accuracy"] for r in rows], dtype=float)
    agg = {
        "method": rows[0]["method"],
        "dataset": rows[0]["dataset"],
        "partition": rows[0]["partition"],
        "clients": rows[0]["clients"],
        "seed": rows[0]["seed"],
        "run_tag": rows[0]["run_tag"],
        "mean_accuracy": float(accs.mean()),
        "worst_accuracy": float(accs.min()),
        "mean_macro_f1": float(f1s.mean()),
        "worst_macro_f1": float(f1s.min()),
        "train_seconds": rows[0]["train_seconds"],
        "model_parameter_count": rows[0]["model_parameter_count"],
        "personal_head_parameter_count": rows[0]["personal_head_parameter_count"],
        "descriptor_dim": rows[0]["descriptor_dim"],
        "estimated_upload_bytes": rows[0]["estimated_upload_bytes"],
        "estimated_download_bytes": rows[0]["estimated_download_bytes"],
        "estimated_model_upload_bytes": rows[0]["estimated_model_upload_bytes"],
        "estimated_model_download_bytes": rows[0]["estimated_model_download_bytes"],
        "estimated_descriptor_upload_bytes": rows[0]["estimated_descriptor_upload_bytes"],
        "estimated_candidate_download_bytes": rows[0]["estimated_candidate_download_bytes"],
        "fedpub_proxy_nodes": rows[0]["fedpub_proxy_nodes"],
    }
    agg_path = path.with_name(path.stem + "_aggregated.csv")
    with open(agg_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(agg.keys()))
        writer.writeheader()
        writer.writerow(agg)


def run_label_aware_oracle(args, task):
    prefix = output_prefix(args, task)
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
        "spc_guard_pp",
        "--descriptor-mode",
        "with_label",
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
        "label_aware_oracle",
    ]
    if args.cpu:
        cmd.append("--cpu")
    log_path = Path(f"{prefix}.log")
    with open(log_path, "w", encoding="utf-8") as log:
        log.write(" ".join(shlex.quote(x) for x in cmd) + "\n")
        log.flush()
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
    if proc.returncode == 0:
        relabel_result_method(result_csv_path(args, task), "label_aware_oracle")
        relabel_result_method(aggregated_csv_path(args, task), "label_aware_oracle")
    return proc.returncode == 0


def relabel_result_method(path, method):
    path = Path(path)
    if not path.exists():
        return
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
        fieldnames = rows[0].keys() if rows else []
    if not rows or "method" not in rows[0]:
        return
    for row in rows:
        row["method"] = method
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def run_task(args, task):
    prefix = output_prefix(args, task)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    log_path = Path(f"{prefix}.log")
    if task.method == "label_aware_oracle":
        ok = run_label_aware_oracle(args, task)
        elapsed = time.time() - start
        if not ok:
            print(f"[failed] {task.dataset}/{task.method}/seed={task.seed}; see {log_path}")
            if not args.keep_going:
                raise SystemExit(1)
            return False
    else:
        try:
            rows = train_local_or_centralized(args, task)
            write_rows(result_csv_path(args, task), rows)
            with open(log_path, "w", encoding="utf-8") as log:
                log.write(f"method={task.method}\ncompleted_seconds={time.time() - start:.2f}\n")
        except Exception as exc:
            with open(log_path, "w", encoding="utf-8") as log:
                log.write(f"failed: {exc}\n")
            print(f"[failed] {task.dataset}/{task.method}/seed={task.seed}; see {log_path}")
            if not args.keep_going:
                raise
            return False
        elapsed = time.time() - start
    print(f"[done] {task.dataset}/{task.method}/seed={task.seed} in {elapsed/60:.1f} min")
    return True


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
                    "output": str(result_csv_path(args, task)),
                }
            )
    return path


def summarize(args):
    stage_dir = Path(args.output_dir) / args.stage
    agg_files = sorted(stage_dir.glob("*_aggregated.csv"))
    if not agg_files:
        return
    out = stage_dir / "aggregated_overview.csv"
    rows = []
    for path in agg_files:
        with open(path, newline="", encoding="utf-8") as f:
            rows.extend(csv.DictReader(f))
    rows.sort(key=lambda r: (r["dataset"], r["method"], int(r["seed"])))
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Run local-only, centralized, and label-aware oracle bounds.")
    parser.add_argument("--stage", required=True, choices=sorted(STAGE_CONFIGS))
    parser.add_argument("--output-dir", default="results/oracle_bounds")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--datasets")
    parser.add_argument("--partitions")
    parser.add_argument("--clients")
    parser.add_argument("--seeds")
    parser.add_argument("--methods")
    parser.add_argument("--rounds", type=int)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--max-tasks", type=int)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--keep-going", action="store_true")
    parser.add_argument("--no-summarize", action="store_true")
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
    print(json.dumps({"stage": args.stage, "tasks": len(tasks), "pending": len(pending), "plan": str(plan_path)}, indent=2))
    if args.dry_run:
        return
    ok_count = 0
    for task in pending:
        ok_count += int(run_task(args, task))
        write_plan(args, tasks)
    if not args.no_summarize:
        summarize(args)
    print(json.dumps({"completed": ok_count, "attempted": len(pending)}, indent=2))


if __name__ == "__main__":
    main()
