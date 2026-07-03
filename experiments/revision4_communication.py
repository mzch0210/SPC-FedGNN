import argparse
import csv
from pathlib import Path


METHOD_LABELS = {
    "spc_guard_pp": "SPC-FedGNN",
    "fedath": "FedATH-style",
    "fedssp": "FedSSP",
    "fedpub_gcn": "FedPub-style",
    "fedavg_gcn": "FedAvg-GCN",
    "fedprox_gcn": "FedProx-GCN",
}


def read_rows(paths):
    rows = []
    for path in paths:
        with open(path, newline="", encoding="utf-8") as f:
            rows.extend(csv.DictReader(f))
    return rows


def to_float(row, key):
    try:
        return float(row.get(key, 0.0))
    except Exception:
        return 0.0


def main():
    parser = argparse.ArgumentParser(description="Build communication/runtime accounting table.")
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--candidate-count", type=int, default=3)
    args = parser.parse_args()

    rows = read_rows([Path(p) for p in args.inputs])
    out = []
    for row in rows:
        method = row.get("method", "")
        model_download = to_float(row, "estimated_model_download_bytes")
        teacher_extra = 0.0
        if method == "spc_guard_pp":
            teacher_extra = model_download * args.candidate_count
        total_counted = to_float(row, "estimated_upload_bytes") + to_float(row, "estimated_download_bytes")
        total_with_teacher = total_counted + teacher_extra
        out.append(
            {
                "dataset": row.get("dataset", ""),
                "method": METHOD_LABELS.get(method, method),
                "seed": row.get("seed", ""),
                "train_seconds": f"{to_float(row, 'train_seconds'):.2f}",
                "model_upload_MB": f"{to_float(row, 'estimated_model_upload_bytes') / 1e6:.2f}",
                "model_download_MB": f"{model_download / 1e6:.2f}",
                "descriptor_or_embedding_upload_KB": f"{to_float(row, 'estimated_descriptor_upload_bytes') / 1e3:.2f}",
                "candidate_index_download_KB": f"{to_float(row, 'estimated_candidate_download_bytes') / 1e3:.2f}",
                "teacher_state_extra_MB": f"{teacher_extra / 1e6:.2f}",
                "total_without_teacher_MB": f"{total_counted / 1e6:.2f}",
                "total_with_teacher_MB": f"{total_with_teacher / 1e6:.2f}",
                "model_parameters": row.get("model_parameter_count", ""),
                "descriptor_dim": row.get("descriptor_dim", ""),
            }
        )
    out.sort(key=lambda r: (r["dataset"], r["method"]))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        writer.writeheader()
        writer.writerows(out)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
