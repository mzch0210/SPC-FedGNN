import argparse
import csv
import math
import subprocess
import sys
from pathlib import Path


def read_rows(path):
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_rows(path, rows):
    if not rows:
        raise SystemExit("No rows to write.")
    keys = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def row_score(row):
    useful = [
        "candidate_label_conflict_rate",
        "candidate_label_js_mean",
        "guard_accept_rate",
        "guard_pp_class_accept_rate",
        "train_seconds",
        "estimated_upload_bytes",
        "estimated_download_bytes",
        "estimated_model_upload_bytes",
        "estimated_model_download_bytes",
        "estimated_descriptor_upload_bytes",
        "estimated_candidate_download_bytes",
        "fedpub_proxy_nodes",
    ]
    score = 0
    for key in useful:
        value = row.get(key, "")
        if value not in {"", "nan", "NaN", "None"}:
            score += 1
            try:
                if not math.isnan(float(value)):
                    score += 1
            except Exception:
                pass
    return score


def experiment_key(row):
    return (
        row.get("dataset", ""),
        row.get("partition", ""),
        row.get("clients", ""),
        row.get("seed", ""),
        row.get("method", ""),
        row.get("run_tag", "base") or "base",
    )


def main():
    parser = argparse.ArgumentParser(description="Merge dual-Colab supplemental SPC-FedGNN result folders.")
    parser.add_argument("--inputs", nargs="+", required=True, help="Result roots, zip-extracted roots, or stage folders.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--reference-methods", default="fedath,fedssp,fedpub_gcn,spc_guard")
    args = parser.parse_args()

    enhanced_files = []
    stage_table_files = []
    for item in args.inputs:
        root = Path(item)
        if root.is_file() and root.name in {"enhanced_overview.csv", "combined_enhanced_overview.csv"}:
            enhanced_files.append(root)
        elif root.is_file() and root.name == "stage_table.csv":
            stage_table_files.append(root)
        elif root.exists():
            enhanced_files.extend(root.rglob("enhanced_overview.csv"))
            enhanced_files.extend(root.rglob("combined_enhanced_overview.csv"))
            stage_table_files.extend(root.rglob("stage_table.csv"))

    if not enhanced_files:
        raise SystemExit("No enhanced_overview.csv files found.")

    best_by_key = {}
    for path in sorted(set(enhanced_files)):
        for row in read_rows(path):
            row = dict(row)
            row["merge_source_file"] = str(path)
            key = experiment_key(row)
            if key not in best_by_key or row_score(row) > row_score(best_by_key[key]):
                best_by_key[key] = row

    merged_rows = sorted(
        best_by_key.values(),
        key=lambda r: (
            r.get("dataset", ""),
            r.get("partition", ""),
            int(float(r.get("clients", 0) or 0)),
            int(float(r.get("seed", 0) or 0)),
            r.get("method", ""),
            r.get("run_tag", "base") or "base",
        ),
    )

    output_dir = Path(args.output_dir)
    merged_path = output_dir / "combined_enhanced_overview.csv"
    write_rows(merged_path, merged_rows)

    subprocess.run(
        [
            sys.executable,
            "experiments/aggregate_stage_tables.py",
            "--input",
            str(merged_path),
            "--output",
            str(output_dir / "combined_stage_table.csv"),
            "--reference-methods",
            args.reference_methods,
        ],
        check=True,
    )

    if stage_table_files:
        stage_rows = []
        for path in sorted(set(stage_table_files)):
            for row in read_rows(path):
                row = dict(row)
                row["merge_source_file"] = str(path)
                stage_rows.append(row)
        write_rows(output_dir / "combined_stage_tables_raw.csv", stage_rows)

    print(f"Wrote {merged_path}")
    print(f"Wrote {output_dir / 'combined_stage_table.csv'}")


if __name__ == "__main__":
    main()
