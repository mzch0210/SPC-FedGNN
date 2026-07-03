import argparse
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path


def extract_zip(zip_path, extract_root):
    target = extract_root / zip_path.stem
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(target)
    return target


def main():
    parser = argparse.ArgumentParser(description="Merge oracle-bound result zips downloaded from separate Colab accounts.")
    parser.add_argument("--zips", nargs="+", required=True)
    parser.add_argument("--output-dir", default="results/oracle_bounds_merged")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    extract_root = output_dir / "_extracted"
    if extract_root.exists():
        shutil.rmtree(extract_root)
    extract_root.mkdir(parents=True, exist_ok=True)

    extracted = [extract_zip(Path(p), extract_root) for p in args.zips]
    patterns = [str(path / "**" / "*_aggregated.csv") for path in extracted]
    summary_path = output_dir / "oracle_bounds_summary.csv"

    subprocess.check_call(
        [
            sys.executable,
            "experiments/oracle_bounds/summarize_oracle_bounds.py",
            "--inputs",
            *patterns,
            "--output",
            str(summary_path),
        ]
    )
    print(f"Merged {len(args.zips)} zip files.")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
