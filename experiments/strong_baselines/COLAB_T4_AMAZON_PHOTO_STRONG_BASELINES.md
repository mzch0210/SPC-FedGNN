# Colab T4 runbook: Amazon-Photo strong baselines

This runbook only completes the missing Amazon-Photo strong-baseline batch. It is independent from the previous strong-baseline runs and writes to:

```text
/content/drive/MyDrive/SPC-FedGNN/results/strong_baselines/t4_photo_strong
```

The batch contains 9 tasks:

- 1 dataset: `amazon_photo`
- 3 seeds: `0,1,2`
- 3 matched baselines: `fedgta_matched`, `adafgl_matched`, `fediih_matched`

These are unified-protocol matched baselines, not official-code results.

## Step 0. Upload code zip

Upload the latest code package to Google Drive:

```text
/content/drive/MyDrive/spc_fedgnn_strong_baselines_photo_code_20260702.zip
```

## Step 1. Mount Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

## Step 2. Unpack the latest code

```python
import os, shutil, zipfile

ZIP = "/content/drive/MyDrive/spc_fedgnn_strong_baselines_photo_code_20260702.zip"
ROOT = "/content/drive/MyDrive/SPC-FedGNN-strong-photo"

if os.path.exists(ROOT):
    shutil.rmtree(ROOT)
os.makedirs(ROOT, exist_ok=True)

with zipfile.ZipFile(ZIP, "r") as z:
    z.extractall(ROOT)

print("ROOT:", ROOT)
print(os.listdir(ROOT)[:20])
```

## Step 3. Install dependencies

```python
import os, sys, subprocess, torch

ROOT = "/content/drive/MyDrive/SPC-FedGNN-strong-photo"
os.chdir(ROOT)

print("torch:", torch.__version__)
print("cuda:", torch.version.cuda)
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "torch-geometric", "ogb"])
```

Verify GPU:

```python
import torch, torch_geometric
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO GPU")
print("PyG:", torch_geometric.__version__)
```

If the output says `NO GPU`, change Colab runtime to T4 GPU before continuing.

## Step 4. Dry-run task plan

```python
import os, subprocess, sys

ROOT = "/content/drive/MyDrive/SPC-FedGNN-strong-photo"
os.chdir(ROOT)

subprocess.check_call([
    sys.executable, "experiments/strong_baselines/run_strong_baselines.py",
    "--stage", "t4_photo_strong",
    "--output-dir", "/content/drive/MyDrive/SPC-FedGNN/results/strong_baselines",
    "--data-root", "/content/drive/MyDrive/SPC-FedGNN-strong-photo/data",
    "--dry-run",
])
```

Expected summary:

```text
"tasks": 9
"pending": 9
```

If you have partially run this stage before, `pending` can be smaller than 9.

## Step 5. Run Amazon-Photo strong baselines

```python
import os, subprocess, sys

ROOT = "/content/drive/MyDrive/SPC-FedGNN-strong-photo"
os.chdir(ROOT)

subprocess.check_call([
    sys.executable, "experiments/strong_baselines/run_strong_baselines.py",
    "--stage", "t4_photo_strong",
    "--output-dir", "/content/drive/MyDrive/SPC-FedGNN/results/strong_baselines",
    "--data-root", "/content/drive/MyDrive/SPC-FedGNN-strong-photo/data",
    "--keep-going",
])
```

Approximate runtime on T4: usually tens of minutes, depending on Colab I/O and whether Amazon-Photo needs to be downloaded.

## Step 6. One-line completion check

```python
!cat /content/drive/MyDrive/SPC-FedGNN/results/strong_baselines/t4_photo_strong/task_plan.csv | tail -n +2 | cut -d, -f7 | sort | uniq -c
```

Expected:

```text
9 1
```

If you see `0`, rerun Step 5. The script skips completed tasks unless `--force` is used.

## Step 7. Check logs

```python
!grep -R -i "error\\|traceback\\|failed\\|nan" /content/drive/MyDrive/SPC-FedGNN/results/strong_baselines/t4_photo_strong/*.log || true
```

Expected: no output.

You can also inspect the final aggregate table:

```python
!cat /content/drive/MyDrive/SPC-FedGNN/results/strong_baselines/t4_photo_strong/aggregated_overview.csv
```

## Step 8. Package only the Amazon-Photo supplement

```python
import os, shutil

ROOT = "/content/drive/MyDrive/SPC-FedGNN/results/strong_baselines"
OUT = "/content/drive/MyDrive/spc_fedgnn_strong_baselines_amazon_photo_results"

if os.path.exists(OUT + ".zip"):
    os.remove(OUT + ".zip")

shutil.make_archive(OUT, "zip", ROOT, "t4_photo_strong")
print(OUT + ".zip")
```

Download this file and put it in the project folder:

```text
spc_fedgnn_strong_baselines_amazon_photo_results.zip
```

## Optional resume command

If Colab disconnects, remount Drive, unpack the same code if needed, install dependencies, and rerun Step 5. Completed CSV files will be skipped automatically.

## Optional rerun from scratch

Only use this if the stage is corrupted:

```python
!rm -rf /content/drive/MyDrive/SPC-FedGNN/results/strong_baselines/t4_photo_strong
```

Then rerun Step 5.
