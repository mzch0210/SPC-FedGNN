# Strong-baseline Colab T4 runbook

This runbook is independent from the current `revision_priority` experiment batch. It writes only to:

```text
/content/drive/MyDrive/SPC-FedGNN/results/strong_baselines
```

Do not run these cells inside the same Colab notebook that is currently running revision-priority experiments.

## What this batch does

Official-code status:

- FedGTA official repository exists: `https://github.com/xkLi-Allen/FedGTA`
- AdaFGL official repository exists: `https://github.com/xkLi-Allen/AdaFGL`
- FedIIH official repository was not found during targeted search.

Unified-protocol matched baselines:

- `fedgta_matched`
- `adafgl_matched`
- `fediih_matched`

These are clearly marked as matched baselines, not official-code results.

## Account split

Account A:

- `smoke`
- `t4a_strong_core`

Account B:

- `t4b_strong_extension`
- optional `t4b_strong_tolokers`

Expected task counts:

- `smoke`: 3 tasks
- `t4a_strong_core`: 27 tasks
- `t4b_strong_extension`: 18 tasks
- `t4_photo_strong`: 9 tasks
- `t4b_strong_tolokers`: 9 tasks

## Step 0. Upload code

Upload `spc_fedgnn_strong_baselines_code_20260702.zip` to `/content/drive/MyDrive/`.

## Step 1. Mount Drive and unpack

```python
from google.colab import drive
drive.mount('/content/drive')
```

```python
import os, shutil, zipfile

ZIP = "/content/drive/MyDrive/spc_fedgnn_strong_baselines_code_20260702.zip"
ROOT = "/content/drive/MyDrive/SPC-FedGNN-strong"

if os.path.exists(ROOT):
    shutil.rmtree(ROOT)
os.makedirs(ROOT, exist_ok=True)

with zipfile.ZipFile(ZIP, "r") as z:
    z.extractall(ROOT)

print(os.listdir(ROOT)[:20])
```

## Step 2. Install dependencies

```python
import os, sys, subprocess, torch

ROOT = "/content/drive/MyDrive/SPC-FedGNN-strong"
os.chdir(ROOT)

print("torch:", torch.__version__)
print("cuda:", torch.version.cuda)
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "torch-geometric", "ogb"])
```

Verify:

```python
import torch, torch_geometric
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO GPU")
print("PyG:", torch_geometric.__version__)
```

## Step 3. Smoke test

Run in either account first:

```python
import os, subprocess, sys
ROOT = "/content/drive/MyDrive/SPC-FedGNN-strong"
os.chdir(ROOT)

subprocess.check_call([
    sys.executable, "experiments/strong_baselines/run_strong_baselines.py",
    "--stage", "smoke",
    "--output-dir", "/content/drive/MyDrive/SPC-FedGNN/results/strong_baselines",
    "--data-root", "/content/drive/MyDrive/SPC-FedGNN-strong/data",
    "--keep-going",
])
```

One-line check:

```python
!cat /content/drive/MyDrive/SPC-FedGNN/results/strong_baselines/smoke/task_plan.csv | tail -n +2 | cut -d, -f7 | sort | uniq -c
```

Expected:

```text
3 1
```

## Step 4. Account A core datasets

```python
import os, subprocess, sys
ROOT = "/content/drive/MyDrive/SPC-FedGNN-strong"
os.chdir(ROOT)

subprocess.check_call([
    sys.executable, "experiments/strong_baselines/run_strong_baselines.py",
    "--stage", "t4a_strong_core",
    "--output-dir", "/content/drive/MyDrive/SPC-FedGNN/results/strong_baselines",
    "--data-root", "/content/drive/MyDrive/SPC-FedGNN-strong/data",
    "--keep-going",
])
```

Check:

```python
!cat /content/drive/MyDrive/SPC-FedGNN/results/strong_baselines/t4a_strong_core/task_plan.csv | tail -n +2 | cut -d, -f7 | sort | uniq -c
```

Expected:

```text
27 1
```

## Step 5. Account B extension datasets

```python
import os, subprocess, sys
ROOT = "/content/drive/MyDrive/SPC-FedGNN-strong"
os.chdir(ROOT)

subprocess.check_call([
    sys.executable, "experiments/strong_baselines/run_strong_baselines.py",
    "--stage", "t4b_strong_extension",
    "--output-dir", "/content/drive/MyDrive/SPC-FedGNN/results/strong_baselines",
    "--data-root", "/content/drive/MyDrive/SPC-FedGNN-strong/data",
    "--keep-going",
])
```

Check:

```python
!cat /content/drive/MyDrive/SPC-FedGNN/results/strong_baselines/t4b_strong_extension/task_plan.csv | tail -n +2 | cut -d, -f7 | sort | uniq -c
```

Expected:

```text
18 1
```

## Step 6. Optional Tolokers strong-baseline audit

Run only after the extension stage is complete:

```python
import os, subprocess, sys
ROOT = "/content/drive/MyDrive/SPC-FedGNN-strong"
os.chdir(ROOT)

subprocess.check_call([
    sys.executable, "experiments/strong_baselines/run_strong_baselines.py",
    "--stage", "t4b_strong_tolokers",
    "--output-dir", "/content/drive/MyDrive/SPC-FedGNN/results/strong_baselines",
    "--data-root", "/content/drive/MyDrive/SPC-FedGNN-strong/data",
    "--keep-going",
])
```

Check:

```python
!cat /content/drive/MyDrive/SPC-FedGNN/results/strong_baselines/t4b_strong_tolokers/task_plan.csv | tail -n +2 | cut -d, -f7 | sort | uniq -c
```

Expected:

```text
9 1
```

## Optional Amazon-Photo supplement

Amazon-Photo can be run as a separate supplement after the main strong-baseline batch:

```python
import os, subprocess, sys
ROOT = "/content/drive/MyDrive/SPC-FedGNN-strong"
os.chdir(ROOT)

subprocess.check_call([
    sys.executable, "experiments/strong_baselines/run_strong_baselines.py",
    "--stage", "t4_photo_strong",
    "--output-dir", "/content/drive/MyDrive/SPC-FedGNN/results/strong_baselines",
    "--data-root", "/content/drive/MyDrive/SPC-FedGNN-strong/data",
    "--keep-going",
])
```

Check:

```python
!cat /content/drive/MyDrive/SPC-FedGNN/results/strong_baselines/t4_photo_strong/task_plan.csv | tail -n +2 | cut -d, -f7 | sort | uniq -c
```

Expected:

```text
9 1
```

For a cleaner standalone guide, use `experiments/strong_baselines/COLAB_T4_AMAZON_PHOTO_STRONG_BASELINES.md`.

## Step 7. Package results

Run this in each account after its stages finish:

```python
import os, shutil
ROOT = "/content/drive/MyDrive/SPC-FedGNN"
OUT = "/content/drive/MyDrive/spc_fedgnn_strong_baselines_results"

if os.path.exists(OUT + ".zip"):
    os.remove(OUT + ".zip")

shutil.make_archive(OUT, "zip", ROOT, "results/strong_baselines")
print(OUT + ".zip")
```

Download and rename locally:

- `spc_fedgnn_strong_baselines_account_a.zip`
- `spc_fedgnn_strong_baselines_account_b.zip`

## Optional: official repository smoke only

This checks that official FedGTA/AdaFGL repositories exist and can be cloned. It does not produce same-protocol results.

```python
%cd /content/drive/MyDrive
!rm -rf official_fgl_repos
!mkdir -p official_fgl_repos
%cd official_fgl_repos
!git clone https://github.com/xkLi-Allen/FedGTA.git
!git clone https://github.com/xkLi-Allen/AdaFGL.git
!find FedGTA AdaFGL -maxdepth 2 -name README.md -print
```

Do not merge default official-repo numbers into the main SPC-FedGNN table unless the split protocol is aligned.
