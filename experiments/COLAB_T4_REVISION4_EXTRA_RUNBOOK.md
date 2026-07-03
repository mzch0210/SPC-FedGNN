# Revision-4 extra experiments on Colab T4

This runbook covers only the newly requested experiments:

- sparse / missing validation evidence for class-wise gates;
- real communication and runtime accounting.

It does not run official-code baselines and does not include local-only / centralized / oracle-bound experiments.

## Output directories

```text
/content/drive/MyDrive/SPC-FedGNN/results/revision4_extra/rev_sparse_validation
/content/drive/MyDrive/SPC-FedGNN/results/revision4_extra/rev_comm_runtime
```

## Account split

Account A:

- `rev_sparse_validation`

Account B:

- `rev_comm_runtime`

Expected task counts:

- `rev_sparse_validation`: 90 tasks
- `rev_comm_runtime`: 42 tasks

## Step 0. Upload code

Upload this zip to each account's Google Drive root:

```text
spc_fedgnn_revision4_extra_code_20260703_fixed.zip
```

## Step 1. Mount Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

## Step 2. Unpack cleanly

```python
import os, shutil, zipfile

ZIP = "/content/drive/MyDrive/spc_fedgnn_revision4_extra_code_20260703_fixed.zip"
ROOT = "/content/drive/MyDrive/SPC-FedGNN-revision4-extra"

if os.path.exists(ROOT):
    shutil.rmtree(ROOT)
os.makedirs(ROOT, exist_ok=True)

with zipfile.ZipFile(ZIP, "r") as z:
    z.extractall(ROOT)

print(os.listdir(ROOT)[:20])
```

## Step 3. Install dependencies

```python
import os, sys, subprocess, torch

ROOT = "/content/drive/MyDrive/SPC-FedGNN-revision4-extra"
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

## Step 4. Account A: sparse validation

Run in Account A:

```python
import os, subprocess, sys

ROOT = "/content/drive/MyDrive/SPC-FedGNN-revision4-extra"
os.chdir(ROOT)

subprocess.check_call([
    sys.executable, "experiments/colab_full_experiment.py",
    "--stage", "rev_sparse_validation",
    "--output-dir", "/content/drive/MyDrive/SPC-FedGNN/results/revision4_extra",
    "--data-root", "/content/drive/MyDrive/SPC-FedGNN-revision4-extra/data",
    "--keep-going",
])
```

One-line completion check:

```python
!cat /content/drive/MyDrive/SPC-FedGNN/results/revision4_extra/rev_sparse_validation/task_plan.csv | tail -n +2 | cut -d, -f8 | sort | uniq -c
```

Expected:

```text
90 1
```

If some tasks fail:

```python
!for f in /content/drive/MyDrive/SPC-FedGNN/results/revision4_extra/rev_sparse_validation/*.log; do echo "==== $f"; tail -n 20 "$f"; done
```

If a previous run failed with a tensor-size mismatch in sparse-validation tasks, upload the fixed zip, rerun Step 2 to overwrite the code, and rerun Step 4. Completed tasks are skipped automatically; failed tasks are retried because their aggregated CSV files are missing.

Package Account A results:

```python
import shutil
SRC = "/content/drive/MyDrive/SPC-FedGNN/results/revision4_extra/rev_sparse_validation"
ZIP = "/content/drive/MyDrive/spc_fedgnn_revision4_sparse_validation_account_a.zip"
shutil.make_archive(ZIP.replace(".zip", ""), "zip", SRC)
print(ZIP)
```

## Step 5. Account B: communication/runtime

Run in Account B:

```python
import os, subprocess, sys

ROOT = "/content/drive/MyDrive/SPC-FedGNN-revision4-extra"
os.chdir(ROOT)

subprocess.check_call([
    sys.executable, "experiments/colab_full_experiment.py",
    "--stage", "rev_comm_runtime",
    "--output-dir", "/content/drive/MyDrive/SPC-FedGNN/results/revision4_extra",
    "--data-root", "/content/drive/MyDrive/SPC-FedGNN-revision4-extra/data",
    "--keep-going",
])
```

One-line completion check:

```python
!cat /content/drive/MyDrive/SPC-FedGNN/results/revision4_extra/rev_comm_runtime/task_plan.csv | tail -n +2 | cut -d, -f8 | sort | uniq -c
```

Expected:

```text
42 1
```

Package Account B results:

```python
import shutil
SRC = "/content/drive/MyDrive/SPC-FedGNN/results/revision4_extra/rev_comm_runtime"
ZIP = "/content/drive/MyDrive/spc_fedgnn_revision4_comm_runtime_account_b.zip"
shutil.make_archive(ZIP.replace(".zip", ""), "zip", SRC)
print(ZIP)
```

## Step 6. Download results

Download these files and place them in the local project root:

```text
spc_fedgnn_revision4_sparse_validation_account_a.zip
spc_fedgnn_revision4_comm_runtime_account_b.zip
```

After they are placed in the project, ask Codex to merge and update the manuscript.

## Interpretation notes

- Sparse-validation results should be reported as robustness diagnostics for the class-wise gate, not as a new baseline table.
- Communication/runtime results should distinguish compact descriptor/candidate traffic from teacher-state transfer. Do not describe the sub-KB retrieval descriptor as total communication cost.
