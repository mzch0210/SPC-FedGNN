# Colab T4 Revision-6 Experiments Runbook

This runbook is for the revision-6 experiment batch:

- fair class-balanced baselines;
- contribution disentanglement;
- independent gate-evidence diagnostics.

It is independent from the previous Colab result folders. Run it on a Colab T4 15G runtime.

## Step 1. Start Colab

1. Open a new Colab notebook.
2. Runtime -> Change runtime type -> T4 GPU.
3. Mount Google Drive:

```python
from google.colab import drive
drive.mount('/content/drive')
```

## Step 2. Put the project in Drive

If the project folder already exists:

```python
ROOT = "/content/drive/MyDrive/SPC-FedGNN"
%cd "$ROOT"
```

If you are uploading a fresh zip, upload it to `/content/drive/MyDrive/`, then run:

```python
import os, zipfile, shutil
ZIP = "/content/drive/MyDrive/spc_fedgnn_revision6_code.zip"
ROOT = "/content/drive/MyDrive/SPC-FedGNN"
if os.path.exists(ROOT):
    shutil.rmtree(ROOT)
os.makedirs(ROOT, exist_ok=True)
with zipfile.ZipFile(ZIP, "r") as z:
    z.extractall(ROOT)
%cd "$ROOT"
```

## Step 3. Install dependencies

```python
!pip -q install torch_geometric
```

Then check GPU:

```python
!nvidia-smi
```

## Step 4. Smoke test

```python
!python3 experiments/revision6/run_revision6_experiments.py \
  --stage smoke \
  --root /content/drive/MyDrive/SPC-FedGNN/results/revision6 \
  --data-root /content/drive/MyDrive/SPC-FedGNN/data
```

One-line completion check:

```python
!cat /content/drive/MyDrive/SPC-FedGNN/results/revision6/smoke/task_plan.csv | tail -n +2 | cut -d, -f10 | sort | uniq -c
```

Expected format after completion:

```text
      3 1
```

If it returns any `0`, inspect logs:

```python
!for f in /content/drive/MyDrive/SPC-FedGNN/results/revision6/smoke/*.log; do echo "==== $f"; tail -n 25 "$f"; done
```

## Step 5. Fair class-balanced baselines

This stage tests whether SPC-FedGNN still looks competitive when standard baselines also receive class-balanced supervised loss.

```python
!python3 experiments/revision6/run_revision6_experiments.py \
  --stage fair_cb \
  --root /content/drive/MyDrive/SPC-FedGNN/results/revision6 \
  --data-root /content/drive/MyDrive/SPC-FedGNN/data
```

Completion check:

```python
!cat /content/drive/MyDrive/SPC-FedGNN/results/revision6/fair_cb/task_plan.csv | tail -n +2 | cut -d, -f10 | sort | uniq -c
```

Expected:

```text
     81 1
```

## Step 6. Contribution disentanglement

This stage separates the role of Base Guard, prototype retrieval, coverage-aware reliability, class balance, random candidates, no personalization, and direct prototype aggregation.

```python
!python3 experiments/revision6/run_revision6_experiments.py \
  --stage contribution \
  --root /content/drive/MyDrive/SPC-FedGNN/results/revision6 \
  --data-root /content/drive/MyDrive/SPC-FedGNN/data
```

Completion check:

```python
!cat /content/drive/MyDrive/SPC-FedGNN/results/revision6/contribution/task_plan.csv | tail -n +2 | cut -d, -f10 | sort | uniq -c
```

Expected:

```text
     63 1
```

## Step 7. Independent gate-evidence diagnostics

This stage reserves 20% of local training evidence as a disjoint gate-evidence subset. The supervised update uses the remaining local training nodes. This directly tests whether the gate is overly dependent on the same labels used for supervised updates.

```python
!python3 experiments/revision6/run_revision6_experiments.py \
  --stage gate_independent \
  --root /content/drive/MyDrive/SPC-FedGNN/results/revision6 \
  --data-root /content/drive/MyDrive/SPC-FedGNN/data
```

Completion check:

```python
!cat /content/drive/MyDrive/SPC-FedGNN/results/revision6/gate_independent/task_plan.csv | tail -n +2 | cut -d, -f10 | sort | uniq -c
```

Expected:

```text
     18 1
```

## Step 8. Summaries

Each stage writes a `summary.csv`. To view compact summaries:

```python
!for f in /content/drive/MyDrive/SPC-FedGNN/results/revision6/*/summary.csv; do echo "==== $f"; cat "$f"; done
```

## Step 9. Package results

```python
%cd /content/drive/MyDrive/SPC-FedGNN
!zip -qr spc_fedgnn_revision6_results.zip results/revision6
```

Download:

```python
from google.colab import files
files.download("/content/drive/MyDrive/SPC-FedGNN/spc_fedgnn_revision6_results.zip")
```

## Notes

- These experiments are revision diagnostics. They should not overwrite previous main-result folders.
- If Colab disconnects, rerun the same command. Completed tasks with aggregated CSV files are skipped.
- The most important outputs are:
  - `results/revision6/fair_cb/summary.csv`;
  - `results/revision6/contribution/summary.csv`;
  - `results/revision6/gate_independent/summary.csv`;
  - all `task_plan.csv` files and logs.
