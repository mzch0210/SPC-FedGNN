# SPC-FedGNN Review Artifact Checklist

Use this checklist before submitting the Neurocomputing review package.

## Code

- `experiments/run_benchmark.py`
- `experiments/colab_full_experiment.py`
- `experiments/strong_baselines/`
- `experiments/revision6/run_revision6_experiments.py`
- table and figure generation scripts
- Colab runbooks:
  - `COLAB_T4_REVISION6_RUNBOOK.md`
  - previous full/supplement runbooks used for the reported tables

## Environment

- Python version
- PyTorch version
- PyTorch Geometric version
- CUDA runtime shown by Colab
- Hardware note: Colab T4 15G for revision diagnostics unless otherwise stated
- Exact install cell used in Colab

## Data and Splits

- Dataset names and sources
- Data root layout
- Split-generation protocol:
  - Dirichlet partition parameter
  - number of clients
  - train/test split within each client
  - edge retention rule
  - random seeds
- Any fixed split files, if exported

## Result Files

- Seed-level client CSV files
- Per-seed aggregated CSV files
- Stage-level `task_plan.csv`
- Logs for failed or retried tasks
- Final summary CSV files
- Generated manuscript tables and figures

## Baseline Notes

- FedAvg-GCN and FedProx-GCN implementation details
- FedSSP implementation details
- FedATH-style caveat
- FedPub-style GCN caveat
- FedGTA/AdaFGL/FedIIH matched-baseline caveats
- Official-vs-matched audit table
- Revision-6 fair class-balanced baseline notes

## Reproducibility Checks

Before packaging, run:

```bash
python3 -m py_compile experiments/run_benchmark.py experiments/revision6/run_revision6_experiments.py
```

For each Colab result folder, check:

```bash
cat results/<stage>/task_plan.csv | tail -n +2 | cut -d, -f10 | sort | uniq -c
```

Expected: all tasks should be `1`.

## Manuscript Alignment

- Claims match matched/official baseline status.
- Coverage-aware reliability is described as diagnostic/reliability support.
- Communication is described as a trade-off, not as low communication.
- Full SPC-FedGNN is not claimed to significantly dominate Base Guard unless revision-6 results support it.
- Revision-6 results are added only after the returned result zip is analyzed.
