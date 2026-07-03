# SPC-FedGNN Reproducibility Artifact

This repository contains the reproducibility artifact for
`SPC-FedGNN: Guarded Structural Prototype Collaboration for Robust Federated Graph Learning`.

The artifact is intended for peer review. It includes the experimental code,
Colab runbooks, figure-generation scripts, and processed result tables used by
the manuscript. It does not include raw private data; all graph datasets used in
the experiments are public benchmark datasets downloaded by PyTorch Geometric.

## Repository Layout

- `experiments/`: core benchmark runner and Colab experiment orchestration.
- `experiments/spc_fedgnn/`: data loading, models, and metrics.
- `experiments/strong_baselines/`: matched recent-baseline experiments and
  official-vs-matched protocol audit.
- `experiments/revision6/`: fairness and gate-validity diagnostic experiments.
- `scripts/`: figure-generation scripts.
- `paper/`: processed CSV tables and analysis notes used by the manuscript.
- `manuscript_neurocomputing/figures/`: generated manuscript figures.
- `requirements.txt`: Python package requirements.

## Environment

The main experiments were run on Google Colab T4 GPUs. A typical setup is:

```bash
python --version
pip install -r requirements.txt
```

For Colab, install PyTorch and PyTorch Geometric wheels matching the active CUDA
runtime before installing the remaining packages. The runbooks under
`experiments/` provide step-by-step Colab cells.

## Quick Smoke Test

```bash
python experiments/run_benchmark.py \
  --dataset cora \
  --partition dirichlet_0.1 \
  --clients 20 \
  --seed 0 \
  --method spc_guard_pp \
  --rounds 5 \
  --local-epochs 1 \
  --hidden-dim 64 \
  --data-root data \
  --output results/smoke_cora_spc_guard_pp.csv \
  --run-tag smoke
```

Expected output files:

- `results/smoke_cora_spc_guard_pp.csv`
- `results/smoke_cora_spc_guard_pp_aggregated.csv`

## Main Experiment Entry Points

- Full unified-protocol experiments:
  `experiments/colab_full_experiment.py`
- Strong matched baselines:
  `experiments/strong_baselines/run_strong_baselines.py`
- Sparse-validation and communication/runtime diagnostics:
  `experiments/COLAB_T4_REVISION4_EXTRA_RUNBOOK.md`
- Fair class-balanced and gate-evidence diagnostics:
  `experiments/revision6/run_revision6_experiments.py`

The most useful step-by-step runbooks are:

- `experiments/COLAB_STEP_BY_STEP_RUNBOOK.md`
- `experiments/COLAB_T4_NEUROCOMPUTING_SUPPLEMENT.md`
- `experiments/strong_baselines/COLAB_T4_STRONG_BASELINES_RUNBOOK.md`
- `experiments/COLAB_T4_REVISION4_EXTRA_RUNBOOK.md`
- `experiments/COLAB_T4_REVISION6_RUNBOOK.md`

## Result Tables

Processed manuscript tables are provided under `paper/`:

- `paper/tables_20260701/`: primary seven-dataset results, ablations,
  heterogeneity stress tests, sensitivity, coverage robustness, and efficiency.
- `paper/tables_strong_baselines_20260702/`: matched FedGTA/AdaFGL/FedIIH-style
  baseline summaries.
- `paper/tables_revision4_20260703/`: sparse-validation and
  communication/runtime diagnostics.
- `paper/tables_revision6_20260703/`: class-balanced fairness and independent
  gate-evidence diagnostics.

## Baseline Caveat

FedATH-style, FedPub-style, FedGTA-matched, AdaFGL-matched, and FedIIH-matched
rows are implemented under a unified evaluation protocol to isolate mechanism
differences. They should not be interpreted as official-code leaderboard
results. The official-vs-matched audit is included in
`paper/tables_revision4_20260702/official_vs_matched_baseline_audit.md`.

## Rebuilding Figures

```bash
python scripts/make_revision_figures.py
```

Generated figures are written to `manuscript_neurocomputing/figures/`.

## Anonymity Note

This artifact is prepared for anonymous review. Please do not add author names,
institutional email addresses, local machine paths, SSH configuration, or
private Google Drive links before uploading it to an anonymous repository.
