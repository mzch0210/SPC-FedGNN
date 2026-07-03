# Sparse-validation robustness analysis

Source zip: `spc_fedgnn_revision4_sparse_validation_account_a.zip`

Imported directory:
`results/revision4_extra_sparse_import_20260703/spc_fedgnn_revision4_sparse_validation_account_a`

## Completion

- Planned tasks: 90
- Completed tasks: 90
- Aggregated result files: 90
- Datasets: Cora, PubMed, Amazon-Photo
- Methods: SPC-FedGNN (`spc_guard_pp`) and Base Guard (`spc_guard`)
- Evidence settings: 100%, 20%, 10%, 5%, and 20% with one dropped local class.

## Main observations

1. Cora and PubMed are stable under sparse or missing validation evidence.
   - SPC-FedGNN worst delta vs 100% evidence: -0.0019 on Cora and -0.0003 on PubMed.
   - Base Guard worst delta vs 100% evidence: -0.0015 on Cora and -0.0045 on PubMed.

2. Amazon-Photo is the only sensitive setting for full SPC-FedGNN.
   - SPC-FedGNN drops from 0.2136 to 0.1919 at 5% validation evidence.
   - Worst delta is -0.0217, so this setting should be reported conservatively.
   - Base Guard remains stable and even improves in the sparse/missing validation settings.

3. The results support a diagnostic robustness claim, not a strong universal robustness claim.
   - Suitable claim: SPC-FedGNN remains stable on Cora and PubMed, while Amazon-Photo reveals sensitivity under extremely sparse local validation evidence.
   - Avoid claiming that the full method is uniformly robust to validation sparsity.

4. The full method should not be claimed to dominate Base Guard in this experiment.
   - Base Guard is comparable or better on most sparse settings.
   - This reinforces the current manuscript position that Base Guard captures much of the empirical benefit, while the full method provides a more complete class-conditional and coverage-aware diagnostic framework.

## Generated artifacts

- `sparse_validation_summary.csv`
- `sparse_validation_summary.md`
- `sparse_validation_full_vs_base.csv`
- `sparse_validation_guard_diagnostics_clean.csv`
- `sparse_validation_robustness_takeaway.csv`
- `sparse_validation_robustness_takeaway.md`
- `manuscript_neurocomputing/figures/sparse_validation_robustness.pdf`
- `manuscript_neurocomputing/figures/sparse_validation_robustness.png`

## Recommended manuscript use

Place this result in the robustness or diagnostics part of the Experiments section, preferably after coverage-aware robustness.

Recommended wording:

> Sparse-validation diagnostics show that both SPC-FedGNN and Base Guard are stable on Cora and PubMed when the local evidence used by the gate is reduced to 5% or when one local class is hidden from the gate. Amazon-Photo is more sensitive: the full SPC-FedGNN variant loses about 0.022 Macro-F1 at 5% evidence, whereas Base Guard remains stable. We therefore interpret sparse-validation robustness as diagnostic evidence for the gated collaboration mechanism rather than as a universal robustness guarantee.

