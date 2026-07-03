# Communication and runtime analysis

Source zip: `spc_fedgnn_revision4_comm_runtime_account_b.zip`

Imported directory:
`results/revision4_extra_comm_import_20260703/account_b`

## Completion

- Planned tasks: 42
- Completed tasks: 42
- Aggregated result files: 42
- Datasets: Cora, CiteSeer, PubMed, Amazon-Photo, Amazon-Computers, Roman-empire, Tolokers
- Methods: SPC-FedGNN, FedATH-style, FedSSP, FedPub-style, FedAvg-GCN, FedProx-GCN
- Seed: 0

## Main observations

1. Descriptor/candidate retrieval communication is genuinely compact.
   - SPC-FedGNN uses 0.56 KB descriptor upload plus 0.24 KB candidate-index download per run summary, i.e. 0.80 KB total retrieval overhead.
   - This supports the narrow claim that the non-model retrieval signal is compact.

2. The complete guarded-distillation protocol is not communication-cheap.
   - Teacher-state transfer dominates the extra cost.
   - With up to three teacher models evaluated per client, the teacher-state extra term is about three model downloads.
   - Total counted traffic is about 2.5x FedAvg-GCN across all seven datasets.

3. Runtime overhead is material but bounded in this Colab/T4 audit.
   - SPC-FedGNN runtime is 3.02x to 4.98x FedAvg-GCN.
   - Mean runtime ratio vs FedAvg-GCN is 3.90x.
   - SPC-FedGNN is the slowest method on five datasets. FedATH-style is slower on Amazon-Computers and Tolokers.

4. The current manuscript communication table should be replaced or expanded.
   - The previous table used only Cora/CiteSeer/PubMed examples.
   - The new table covers all seven datasets under a consistent seed-0 Colab/T4 audit.
   - The revised manuscript should continue to state that the descriptor is compact but the full protocol trades additional teacher-state traffic and runtime for safer transfer.

## Compact dataset-level table

| Dataset | SPC sec | FedAvg sec | Runtime x | Retrieval KB | Teacher MB | Total MB | Traffic x |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Cora | 58.4 | 14.0 | 4.17 | 0.80 | 1328.13 | 2213.54 | 2.50 |
| CiteSeer | 69.7 | 17.6 | 3.97 | 0.80 | 3419.22 | 5698.70 | 2.50 |
| PubMed | 56.5 | 14.5 | 3.91 | 0.80 | 464.53 | 774.22 | 2.50 |
| Amazon-Photo | 59.2 | 16.7 | 3.55 | 0.80 | 695.00 | 1158.34 | 2.50 |
| Amazon-Computers | 77.1 | 20.9 | 3.68 | 0.80 | 717.15 | 1195.25 | 2.50 |
| Roman-empire | 69.3 | 13.9 | 4.98 | 0.80 | 294.25 | 490.42 | 2.50 |
| Tolokers | 84.7 | 28.0 | 3.02 | 0.80 | 12.01 | 20.02 | 2.50 |

## Recommended manuscript wording

> A seed-0 Colab/T4 audit over all seven datasets separates compact retrieval traffic from model-state traffic. The descriptor and candidate-index overhead of SPC-FedGNN is only 0.80 KB in the current implementation. However, this is not the total communication cost: because the target client evaluates up to three candidate teachers, teacher-state transfer contributes about three additional model downloads. Consequently, the counted total traffic is about 2.5x FedAvg-GCN, and wall-clock time is 3.0x--5.0x FedAvg-GCN. We therefore interpret SPC-FedGNN as a method that trades additional teacher-state communication and runtime for safer guarded transfer, rather than as a low-communication method.

## Generated artifacts

- `communication_runtime_rev4_full.csv`
- `communication_runtime_rev4_method_clean.csv`
- `communication_runtime_rev4_dataset_compact.csv`
- `communication_runtime_rev4_dataset_compact.md`
- `communication_runtime_rev4_summary.txt`
- `manuscript_neurocomputing/figures/communication_runtime_tradeoff.pdf`
- `manuscript_neurocomputing/figures/communication_runtime_tradeoff.png`

