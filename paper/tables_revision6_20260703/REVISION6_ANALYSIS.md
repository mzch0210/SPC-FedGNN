# Revision-6 Result Analysis

Imported sources:

- `spc_fedgnn_revision6_results-a.zip`
- `spc_fedgnn_revision6_results-b.zip`

## Completion

All planned tasks completed: smoke 3/3, fair_cb 81/81, contribution 63/63, gate_independent 18/18.

## Fair class-balanced baselines

These results are a diagnostic fairness check under the revision-6 protocol, not a replacement for the original full main table.

| Dataset | SPC-FedGNN | Best external | Delta | SPC wins/losses/ties |
|---|---:|---|---:|---|
| amazon_photo | 0.1968 | fedath_cb 0.2434 | -0.0466 | 0/3/0 |
| cora | 0.1912 | fedath_cb 0.2062 | -0.0151 | 0/3/0 |
| pubmed | 0.3843 | fedpub_gcn_cb 0.3506 | +0.0337 | 2/1/0 |

Interpretation: class-balanced CE substantially strengthens several baselines. SPC-FedGNN remains clearly ahead on PubMed, but FedATH-CB is stronger on Cora and Amazon-Photo in this diagnostic batch. Therefore the paper should not claim that the main gain is independent of class balancing. It should present fair-CB baselines as an important caveat and shrink the superiority claim.

## Contribution disentanglement

| Dataset | Full SPC | Base Guard | No coverage delta | No class balance delta | Random candidates delta | No personalization delta | Direct ProtoAgg delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| amazon_photo | 0.2152 | 0.2136 | -0.0013 | -0.0262 | -0.0265 | -0.0580 | -0.0631 |
| cora | 0.1931 | 0.1937 | -0.0000 | -0.0391 | -0.0411 | -0.0791 | -0.0595 |
| pubmed | 0.3844 | 0.3884 | -0.0001 | -0.0872 | -0.0883 | -0.1485 | -0.0985 |

Interpretation: Base Guard is essentially tied with full SPC-FedGNN; coverage removal has almost no mean-Macro-F1 penalty; removing class balance, meaningful candidates, or personalization causes clear degradation. Direct prototype aggregation remains weak. This supports the revised contribution framing: guarded class-wise transfer, class-balanced training, and meaningful candidate retrieval are the main empirical sources; coverage-aware reliability is diagnostic/reliability support.

## Independent gate-evidence diagnostics

| Dataset | Method | Train-evidence gate | Independent gate | Delta |
|---|---|---:|---:|---:|
| amazon_photo | spc_guard | 0.2136 | 0.1919 | -0.0218 |
| amazon_photo | spc_guard_pp | 0.2152 | 0.1988 | -0.0163 |
| cora | spc_guard | 0.1937 | 0.1710 | -0.0227 |
| cora | spc_guard_pp | 0.1931 | 0.1698 | -0.0232 |
| pubmed | spc_guard | 0.3884 | 0.3666 | -0.0219 |
| pubmed | spc_guard_pp | 0.3844 | 0.3742 | -0.0102 |

Interpretation: reserving 20% of local training evidence for gate decisions causes moderate but not catastrophic drops. This reduces, but does not eliminate, concern that the gate only works by using the same evidence for supervised updates and teacher acceptance. Because this protocol also reduces supervised training evidence, the result should be described as a stricter diagnostic rather than a pure causal isolation of gate overfitting.

## Manuscript implications

1. Add a compact fairness-check table rather than merging these numbers into the original main table.
2. Make the contribution claim more explicit: full SPC-FedGNN is not consistently better than Base Guard; Base Guard captures most average-performance gain.
3. Keep coverage-aware reliability as reliability/diagnostic support.
4. Mention independent gate-evidence results as a diagnostic that the gate remains usable under stricter evidence separation, with moderate performance cost.
5. Avoid claiming that SPC-FedGNN is broadly superior to class-balanced strong baselines.
