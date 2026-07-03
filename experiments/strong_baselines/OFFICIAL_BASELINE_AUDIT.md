# Official-code audit for high-value FGL baselines

Date checked: 2026-07-02

## FedGTA

- Paper: **FedGTA: Topology-aware Averaging for Federated Graph Learning**
- Venue record in arXiv text: PVLDB 17(1), 2023.
- Official/artifact code: `https://github.com/xkLi-Allen/FedGTA`
- Evidence: the paper page states that source code/data/artifacts are available at this GitHub repository. The repository README describes the FedGTA training command and environment.
- Compatibility note: the official repository uses its own data layout and example Louvain 10-client split for Cora/CiteSeer/PubMed. It should not be mixed directly into the current Dirichlet induced-subgraph main table unless we export compatible splits or clearly report it as an official-protocol reproduction.

## AdaFGL

- Paper: **AdaFGL: A New Paradigm for Federated Node Classification with Topology Heterogeneity**
- Paper status in arXiv page: accepted by ICDE 2024.
- Official/artifact code: `https://github.com/xkLi-Allen/AdaFGL`
- Evidence: the repository README describes AdaFGL, environment requirements, and `python main.py` training.
- Compatibility note: AdaFGL is a decoupled two-step method and the official repository uses its own dataset/split workflow. It is valuable as an official-code baseline, but protocol alignment must be documented carefully.

## FedIIH

- Paper: **Modeling Inter-Intra Heterogeneity for Graph Federated Learning**
- arXiv: `https://arxiv.org/abs/2412.11402`
- Official code status: no public official GitHub repository was found from targeted searches for `FedIIH`, the full paper title, and GitHub/code keywords.
- Baseline choice: use a clearly named `fediih_matched` unified-protocol approximation until official code is located. It must be described as a carefully matched implementation, not official-code FedIIH.

## Current implementation decision

To avoid interfering with the current revision-priority experiment batch, all strong-baseline work is isolated under:

- code: `experiments/strong_baselines/`
- output: `results/strong_baselines/`
- Colab package: `spc_fedgnn_strong_baselines_code_20260702.zip`

The matched baselines are deliberately named with `_matched`:

- `fedgta_matched`: topology-aware personalized aggregation using structural similarity under the current unified protocol.
- `adafgl_matched`: decoupled collaborative training with client-side personalization head under the current unified protocol.
- `fediih_matched`: functional-similarity personalization plus client-side personalization head under the current unified protocol.

These are not official-code results. Official FedGTA/AdaFGL results should be reported separately unless we fully adapt their repositories to the exact exported splits.
