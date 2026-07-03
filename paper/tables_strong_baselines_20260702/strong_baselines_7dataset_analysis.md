# Seven-dataset strong-baseline analysis

## Completion

- Previous strong-baseline batch: core, extension, and Tolokers completed.
- Amazon-Photo supplement: 9 aggregated result files imported.
- Seven datasets now covered: Amazon-Computers, Amazon-Photo, CiteSeer, Cora, PubMed, Roman-empire, and Tolokers.

## Best matched strong baseline by dataset

| Dataset | Best matched strong baseline | Strong Macro-F1 | SPC-FedGNN Macro-F1 | SPC - strong | SPC wins/losses/ties |
|---|---:|---:|---:|---:|---:|
| amazon_computers | adafgl_matched | 0.2011 ± 0.0330 | 0.1958 | -0.0053 | 1/2/0 |
| amazon_photo | adafgl_matched | 0.2166 ± 0.0420 | 0.2191 | 0.0025 | 1/2/0 |
| citeseer | adafgl_matched | 0.2164 ± 0.0260 | 0.2279 | 0.0115 | 3/0/0 |
| cora | fediih_matched | 0.1960 ± 0.0050 | 0.1910 | -0.0051 | 1/2/0 |
| pubmed | fediih_matched | 0.3396 ± 0.0991 | 0.3844 | 0.0447 | 3/0/0 |
| roman_empire | adafgl_matched | 0.2047 ± 0.0087 | 0.2084 | 0.0037 | 3/0/0 |
| tolokers | fedgta_matched | 0.3749 ± 0.0275 | 0.3985 | 0.0236 | 2/0/1 |

## Interpretation

- Against the best matched strong baseline on each dataset, SPC-FedGNN wins 5/7 datasets by mean Macro-F1 and loses 2/7.
- The mean-level losses are Cora and Amazon-Computers. FedIIH-matched is slightly stronger on Cora, while AdaFGL-matched is stronger on Amazon-Computers.
- Amazon-Photo is a borderline case: SPC-FedGNN is slightly higher in mean Macro-F1 than AdaFGL-matched, but the seed-level comparison is 1/2/0. It should be described as competitive rather than clearly superior.
- SPC-FedGNN remains stronger on CiteSeer, PubMed, Roman-empire, and Tolokers.
- These results strengthen the paper by adding recent matched personalized/topology-aware competitors, but they require a conservative main claim. The paper should not state that SPC-FedGNN is the best on all datasets once these strong baselines are included.
- Recommended positioning: SPC-FedGNN achieves the best or competitive performance on most datasets and remains particularly strong under severe label skew on CiteSeer, PubMed, Roman-empire, and Tolokers, while recent matched personalized baselines are competitive or stronger on Cora and Amazon product graphs.
