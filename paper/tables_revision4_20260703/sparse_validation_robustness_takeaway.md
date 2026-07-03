| Dataset | Method | 100% evidence | 5% evidence | Missing class | Worst delta | Takeaway |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| amazon_photo | spc_guard_pp | 0.2136 | 0.1919 | 0.1997 | -0.0217 | sensitive under most sparse evidence; report conservatively |
| amazon_photo | spc_guard | 0.1956 | 0.2127 | 0.2241 | +0.0029 | stable under sparse/missing validation evidence |
| cora | spc_guard_pp | 0.1914 | 0.1920 | 0.1946 | -0.0019 | stable under sparse/missing validation evidence |
| cora | spc_guard | 0.1936 | 0.1920 | 0.1949 | -0.0015 | stable under sparse/missing validation evidence |
| pubmed | spc_guard_pp | 0.3843 | 0.3893 | 0.3844 | -0.0003 | stable under sparse/missing validation evidence |
| pubmed | spc_guard | 0.3885 | 0.3883 | 0.3840 | -0.0045 | stable under sparse/missing validation evidence |
