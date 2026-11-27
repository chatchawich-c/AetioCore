# AetioCore `v0.1.6` Benchmark Summary

- **Peak RSS:** `196.1 MiB` (Budget `<= 150.0 MiB`)

| Suite | Engine | Fit (ms) | Infer p50 (ms) | Infer p99 (ms) | Metric (Accuracy / Error / Guarantee) |
| :--- | :--- | ---: | ---: | ---: | :--- |
| [Tier I: Synthetic] do(LX)-Intervention | `aetiocore-sem` | `7.841` | `0.007` | `0.008` | `Intervention_RMSE=0.1470, ATE_Error=0.0000, conv=1e-6` |
| [Tier I: Synthetic] do(LX)-Intervention | `aetiocore-sem-dl` | `21.583` | `0.006` | `0.007` | `Intervention_RMSE=0.1470, ATE_Error=0.0000` |
| [Tier I: Synthetic] do(LX)-Intervention | `XGBoost` | `70.610` | `0.402` | `0.542` | `Intervention_RMSE=1.0990, ATE_Error=0.5208` |
| [Tier I: Synthetic] do(LX)-Intervention | `NGBoost` | `590.652` | `18.460` | `25.760` | `Intervention_RMSE=4.4899, ATE_Error=3.4525` |
