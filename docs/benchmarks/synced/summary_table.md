# AetioCore `v0.1.6` Benchmark Summary

- **Peak RSS:** `147.5 MiB` (Budget `<= 150.0 MiB`)

| Suite | Engine | Fit (ms) | Infer p50 (ms) | Infer p99 (ms) | Metric (Accuracy / Error / Guarantee) |
| :--- | :--- | ---: | ---: | ---: | :--- |
| do(LX)-Intervention | `AetioCore (PLSc-SEM Proxy)` | `130.888` | `0.219` | `0.570` | `Intervention_RMSE=0.1459, ATE_Error=0.0035, conv=1e-6` |
| do(LX)-Intervention | `XGBoost` | `182.400` | `0.429` | `0.768` | `Intervention_RMSE=1.0990, ATE_Error=0.5208` |
