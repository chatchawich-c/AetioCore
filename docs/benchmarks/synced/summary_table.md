# AetioCore `v0.1.6` Benchmark Summary

- **Zenodo DOI:** [`10.5281/zenodo.24568459`](https://doi.org/10.5281/zenodo.24568459)
- **Comparative Harness Peak RSS:** `196.1 MiB` (includes co-loaded Python baselines: `pandas`, `pyarrow`, `xgboost`, `lightgbm`, `catboost`, `ngboost`)
- **Standalone Pure-Rust Engine Footprint:** `28.86 – 30.13 MiB` (~5x below the `150.0 MiB` budget)

## 1. Breakdown by Dataset

### [Tier I: Synthetic] do(LX)-Intervention

| Engine | Fit (ms) | Infer p50 (ms) | Infer p99 (ms) | Metric (Accuracy / Error / Guarantee) |
| :--- | ---: | ---: | ---: | :--- |
| `aetiocore-sem` | `7.841` | `0.007` | `0.008` | `Intervention_RMSE=0.1470, ATE_Error=0.0000, conv=1e-6` |
| `aetiocore-sem-dl` | `21.583` | `0.006` | `0.007` | `Intervention_RMSE=0.1470, ATE_Error=0.0000` |
| `XGBoost` | `70.610` | `0.402` | `0.542` | `Intervention_RMSE=1.0990, ATE_Error=0.5208` |
| `NGBoost` | `590.652` | `18.460` | `25.760` | `Intervention_RMSE=4.4899, ATE_Error=3.4525` |

## 2. Breakdown by Model / Engine

### `aetiocore-sem`

| Dataset / Suite | Fit (ms) | Infer p50 (ms) | Infer p99 (ms) | Metric (Accuracy / Error / Guarantee) |
| :--- | ---: | ---: | ---: | :--- |
| [Tier I: Synthetic] do(LX)-Intervention | `7.841` | `0.007` | `0.008` | `Intervention_RMSE=0.1470, ATE_Error=0.0000, conv=1e-6` |

### `aetiocore-sem-dl`

| Dataset / Suite | Fit (ms) | Infer p50 (ms) | Infer p99 (ms) | Metric (Accuracy / Error / Guarantee) |
| :--- | ---: | ---: | ---: | :--- |
| [Tier I: Synthetic] do(LX)-Intervention | `21.583` | `0.006` | `0.007` | `Intervention_RMSE=0.1470, ATE_Error=0.0000` |

### `XGBoost`

| Dataset / Suite | Fit (ms) | Infer p50 (ms) | Infer p99 (ms) | Metric (Accuracy / Error / Guarantee) |
| :--- | ---: | ---: | ---: | :--- |
| [Tier I: Synthetic] do(LX)-Intervention | `70.610` | `0.402` | `0.542` | `Intervention_RMSE=1.0990, ATE_Error=0.5208` |

### `NGBoost`

| Dataset / Suite | Fit (ms) | Infer p50 (ms) | Infer p99 (ms) | Metric (Accuracy / Error / Guarantee) |
| :--- | ---: | ---: | ---: | :--- |
| [Tier I: Synthetic] do(LX)-Intervention | `590.652` | `18.460` | `25.760` | `Intervention_RMSE=4.4899, ATE_Error=3.4525` |

