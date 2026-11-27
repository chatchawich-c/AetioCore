# AetioCore `v0.1.6` Benchmark Summary

- **Peak RSS:** `29.5 MiB` (Budget `<= 150.0 MiB`)

| Suite | Engine | p50 (ms) | p99 (ms) | Metric |
| :--- | :--- | :--- | :--- | :--- |
| PLSc-SEM (n=300) | `aetiocore-sem` | `0.159` | `13.238` | `converged` |
| WhiteBox Causal DL + do(LX) | `aetiocore-sem-dl` | `8.380` | `24.726` | `total_eff=0.950` |
| Tabular Outcome Proxy | `XGBoost` | `59.901` | `74.952` | `baseline` |
| Tabular Outcome Proxy | `NGBoost` | `688.510` | `846.310` | `baseline` |
