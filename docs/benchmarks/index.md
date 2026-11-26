# Benchmarks & Cross-Disciplinary Gallery

All external benchmark suites are maintained under the Zero-Duplication Policy in `../ecowind-benchmarks` and `Data for Benchmark Gallery/` (excluded from Git).

## Reproducing Benchmarks

```bash
cd ../ecowind-benchmarks && python3 scripts/run_all_benchmarks.py
```

## Engineering Budgets (`v0.1.0`)

- **Wheel Size:** `< 5 MiB` across Linux, macOS, and Windows.
- **Peak Memory:** `<= 150.0 MiB` RSS on target workloads.
- **Numerical Parity:** `< 1e-5` vs. `lavaan` (CB-SEM) and `< 1e-11` vs. `cSEM` (PLS/PLSc).
