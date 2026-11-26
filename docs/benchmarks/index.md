# Benchmarks & Cross-Disciplinary Gallery (`v0.1.4`)

- **Architecture**: External benchmark suite maintained in [`ecowind-benchmarks`](https://github.com/chatchawich-c/ecowind-benchmarks) under the **Zero-Duplication Policy**.
- **Sync Status**: Synced from verified `ecowind-benchmarks` baseline (`v0.1.4`).
- **Funding**: European Union Erasmus+ CBHE **ECO-WIND** (Grant No. `ERASMUS-EDU-2025-CBHE-101177220`).

## Release v0.1.4 Runtime & Memory Footprint

| Metric | Target Budget | `AetioCore-SEM` / `AetioCore-DL` (`v0.1.4`) | Status |
| :--- | :--- | :--- | :--- |
| **Stripped Wheel Size** | `< 5.0 MiB` | `0.71 MiB` (`semflow`) / `0.35 MiB` (`semflow-dl`) | Pass |
| **Peak Memory RSS** | `<= 150.0 MiB` | `19.76 MiB` (base) / `86.8 MiB` ($N=10^5$ DL) | Pass |
| **CB-SEM Numerical Parity** | `< 1e-5` vs. `lavaan` | `3.26e-6` (HS1939) / `5.62e-6` (PolDem) | Pass |
| **PLS/PLSc Parity** | `< 1e-11` vs. `cSEM` | `4.18e-12` (PLS) / `4.30e-12` (PLSc) | Pass |
| **Monotonicity Guarantee** | `0.0%` violations | `0.0%` across full domain (I-Spline) | Pass |

## How to Run External Benchmarks

```bash
cd ../ecowind-benchmarks && python3 scripts/run_all_benchmarks.py
```
