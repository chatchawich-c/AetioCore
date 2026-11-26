# semflow

[![CI](https://github.com/<your-username>/semflow/actions/workflows/ci.yml/badge.svg)](https://github.com/<your-username>/semflow/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

**semflow** is a high-performance, Rust-backed Structural Equation Modeling (SEM) library with zero-copy native bindings for both **Python** (`PyO3`/`maturin`) and **R** (`extendr`).

By executing all linear algebra (`faer`), reverse-mode automatic differentiation (`sem-autodiff`), and parallel resampling (`rayon`) in a single pure-Rust engine, `semflow` delivers **bit-for-bit identical estimates across Python and R** while scaling to millions of observations and thousands of bootstrap resamples on 8 GB RAM laptops.

## Key Features

- **Unified `lavaan+` DSL (`sem-dsl`):** Supports reflective (`=~`), formative (`<~`), structural (`~`), residual covariance (`~~`), fixed parameters (`1*y1`), equality labels (`b*x1`), and higher-order constructs.
- **CB-SEM (`sem-cb`):** Reticular Action Model (RAM) solver with ML, GLS, ULS, pattern-grouped **FIML** (missing data), and **RegSEM** (smooth-L1/L2 penalties).
- **PLS-SEM & PLSc (`sem-pls`):** Correlation-matrix Wold algorithm (Mode A & Mode B), Dijkstra's $\rho_A$ disattenuation (`PLSc`), AVE/CR/HTMT, and parallel `rayon` bootstrapping.
- **Zero-Copy Dual-Language Core:** Direct `RowMajor` (`float32`/`float64`) views for Python (Polars/NumPy/Pandas) and `ColMajor` (`float64`) views for R matrices.

## Installation

### Python

```bash
pip install semflow
```

### R (via R-universe)

```r
install.packages(
  "semflow",
  repos = c("https://<your-username>.r-universe.dev", "https://cloud.r-project.org")
)
```

## Quickstart

### Python

```python
import polars as pl
import semflow as sf

formula = """
    visual  =~ x1 + x2 + x3
    textual =~ x4 + x5 + x6
    speed   =~ x7 + x8 + x9
"""

df = pl.read_csv("benchmarks/golden/HolzingerSwineford1939.csv")
fit = sf.SEM(formula, method="cb", estimator="ml").fit(df)
print(fit.summary())
```

### R

```r
library(semflow)
fit <- sem(formula, data = df, method = "cb", estimator = "ml")
print(fit)
```

## Verification & Performance Highlights

* **Numerical Parity:** Matches `lavaan` within $< 5.7 \times 10^{-6}$ (SE $< 1.7 \times 10^{-6}$) on `HolzingerSwineford1939` and `PoliticalDemocracy`, and matches `cSEM` within $< 4.6 \times 10^{-12}$ on PLS/PLSc.
* **Scalability ($N = 1{,}000{,}000, p = 12$):** CB-SEM fits in **149 ms** (268 MB RSS) — **3.6× faster** and **3.05× less RAM** than `semopy`.
* **Parallel Bootstrap ($B = 5{,}000$):** **83×–122× faster** than `cSEM` with nearly **4× lower RAM**.

## Companion Ecosystem

* **`semflow-dl`**: External white-box causal deep learning plugin (Monotonic I-Splines, exact local elasticity, Pearl's `do(X)` intervention, and Split-Conformal intervals).

## Citation

If you use `semflow` in academic work, please cite `paper/paper.md` (Zenodo DOI pending release).

## License

Licensed under the Apache License, Version 2.0.
