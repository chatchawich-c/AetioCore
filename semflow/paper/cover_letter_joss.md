# Cover Letter / Submission Note to the Editors — JOSS

Dear JOSS Editorial Team,

We are pleased to submit our software paper, **"semflow: A Unified Rust-Backed Engine for High-Performance Structural Equation Modeling in Python and R"**, for consideration in the *Journal of Open Source Software* (JOSS).

## Overview & Scholarly Contribution

`semflow` is an open-source, pure-Rust Structural Equation Modeling (SEM) library providing zero-copy native bindings for both Python (`PyO3`/`maturin`) and R (`extendr`). Currently, the open-source SEM ecosystem is fragmented across languages: R hosts mature packages such as `lavaan` (CB-SEM) and `cSEM` (PLS-SEM/PLSc), while Python relies primarily on `semopy` (CB-SEM only). Researchers working across Python and R face subtle numerical discrepancies and severe memory/runtime bottlenecks when scaling to millions of observations or thousands of bootstrap resamples.

`semflow` resolves both challenges through a single unified pure-Rust engine (`sem-traits`, `sem-dsl`, `sem-autodiff`, `sem-pls`, `sem-cb`) operating on zero-copy `RowMajor` (Python) and `ColMajor` (R) memory views:

- **Numerical Parity:** Verified against `lavaan` (Wishart likelihood, observed information) within $< 5.7 \times 10^{-6}$ for parameter estimates (SE $< 1.7 \times 10^{-6}$) on `HolzingerSwineford1939` and `PoliticalDemocracy`, and against `cSEM` within $< 4.6 \times 10^{-12}$ for PLS/PLSc.
- **Scalability & Memory Efficiency:** On an 8 GB RAM consumer laptop ($N = 1{,}000{,}000, p = 12$), `semflow` CB-SEM completes in **146.93 ms** (**274.12 MB** peak RSS), achieving a **3.4× speedup** over `semopy` and a **17.6× speedup** over `lavaan`. At $N = 5{,}000{,}000$, `semflow` PLS (665.07 ms) and CB-SEM (1,599.57 ms) complete cleanly within 1 GB RSS while `lavaan` and `cSEM` exceed the 90 s timeout.
- **Parallel Bootstrapping:** With `rayon`-parallelized correlation-matrix PLS, `semflow` executes $B = 5{,}000$ bootstrap resamples ($N = 2{,}000$) in **0.97 s** versus **110.70 s** for `cSEM` (**114× speedup**, **3.8× lower peak RSS**).

## Repository & Availability

- **Repository:** `https://github.com/<your-github-username>/semflow`
- **Version / Archive DOI:** `v0.2.0` (`<Zenodo DOI>`)
- **License:** Apache-2.0
- **Installation:** `pip install semflow` (PyPI pre-built wheels) and R-universe (`semflow`)
- **Tests & Reproducibility:** Automated Rust unit tests (`cargo test`), Python golden parity tests (`benchmarks/test_parity.py`), and cross-language benchmark scripts (`benchmarks/bench_paper.py`) are included and verified via GitHub Actions CI.

Thank you for your time and consideration.

Sincerely,

Chatchawich Chaihong (ORCID: 0009-0000-7379-5738)  
Department of Interdisciplinary Engineering, Faculty of Engineering  
Prince of Songkla University, Hat Yai, Songkhla, Thailand  
Email: `chatchawich.work@gmail.com`
