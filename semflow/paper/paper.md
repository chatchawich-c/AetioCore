---
title: 'semflow: A Unified Rust-Backed Engine for High-Performance Structural Equation Modeling in Python and R'
tags:
  - Rust
  - Python
  - R
  - structural equation modeling
  - PLS-SEM
  - CB-SEM
  - zero-copy
authors:
  - name: Chatchawich Chaihong
    orcid: 0009-0000-7379-5738
    corresponding: true
    affiliation: 1
affiliations:
  - name: Department of Interdisciplinary Engineering, Faculty of Engineering, Prince of Songkla University, Hat Yai, Songkhla, Thailand
    index: 1
date: 22 November 2026
bibliography: paper.bib
---

# Summary

Structural Equation Modeling (SEM) is a foundational statistical framework for testing latent-variable measurement models and directed causal pathways across the social sciences, biostatistics, finance, and engineering. `semflow` is an open-source, high-performance SEM library written in pure Rust with zero-copy native bindings for both Python (`PyO3`/`maturin`) and R (`extendr`). By compiling a single unified numerical core to both ecosystems, `semflow` guarantees bit-for-bit identical parameter estimates across Python and R while scaling to millions of observations and thousands of bootstrap resamples on standard consumer laptops (e.g., 8 GB RAM, CPU-only).

`semflow` unifies Covariance-Based SEM (CB-SEM via Maximum Likelihood, Generalized Least Squares, Unweighted Least Squares, Pattern-Grouped Full Information Maximum Likelihood for missing data, and Regularized SEM) and Variance-Based SEM (PLS-SEM Mode A/B and Consistent PLS, $\text{PLSc}$) behind a single `lavaan`-compatible Domain-Specific Language (`sem-dsl`) supporting fixed parameters, equality constraints, and higher-order constructs.

# Statement of Need

The open-source SEM ecosystem is currently bifurcated along language and performance boundaries:

1. **Ecosystem Fragmentation:** The R ecosystem hosts mature statistical packages such as `lavaan` [@rosseel2012lavaan] for CB-SEM and `cSEM` [@rademaker2020csem] for PLS-SEM/PLSc, whereas Python users rely primarily on `semopy` [@igolkina2020semopy], which focuses on CB-SEM and lacks native PLS/PLSc support. Researchers collaborating across R and Python frequently encounter subtle numerical discrepancies due to differing optimizers, likelihood conventions, and missing-data handlers.
2. **Scalability and Memory Bottlenecks:** Modern telemetry, psychometric, and engineering datasets (e.g., wind turbine SCADA records or large-scale online assessments) routinely exceed $10^6$ observations. Existing interpreted implementations incur substantial memory overhead when copying tabular frames and become prohibitively slow during large-scale nonparametric bootstrapping (1,000–5,000 resamples).

`semflow` resolves both gaps by decoupling statistical computation from the host language. All linear algebra (`faer`), reverse-mode automatic differentiation (`sem-autodiff`), and parallel resampling (`rayon`) execute in pure Rust over zero-copy memory views (`RowMajor` for Python NumPy/Polars and `ColMajor` for R matrices).

# Architecture and Mathematical Formulation

`semflow` is structured as a modular Rust workspace (`sem-traits`, `sem-dsl`, `sem-autodiff`, `sem-pls`, `sem-cb`) exposed through thin FFI crates (`sem-py`, `r-pkg`).

## Unified RAM CB-SEM, FIML, and RegSEM

To support confirmatory factor analysis (CFA), full structural models, higher-order constructs, and latent growth models in a single solver, `sem-cb` adopts the Reticular Action Model (RAM) formulation [@mcardle1984ram]:

$$\boldsymbol{\Sigma}(\boldsymbol{\theta}) = \mathbf{F}(\mathbf{I}-\mathbf{A})^{-1}\mathbf{S}_{\text{ram}}(\mathbf{I}-\mathbf{A})^{-\top}\mathbf{F}^\top$$

where $\mathbf{A} \in \mathbb{R}^{m \times m}$ encodes asymmetric paths (`=~`, `~`), $\mathbf{S}_{\text{ram}} \in \mathbb{R}^{m \times m}$ encodes symmetric variances and covariances (`~~`), and $\mathbf{F} = [\mathbf{I}_p \mid \mathbf{0}]$ selects the $p$ observed indicators. Variances are parameterized via $\sigma_{ii}^2 = \ln(1 + e^{\tilde{\theta}_i})$ to preserve positive definiteness throughout L-BFGS optimization, with asymptotic standard errors recovered via the observed information matrix and the delta method.

`sem-cb` evaluates analytical gradients $\nabla_{\boldsymbol{\Sigma}} F$ for ML, GLS, and ULS, and propagates them through `sem-autodiff`. For incomplete data, observations are grouped into $G$ missingness patterns $O_g \subseteq \{1,\dots,p\}$ to compute pattern-grouped FIML:

$$F_{\text{FIML}}(\boldsymbol{\theta}, \boldsymbol{\mu}) = \frac{1}{N}\sum_{g=1}^{G} n_g \left[\log|\boldsymbol{\Sigma}_{O_g}| + \operatorname{tr}\!\left(\boldsymbol{\Sigma}_{O_g}^{-1}\mathbf{T}_g\right)\right], \quad \mathbf{T}_g = \mathbf{S}_g + (\bar{\mathbf{y}}_g - \boldsymbol{\mu}_{O_g})(\bar{\mathbf{y}}_g - \boldsymbol{\mu}_{O_g})^\top$$

paired with a pattern-level EM saturated baseline to report exact FIML $\chi^2$, CFI, TLI, and RMSEA. Optional Regularized SEM (RegSEM) augments $F(\boldsymbol{\theta})$ with smooth-L1 (Charbonnier, $\epsilon = 10^{-6}$) and L2 penalties on structural paths.

## Correlation-Matrix PLS-SEM and Consistent PLS (PLSc)

Rather than iterating over $N \times p$ data matrices inside Wold's fixed-point loop, `sem-pls` computes the sample correlation matrix $\mathbf{R} \in \mathbb{R}^{p \times p}$ in a single parallel pass (supporting listwise, pairwise, and mean missing-data handling) and executes Mode A (reflective) and Mode B (formative) weight updates directly on $\mathbf{R}$, reducing complexity from $O(T N p)$ to $O(N p^2 + T p^2)$. For reflective models, `method="plsc"` applies Dijkstra's $\rho_A$ disattenuation [@dijkstra2015consistent] to produce consistent common-factor path estimates. Parallel bootstrapping across CPU cores is executed via `rayon`.

# Verification and Benchmarks

Numerical accuracy was verified against `lavaan` (Wishart likelihood, observed information) and `cSEM` (centroid inner weighting) on canonical datasets (`HolzingerSwineford1939` and `PoliticalDemocracy`):

| Benchmark Dataset & Mode | Reference | Max $|\Delta \text{Estimate}|$ | Max $|\Delta \text{SE}|$ |
|---|---|---:|---:|
| `HolzingerSwineford1939` (CB-SEM ML) | `lavaan` | $3.26 \times 10^{-6}$ | $5.00 \times 10^{-7}$ |
| `PoliticalDemocracy` (CB-SEM ML) | `lavaan` | $5.62 \times 10^{-6}$ | $1.65 \times 10^{-6}$ |
| `PoliticalDemocracy` (PLS Mode A) | `cSEM` | $4.18 \times 10^{-12}$ | — |
| `PoliticalDemocracy` (PLSc) | `cSEM` | $4.30 \times 10^{-12}$ | — |
| Python vs. R Cross-Language (FIML 15% NA) | `semflow` (R) | $< 10^{-12}$ | $< 10^{-12}$ |

Isolated subprocess benchmarks on an 8 GB RAM Intel MacBook Pro (2017) demonstrate substantial speedups and memory reductions across row scaling, high-dimensional models, and parallel bootstrapping (Table 2 and Figure 1). At $N = 5{,}000{,}000$, both `lavaan` and `cSEM` exceed the 90 s timeout while consuming $>2.1$–$3.25$ GB RSS, whereas `semflow` completes in $0.67$–$1.60$ s within $1$ GB RSS.

| Benchmark Scenario | Engine | Fit Time | Peak RSS (MB) | Speedup vs. Baseline |
|---|---|---:|---:|---|
| $N=1\text{M}, p=12$ (CB-SEM) | **`semflow` (Py CB)** | **146.93 ms** | **274.12** | **3.4×** (`semopy`: 501.46 ms), **17.6×** (`lavaan`: 2,586 ms) |
| $N=1\text{M}, p=12$ (PLS) | **`semflow` (Py PLS)** | **122.13 ms** | **273.83** | **24.5×** (`cSEM`: 2,988 ms, 1,006.50 MB) |
| $N=5\text{M}, p=12$ (CB / PLS) | **`semflow` (CB / PLS)** | **1.60 s / 0.67 s** | **892.82 / 962.23** | `lavaan`/`cSEM` > 90 s TIMEOUT; `semopy` 3.61 s, 2,996.27 MB |
| $N=100\text{K}, p=60$ (PLS) | **`semflow` (Py PLS)** | **128.54 ms** | **194.40** | **28.7×** (`cSEM`: 3,689 ms, 569.53 MB), **2.93×** less RAM |
| $N=2\text{K}, B=5\text{K}$ (Boot) | **`semflow` (Py PLS)** | **0.97 s** | **56.19** | **114×** (`cSEM`: 110.70 s, 216.12 MB) |
| $N=10\text{K}, B=5\text{K}$ (Boot) | **`semflow` (Py PLS)** | **4.45 s** | **59.40** | `cSEM` > 180 s TIMEOUT (189.67 MB) |

![Scalability, memory footprint, and parallel bootstrap benchmarks comparing `semflow` against `semopy`, `lavaan`, and `cSEM`. In panels (a)–(b), `lavaan` and `cSEM` terminate at $N = 10^6$ because both exceed the 90 s timeout at $N = 5 \times 10^6$.](../benchmarks/figures/fig_semflow_benchmarks.png){#fig:benchmarks}

# Acknowledgements

This work was supported in part by the European Union Erasmus+ Capacity Building in Higher Education (CBHE) framework under the ECO-WIND project (Grant No. ERASMUS-EDU-2025-CBHE-101177220).

# References
