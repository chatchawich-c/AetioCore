# API Reference (`v0.1.1`)

## 1. Tier 1: `semflow` (Classical CB-SEM, PLS/PLSc & Risk Flow)

| Function / Class | Signature | Description |
| :--- | :--- | :--- |
| `parse_spec` | `parse_spec(dsl_str: str) -> dict` | Validates `lavaan`-style DSL (`=~`, `<~`, `~`, `~~`) via Rust `sem-dsl` and returns structured equations. |
| `SEM` | `SEM(formula, method="pls", estimator="ml", n_boot=0, max_iter=500, tol=1e-6)` | Unified estimator supporting `method="cb"` (ML/RAM), `"pls"`, and `"plsc"` (Dijkstra $\rho_A$). |
| `fit_sem` | `fit_sem(spec, data, method="cb", ...)` | Functional wrapper returning a fitted `SEM` instance. |
| `fit_indices` | `fit_indices(fit_result) -> dict` | Returns `chi2`, `df`, `cfi`, `tli`, `rmsea`, `srmr`, `f_ml` (CB-SEM) or `rho_A`, `ave`, `cr`, `r2` (PLS/PLSc). |
| `propagate_risk` | `propagate_risk(fit_result, shocks: dict) -> dict` | Computes total causal shock propagation $\Delta\boldsymbol{\eta} = (\mathbf{I}-\mathbf{B})^{-1}\boldsymbol{\Gamma}\Delta\boldsymbol{\xi}$. |

## 2. Tier 2: `semflow_dl` (White-Box Causal Deep Learning)

| Function / Class | Signature | Description |
| :--- | :--- | :--- |
| `EdgeWhiteBoxSEM` | `EdgeWhiteBoxSEM(formula, knots=6, smooth_lambda=1e-4, max_iter=300)` | DAG-masked neural SEM with Monotonic I-Spline edge operators and Split-Conformal calibration. |
| `fit_whitebox_sem` | `fit_whitebox_sem(formula, df_train, df_cal=None, conformal_alpha=0.05)` | Trains `EdgeWhiteBoxSEM` and optionally calibrates finite-sample conformal intervals. |
| `intervene_do` | `intervene_do(model, df, do_var, delta, target) -> dict` | Executes Pearl's counterfactual `do(X)` mediation (Abduction $\to$ Action $\to$ Prediction) returning Total, Direct, and Indirect effects. |
