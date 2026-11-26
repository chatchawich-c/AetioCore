# Cover Letter — JMLR (Machine Learning Open Source Software Track)

Dear Editors-in-Chief and MLOSS Action Editors of the *Journal of Machine Learning Research*,

We are pleased to submit our manuscript, **"semflow-dl: Edge-Level White-Box Causal Deep Learning with Monotonic I-Splines and Conformal Guarantees"**, for consideration in the **Machine Learning Open Source Software (MLOSS)** track of JMLR.

## Significance and Software Contribution

Contemporary interpretable tabular machine learning faces a fundamental tension between flat glass-box/post-hoc models (`EBM`, `NAM`, `XGBoost+SHAP`) and classical Structural Equation Modeling (`SEM`). Flat models ignore directed causal structure among noisy multi-indicator constructs, causing severe *mediator blocking* under intervention ($\mathrm{AIE}_{\mathrm{Indirect}} \equiv 0$) and non-monotonic oscillations in sparse distribution tails. Conversely, classical SEM models latent causal graphs but is restricted to linear equations and lacks distribution-free uncertainty quantification.

`semflow-dl` bridges this gap as an open-source, pure-Rust edge-level white-box causal deep learning library with zero-copy Python bindings (`PyO3`/`maturin`), built as a modular external plugin on top of the `semflow` Rust foundation:

1. **Guaranteed Smooth Monotonicity & Exact Local Elasticity:** Structural DAG edges are parameterized with Ramsay's monotonic I-Splines (`order >= 3`), guaranteeing $0.0\%$ monotonicity violations everywhere on $\mathbb{R}$ and providing closed-form local causal elasticity $\phi'(z)$ via exact M-Spline derivatives.
2. **Exact Counterfactual Decomposition:** Evaluates Pearl's $\mathrm{do}(\eta_i + \Delta)$ interventions via abduction–action–prediction along the topological order of the DAG, decomposing total effects into exact direct and indirect contributions without approximation error.
3. **Distribution-Free Uncertainty & Extensions:** Integrates finite-sample split-conformal calibration, ANOVA-purified 2D interactions, and sparse CSR spatial/network coupling ($\rho = \tanh(\tilde{\rho})$).
4. **Empirical Advantage & Efficiency:** Across controlled nonlinear mediation benchmarks ($N = 10^4, 10^5$) on an 8 GB RAM CPU-only laptop, `semflow-dl` reduces $\mathrm{do}(\mathrm{Load} + 1\,\mathrm{SD})$ intervention error by over **5×** compared to `EBM` and `XGBoost+SHAP` ($0.082\text{--}0.094$ vs.\ $0.472\text{--}0.517$), eliminates tail monotonicity violations ($0.0\%$ vs.\ $1.0\%\text{--}33.2\%$), and consumes **2.4×–2.9× less peak memory** ($58\text{--}87\text{ MB}$ RSS).

## Open-Source Availability & Reproducibility

- **Source Repository:** `https://github.com/<your-github-username>/semflow-dl` (Apache-2.0 License)
- **PyPI Package:** `pip install semflow-dl` (`v0.1.0`)
- **Reproducibility:** All unit tests (`cargo test -p semflow-dl-core`), case studies (`examples/case_study_scada.py`), benchmark scripts (`benchmarks/bench_whitebox_vs_ebm_shap.py`), and publication figure scripts (`benchmarks/plot_publication_figure_4panel.py`) are openly provided.

We confirm that this manuscript is original, has not been published elsewhere, and is not under consideration by another journal. Thank you for considering our submission to the JMLR MLOSS track.

Sincerely,

Chatchawich Chaihong (ORCID: 0009-0000-7379-5738)  
Department of Interdisciplinary Engineering, Faculty of Engineering  
Prince of Songkla University, Hat Yai, Songkhla, Thailand  
Email: `chatchawich.work@gmail.com`
