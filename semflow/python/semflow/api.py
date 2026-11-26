from __future__ import annotations
import numpy as np
from . import _core


def _extract_array(df, prefer_f32: bool = False) -> tuple[np.ndarray, list[str]]:
    if hasattr(df, "to_numpy") and hasattr(df, "columns"):
        cols = list(df.columns)
        try:
            arr = df.to_numpy(writable=False)
        except TypeError:
            arr = df.to_numpy()
    else:
        raise TypeError("Expected a Polars or Pandas DataFrame")
    target_dtype = np.float32 if prefer_f32 else np.float64
    if arr.dtype != target_dtype:
        arr = arr.astype(target_dtype)
    return np.ascontiguousarray(arr), cols


def parse_spec(dsl_str: str) -> dict:
    """Validate lavaan-style DSL via Rust sem-dsl and return structured equations."""
    _core.validate_dsl(dsl_str)
    meas, reg, cov = [], [], []
    for raw in dsl_str.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        for op, bucket in (("=~", meas), ("<~", meas), ("~~", cov), ("~", reg)):
            if op in line:
                lhs, rhs = [x.strip() for x in line.split(op, 1)]
                bucket.append({"lhs": lhs, "op": op, "rhs": [t.strip() for t in rhs.split("+")]})
                break
    return {"formula": dsl_str, "measurements": meas, "regressions": reg, "covariances": cov}


class SEM:
    def __init__(self, formula: str, method: str = "pls", estimator: str = "ml", n_boot: int = 0, max_iter: int = 500, tol: float = 1e-6):
        self.formula = formula
        self.method = method.lower()
        self.estimator = estimator.lower()
        self.n_boot = n_boot
        self.max_iter = max_iter
        self.tol = tol
        self.result_ = None

    def fit(self, df):
        arr, cols = _extract_array(df, prefer_f32=True)
        if self.method in ("pls", "plsc"):
            fn = _core.fit_pls_f32 if arr.dtype == np.float32 else _core.fit_pls_f64
            self.result_ = fn(self.formula, arr, cols, self.method == "plsc", self.n_boot, self.max_iter, self.tol)
        elif self.method == "cb":
            fn = _core.fit_cb_f32 if arr.dtype == np.float32 else _core.fit_cb_f64
            self.result_ = fn(self.formula, arr, cols, self.max_iter)
        else:
            raise ValueError("method must be one of: 'pls', 'plsc', 'cb'")
        return self

    def summary(self) -> str:
        if self.result_ is None:
            return "Model not fitted."
        r = self.result_
        if self.method in ("pls", "plsc"):
            lines = [f"{'PLSc' if r.consistent else 'PLS-SEM'} | converged={r.converged} ({r.iterations} iters)"]
            lines += [f"  {s} -> {d} : {e:+.5f}" + (f" (SE={se:.4f}, t={t:.2f})" if se is not None else "") for s, d, e, se, t in r.paths]
            return "\n".join(lines)
        return f"CB-SEM (ML) | Chi2={r.chi_square:.3f} (df={r.df}), CFI={r.cfi:.4f}, TLI={r.tli:.4f}, RMSEA={r.rmsea:.4f}, SRMR={r.srmr:.4f}"


def fit_sem(spec: str | dict, data, method: str = "cb", estimator: str = "ml", n_boot: int = 0, max_iter: int = 500, tol: float = 1e-6) -> SEM:
    formula = spec["formula"] if isinstance(spec, dict) else spec
    return SEM(formula, method=method, estimator=estimator, n_boot=n_boot, max_iter=max_iter, tol=tol).fit(data)


def fit_indices(fit_result) -> dict:
    r = fit_result.result_ if isinstance(fit_result, SEM) else fit_result
    if r is None:
        raise ValueError("Model must be fitted first")
    if hasattr(r, "chi_square"):
        return {"converged": r.converged, "iterations": r.iterations, "f_ml": r.f_ml, "chi2": r.chi_square, "df": r.df, "cfi": r.cfi, "tli": r.tli, "rmsea": r.rmsea, "srmr": r.srmr}
    return {"converged": r.converged, "iterations": r.iterations, "consistent": r.consistent, "constructs": {l: {"mode": m, "rho_A": rho, "ave": ave, "cr": cr, "r2": r2} for l, m, rho, ave, cr, r2 in r.metrics}}


def propagate_risk(fit_result, shocks: dict[str, float]) -> dict[str, float]:
    """Compute total causal propagation Delta_eta = (I - B)^(-1) * Gamma * Delta_xi."""
    r = fit_result.result_ if isinstance(fit_result, SEM) else fit_result
    if r is None:
        raise ValueError("Model must be fitted first")
    edges = [(s, d, e) for s, d, e, _, _ in r.paths] if hasattr(r, "paths") else [(rhs, lhs, e) for lhs, op, rhs, e, _, _ in r.params if op == "~"]
    nodes = list(dict.fromkeys([n for s, d, _ in edges for n in (s, d)] + list(shocks.keys())))
    idx = {n: i for i, n in enumerate(nodes)}
    b_mat = np.zeros((len(nodes), len(nodes)), dtype=np.float64)
    for src, dst, est in edges:
        b_mat[idx[dst], idx[src]] += float(est)
    xi = np.array([float(shocks.get(n, 0.0)) for n in nodes], dtype=np.float64)
    eta = np.linalg.solve(np.eye(len(nodes)) - b_mat, xi)
    return {n: float(eta[i]) for i, n in enumerate(nodes)}
