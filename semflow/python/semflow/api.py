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
    if arr.dtype not in (np.float32, np.float64):
        arr = arr.astype(target_dtype)
    elif not prefer_f32 and arr.dtype == np.float32:
        arr = arr.astype(np.float64)
    return np.ascontiguousarray(arr), cols


class SEM:
    def __init__(
        self,
        formula: str,
        method: str = "pls",  # "pls", "plsc", "cb"
        estimator: str = "ml",
        n_boot: int = 0,
        max_iter: int = 500,
        tol: float = 1e-6,
    ):
        self.formula = formula
        self.method = method.lower()
        self.estimator = estimator.lower()
        self.n_boot = n_boot
        self.max_iter = max_iter
        self.tol = tol
        self.result_ = None

    def fit(self, df):
        if self.method in ("pls", "plsc"):
            arr, cols = _extract_array(df, prefer_f32=True)
            consistent = self.method == "plsc"
            fn = _core.fit_pls_f32 if arr.dtype == np.float32 else _core.fit_pls_f64
            self.result_ = fn(
                self.formula, arr, cols, consistent, self.n_boot, self.max_iter, self.tol
            )
        elif self.method == "cb":
            arr, cols = _extract_array(df, prefer_f32=True)
            fn = _core.fit_cb_f32 if arr.dtype == np.float32 else _core.fit_cb_f64
            self.result_ = fn(self.formula, arr, cols, self.max_iter)
        else:
            raise ValueError("method must be one of: 'pls', 'plsc', 'cb'")
        return self

    def summary(self) -> str:
        if self.result_ is None:
            return "Model not fitted."
        if self.method in ("pls", "plsc"):
            r = self.result_
            tag = "PLSc (Consistent)" if r.consistent else "PLS-SEM"
            lines = [f"{tag} | converged={r.converged} in {r.iterations} iters"]
            lines.append("--- Path Coefficients ---")
            for src, dst, est, se, t in r.paths:
                extra = f" | SE={se:.4f}, t={t:.2f}" if se is not None else ""
                lines.append(f"  {src} -> {dst} : {est:+.5f}{extra}")
            lines.append("--- Construct Reliability & Validity ---")
            for lat, mode, rho, ave, cr, r2 in r.metrics:
                r2_str = f", R2={r2:.4f}" if r2 is not None else ""
                lines.append(f"  {lat} [{mode}]: rho_A={rho:.4f}, AVE={ave:.4f}, CR={cr:.4f}{r2_str}")
            return "\n".join(lines)

        r = self.result_
        lines = [
            f"CB-SEM (ML / RAM) | converged={r.converged} in {r.iterations} iters | F_ML={r.f_ml:.6f}",
            f"Fit: Chi2={r.chi_square:.3f} (df={r.df}), CFI={r.cfi:.4f}, TLI={r.tli:.4f}, RMSEA={r.rmsea:.4f}, SRMR={r.srmr:.4f}",
            "--- Parameter Estimates ---",
        ]
        for lhs, op, rhs, est, se, fixed in r.params:
            se_str = " (fixed)" if fixed else f" | SE={se:.4f}"
            lines.append(f"  {lhs:>6} {op:<2} {rhs:<6} : {est:+.5f}{se_str}")
        return "\n".join(lines)
