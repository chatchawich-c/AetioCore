from __future__ import annotations
import numpy as np
from . import _core


def _to_f32(df) -> tuple[np.ndarray, list[str]]:
    cols = list(df.columns)
    try:
        arr = df.to_numpy(writable=False)
    except TypeError:
        arr = df.to_numpy()
    return np.ascontiguousarray(arr, dtype=np.float32), cols


class EdgeWhiteBoxSEM:
    def __init__(self, formula: str, knots: int = 6, smooth_lambda: float = 1e-4, max_iter: int = 300):
        self.formula = formula
        self.knots = knots
        self.smooth_lambda = smooth_lambda
        self.max_iter = max_iter
        self._model = None

    def fit(self, df_train, df_cal=None, conformal_alpha: float = 0.05):
        arr, cols = _to_f32(df_train)
        self._model = _core.fit_whitebox_f32(
            self.formula, arr, cols, self.knots, self.smooth_lambda, self.max_iter
        )
        if df_cal is not None:
            cal_arr, cal_cols = _to_f32(df_cal)
            self._model.calibrate_f32(cal_arr, cal_cols, conformal_alpha)
        return self

    def edge_curve(self, from_var: str, to_var: str, n_points: int = 100) -> dict:
        etype, x, y, dy = self._model.edge_curve(from_var, to_var, n_points)
        return {
            "edge_type": etype,
            "x": np.asarray(x),
            "y_centered": np.asarray(y),
            "local_elasticity": np.asarray(dy),
        }

    def predict(self, df, target: str) -> dict:
        arr, cols = _to_f32(df)
        y_true, y_pred, hw = self._model.predict_f32(arr, cols, target)
        return {
            "y_true": np.asarray(y_true),
            "y_pred": np.asarray(y_pred),
            "conformal_half_width": hw,
        }

    def intervene(self, df, do_var: str, delta: float, target: str) -> dict:
        arr, cols = _to_f32(df)
        tot, direct, indirect, hw = self._model.intervene_f32(arr, cols, do_var, delta, target)
        return {
            "do": f"{do_var} + {delta:+.2f} SD",
            "target": target,
            "total_effect": tot,
            "direct_effect": direct,
            "indirect_effect": indirect,
            "conformal_half_width": hw,
        }

    def summary(self) -> str:
        lines = ["EdgeWhiteBoxSEM (semflow-dl v0.1)"]
        lines.append("--- Edge Operators & Mean Local Elasticity E[phi'(z)] ---")
        for src, dst, etype, elast in self._model.edges_summary:
            lines.append(f"  {src} -> {dst} [{etype:7s}] : mean phi'(z) = {elast:+.4f}")
        lines.append("--- R-Squared ---")
        for lat, r2 in self._model.r_squared.items():
            lines.append(f"  {lat} : {r2:.4f}")
        return "\n".join(lines)
