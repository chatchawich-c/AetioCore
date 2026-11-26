#!/usr/bin/env python3
"""
4-Panel Publication Figure (2x2, PDF Vector + 300 DPI PNG) สำหรับเปเปอร์ semflow-dl:
  (a) Edge 1: Load -> Stress (Saturating Monotonic I-Spline vs Oracle vs EBM)
  (b) Edge 2: Stress -> Risk (Convex I-Spline + 95% Conformal Band vs EBM vs XGBoost SHAP)
  (c) Exact Local Causal Elasticity phi'(Stress -> Risk) vs Linear-SEM vs Oracle
  (d) Structural Intervention do(Load +1.0 SD): Direct vs Indirect (Mediator Blocking)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import shap
import xgboost as xgb
from interpret.glassbox import ExplainableBoostingRegressor
from semflow_dl import EdgeWhiteBoxSEM

WHITEBOX_FORMULA = """
    Load   =~ power + torque
    Stress =~ temp + vib
    Risk   =~ risk_1 + risk_2
    Stress ~ Load   [edge=ispline, sign=+1, knots=6]
    Risk   ~ Stress [edge=ispline, sign=+1, knots=6] + Load [edge=bspline, knots=5]
"""

LINEAR_FORMULA = """
    Load   =~ power + torque
    Stress =~ temp + vib
    Risk   =~ risk_1 + risk_2
    Stress ~ Load   [edge=linear]
    Risk   ~ Stress [edge=linear] + Load [edge=linear]
"""

FEATURE_COLS = ["power", "torque", "temp", "vib"]
SEM_COLS = ["power", "torque", "temp", "vib", "risk_1", "risk_2"]
LOAD_LAMBDAS = (0.90, 0.85)
STRESS_LAMBDAS = (0.90, 0.85)
RISK_LAMBDAS = (0.92, 0.88)


def set_journal_rcparams() -> None:
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "STIXGeneral"],
        "mathtext.fontset": "stix",
        "font.size": 9.5,
        "axes.titlesize": 10.5,
        "axes.titleweight": "bold",
        "axes.labelsize": 9.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def generate_data(n_total: int = 30_000, seed: int = 2026) -> dict:
    rng = np.random.default_rng(seed)
    load = rng.standard_normal(n_total)
    eps_s = rng.standard_normal(n_total)
    eps_y = rng.standard_normal(n_total)

    def f_ls(l_val: np.ndarray) -> np.ndarray:
        return 1.15 * np.tanh(0.85 * l_val) + 0.25 * l_val

    def f_sy(s_std: np.ndarray) -> np.ndarray:
        return 0.65 * np.log1p(np.exp(1.25 * s_std))

    def df_sy(s_std: np.ndarray) -> np.ndarray:
        return 0.65 * 1.25 / (1.0 + np.exp(-1.25 * s_std))

    def f_ly(l_val: np.ndarray) -> np.ndarray:
        return 0.14 * l_val

    s_raw = f_ls(load) + 0.45 * eps_s
    n_train = int(n_total * 0.70)
    n_cal = int(n_total * 0.15)

    s_mu, s_sd = float(s_raw[:n_train].mean()), float(s_raw[:n_train].std())
    s_std = (s_raw - s_mu) / s_sd

    y_raw = f_sy(s_std) + f_ly(load) + 0.40 * eps_y
    y_mu, y_sd = float(y_raw[:n_train].mean()), float(y_raw[:n_train].std())
    y_std = (y_raw - y_mu) / y_sd

    # Oracle Counterfactual do(Load +1.0 SD)
    sl_t = slice(n_train + n_cal, n_total)
    l_cf = load[sl_t] + 1.0
    s_std_cf = (f_ls(l_cf) + 0.45 * eps_s[sl_t] - s_mu) / s_sd
    y_cf_tot = f_sy(s_std_cf) + f_ly(l_cf) + 0.40 * eps_y[sl_t]
    y_cf_dir = f_sy(s_std[sl_t]) + f_ly(l_cf) + 0.40 * eps_y[sl_t]

    oracle_total = float(np.mean((y_cf_tot - y_raw[sl_t]) / y_sd))
    oracle_direct = float(np.mean((y_cf_dir - y_raw[sl_t]) / y_sd))

    def noisy(z: np.ndarray, lam: float) -> np.ndarray:
        return (lam * z + np.sqrt(1.0 - lam**2) * rng.standard_normal(n_total)).astype(np.float32)

    df = pl.DataFrame({
        "power": noisy(load, LOAD_LAMBDAS[0]),
        "torque": noisy(load, LOAD_LAMBDAS[1]),
        "temp": noisy(s_std, STRESS_LAMBDAS[0]),
        "vib": noisy(s_std, STRESS_LAMBDAS[1]),
        "risk_1": noisy(y_std, RISK_LAMBDAS[0]),
        "risk_2": noisy(y_std, RISK_LAMBDAS[1]),
    })

    # Composite target สำหรับ EBM & XGBoost
    def composite(c1: np.ndarray, c2: np.ndarray) -> np.ndarray:
        raw = 0.5 * (
            (c1 - c1[:n_train].mean()) / c1[:n_train].std()
            + (c2 - c2[:n_train].mean()) / c2[:n_train].std()
        )
        return ((raw - raw[:n_train].mean()) / raw[:n_train].std()).astype(np.float32)

    df = df.with_columns([
        pl.Series("stress_comp", composite(df["temp"].to_numpy(), df["vib"].to_numpy())),
        pl.Series("risk_comp", composite(df["risk_1"].to_numpy(), df["risk_2"].to_numpy())),
    ])

    grid = np.linspace(-2.5, 2.5, 200)
    return {
        "train": df.slice(0, n_train),
        "cal": df.slice(n_train, n_cal),
        "test": df.slice(n_train + n_cal, n_total - n_train - n_cal),
        "s_test_true": s_std[sl_t],
        "grid": grid,
        "oracle_ls_curve": (f_ls(grid) - float(np.mean(f_ls(load[:n_train])))) / s_sd,
        "oracle_sy_curve": (f_sy(grid) - float(np.mean(f_sy(s_std[:n_train])))) / y_sd,
        "oracle_sy_deriv": df_sy(grid) / y_sd,
        "oracle_total": oracle_total,
        "oracle_direct": oracle_direct,
        "oracle_indirect": oracle_total - oracle_direct,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-total", type=int, default=30_000)
    parser.add_argument("--out-dir", type=Path, default=Path("benchmarks/figures"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    set_journal_rcparams()

    d = generate_data(args.n_total)
    df_tr, df_cal, df_te = d["train"], d["cal"], d["test"]
    grid_f32 = d["grid"].astype(np.float32)

    # 1. Fit semflow-dl (Monotonic I-Spline + 95% Split-Conformal) และ Linear-SEM
    m_dl = EdgeWhiteBoxSEM(WHITEBOX_FORMULA, knots=6, smooth_lambda=1e-4).fit(
        df_tr.select(SEM_COLS), df_cal=df_cal.select(SEM_COLS), conformal_alpha=0.05
    )
    m_lin = EdgeWhiteBoxSEM(LINEAR_FORMULA, knots=6, smooth_lambda=1e-4).fit(
        df_tr.select(SEM_COLS), df_cal=df_cal.select(SEM_COLS), conformal_alpha=0.05
    )

    c_ls = m_dl.edge_curve("Load", "Stress", n_points=200)
    c_sr = m_dl.edge_curve("Stress", "Risk", n_points=200)
    c_sr_lin = m_lin.edge_curve("Stress", "Risk", n_points=200)
    eff_dl = m_dl.intervene(df_te.select(SEM_COLS), do_var="Load", delta=1.0, target="Risk")
    eff_lin = m_lin.intervene(df_te.select(SEM_COLS), do_var="Load", delta=1.0, target="Risk")
    hw_risk = eff_dl["conformal_half_width"]

    # 2. Fit EBM (สำหรับทั้ง Load->Stress และ Flat Risk)
    x_tr_load = df_tr.select(["power", "torque"]).to_numpy()
    y_tr_stress = df_tr["stress_comp"].to_numpy()
    m_ebm_ls = ExplainableBoostingRegressor(
        feature_names=["power", "torque"], interactions=0, random_state=42, n_jobs=-1
    ).fit(x_tr_load, y_tr_stress)
    x_grid_load = np.column_stack([LOAD_LAMBDAS[0] * grid_f32, LOAD_LAMBDAS[1] * grid_f32])
    ebm_ls_curve = m_ebm_ls.predict(x_grid_load)
    ebm_ls_curve -= float(np.mean(ebm_ls_curve))

    x_tr = df_tr.select(FEATURE_COLS).to_numpy()
    y_tr = df_tr["risk_comp"].to_numpy()
    x_te = df_te.select(FEATURE_COLS).to_numpy()
    x_te_cf = x_te.copy()
    x_te_cf[:, 0] += np.float32(LOAD_LAMBDAS[0])
    x_te_cf[:, 1] += np.float32(LOAD_LAMBDAS[1])

    m_ebm = ExplainableBoostingRegressor(
        feature_names=FEATURE_COLS, interactions=0, random_state=42, n_jobs=-1
    ).fit(x_tr, y_tr)
    x_grid_stress = np.column_stack([
        np.zeros_like(grid_f32), np.zeros_like(grid_f32),
        STRESS_LAMBDAS[0] * grid_f32, STRESS_LAMBDAS[1] * grid_f32,
    ])
    ebm_sr_curve = m_ebm.predict(x_grid_stress)
    ebm_sr_curve -= float(np.mean(ebm_sr_curve))
    ebm_do = float(np.mean(m_ebm.predict(x_te_cf) - m_ebm.predict(x_te)))

    # 3. Fit XGBoost + TreeSHAP
    m_xgb = xgb.XGBRegressor(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, tree_method="hist", n_jobs=-1, random_state=42,
    ).fit(x_tr, y_tr)
    xgb_do = float(np.mean(m_xgb.predict(x_te_cf) - m_xgb.predict(x_te)))

    n_shap = min(1_500, len(x_te))
    sv = shap.TreeExplainer(m_xgb).shap_values(x_te[:n_shap])
    shap_stress_sum = sv[:, 2] + sv[:, 3]
    s_shap_x = d["s_test_true"][:n_shap]

    # บันทึกข้อมูลกราฟลง .npz เพื่อปรับแต่งสไตล์ภายหลังได้ทันที
    np.savez(
        args.out_dir / "fig_4panel_cache.npz",
        grid=d["grid"],
        oracle_ls=d["oracle_ls_curve"],
        oracle_sy=d["oracle_sy_curve"],
        oracle_sy_deriv=d["oracle_sy_deriv"],
        dl_ls_x=c_ls["x"], dl_ls_y=c_ls["y_centered"],
        dl_sr_x=c_sr["x"], dl_sr_y=c_sr["y_centered"], dl_sr_dy=c_sr["local_elasticity"],
        ebm_ls=ebm_ls_curve, ebm_sr=ebm_sr_curve,
        hw_risk=hw_risk,
    )

    # -------------------------------------------------------------------------
    # วาดภาพ 2x2 Publication Figure (Okabe-Ito Palette)
    # -------------------------------------------------------------------------
    C_ORACLE = "#111111"
    C_DL = "#0072B2"
    C_EBM = "#D55E00"
    C_XGB = "#009E73"
    C_LIN = "#CC79A7"
    C_INDIRECT = "#56B4E9"

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.6), constrained_layout=True)

    # --- Panel (a): Load -> Stress (Saturating Monotonic Curve) ---
    ax = axes[0, 0]
    ax.plot(d["grid"], ebm_ls_curve, color=C_EBM, lw=1.6, ls="-.", label="EBM 1D Shape (Histogram)")
    ax.plot(c_ls["x"], c_ls["y_centered"], color=C_DL, lw=2.4, label="semflow-dl (Monotonic I-Spline)")
    ax.plot(d["grid"], d["oracle_ls_curve"], color=C_ORACLE, lw=1.8, ls="--", label="Oracle Ground Truth")
    ax.set_xlim(-2.5, 2.5)
    ax.set_title("(a) Saturating Edge: Load $\\rightarrow$ Stress")
    ax.set_xlabel("Latent Load ($z$, SD)")
    ax.set_ylabel("Centered Effect on Stress ($\\phi(z)$)")
    ax.legend(loc="upper left", frameon=True, framealpha=0.92)

    # --- Panel (b): Stress -> Risk + 95% Split-Conformal Band ---
    ax = axes[0, 1]
    ax.scatter(
        s_shap_x, shap_stress_sum, s=8, alpha=0.20, color=C_XGB,
        edgecolors="none", label="XGBoost SHAP (temp + vib)", zorder=1
    )
    ax.fill_between(
        c_sr["x"], c_sr["y_centered"] - hw_risk, c_sr["y_centered"] + hw_risk,
        color=C_DL, alpha=0.12, label=f"95% Conformal Band ($\\pm {hw_risk:.2f}$)"
    )
    ax.plot(d["grid"], ebm_sr_curve, color=C_EBM, lw=1.6, ls="-.", label="EBM 1D Shape (Histogram)")
    ax.plot(c_sr["x"], c_sr["y_centered"], color=C_DL, lw=2.4, label="semflow-dl (Monotonic I-Spline)")
    ax.plot(d["grid"], d["oracle_sy_curve"], color=C_ORACLE, lw=1.8, ls="--", label="Oracle Ground Truth")
    ax.set_xlim(-2.5, 2.5)
    ax.set_title("(b) Convex Edge: Stress $\\rightarrow$ Risk (+ Conformal)")
    ax.set_xlabel("Latent Stress ($z$, SD)")
    ax.set_ylabel("Centered Effect on Risk ($\\phi(z)$)")
    ax.legend(loc="upper left", frameon=True, framealpha=0.92)

    # --- Panel (c): Exact Local Causal Elasticity phi'(z) ---
    ax = axes[1, 0]
    ax.plot(
        c_sr_lin["x"], c_sr_lin["local_elasticity"], color=C_LIN, lw=1.9, ls=":",
        label=f"Linear-SEM (const $\\beta={c_sr_lin['local_elasticity'][0]:.2f}$)"
    )
    ax.plot(c_sr["x"], c_sr["local_elasticity"], color=C_DL, lw=2.4, label="semflow-dl Exact M-Spline $\\phi'(z)$")
    ax.plot(d["grid"], d["oracle_sy_deriv"], color=C_ORACLE, lw=1.8, ls="--", label="Oracle Derivative $\\partial f/\\partial z$")
    ax.fill_between(c_sr["x"], 0, c_sr["local_elasticity"], color=C_DL, alpha=0.08)
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(bottom=0.0)
    ax.set_title("(c) Local Causal Elasticity $\\phi'(\\mathrm{Stress} \\rightarrow \\mathrm{Risk})$")
    ax.set_xlabel("Latent Stress ($z$, SD)")
    ax.set_ylabel("Local Sensitivity $\\phi'(z)$")
    ax.legend(loc="upper left", frameon=True, framealpha=0.92)

    # --- Panel (d): Causal Intervention do(Load +1.0 SD) ---
    ax = axes[1, 1]
    labels = ["Oracle\nTruth", "semflow-dl\n(I-Spline)", "Linear\nSEM", "EBM\n(Flat)", "XGBoost\n+ SHAP"]
    directs = np.array([d["oracle_direct"], eff_dl["direct_effect"], eff_lin["direct_effect"], ebm_do, xgb_do])
    indirects = np.array([d["oracle_indirect"], eff_dl["indirect_effect"], eff_lin["indirect_effect"], 0.0, 0.0])
    totals = directs + indirects
    x_pos = np.arange(len(labels))

    ax.bar(x_pos, directs, width=0.55, color=C_DL, edgecolor="white", label="Direct Effect ($L \\rightarrow R$)")
    ax.bar(
        x_pos, indirects, width=0.55, bottom=directs, color=C_INDIRECT,
        edgecolor="white", hatch="//", label="Indirect Effect ($L \\rightarrow S \\rightarrow R$)"
    )
    ax.axhline(d["oracle_total"], color=C_ORACLE, ls="--", lw=1.2, alpha=0.8)

    for i, tot in enumerate(totals):
        ax.text(i, tot + 0.025, f"{tot:+.2f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    ax.annotate(
        "Mediator Blocking\n(Indirect = 0)",
        xy=(3.5, max(ebm_do, xgb_do) + 0.07),
        xytext=(3.2, d["oracle_total"] * 0.62),
        ha="center", fontsize=8.0, color=C_EBM, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=C_EBM, lw=1.2),
    )

    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, max(totals.max(), d["oracle_total"]) * 1.24)
    ax.set_title("(d) Intervention $\\mathrm{do}(\\mathrm{Load} + 1\\,\\mathrm{SD})$")
    ax.set_ylabel("Predicted Change in Risk (SD)")
    ax.legend(loc="upper right", frameon=True, framealpha=0.92)

    pdf_path = args.out_dir / "fig_whitebox_4panel.pdf"
    png_path = args.out_dir / "fig_whitebox_4panel.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved 4-panel figures:\n  - {pdf_path}\n  - {png_path}")


if __name__ == "__main__":
    main()
