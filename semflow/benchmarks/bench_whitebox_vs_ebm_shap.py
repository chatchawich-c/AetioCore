#!/usr/bin/env python3
"""
White-Box Causal DL Benchmark:
semflow-dl (Monotonic I-Spline + Latent DAG) vs Linear-SEM vs EBM (InterpretML) vs XGBoost+SHAP

วัดครบ 5 มิติใน Subprocess แยก (Peak RSS แม่นยำผ่าน os.wait4):
  1. Predictive Accuracy (Test R2 vs Observed & vs Noise-Free Oracle)
  2. Causal Intervention do(Load +1.0 SD): Total, Direct, Indirect Error vs Oracle Ground Truth
  3. Mediator Blocking Diagnostic: สัดส่วนความสำคัญที่ให้แก่ Load vs Stress
  4. Tail Monotonicity Violation (%) บนกราฟ Stress -> Risk (-2.5 SD ถึง +2.5 SD)
  5. Fit Time, Explain/Intervene Time (ms) และ Peak RSS RAM (MB)
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import polars as pl

WHITEBOX_FORMULA = """
    Load   =~ power + torque
    Stress =~ temp + vib
    Risk   =~ risk_1 + risk_2
    Stress ~ Load   [edge=ispline, sign=+1, knots=6]
    Risk   ~ Stress [edge=ispline, sign=+1, knots=6] + Load [edge=bspline, knots=5]
"""

LINEAR_ABLATION_FORMULA = """
    Load   =~ power + torque
    Stress =~ temp + vib
    Risk   =~ risk_1 + risk_2
    Stress ~ Load   [edge=linear]
    Risk   ~ Stress [edge=linear] + Load [edge=linear]
"""

FEATURE_COLS = ["power", "torque", "temp", "vib"]
LOAD_LAMBDAS = (0.90, 0.85)
STRESS_LAMBDAS = (0.90, 0.85)
RISK_LAMBDAS = (0.92, 0.88)


def generate_causal_dataset(n_total: int, out_dir: Path, seed: int = 2026) -> dict:
    """
    สร้างชุดข้อมูล Train / Cal / Test พร้อมคำนวณ Oracle Counterfactual Ground Truth
    สำหรับ do(Load += +1.0 SD) อย่างแม่นยำจากสมการโครงสร้างจริง
    """
    rng = np.random.default_rng(seed)

    # 1. Ground-Truth Structural Equations (Nonlinear + Monotonic Mediation)
    load = rng.standard_normal(n_total)
    eps_s = rng.standard_normal(n_total)
    eps_y = rng.standard_normal(n_total)

    def f_ls(l_val: np.ndarray) -> np.ndarray:
        # ความเค้นเพิ่มแบบอิ่มตัว (Saturating Monotonic)
        return 1.15 * np.tanh(0.85 * l_val) + 0.25 * l_val

    def f_sy(s_std: np.ndarray) -> np.ndarray:
        # ความเสี่ยงเร่งตัวแบบนูนเมื่อความเค้นสูง (Convex Thermal/Arrhenius Acceleration)
        return 0.65 * np.log1p(np.exp(1.25 * s_std))

    def f_ly(l_val: np.ndarray) -> np.ndarray:
        # อิทธิพลทางตรงเล็กน้อย (~20% ของ Total Effect)
        return 0.14 * l_val

    s_raw = f_ls(load) + 0.45 * eps_s

    # แบ่ง Train (70%) / Calibration (15%) / Test (15%)
    n_train = int(n_total * 0.70)
    n_cal = int(n_total * 0.15)
    n_test = n_total - n_train - n_cal

    # Standardize ด้วยสถิติของ Train เท่านั้น (ป้องกัน Data Leakage)
    s_mu, s_sd = float(s_raw[:n_train].mean()), float(s_raw[:n_train].std())
    s_std = (s_raw - s_mu) / s_sd

    y_raw = f_sy(s_std) + f_ly(load) + 0.40 * eps_y
    y_mu, y_sd = float(y_raw[:n_train].mean()), float(y_raw[:n_train].std())
    y_std = (y_raw - y_mu) / y_sd

    # 2. คำนวณ Oracle Ground Truth สำหรับ do(Load += +1.0 SD) บนชุด Test
    sl_test = slice(n_train + n_cal, n_total)
    l_t = load[sl_test]
    eps_s_t = eps_s[sl_test]
    eps_y_t = eps_y[sl_test]
    s_std_t = s_std[sl_test]
    y_raw_t = y_raw[sl_test]

    l_cf = l_t + 1.0
    s_std_cf = (f_ls(l_cf) + 0.45 * eps_s_t - s_mu) / s_sd
    y_raw_cf_total = f_sy(s_std_cf) + f_ly(l_cf) + 0.40 * eps_y_t
    y_raw_cf_direct = f_sy(s_std_t) + f_ly(l_cf) + 0.40 * eps_y_t

    oracle_total = float(np.mean((y_raw_cf_total - y_raw_t) / y_sd))
    oracle_direct = float(np.mean((y_raw_cf_direct - y_raw_t) / y_sd))
    oracle_indirect = oracle_total - oracle_direct

    # 3. สร้างตัวชี้วัดที่มี Measurement Noise และ Collinearity
    def noisy_ind(latent: np.ndarray, lam: float) -> np.ndarray:
        sd_e = np.sqrt(max(1.0 - lam**2, 1e-6))
        return (lam * latent + sd_e * rng.standard_normal(n_total)).astype(np.float32)

    df_all = pl.DataFrame({
        "power": noisy_ind(load, LOAD_LAMBDAS[0]),
        "torque": noisy_ind(load, LOAD_LAMBDAS[1]),
        "temp": noisy_ind(s_std, STRESS_LAMBDAS[0]),
        "vib": noisy_ind(s_std, STRESS_LAMBDAS[1]),
        "risk_1": noisy_ind(y_std, RISK_LAMBDAS[0]),
        "risk_2": noisy_ind(y_std, RISK_LAMBDAS[1]),
        "oracle_y": y_std.astype(np.float32),
    })

    # สร้าง Composite Target (Zero Mean, Unit SD บน Train) ให้ EBM/XGBoost ใช้สเกลเดียวกับ semflow-dl
    r1 = df_all["risk_1"].to_numpy()
    r2 = df_all["risk_2"].to_numpy()
    comp_raw = 0.5 * (
        (r1 - r1[:n_train].mean()) / r1[:n_train].std()
        + (r2 - r2[:n_train].mean()) / r2[:n_train].std()
    )
    comp_std = ((comp_raw - comp_raw[:n_train].mean()) / comp_raw[:n_train].std()).astype(np.float32)
    df_all = df_all.with_columns(pl.Series("risk_comp", comp_std))

    df_all.slice(0, n_train).write_parquet(out_dir / "train.parquet")
    df_all.slice(n_train, n_cal).write_parquet(out_dir / "cal.parquet")
    df_all.slice(n_train + n_cal, n_test).write_parquet(out_dir / "test.parquet")

    meta = {
        "n_total": n_total,
        "n_train": n_train,
        "n_cal": n_cal,
        "n_test": n_test,
        "oracle_total": oracle_total,
        "oracle_direct": oracle_direct,
        "oracle_indirect": oracle_indirect,
        "oracle_load_true_share_pct": 100.0 * abs(oracle_total) / (abs(oracle_total) + 0.70),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def r2_and_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    err = y_true - y_pred
    rmse = float(np.sqrt(np.mean(err**2)))
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
    return r2, rmse


def monotonicity_violation_pct(y_curve: np.ndarray, tol: float = 1e-6) -> float:
    """ร้อยละของช่วงบนกราฟ Stress -> Risk ที่ความชันติดลบ (ละเมิดกฎทางเดียว)"""
    diffs = np.diff(y_curve)
    return float(100.0 * np.mean(diffs < -tol))


def run_worker(engine: str, data_dir: Path, n_runs: int, out_json: Path) -> None:
    meta = json.loads((data_dir / "meta.json").read_text())
    df_train = pl.read_parquet(data_dir / "train.parquet")
    df_cal = pl.read_parquet(data_dir / "cal.parquet")
    df_test = pl.read_parquet(data_dir / "test.parquet")

    sem_cols = ["power", "torque", "temp", "vib", "risk_1", "risk_2"]
    y_test_obs = df_test["risk_comp"].to_numpy().astype(np.float64)
    y_test_oracle = df_test["oracle_y"].to_numpy().astype(np.float64)

    # กริดสำหรับตรวจ Monotonicity ของ Stress -> Risk (-2.5 SD ถึง +2.5 SD, 200 จุด)
    s_grid = np.linspace(-2.5, 2.5, 200, dtype=np.float32)
    x_grid_flat = np.column_stack([
        np.zeros_like(s_grid),
        np.zeros_like(s_grid),
        STRESS_LAMBDAS[0] * s_grid,
        STRESS_LAMBDAS[1] * s_grid,
    ])

    fit_times, explain_times = [], []
    last_metrics = {}

    if engine in ("semflow_dl", "semflow_linear"):
        from semflow_dl import EdgeWhiteBoxSEM

        formula = WHITEBOX_FORMULA if engine == "semflow_dl" else LINEAR_ABLATION_FORMULA
        tr_sem = df_train.select(sem_cols)
        cal_sem = df_cal.select(sem_cols)
        te_sem = df_test.select(sem_cols)

        # Warmup 1 รอบ
        EdgeWhiteBoxSEM(formula, knots=6, smooth_lambda=1e-4).fit(tr_sem, df_cal=cal_sem, conformal_alpha=0.05)

        for _ in range(n_runs):
            t0 = time.perf_counter()
            model = EdgeWhiteBoxSEM(formula, knots=6, smooth_lambda=1e-4).fit(
                tr_sem, df_cal=cal_sem, conformal_alpha=0.05
            )
            t_fit = time.perf_counter()

            # Explain (Edge Curves) + Structural Intervention do(Load +1.0 SD)
            pred_out = model.predict(te_sem, target="Risk")
            eff = model.intervene(te_sem, do_var="Load", delta=1.0, target="Risk")
            curve_sr = model.edge_curve("Stress", "Risk", n_points=200)
            t_exp = time.perf_counter()

            fit_times.append((t_fit - t0) * 1000.0)
            explain_times.append((t_exp - t_fit) * 1000.0)

            y_true_lat = pred_out["y_true"]
            y_pred_lat = pred_out["y_pred"]
            hw = pred_out["conformal_half_width"]
            coverage = float(100.0 * np.mean(np.abs(y_true_lat - y_pred_lat) <= hw))

            r2_obs, rmse_obs = r2_and_rmse(y_true_lat, y_pred_lat)
            r2_orc, _ = r2_and_rmse(y_test_oracle, y_pred_lat)

            # สัดส่วนความสำคัญเชิงเหตุผลของ Load (Total Effect) เทียบ Stress (Mean Elasticity)
            stress_elast = float(np.mean(np.abs(curve_sr["local_elasticity"])))
            load_share = 100.0 * abs(eff["total_effect"]) / max(abs(eff["total_effect"]) + stress_elast, 1e-12)

            last_metrics = {
                "r2_obs": r2_obs,
                "r2_oracle": r2_orc,
                "rmse_obs": rmse_obs,
                "do_total_est": eff["total_effect"],
                "do_direct_est": eff["direct_effect"],
                "do_indirect_est": eff["indirect_effect"],
                "do_total_abs_err": abs(eff["total_effect"] - meta["oracle_total"]),
                "load_credit_pct": load_share,
                "mono_viol_pct": monotonicity_violation_pct(curve_sr["y_centered"]),
                "conformal_cov_pct": coverage,
            }

    elif engine == "ebm":
        from interpret.glassbox import ExplainableBoostingRegressor

        x_tr = df_train.select(FEATURE_COLS).to_numpy()
        y_tr = df_train["risk_comp"].to_numpy()
        x_te = df_test.select(FEATURE_COLS).to_numpy()

        # ข้อมูลจำลองแทรกแซงแบบ Flat Ceteris Paribus (เพิ่ม Load +1 SD ที่เซ็นเซอร์ power/torque โดยตรึง temp/vib)
        x_te_cf = x_te.copy()
        x_te_cf[:, 0] += np.float32(LOAD_LAMBDAS[0])
        x_te_cf[:, 1] += np.float32(LOAD_LAMBDAS[1])

        for _ in range(n_runs):
            t0 = time.perf_counter()
            model = ExplainableBoostingRegressor(
                feature_names=FEATURE_COLS,
                interactions=0,  # Pure 1D Glass-Box เพื่อเทียบกราฟ 1D กับ Edge-Level Spline โดยตรง
                random_state=42,
                n_jobs=-1,
            )
            model.fit(x_tr, y_tr)
            t_fit = time.perf_counter()

            y_pred = model.predict(x_te)
            y_pred_cf = model.predict(x_te_cf)
            expl = model.explain_global().data()
            y_curve_sr = model.predict(x_grid_flat)
            t_exp = time.perf_counter()

            fit_times.append((t_fit - t0) * 1000.0)
            explain_times.append((t_exp - t_fit) * 1000.0)

            r2_obs, rmse_obs = r2_and_rmse(y_test_obs, y_pred)
            r2_orc, _ = r2_and_rmse(y_test_oracle, y_pred)
            est_tot = float(np.mean(y_pred_cf - y_pred))

            # วัดสัดส่วน Global Importance ของ Load (power+torque) vs Stress (temp+vib)
            imp_map = dict(zip(expl["names"], expl["scores"]))
            load_imp = imp_map.get("power", 0.0) + imp_map.get("torque", 0.0)
            stress_imp = imp_map.get("temp", 0.0) + imp_map.get("vib", 0.0)
            load_share = 100.0 * load_imp / max(load_imp + stress_imp, 1e-12)

            last_metrics = {
                "r2_obs": r2_obs,
                "r2_oracle": r2_orc,
                "rmse_obs": rmse_obs,
                "do_total_est": est_tot,
                "do_direct_est": est_tot,   # Flat model แยกทางอ้อมไม่ได้
                "do_indirect_est": 0.0,
                "do_total_abs_err": abs(est_tot - meta["oracle_total"]),
                "load_credit_pct": load_share,
                "mono_viol_pct": monotonicity_violation_pct(y_curve_sr),
                "conformal_cov_pct": None,
            }

    elif engine == "xgb_shap":
        import shap
        import xgboost as xgb

        x_tr = df_train.select(FEATURE_COLS).to_numpy()
        y_tr = df_train["risk_comp"].to_numpy()
        x_te = df_test.select(FEATURE_COLS).to_numpy()

        x_te_cf = x_te.copy()
        x_te_cf[:, 0] += np.float32(LOAD_LAMBDAS[0])
        x_te_cf[:, 1] += np.float32(LOAD_LAMBDAS[1])
        # จำกัดแถว SHAP สูงสุด 2,000 แถวตามแนวปฏิบัติมาตรฐานเพื่อไม่ให้กินเวลานานเกินไป
        x_shap = x_te[: min(2000, len(x_te))]

        for _ in range(n_runs):
            t0 = time.perf_counter()
            model = xgb.XGBRegressor(
                n_estimators=300,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                tree_method="hist",
                n_jobs=-1,
                random_state=42,
            )
            model.fit(x_tr, y_tr)
            t_fit = time.perf_counter()

            y_pred = model.predict(x_te)
            y_pred_cf = model.predict(x_te_cf)
            explainer = shap.TreeExplainer(model)
            sv = explainer.shap_values(x_shap)
            y_curve_sr = model.predict(x_grid_flat)
            t_exp = time.perf_counter()

            fit_times.append((t_fit - t0) * 1000.0)
            explain_times.append((t_exp - t_fit) * 1000.0)

            r2_obs, rmse_obs = r2_and_rmse(y_test_obs, y_pred)
            r2_orc, _ = r2_and_rmse(y_test_oracle, y_pred)
            est_tot = float(np.mean(y_pred_cf - y_pred))

            mean_abs_shap = np.mean(np.abs(sv), axis=0)
            load_shap = float(mean_abs_shap[0] + mean_abs_shap[1])
            stress_shap = float(mean_abs_shap[2] + mean_abs_shap[3])
            load_share = 100.0 * load_shap / max(load_shap + stress_shap, 1e-12)

            last_metrics = {
                "r2_obs": r2_obs,
                "r2_oracle": r2_orc,
                "rmse_obs": rmse_obs,
                "do_total_est": est_tot,
                "do_direct_est": est_tot,
                "do_indirect_est": 0.0,
                "do_total_abs_err": abs(est_tot - meta["oracle_total"]),
                "load_credit_pct": load_share,
                "mono_viol_pct": monotonicity_violation_pct(y_curve_sr),
                "conformal_cov_pct": None,
            }
    else:
        raise ValueError(f"Unknown engine: {engine}")

    out_payload = {
        "engine": engine,
        "fit_ms": round(statistics.median(fit_times), 2),
        "explain_ms": round(statistics.median(explain_times), 2),
        **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in last_metrics.items()},
    }
    out_json.write_text(json.dumps(out_payload))


def run_child_with_wait4(cmd: list[str], out_json: Path, timeout_sec: int) -> dict:
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    t_start = time.monotonic()
    while True:
        pid, status, rusage = os.wait4(proc.pid, os.WNOHANG)
        if pid != 0:
            rss_mb = (
                rusage.ru_maxrss / (1024.0 * 1024.0)
                if sys.platform == "darwin"
                else rusage.ru_maxrss / 1024.0
            )
            ok = os.WIFEXITED(status) and os.WEXITSTATUS(status) == 0
            err = proc.stderr.read().strip() if proc.stderr else ""
            if not ok or not out_json.exists():
                msg = err.splitlines()[-1] if err else f"exit={status}"
                return {"error": msg[:80], "peak_rss_mb": round(rss_mb, 2)}
            res = json.loads(out_json.read_text())
            res["peak_rss_mb"] = round(rss_mb, 2)
            return res

        if time.monotonic() - t_start > timeout_sec:
            proc.kill()
            _, _, rusage = os.wait4(proc.pid, 0)
            rss_mb = (
                rusage.ru_maxrss / (1024.0 * 1024.0)
                if sys.platform == "darwin"
                else rusage.ru_maxrss / 1024.0
            )
            return {"error": f"TIMEOUT ({timeout_sec}s)", "peak_rss_mb": round(rss_mb, 2)}
        time.sleep(0.05)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=int, nargs="+", default=[10_000, 100_000])
    parser.add_argument("--engines", type=str, nargs="+", default=["semflow_dl", "semflow_linear", "ebm", "xgb_shap"])
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=300)
    # Internal worker arguments
    parser.add_argument("--worker", type=str, default=None)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--out-json", type=Path, default=None)
    args = parser.parse_args()

    if args.worker:
        run_worker(args.worker, args.data_dir, args.runs, args.out_json)
        return

    rows = []
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        for n in args.sizes:
            d_dir = root / f"n_{n}"
            d_dir.mkdir()
            print(f"[GEN] Creating Nonlinear Mediation Dataset N = {n:,} ...", file=sys.stderr)
            meta = generate_causal_dataset(n, d_dir)
            print(
                f"      Oracle do(Load +1 SD) -> Total={meta['oracle_total']:+.4f}, "
                f"Direct={meta['oracle_direct']:+.4f}, Indirect={meta['oracle_indirect']:+.4f}",
                file=sys.stderr,
            )

            for eng in args.engines:
                out_json = d_dir / f"res_{eng}.json"
                print(f"[RUN] N={n:,} engine={eng} ...", file=sys.stderr)
                cmd = [
                    sys.executable, str(Path(__file__).resolve()),
                    "--worker", eng,
                    "--data-dir", str(d_dir),
                    "--runs", str(args.runs),
                    "--out-json", str(out_json),
                ]
                res = run_child_with_wait4(cmd, out_json, args.timeout)
                rows.append({
                    "n_total": n,
                    "engine": eng,
                    "oracle_total": round(meta["oracle_total"], 4),
                    "oracle_direct": round(meta["oracle_direct"], 4),
                    "oracle_indirect": round(meta["oracle_indirect"], 4),
                    **res,
                })

    csv_path = Path(__file__).parent / "results_whitebox_vs_ebm_shap.csv"
    fields = [
        "n_total", "engine", "r2_obs", "r2_oracle", "rmse_obs",
        "oracle_total", "do_total_est", "do_direct_est", "do_indirect_est", "do_total_abs_err",
        "load_credit_pct", "mono_viol_pct", "conformal_cov_pct",
        "fit_ms", "explain_ms", "peak_rss_mb", "error",
    ]
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    print("\n### White-Box Causal DL Benchmark: semflow-dl vs Linear-SEM vs EBM vs XGBoost+SHAP")
    print(
        "| N | Engine | R2 (Oracle) | Oracle do(L) | Est do(L) | |Δdo| Err | Indirect Est | "
        "Load Credit% | Mono Viol% | Fit (ms) | Exp (ms) | Peak RSS (MB) |"
    )
    print("|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        if r.get("error"):
            print(f"| {r['n_total']:,} | `{r['engine']}` | ERR: {r['error']} | | | | | | | | | {r.get('peak_rss_mb')} |")
            continue
        print(
            f"| {r['n_total']:,} | `{r['engine']}` | {r['r2_oracle']:.4f} | {r['oracle_total']:+.3f} | "
            f"{r['do_total_est']:+.3f} | **{r['do_total_abs_err']:.3f}** | {r['do_indirect_est']:+.3f} | "
            f"{r['load_credit_pct']:.1f}% | {r['mono_viol_pct']:.1f}% | "
            f"{r['fit_ms']} | {r['explain_ms']} | {r['peak_rss_mb']} |"
        )
    print(f"\nSaved results to: {csv_path}")


if __name__ == "__main__":
    main()
