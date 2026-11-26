#!/usr/bin/env python3
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


def build_formula_and_parquet(n_rows: int, n_vars: int, pq_path: Path, fml_path: Path) -> None:
    """สร้างโมเดล SEM ขนาด p ตัวแปร (3 indicators ต่อ 1 latent) พร้อมข้อมูลสังเคราะห์ float32"""
    assert n_vars % 3 == 0 and n_vars >= 6, "n_vars must be a multiple of 3 (>= 6)"
    k = n_vars // 3
    rng = np.random.default_rng(2026)

    # สร้างโครงสร้าง DAG แบบโซ่ + skip connection เพื่อให้โมเดลสมจริงทั้ง p=12, 30, 60
    latents = []
    for j in range(k):
        eps = rng.standard_normal(n_rows, dtype=np.float32)
        if j == 0:
            z = eps
        elif j == 1:
            z = np.float32(0.70) * latents[0] + np.float32(0.71) * eps
        else:
            z = (
                np.float32(0.55) * latents[j - 1]
                + np.float32(0.20) * latents[j - 2]
                + np.float32(0.65) * eps
            )
        latents.append(z)

    cols = {}
    meas_lines = []
    for j in range(k):
        inds = []
        for m, lam in enumerate((0.88, 0.84, 0.80), start=1):
            col_name = f"x{j + 1}_{m}"
            sd = np.float32(np.sqrt(1.0 - lam**2))
            cols[col_name] = np.float32(lam) * latents[j] + sd * rng.standard_normal(n_rows, dtype=np.float32)
            inds.append(col_name)
        meas_lines.append(f"    F{j + 1} =~ " + " + ".join(inds))

    struct_lines = []
    for j in range(1, k):
        if j == 1:
            struct_lines.append(f"    F{j + 1} ~ F{j}")
        else:
            struct_lines.append(f"    F{j + 1} ~ F{j} + F{j - 1}")

    formula = "\n".join(meas_lines + struct_lines)
    fml_path.write_text(formula)
    pl.DataFrame(cols).write_parquet(pq_path)


def run_python_worker(engine: str, pq_path: Path, fml_path: Path, n_boot: int, n_runs: int, out_json: Path) -> None:
    """รันฝั่ง Python แยก Subprocess (แยกเวลา import ออกจาก load และ fit)"""
    formula = fml_path.read_text()

    if engine.startswith("semflow_py"):
        import semflow as sf
    elif engine == "semopy_cb":
        import pandas as pd
        import semopy
    else:
        raise ValueError(f"Unknown Python engine: {engine}")

    def run_once():
        t0 = time.perf_counter()
        if engine.startswith("semflow_py"):
            df = pl.read_parquet(pq_path)
            t_load = time.perf_counter()
            if engine == "semflow_py_cb":
                m = sf.SEM(formula, method="cb", estimator="ml", max_iter=800).fit(df)
            elif engine == "semflow_py_pls":
                m = sf.SEM(formula, method="pls", n_boot=n_boot, max_iter=300, tol=1e-6).fit(df)
            elif engine == "semflow_py_plsc":
                m = sf.SEM(formula, method="plsc", n_boot=n_boot, max_iter=300, tol=1e-6).fit(df)
            t_fit = time.perf_counter()
            conv = bool(m.result_.converged)
        else:
            df = pd.read_parquet(pq_path)
            t_load = time.perf_counter()
            m = semopy.Model(formula)
            opt = m.fit(df, obj="MLW")
            t_fit = time.perf_counter()
            conv = bool(getattr(opt, "success", True))
        return (t_load - t0) * 1000.0, (t_fit - t_load) * 1000.0, (t_fit - t0) * 1000.0, conv

    # Warmup 1 รอบ
    run_once()

    loads, fits, tots, convs = [], [], [], []
    for _ in range(n_runs):
        ld, ft, tt, cv = run_once()
        loads.append(ld)
        fits.append(ft)
        tots.append(tt)
        convs.append(cv)

    out_json.write_text(json.dumps({
        "engine": engine,
        "load_ms": round(statistics.median(loads), 2),
        "fit_ms": round(statistics.median(fits), 2),
        "total_ms": round(statistics.median(tots), 2),
        "converged": all(convs),
    }))


def run_child_with_wait4(cmd: list[str], out_json: Path, timeout_sec: int = 600) -> dict:
    """ใช้ os.wait4(pid, WNOHANG) วัด Peak RSS แม่นยำทั้ง Python และ R บน macOS/Linux"""
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
            exit_ok = os.WIFEXITED(status) and os.WEXITSTATUS(status) == 0
            stderr_txt = proc.stderr.read().strip() if proc.stderr else ""
            proc.returncode = os.WEXITSTATUS(status) if os.WIFEXITED(status) else -1
            if not exit_ok or not out_json.exists():
                last_err = stderr_txt.splitlines()[-1] if stderr_txt else f"exit={status}"
                return {
                    "load_ms": None, "fit_ms": None, "total_ms": None,
                    "peak_rss_mb": round(rss_mb, 2),
                    "converged": f"ERR: {last_err[:60]}",
                }
            data = json.loads(out_json.read_text())
            data["peak_rss_mb"] = round(rss_mb, 2)
            return data

        if time.monotonic() - t_start > timeout_sec:
            proc.kill()
            _, _, rusage = os.wait4(proc.pid, 0)
            rss_mb = (
                rusage.ru_maxrss / (1024.0 * 1024.0)
                if sys.platform == "darwin"
                else rusage.ru_maxrss / 1024.0
            )
            return {
                "load_ms": None, "fit_ms": None, "total_ms": None,
                "peak_rss_mb": round(rss_mb, 2),
                "converged": f"TIMEOUT ({timeout_sec}s)",
            }
        time.sleep(0.05)


def dispatch_engine(
    engine: str, pq_path: Path, fml_path: Path, n_boot: int, n_runs: int, tmp: Path, timeout_sec: int
) -> dict:
    out_json = tmp / f"out_{engine}_{n_boot}.json"
    if out_json.exists():
        out_json.unlink()

    if engine.startswith("semflow_py") or engine == "semopy_cb":
        cmd = [
            sys.executable, str(Path(__file__).resolve()),
            "--py-worker", engine,
            "--parquet", str(pq_path),
            "--formula", str(fml_path),
            "--boot", str(n_boot),
            "--runs", str(n_runs),
            "--out-json", str(out_json),
        ]
    else:
        r_worker = Path(__file__).parent / "r_worker.R"
        cmd = [
            "Rscript", str(r_worker),
            engine, str(pq_path), str(fml_path),
            str(n_boot), str(n_runs), str(out_json),
        ]
    return run_child_with_wait4(cmd, out_json, timeout_sec=timeout_sec)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=["rows", "dims", "boot"], default="rows")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=300)
    # Internal Python worker flags
    parser.add_argument("--py-worker", type=str, default=None)
    parser.add_argument("--parquet", type=Path, default=None)
    parser.add_argument("--formula", type=Path, default=None)
    parser.add_argument("--boot", type=int, default=0)
    parser.add_argument("--out-json", type=Path, default=None)
    args = parser.parse_args()

    if args.py_worker:
        run_python_worker(args.py_worker, args.parquet, args.formula, args.boot, args.runs, args.out_json)
        return

    # กำหนดแผนการทดลองทั้ง 3 Suite
    if args.suite == "rows":
        cases = [(n, 12, 0) for n in (10_000, 100_000, 1_000_000, 5_000_000)]
        engines = [
            "semflow_py_cb", "semflow_r_cb", "semopy_cb", "lavaan_cb",
            "semflow_py_pls", "semflow_r_pls", "csem_pls",
        ]
    elif args.suite == "dims":
        cases = [(100_000, p, 0) for p in (12, 30, 60)]
        engines = [
            "semflow_py_cb", "semflow_r_cb", "semopy_cb", "lavaan_cb",
            "semflow_py_pls", "csem_pls",
        ]
    else:  # boot
        cases = [(2_000, 12, 1_000), (2_000, 12, 5_000), (10_000, 12, 1_000), (10_000, 12, 5_000)]
        engines = ["semflow_py_pls", "semflow_r_pls", "semflow_py_plsc", "csem_pls", "csem_pls_par"]

    results = []
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        for n_rows, n_vars, n_boot in cases:
            pq_path = tmp / f"data_{n_rows}_{n_vars}.parquet"
            fml_path = tmp / f"model_{n_vars}.txt"
            if not pq_path.exists():
                print(f"[GEN] N={n_rows:,}, p={n_vars} ...", file=sys.stderr)
                build_formula_and_parquet(n_rows, n_vars, pq_path, fml_path)

            for eng in engines:
                # 5,000 bootstrap บน cSEM ใช้เวลาสูง ให้รัน 1 รอบพอเพื่อไม่ให้รอหลายชั่วโมง
                runs = 1 if (eng.startswith("csem") and n_boot >= 1000) else args.runs
                print(f"[RUN] suite={args.suite} N={n_rows:,} p={n_vars} B={n_boot} engine={eng}", file=sys.stderr)
                res = dispatch_engine(eng, pq_path, fml_path, n_boot, runs, tmp, args.timeout)
                results.append({
                    "suite": args.suite,
                    "n_rows": n_rows,
                    "n_vars": n_vars,
                    "n_boot": n_boot,
                    "engine": eng,
                    "load_ms": res.get("load_ms"),
                    "fit_ms": res.get("fit_ms"),
                    "total_ms": res.get("total_ms"),
                    "peak_rss_mb": res.get("peak_rss_mb"),
                    "converged": res.get("converged"),
                })

    csv_path = Path(__file__).parent / f"results_paper_{args.suite}.csv"
    fields = ["suite", "n_rows", "n_vars", "n_boot", "engine", "load_ms", "fit_ms", "total_ms", "peak_rss_mb", "converged"]
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(results)

    print(f"\n### Benchmark Suite: `{args.suite}`")
    print("| N Rows | p | Boot | Engine | Load (ms) | Fit (ms) | Total (ms) | Peak RSS (MB) | Status |")
    print("|---:|---:|---:|---|---:|---:|---:|---:|---|")
    for r in results:
        print(
            f"| {r['n_rows']:,} | {r['n_vars']} | {r['n_boot']:,} | `{r['engine']}` | "
            f"{r['load_ms']} | {r['fit_ms']} | {r['total_ms']} | {r['peak_rss_mb']} | {r['converged']} |"
        )
    print(f"\nSaved: {csv_path}")


if __name__ == "__main__":
    main()
