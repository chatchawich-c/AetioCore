#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import resource
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import polars as pl

SCADA_FORMULA = """
    Wind   =~ w1 + w2 + w3
    Load   =~ l1 + l2 + l3
    Stress =~ s1 + s2 + s3
    Deg    =~ d1 + d2 + d3
    Load   ~ Wind
    Stress ~ Load + Wind
    Deg    ~ Stress + Load
"""


def get_peak_rss_mb() -> float:
    """คืนค่า Peak RSS เป็น MB ให้ถูกต้องทั้ง macOS (bytes) และ Linux (KB)"""
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return usage / (1024.0 * 1024.0)
    return usage / 1024.0


def generate_scada_parquet(n_rows: int, path: Path) -> None:
    """สร้างชุดข้อมูลสังเคราะห์ 12 คอลัมน์ float32 ขนาดเบา (1M แถว ~ 48 MB)"""
    rng = np.random.default_rng(42)
    wind = rng.standard_normal(n_rows, dtype=np.float32)
    load = 0.75 * wind + rng.standard_normal(n_rows, dtype=np.float32) * 0.66
    stress = 0.65 * load + 0.20 * wind + rng.standard_normal(n_rows, dtype=np.float32) * 0.55
    deg = 0.70 * stress + 0.15 * load + rng.standard_normal(n_rows, dtype=np.float32) * 0.50

    def noisy(base: np.ndarray, loading: float) -> np.ndarray:
        sd = np.float32(np.sqrt(max(1.0 - loading**2, 0.05)))
        return (np.float32(loading) * base + rng.standard_normal(n_rows, dtype=np.float32) * sd)

    df = pl.DataFrame({
        "w1": noisy(wind, 0.88), "w2": noisy(wind, 0.84), "w3": noisy(wind, 0.80),
        "l1": noisy(load, 0.86), "l2": noisy(load, 0.82), "l3": noisy(load, 0.79),
        "s1": noisy(stress, 0.87), "s2": noisy(stress, 0.83), "s3": noisy(stress, 0.78),
        "d1": noisy(deg, 0.89), "d2": noisy(deg, 0.85), "d3": noisy(deg, 0.81),
    })
    df.write_parquet(path)


def run_worker(engine: str, parquet_path: Path, n_runs: int = 3) -> dict:
    """รันใน Subprocess แยกเพื่อวัด Wall-clock Time (แยก import + warmup) และ Peak RSS จริง"""
    import statistics

    if engine.startswith("semflow_"):
        import semflow as sf
        method = {"semflow_pls_f32": "pls", "semflow_plsc_f32": "plsc", "semflow_cb_f32": "cb"}[engine]
        max_iter = 500 if method == "cb" else 300

        def _once():
            t0 = time.perf_counter()
            df = pl.read_parquet(parquet_path)
            t1 = time.perf_counter()
            m = sf.SEM(SCADA_FORMULA, method=method, max_iter=max_iter, tol=1e-6).fit(df)
            t2 = time.perf_counter()
            return (t1 - t0) * 1000.0, (t2 - t1) * 1000.0, bool(m.result_.converged)

    elif engine == "semopy_cb":
        import pandas as pd
        import semopy

        def _once():
            t0 = time.perf_counter()
            df = pd.read_parquet(parquet_path)
            t1 = time.perf_counter()
            m = semopy.Model(SCADA_FORMULA)
            opt = m.fit(df, obj="MLW")
            t2 = time.perf_counter()
            return (t1 - t0) * 1000.0, (t2 - t1) * 1000.0, bool(getattr(opt, "success", True))

    else:
        raise ValueError(f"Unknown engine: {engine}")

    # Warmup 1 รอบ (ตัด Cold-Start Import / JIT / Cache Anomaly)
    _once()

    load_times, fit_times, total_times = [], [], []
    converged = True
    for _ in range(n_runs):
        ld, ft, conv = _once()
        load_times.append(ld)
        fit_times.append(ft)
        total_times.append(ld + ft)
        converged = converged and conv

    return {
        "engine": engine,
        "load_ms": round(statistics.median(load_times), 2),
        "fit_ms": round(statistics.median(fit_times), 2),
        "total_ms": round(statistics.median(total_times), 2),
        "peak_rss_mb": round(get_peak_rss_mb(), 2),
        "converged": converged,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", type=str, default=None)
    parser.add_argument("--parquet", type=Path, default=None)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[10_000, 100_000, 1_000_000],
        help="จำนวนแถวข้อมูลสำหรับ Benchmark",
    )
    parser.add_argument(
        "--engines",
        type=str,
        nargs="+",
        default=["semflow_pls_f32", "semflow_plsc_f32", "semflow_cb_f32", "semopy_cb"],
    )
    args = parser.parse_args()

    # โหมด Worker (ถูกเรียกจาก Subprocess)
    if args.worker is not None:
        out = run_worker(args.worker, args.parquet, args.runs)
        print(json.dumps(out))
        return

    # โหมด Driver หลัก
    rows = []
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        for n in args.sizes:
            pq_path = tmp / f"scada_{n}.parquet"
            print(f"Generating synthetic dataset N = {n:,} ...", file=sys.stderr)
            generate_scada_parquet(n, pq_path)

            for eng in args.engines:
                cmd = [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--worker", eng,
                    "--parquet", str(pq_path),
                ]
                proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                if proc.returncode != 0:
                    err_msg = proc.stderr.strip().splitlines()[-1] if proc.stderr else "failed"
                    rows.append({
                        "n_rows": n, "engine": eng,
                        "load_ms": None, "fit_ms": None, "total_ms": None,
                        "peak_rss_mb": None, "converged": f"ERR: {err_msg}",
                    })
                    continue

                res = json.loads(proc.stdout.strip().splitlines()[-1])
                res["n_rows"] = n
                rows.append(res)

    # บันทึกผลเป็น CSV
    csv_out = Path(__file__).parent / "results_scalability.csv"
    fieldnames = ["n_rows", "engine", "load_ms", "fit_ms", "total_ms", "peak_rss_mb", "converged"]
    with csv_out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # พิมพ์ตาราง Markdown พร้อมนำไปใช้ในเอกสารวิจัย
    print("\n| N Rows | Engine | Load (ms) | Fit (ms) | Total (ms) | Peak RSS (MB) | Converged |")
    print("|---:|---|---:|---:|---:|---:|---|")
    for r in rows:
        print(
            f"| {r['n_rows']:,} | `{r['engine']}` | {r['load_ms']} | "
            f"{r['fit_ms']} | {r['total_ms']} | {r['peak_rss_mb']} | {r['converged']} |"
        )
    print(f"\nSaved benchmark results to: {csv_out}")


if __name__ == "__main__":
    main()
