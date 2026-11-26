#!/usr/bin/env python3
"""
สร้างภาพ 3-Panel Publication Figure สำหรับเปเปอร์ที่ 1 (semflow — JOSS / SoftwareX):
  (a) Row Scalability: Fit Time (ms, log scale) vs N Rows (10K -> 5M)
  (b) Memory Efficiency: Peak RSS RAM (MB) vs N Rows (แสดงเพดาน 8 GB Laptop)
  (c) Parallel Bootstrap Speedup (B = 1,000 & 5,000): semflow vs cSEM (พร้อมป้าย 83x-122x / TIMEOUT)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

# Okabe-Ito Colorblind-Safe Palette
ENGINE_STYLE = {
    "semflow_py_cb":   {"label": "semflow (Py CB)",   "color": "#0072B2", "marker": "o", "ls": "-"},
    "semflow_r_cb":    {"label": "semflow (R CB)",    "color": "#56B4E9", "marker": "s", "ls": "--"},
    "semflow_py_pls":  {"label": "semflow (Py PLS)",  "color": "#009E73", "marker": "^", "ls": "-"},
    "semflow_py_plsc": {"label": "semflow (Py PLSc)", "color": "#009E73", "marker": "v", "ls": ":"},
    "semflow_r_pls":   {"label": "semflow (R PLS)",   "color": "#66C2A5", "marker": "D", "ls": "--"},
    "semopy_cb":       {"label": "semopy (Py CB)",    "color": "#D55E00", "marker": "X", "ls": "-."},
    "lavaan_cb":       {"label": "lavaan (R CB)",     "color": "#E69F00", "marker": "P", "ls": "-."},
    "csem_pls":        {"label": "cSEM (R PLS)",      "color": "#CC79A7", "marker": "d", "ls": ":"},
}


def configure_style() -> None:
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
        "legend.fontsize": 7.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def _load_valid_csv(path: Path) -> pl.DataFrame | None:
    if not path.exists():
        print(f"[INFO] Missing CSV (skipping panel data): {path}")
        return None
    df = pl.read_csv(path, infer_schema_length=1000)
    return df


def plot_panel_a_and_b(ax_time, ax_mem, df_rows: pl.DataFrame | None) -> None:
    if df_rows is None or df_rows.height == 0:
        ax_time.text(0.5, 0.5, "Run --suite rows\nto populate", ha="center", va="center", transform=ax_time.transAxes)
        ax_mem.text(0.5, 0.5, "Run --suite rows\nto populate", ha="center", va="center", transform=ax_mem.transAxes)
        return

    # กรองเฉพาะแถวที่รันผ่าน (fit_ms ไม่เป็น null)
    df_ok = df_rows.filter(pl.col("fit_ms").is_not_null())
    engines = [e for e in ENGINE_STYLE if e in df_ok["engine"].unique().to_list()]

    for eng in engines:
        sub = df_ok.filter(pl.col("engine") == eng).sort("n_rows")
        st = ENGINE_STYLE[eng]
        xs = sub["n_rows"].to_numpy()
        ys_time = sub["fit_ms"].to_numpy()
        ys_mem = sub["peak_rss_mb"].to_numpy()

        ax_time.plot(xs, ys_time, label=st["label"], color=st["color"], marker=st["marker"], ls=st["ls"], lw=2.0, ms=5.5)
        ax_mem.plot(xs, ys_mem, label=st["label"], color=st["color"], marker=st["marker"], ls=st["ls"], lw=2.0, ms=5.5)

    ax_time.set_xscale("log")
    ax_time.set_yscale("log")
    ax_time.set_title("(a) Fit Time Scalability ($p = 12$)")
    ax_time.set_xlabel("Number of Observations ($N$, log scale)")
    ax_time.set_ylabel("Median Fit Time (ms, log scale)")
    ax_time.legend(loc="upper left", frameon=True, framealpha=0.92, ncol=1)

    ax_mem.set_xscale("log")
    ax_mem.set_title("(b) Memory Footprint (Peak RSS)")
    ax_mem.set_xlabel("Number of Observations ($N$, log scale)")
    ax_mem.set_ylabel("Peak RSS Memory (MB)")
    ax_mem.legend(loc="upper left", frameon=True, framealpha=0.92, ncol=1)


def plot_panel_c_boot(ax_boot, df_boot: pl.DataFrame | None) -> None:
    if df_boot is None or df_boot.height == 0:
        ax_boot.text(0.5, 0.5, "Run --suite boot\nto populate", ha="center", va="center", transform=ax_boot.transAxes)
        return

    # สร้างป้ายกลุ่ม (N, B)
    cases = (
        df_boot.select(["n_rows", "n_boot"])
        .unique()
        .sort(["n_rows", "n_boot"])
        .iter_rows()
    )
    case_list = list(cases)
    x_labels = [f"N={n // 1000}K\nB={b // 1000}K" for n, b in case_list]
    x = np.arange(len(case_list))

    target_engines = ["semflow_py_pls", "semflow_py_plsc", "csem_pls"]
    width = 0.25

    for idx, eng in enumerate(target_engines):
        st = ENGINE_STYLE[eng]
        vals = []
        is_timeout = []
        for n, b in case_list:
            sub = df_boot.filter(
                (pl.col("n_rows") == n) & (pl.col("n_boot") == b) & (pl.col("engine") == eng)
            )
            if sub.height == 0:
                vals.append(0.0)
                is_timeout.append(False)
                continue
            fit_val = sub["fit_ms"][0]
            status_str = str(sub["converged"][0])
            if fit_val is None or "TIMEOUT" in status_str:
                vals.append(180_000.0)  # แสดงแท่งชนเพดาน 180s
                is_timeout.append(True)
            else:
                vals.append(float(fit_val))
                is_timeout.append(False)

        offset = (idx - 1) * width
        bars = ax_boot.bar(
            x + offset, np.array(vals) / 1000.0, width=width,
            color=st["color"], edgecolor="white", label=st["label"],
            hatch="///" if eng == "csem_pls" else None,
        )
        for bar, val_ms, to_flag in zip(bars, vals, is_timeout):
            if to_flag:
                ax_boot.text(
                    bar.get_x() + bar.get_width() / 2.0, bar.get_height() * 1.08,
                    ">180s\n(TIMEOUT)", ha="center", va="bottom",
                    fontsize=7.0, fontweight="bold", color="#D55E00",
                )
            elif eng == "semflow_py_pls" and val_ms > 0:
                ax_boot.text(
                    bar.get_x() + bar.get_width() / 2.0, bar.get_height() * 1.12,
                    f"{val_ms / 1000.0:.2f}s", ha="center", va="bottom",
                    fontsize=7.2, fontweight="bold", color=st["color"],
                )

    ax_boot.set_yscale("log")
    ax_boot.set_xticks(x)
    ax_boot.set_xticklabels(x_labels)
    ax_boot.set_ylim(0.1, 600.0)
    ax_boot.set_title("(c) Parallel Bootstrap ($B=1\\text{K}, 5\\text{K}$)")
    ax_boot.set_ylabel("Fit Time (seconds, log scale)")
    ax_boot.legend(loc="upper left", frameon=True, framealpha=0.92)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench-dir", type=Path, default=Path("benchmarks"))
    parser.add_argument("--out-dir", type=Path, default=Path("benchmarks/figures"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    configure_style()

    df_rows = _load_valid_csv(args.bench_dir / "results_paper_rows.csv")
    df_boot = _load_valid_csv(args.bench_dir / "results_paper_boot.csv")

    fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.1), constrained_layout=True)
    plot_panel_a_and_b(axes[0], axes[1], df_rows)
    plot_panel_c_boot(axes[2], df_boot)

    pdf_path = args.out_dir / "fig_semflow_benchmarks.pdf"
    png_path = args.out_dir / "fig_semflow_benchmarks.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved semflow benchmark figures:\n  - {pdf_path}\n  - {png_path}")


if __name__ == "__main__":
    main()
