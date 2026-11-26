#!/usr/bin/env python3
"""
Real-World Case Study 2: Large-Scale Social Science / Psychometrics (Big Five IPIP-FFM)
- ข้อมูลเปิดจริง: OpenPsychometrics Big Five (N = 1,015,342, 50 indicators, ค่าว่างระบุเป็น 0 หรือ NULL)
  ดาวน์โหลดได้ที่ Kaggle / OpenPsychometrics (data-final.csv) แล้วแปลงเป็น Parquet
- โหมดเปรียบเทียบ: 5-Factor CFA (CB-SEM ML + FIML) และ Parallel Bootstrap PLS
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import polars as pl
import semflow as sf

# 5 ปัจจัยหลัก (Extraversion, Neuroticism, Agreeableness, Conscientiousness, Openness) x 10 ข้อ = 50 ตัวชี้วัด
BIGFIVE_50_FORMULA = "\n".join([
    "EXT =~ " + " + ".join(f"EXT{i}" for i in range(1, 11)),
    "EST =~ " + " + ".join(f"EST{i}" for i in range(1, 11)),
    "AGR =~ " + " + ".join(f"AGR{i}" for i in range(1, 11)),
    "CSN =~ " + " + ".join(f"CSN{i}" for i in range(1, 11)),
    "OPN =~ " + " + ".join(f"OPN{i}" for i in range(1, 11)),
])


def load_bigfive_ipip(path: Path | None, n_fallback: int = 100_000) -> pl.DataFrame:
    item_cols = [f"{tr}{i}" for tr in ("EXT", "EST", "AGR", "CSN", "OPN") for i in range(1, 11)]

    if path is not None and path.exists():
        # อ่านไฟล์ OpenPsychometrics data-final.csv (tab-separated) หรือ .parquet
        if path.suffix == ".parquet":
            df = pl.read_parquet(path, columns=item_cols)
        else:
            df = pl.read_csv(path, separator="\t", columns=item_cols, null_values=["NONE", "NULL", ""])
        # ในชุด IPIP-FFM ค่า 0 หมายถึงผู้ตอบข้ามข้อ (Missing) -> แปลงเป็น NaN/null สำหรับ FIML
        df = df.select([
            pl.when(pl.col(c).cast(pl.Float32) == 0.0)
            .then(None)
            .otherwise(pl.col(c).cast(pl.Float32))
            .alias(c)
            for c in item_cols
        ])
        print(f"Loaded real IPIP-FFM Big Five: {df.height:,} respondents x {df.width} items")
        return df

    # Fallback จำลองโครงสร้าง 5 ปัจจัย (50 ข้อคำถาม สเกล Likert 1-5 + Missing 5%) สำหรับทดสอบทันที
    rng = np.random.default_rng(42)
    cov_5 = np.full((5, 5), 0.20)
    np.fill_diagonal(cov_5, 1.0)
    factors = rng.multivariate_normal(np.zeros(5), cov_5, size=n_fallback).astype(np.float32)

    cols = {}
    for f_idx, tr in enumerate(("EXT", "EST", "AGR", "CSN", "OPN")):
        for i in range(1, 11):
            raw = 3.0 + 0.75 * factors[:, f_idx] + rng.standard_normal(n_fallback, dtype=np.float32) * 0.65
            likert = np.clip(np.round(raw), 1.0, 5.0).astype(np.float32)
            # จำลองผู้ตอบข้ามข้อ 3%
            miss_mask = rng.random(n_fallback) < 0.03
            likert[miss_mask] = np.nan
            cols[f"{tr}{i}"] = likert
    return pl.DataFrame(cols)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=None, help="Path ถึง data-final.csv หรือ .parquet")
    parser.add_argument("--n-rows", type=int, default=100_000)
    args = parser.parse_args()

    df = load_bigfive_ipip(args.data, n_fallback=args.n_rows)
    null_cells = sum(df[c].is_nan().sum() + df[c].null_count() for c in df.columns)
    print(f"Dataset shape: {df.height:,} x {df.width} | Missing cells: {null_cells:,}")

    # 1. รัน 5-Factor CFA (p = 50 ตัวแปร!) บน Complete Cases ด้วย semflow CB-SEM
    df_complete = df.drop_nulls().drop_nans()
    print(f"\n[1] 5-Factor CFA (50 indicators, Listwise N = {df_complete.height:,})")
    t0 = time.perf_counter()
    m_cb = sf.SEM(BIGFIVE_50_FORMULA, method="cb", estimator="ml", max_iter=500).fit(df_complete)
    r = m_cb.result_
    print(
        f"Elapsed: {(time.perf_counter() - t0) * 1000:.1f} ms | "
        f"converged={r.converged} ({r.iterations} iters) | "
        f"CFI={r.cfi:.4f}, TLI={r.tli:.4f}, RMSEA={r.rmsea:.4f}, SRMR={r.srmr:.4f}"
    )


if __name__ == "__main__":
    main()
