from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import semflow as sf

GOLDEN_DIR = Path(__file__).parent / "golden"

HS_MODEL = """
    visual  =~ x1 + x2 + x3
    textual =~ x4 + x5 + x6
    speed   =~ x7 + x8 + x9
"""

PD_CB_MODEL = """
    ind60 =~ x1 + x2 + x3
    dem60 =~ y1 + y2 + y3 + y4
    dem65 =~ y5 + y6 + y7 + y8
    dem60 ~ ind60
    dem65 ~ ind60 + dem60
    y1 ~~ y5
    y2 ~~ y4 + y6
    y3 ~~ y7
    y4 ~~ y8
    y6 ~~ y8
"""

PD_PLS_MODEL = """
    ind60 =~ x1 + x2 + x3
    dem60 =~ y1 + y2 + y3 + y4
    dem65 =~ y5 + y6 + y7 + y8
    dem60 ~ ind60
    dem65 ~ ind60 + dem60
"""


def _norm_key(lhs: str, op: str, rhs: str) -> tuple[str, str, str]:
    """ทำให้ลำดับตัวแปรของคู่ covariance (~~) เป็นมาตรฐานเดียวกัน"""
    if op == "~~":
        a, b = sorted([lhs, rhs])
        return (a, op, b)
    return (lhs, op, rhs)


def _assert_cb_parity(csv_name: str, json_name: str, formula: str) -> None:
    df = pl.read_csv(GOLDEN_DIR / csv_name)
    golden = json.loads((GOLDEN_DIR / json_name).read_text())

    res = sf.SEM(formula, method="cb", max_iter=1000).fit(df).result_
    assert res.converged, f"CB-SEM did not converge on {csv_name}"

    # 1. ตรวจสอบ Fit Indices
    g_fit = golden["fit"]
    assert res.df == g_fit["df"], f"df mismatch: {res.df} vs {g_fit['df']}"
    for metric in ("chi_square", "cfi", "tli", "rmsea", "srmr"):
        actual = getattr(res, metric)
        expected = g_fit[metric]
        assert np.isclose(actual, expected, atol=1e-4, rtol=1e-4), (
            f"{metric} mismatch on {csv_name}: semflow={actual:.6f} vs lavaan={expected:.6f}"
        )

    # 2. ตรวจสอบ Parameter Estimates และ Standard Errors ทีละตัว
    sf_map = {
        _norm_key(lhs, op, rhs): (est, se, fixed)
        for lhs, op, rhs, est, se, fixed in res.params
    }

    max_est_err = 0.0
    max_se_err = 0.0

    for item in golden["params"]:
        key = _norm_key(item["lhs"], item["op"], item["rhs"])
        assert key in sf_map, f"Missing parameter in semflow: {key}"
        est, se, fixed = sf_map[key]

        est_err = abs(est - item["est"])
        max_est_err = max(max_est_err, est_err)
        assert est_err < 1e-5, (
            f"Estimate mismatch for {key}: semflow={est:.7f} vs lavaan={item['est']:.7f}"
        )

        if not fixed and item["se"] > 0.0:
            se_err = abs(se - item["se"])
            max_se_err = max(max_se_err, se_err)
            assert se_err < 5e-4, (
                f"SE mismatch for {key}: semflow={se:.6f} vs lavaan={item['se']:.6f}"
            )

    print(
        f"\n[PASS] {csv_name} vs lavaan | "
        f"max |Δest| = {max_est_err:.2e}, max |ΔSE| = {max_se_err:.2e}"
    )


def _assert_pls_parity(method: str, json_name: str) -> None:
    df = pl.read_csv(GOLDEN_DIR / "PoliticalDemocracy.csv")
    golden = json.loads((GOLDEN_DIR / json_name).read_text())

    res = sf.SEM(PD_PLS_MODEL, method=method, max_iter=500, tol=1e-8).fit(df).result_
    assert res.converged, f"{method} did not converge"

    # 1. ตรวจสอบ Path Coefficients
    sf_paths = {(src, dst): est for src, dst, est, _, _ in res.paths}
    max_path_err = 0.0
    for p in golden["paths"]:
        actual = sf_paths[(p["from"], p["to"])]
        err = abs(actual - p["est"])
        max_path_err = max(max_path_err, err)
        assert err < 1e-5, (
            f"Path mismatch ({p['from']} -> {p['to']}): "
            f"semflow={actual:.7f} vs cSEM={p['est']:.7f}"
        )

    # 2. ตรวจสอบ Outer Loadings
    sf_loads = {(lat, ind): est for lat, ind, est in res.loadings}
    max_load_err = 0.0
    for ld in golden["loadings"]:
        actual = sf_loads[(ld["latent"], ld["indicator"])]
        err = abs(actual - ld["est"])
        max_load_err = max(max_load_err, err)
        assert err < 1e-5, (
            f"Loading mismatch ({ld['latent']} =~ {ld['indicator']}): "
            f"semflow={actual:.7f} vs cSEM={ld['est']:.7f}"
        )

    # 3. ตรวจสอบ R2, AVE, CR และ rho_A
    sf_metrics = {
        lat: {"rho_a": rho, "ave": ave, "cr": cr, "r2": r2}
        for lat, _, rho, ave, cr, r2 in res.metrics
    }
    for lat, g_r2 in golden["r2"].items():
        assert abs(sf_metrics[lat]["r2"] - g_r2) < 1e-5
    for lat, g_ave in golden["ave"].items():
        assert abs(sf_metrics[lat]["ave"] - g_ave) < 1e-4
    for lat, g_cr in golden["cr"].items():
        assert abs(sf_metrics[lat]["cr"] - g_cr) < 1e-4

    print(
        f"\n[PASS] PoliticalDemocracy ({method}) vs cSEM | "
        f"max |Δpath| = {max_path_err:.2e}, max |Δloading| = {max_load_err:.2e}"
    )


def test_cb_parity_holzinger_swineford():
    _assert_cb_parity("HolzingerSwineford1939.csv", "lavaan_hs1939.json", HS_MODEL)


def test_cb_parity_political_democracy():
    _assert_cb_parity("PoliticalDemocracy.csv", "lavaan_poldem.json", PD_CB_MODEL)


def test_pls_parity_political_democracy():
    _assert_pls_parity("pls", "csem_poldem_pls.json")


def test_plsc_parity_political_democracy():
    _assert_pls_parity("plsc", "csem_poldem_plsc.json")
