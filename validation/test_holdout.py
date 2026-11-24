"""Tests for the Step 5.3 holdout evaluation outputs (outputs/holdout).

Checks that the evaluation used the frozen forecast.py, only pre-cut-off data, the exact baseline,
and that saved metrics are reproducible from the saved forecasts.

Run from the project root:
    PYTHONDONTWRITEBYTECODE=1 python -m pytest validation -v -p no:cacheprovider
"""
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import forecast as FC  # noqa: E402
from src import backtest as B  # noqa: E402
from test_eda import STEP2_SHA256  # noqa: E402
from test_loaders import ORIGINAL_SHA256, sha256  # noqa: E402

HO = os.path.join(ROOT, "outputs", "holdout")
FROZEN = "aa4809046af56166ceed6f3daab1521be093878a7d9c0ffaa9d93b784462c866"


@pytest.fixture(scope="module")
def P():
    return pd.read_csv(os.path.join(HO, "tables", "forecasts_primary_grid.csv"), parse_dates=["cutoff", "date"])


@pytest.fixture(scope="module")
def D():
    return pd.read_csv(os.path.join(HO, "tables", "forecasts_daily_sensitivity.csv"), parse_dates=["cutoff", "date"])


@pytest.fixture(scope="module")
def counts():
    return B.DataStore(ROOT).counts


def test_forecast_py_is_frozen_version():
    assert sha256(os.path.join(ROOT, "forecast.py")) == FROZEN
    assert json.load(open(os.path.join(HO, "tables", "summary.json")))["forecast_py_sha256"] == FROZEN


def test_cutoff_sets(P, D):
    grid = pd.date_range("2017-07-01", "2019-03-18", freq="4D")
    assert list(sorted(P.cutoff.unique())) == list(grid[grid >= "2019-01-01"])
    assert list(sorted(D.cutoff.unique())) == list(pd.date_range("2019-01-01", "2019-03-18"))
    assert P.date.max() <= pd.Timestamp("2019-03-30") and D.date.max() == pd.Timestamp("2019-03-31")
    assert (P.groupby("cutoff").size() == 14).all() and (D.groupby("cutoff").size() == 14).all()


def test_audit_every_folder_ended_before_cutoff():
    a = pd.read_csv(os.path.join(HO, "tables", "leakage_and_runtime_audit.csv"), parse_dates=["cutoff", "folder_last_doa"])
    assert len(a) == 19 + 77
    assert (a.folder_last_doa == a.cutoff - pd.Timedelta(days=1)).all()
    assert not a["stderr"].fillna("").str.contains("FALLBACK").any()
    assert a.seconds.max() < 900


def test_truth_and_baseline(P, counts):
    assert np.array_equal(P.y.to_numpy(), counts.reindex(P.date).to_numpy())
    for c, g in P.groupby("cutoff"):
        # both forecast weeks repeat the last observed week (cutoff-7 .. cutoff-1)
        src = [c - pd.Timedelta(days=7) + pd.Timedelta(days=(h - 1) % 7) for h in g.h]
        assert np.array_equal(g.b0.to_numpy(), counts.reindex(src).to_numpy())


def test_saved_forecasts_reproduce_in_process(P, counts):
    y = counts.astype(float)
    cal = FC.load_calendar(ROOT, pd.Timestamp("2019-03-18"), y.index.min())
    for c in sorted(P.cutoff.unique())[::6]:
        out = FC.forecast_from_history(y[y.index < c], cal, pd.Timestamp(c))
        g = P[P.cutoff == c].sort_values("h")
        assert np.allclose(out["mean"], g["mean"], atol=1e-9)
        assert np.allclose(out["lower_80"], g["lo"], atol=1e-9) and np.allclose(out["upper_80"], g["hi"], atol=1e-9)


def test_contract_holds_for_every_saved_forecast(D):
    assert np.isfinite(D[["mean", "lo", "hi"]].to_numpy()).all()
    assert (D.lo <= D["mean"]).all() and (D["mean"] <= D.hi).all() and (D.lo >= 0).all()


def test_metrics_table_recomputes(P):
    t = pd.read_csv(os.path.join(HO, "tables", "metrics_primary_grid.csv"), index_col=0)
    mae = (P["mean"] - P.y).abs().mean()
    mae0 = (P.b0 - P.y).abs().mean()
    cov = ((P.y >= P.lo) & (P.y <= P.hi)).mean()
    m4 = t.loc["M4 (forecast.py)"]
    assert m4.MAE == pytest.approx(mae) and m4.MAE_ratio_vs_B0 == pytest.approx(mae / mae0)
    assert m4.coverage80 == pytest.approx(cov)
    assert m4.WAPE == pytest.approx((P["mean"] - P.y).abs().sum() / P.y.sum())


def test_report_exists_and_labels_holdout():
    txt = open(os.path.join(HO, "holdout_report.md"), encoding="utf-8").read()
    assert "one-time evaluation of the frozen submission" in txt and "HOLDOUT" in txt


def test_earlier_files_unchanged():
    for name, h in ORIGINAL_SHA256.items():
        assert sha256(os.path.join(ROOT, name)) == h, name
    for name, h in STEP2_SHA256.items():
        assert sha256(os.path.join(ROOT, "outputs", name)) == h, name
    for fx in ("step3_outputs_sha256.json", "src_and_step4_sha256.json"):
        for path, h in json.load(open(os.path.join(HERE, "fixtures", fx))).items():
            assert hashlib.sha256(open(os.path.join(ROOT, path), "rb").read()).hexdigest() == h, path


# ============================================================== Step 5.4 finalisation
def test_step53_holdout_files_unchanged_by_finalisation():
    ref = json.load(open(os.path.join(HERE, "fixtures", "holdout_outputs_sha256.json")))
    assert len(ref) == 15
    for path, h in ref.items():
        assert hashlib.sha256(open(os.path.join(ROOT, path), "rb").read()).hexdigest() == h, path


def test_final_report_sections():
    txt = open(os.path.join(HO, "holdout_report.md"), encoding="utf-8").read()
    assert txt.count("<!-- finalized: build step 5.4 -->") == 1
    for h in ["## 1. Method", "## 2. Leakage", "## 3. Results", "## 4. Sensitivity", "## 5. Simulated harness",
              "## 6. Diagnosis", "### 6.2 Historical seasonal pattern vs recent admission level",
              "### 6.3 The changing duplicate-record rate", "## 7. Tuning, validation and holdout",
              "## 8. Why the holdout must not be reused", "## 9. Limitations", "## Files"]:
        assert txt.count(h) == 1, h
    assert "no alternative was run on the holdout" in txt


def test_finalisation_is_idempotent_and_keeps_sections_1_to_5(tmp_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("fin", os.path.join(ROOT, "analysis", "step5_4_finalize_report.py"))
    fin = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fin)
    cur = open(os.path.join(HO, "holdout_report.md"), encoding="utf-8").read()
    P, d, dup, ctx, met = fin.load()
    P, bm, bc, s = fin.diagnose(P, d, dup)
    assert fin.build(cur, bm, bc, s, ctx, met) == cur          # rebuilding from the final report is a no-op
    core = lambda t: t.split("## 1. Method", 1)[1].split("\n## 6.", 1)[0]  # noqa: E731
    assert core(fin.build(cur, bm, bc, s, ctx, met)) == core(cur)


def test_diagnosis_tables_recompute(P):
    bm = pd.read_csv(os.path.join(HO, "tables", "diagnosis_by_month.csv"), index_col=0)
    d = pd.read_csv(os.path.join(ROOT, "outputs", "daily_admissions.csv"), index_col=0, parse_dates=True)
    for m, g in P.groupby(P.date.dt.strftime("%Y-%m")):
        assert bm.loc[m, "forecast"] == pytest.approx(g["mean"].mean())
        assert bm.loc[m, "actual_raw"] == pytest.approx(g.y.mean())
        assert bm.loc[m, "actual_dedup"] == pytest.approx(d["admissions_dedup_mrd_doa"].reindex(g.date).mean())
    bc = pd.read_csv(os.path.join(HO, "tables", "diagnosis_by_cutoff.csv"), index_col=0)
    assert len(bc) == 19 and (bc.forecast_mean < bc.actual_raw).all()
    assert os.path.getsize(os.path.join(HO, "figures", "fig04_level_diagnosis.png")) > 10_000
