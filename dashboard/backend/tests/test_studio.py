"""Phase 2 (Forecasting Studio) backend tests.

Run from the project root:
    PYTHONDONTWRITEBYTECODE=1 python -m pytest dashboard/backend/tests -v -p no:cacheprovider
"""
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

BACKEND = Path(__file__).resolve().parents[1]
ROOT = BACKEND.parents[1]
sys.path.insert(0, str(BACKEND))

from fastapi.testclient import TestClient  # noqa: E402

from app import config as C  # noqa: E402
from app import forecast_runner as FR  # noqa: E402
from app.datastore import assert_no_pii  # noqa: E402
from app.main import app  # noqa: E402

HOLDOUT = ROOT / "outputs" / "holdout" / "tables"


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def direct():
    os.environ["PS06_DATA_DIR"] = str(ROOT)
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location("submission_direct_studio", ROOT / "forecast.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.fixture(scope="module")
def daily():
    return pd.read_csv(ROOT / "outputs" / "daily_admissions.csv", parse_dates=["date"]).set_index("date")


# ====================================================================== cut-off rules
def test_retro_bounds_never_reach_holdout():
    last_window_end = pd.Timestamp(C.RETRO_MAX_CUTOFF) + pd.Timedelta(days=13)
    assert last_window_end < pd.Timestamp(C.HOLDOUT_START)
    assert (C.RETRO_MIN_CUTOFF, C.RETRO_MAX_CUTOFF) == ("2017-07-01", "2018-12-15")


@pytest.mark.parametrize("cut", ["2017-07-01", "2018-12-15"])
def test_retro_accepts_boundaries(client, cut):
    r = client.get(f"/api/forecast/retrospective?cutoff={cut}")
    assert r.status_code == 200 and r.json()["cutoff"] == cut
    assert max(row["date"] for row in r.json()["forecast"]) < C.HOLDOUT_START


@pytest.mark.parametrize("cut", ["2017-06-30", "2018-12-16", "2018-12-31", "2019-01-01", "2019-02-13", "2019-03-31",
                                 "2019-04-01", "2016-01-01"])
def test_retro_rejects_out_of_range(client, cut):
    r = client.get(f"/api/forecast/retrospective?cutoff={cut}")
    assert r.status_code == 422 and "2018-12-15" in r.json()["detail"]


@pytest.mark.parametrize("cut", ["2018-02-30", "2018/10/02", "02-10-2018", "x", "", "2018-10-02T00:00"])
def test_retro_rejects_malformed(client, cut):
    assert client.get(f"/api/forecast/retrospective?cutoff={cut}").status_code == 422


def test_retro_requires_cutoff(client):
    assert client.get("/api/forecast/retrospective").status_code == 422


def test_explain_rejects_holdout_cutoffs(client):
    assert client.get("/api/model/explain?cutoff=2019-02-01").status_code == 422
    assert client.get("/api/model/explain?cutoff=2018-10-02").status_code == 200


# ============================================================= parity with forecast.py
@pytest.mark.parametrize("cut", ["2017-09-14", "2018-10-02", "2018-12-15"])
def test_retro_matches_direct_forecast_py(client, direct, cut):
    got = pd.DataFrame(client.get(f"/api/forecast/retrospective?cutoff={cut}").json()["forecast"])
    ref = direct.forecast(cut)
    assert list(got["date"]) == list(ref["date"])
    for col in ("mean", "lower_80", "upper_80"):
        assert np.allclose(got[col], ref[col], atol=6e-4), col


def test_retro_matches_step4_backtest_archive(client):
    arc = pd.read_csv(ROOT / "outputs" / "backtest" / "forecasts_selected_models.csv.gz", parse_dates=["cutoff"])
    m4 = arc[(arc.model == "M4_glm_Nall_K2_T0_L0") & (arc.cutoff >= "2018-07-01")]
    for c in sorted(m4.cutoff.unique())[::10]:
        cut = pd.Timestamp(c).strftime("%Y-%m-%d")
        got = pd.DataFrame(client.get(f"/api/forecast/retrospective?cutoff={cut}").json()["forecast"])
        ref = m4[m4.cutoff == c].sort_values("h")
        assert np.allclose(got["mean"], ref["mean"], atol=6e-4) and np.allclose(got["lower_80"], ref["lo"], atol=6e-4)
        assert np.allclose(got["upper_80"], ref["hi"], atol=6e-4)


# ============================================================ separation / no leakage
def test_forecast_rows_never_contain_actuals(client):
    for url in ("/api/forecast/historical", "/api/forecast/retrospective?cutoff=2018-10-02"):
        for row in client.get(url).json()["forecast"]:
            assert set(row) == {"date", "mean", "lower_80", "upper_80", "weekday", "holidays"}


def test_historical_has_no_actual_sections(client):
    h = client.get("/api/forecast/historical").json()
    assert not any(k.startswith("actual") or k == "metrics" for k in h)


def test_retro_actuals_separate_and_correct(client, daily):
    r = client.get("/api/forecast/retrospective?cutoff=2018-10-02").json()
    win = pd.date_range("2018-10-02", periods=14)
    assert r["actual_after_cutoff"]["series"] == "raw"
    assert r["actual_after_cutoff_dedup"]["series"] == "dedup_mrd"
    assert [p["value"] for p in r["actual_after_cutoff"]["points"]] == daily.loc[win, "admissions"].tolist()
    assert [p["value"] for p in r["actual_after_cutoff_dedup"]["points"]] == daily.loc[win, "admissions_dedup_mrd_doa"].tolist()
    assert all(o["date"] < "2018-10-02" for o in r["observed_before_cutoff"])
    assert r["metrics"]["scored_against"] == "raw records"


def test_retro_metrics_recompute(client, daily):
    r = client.get("/api/forecast/retrospective?cutoff=2018-10-02").json()
    f = pd.DataFrame(r["forecast"])
    y = np.array([p["value"] for p in r["actual_after_cutoff"]["points"]], dtype=float)
    e = f["mean"].to_numpy() - y
    m = r["metrics"]
    assert m["MAE"] == pytest.approx(abs(e).mean(), abs=2e-3) and m["bias"] == pytest.approx(e.mean(), abs=2e-3)
    inside = (y >= f["lower_80"].to_numpy()) & (y <= f["upper_80"].to_numpy())
    assert m["days_inside"] == int(inside.sum())
    base = [p["value"] for p in r["baseline"]["points"]]
    start = pd.Timestamp("2018-10-02") - pd.Timedelta(days=7)
    expected = [daily.at[start + pd.Timedelta(days=h % 7), "admissions"] for h in range(14)]
    assert base == expected                                      # exact harness seasonal naive
    assert m["MAE_baseline"] == pytest.approx(abs(np.array(base) - y).mean(), abs=2e-3)


# ================================================================ holdout read-only
def _holdout_hashes():
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in HOLDOUT.iterdir()}


def test_holdout_values_equal_saved_files(client):
    h = client.get("/api/evaluation/holdout").json()
    m = pd.read_csv(HOLDOUT / "metrics_primary_grid.csv", index_col=0)
    for k in ("MAE", "RMSE", "WAPE", "bias", "coverage80", "MAE_ratio_vs_B0"):
        assert h["metrics"]["primary"]["model"][k] == pytest.approx(m.loc["M4 (forecast.py)", k], abs=1e-3)
        assert h["metrics"]["primary"]["baseline"][k] == pytest.approx(m.loc["Seasonal naive (harness baseline)", k], abs=1e-3)
    assert h["metrics"]["primary"]["model"]["bias"] == pytest.approx(-4.581, abs=1e-3)
    assert h["ratio_ci"]["primary"] == pytest.approx(json.loads((HOLDOUT / "summary.json").read_text())["ci_primary"], abs=1e-3)
    assert len(h["by_horizon"]) == 14 and len(h["per_cutoff"]) == 19 and len(h["by_month"]) == 3
    assert h["sha256_matches_current"] is True


def test_holdout_endpoint_runs_no_model_and_writes_nothing(client, monkeypatch):
    before = _holdout_hashes()

    def forbidden(*a, **k):
        raise AssertionError("holdout endpoint must not run the model")
    monkeypatch.setattr(FR, "run_forecast", forbidden)
    monkeypatch.setattr(FR, "run_explain", forbidden)
    for _ in range(3):
        assert client.get("/api/evaluation/holdout").status_code == 200
    assert _holdout_hashes() == before


# ============================================================ model explanation
def test_explain_reproduces_forecast(client, direct):
    e = client.get("/api/model/explain").json()
    assert e["reproduces_forecast"] is True and e["fourier_harmonics"] == 2 and e["ridge"] == 1.0
    assert e["training"] == {"start": "2017-04-01", "end": "2019-03-31", "days": 730}
    assert e["weekday"][0] == {"day": "Monday", "multiplier": 1.0} and len(e["weekday"]) == 7
    assert [t["term"] for t in e["calendar"]] == ["Holiday", "Day before a holiday", "Day after a holiday"]
    assert len(e["annual_curve"]) == 365
    assert np.mean([p["multiplier"] for p in e["annual_curve"]]) == pytest.approx(1.0, abs=1e-3)


def test_explain_refuses_if_it_cannot_reproduce(monkeypatch):
    FR.clear_cache()
    real = FR.run_forecast

    def shifted(cut, *a, **k):
        r = dict(real(cut))
        r["rows"] = [{**row, "mean": row["mean"] + 1} for row in r["rows"]]
        return r
    monkeypatch.setattr(FR, "run_forecast", shifted)
    with pytest.raises(FR.ForecastError, match="does not reproduce"):
        FR.run_explain("2018-10-02")
    FR.clear_cache()


# ================================================================== PII & integrity
@pytest.mark.parametrize("url", ["/api/forecast/retrospective?cutoff=2018-01-10", "/api/evaluation/holdout", "/api/model/explain"])
def test_new_endpoints_pii_free(client, url):
    r = client.get(url)
    assert r.status_code == 200
    assert_no_pii(r.json())


def test_forecast_py_unchanged_after_studio_calls(client):
    client.get("/api/forecast/retrospective?cutoff=2018-05-05")
    client.get("/api/model/explain")
    assert hashlib.sha256((ROOT / "forecast.py").read_bytes()).hexdigest() == C.FROZEN_FORECAST_SHA256
    assert not (ROOT / "__pycache__").exists()
