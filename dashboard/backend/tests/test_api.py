"""Phase 1 dashboard backend tests.

Run from the project root:
    PYTHONDONTWRITEBYTECODE=1 python -m pytest dashboard/backend/tests -v -p no:cacheprovider
"""
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

BACKEND = Path(__file__).resolve().parents[1]
ROOT = BACKEND.parents[1]
sys.path.insert(0, str(BACKEND))
sys.path.insert(0, str(ROOT / "dashboard" / "tools"))
sys.path.insert(0, str(ROOT / "validation"))

from fastapi.testclient import TestClient  # noqa: E402

from app import config as C  # noqa: E402
from app import forecast_runner as FR  # noqa: E402
from app.datastore import PIIError, assert_no_pii  # noqa: E402
from app.main import app  # noqa: E402

ENDPOINTS = ["/api/health", "/api/meta", "/api/overview", "/api/overview?window_days=730",
             "/api/forecast/historical", "/api/admissions/daily", "/api/admissions/daily?series=dedup_exact",
             "/api/admissions/daily?series=dedup_mrd"]


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def daily():
    return pd.read_csv(ROOT / "outputs" / "daily_admissions.csv", parse_dates=["date"]).set_index("date")


@pytest.fixture(scope="module")
def direct_forecast():
    """forecast.py called directly (separate import, PS06_DATA_DIR = project root)."""
    os.environ["PS06_DATA_DIR"] = str(ROOT)
    spec = importlib.util.spec_from_file_location("submission_direct", ROOT / "forecast.py")
    m = importlib.util.module_from_spec(spec)
    sys.dont_write_bytecode = True
    spec.loader.exec_module(m)
    return m.forecast(C.HISTORICAL_CUTOFF)


def _strings(obj):
    if isinstance(obj, dict):
        for v in obj.values():
            yield from _strings(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _strings(v)
    elif isinstance(obj, str):
        yield obj


# ============================================================== endpoints & contracts
@pytest.mark.parametrize("url", ENDPOINTS)
def test_endpoints_ok_and_pii_free(client, url):
    r = client.get(url)
    assert r.status_code == 200, r.text
    assert_no_pii(r.json())


def test_only_get_allowed(client):
    assert client.post("/api/overview").status_code == 405


def test_meta_model_card(client):
    m = client.get("/api/meta").json()
    assert m["model"]["frozen"] is True and m["model"]["sha256"] == C.FROZEN_FORECAST_SHA256
    assert m["historical_forecast"]["cutoff"] == "2019-04-01" and m["historical_forecast"]["horizon_end"] == "2019-04-14"
    assert "not a live forecast" in m["historical_forecast"]["label"]
    assert m["data"]["potentially_incomplete"]["status"] == "potentially incomplete"
    assert "weather" in m["model"]["not_used"] and "air quality" in m["model"]["not_used"]


# ================================================================ historical forecast
def test_historical_forecast_matches_forecast_py(client, direct_forecast):
    h = client.get("/api/forecast/historical").json()
    got = pd.DataFrame(h["forecast"])
    assert list(got["date"]) == list(direct_forecast["date"])
    for col in ("mean", "lower_80", "upper_80"):
        assert np.allclose(got[col], direct_forecast[col], atol=6e-4)       # API rounds to 3 decimals
    assert h["fallback_used"] is False


def test_historical_forecast_contains_no_actuals(client):
    h = client.get("/api/forecast/historical").json()
    assert h["kind"] == "historical_forecast" and "not live" in h["label"]
    for row in h["forecast"]:
        assert set(row) == {"date", "mean", "lower_80", "upper_80", "weekday", "holidays"}
    assert all(o["date"] < h["cutoff"] for o in h["observed_before_cutoff"])
    assert max(o["date"] for o in h["observed_before_cutoff"]) == "2019-03-31"
    assert not any(k in json.dumps(h) for k in ('"y"', '"actual', '"truth'))


def test_forecast_dates_and_holidays(client):
    h = client.get("/api/forecast/historical").json()
    assert [r["date"] for r in h["forecast"]] == pd.date_range("2019-04-01", periods=14).strftime("%Y-%m-%d").tolist()
    hol = {r["date"]: r["holidays"] for r in h["forecast"] if r["holidays"]}
    assert "Vaisakhi" in hol["2019-04-14"] and hol.keys() == {"2019-04-08", "2019-04-13", "2019-04-14"}


# ============================================================================ overview
def test_overview_kpis_recompute(client, daily, direct_forecast):
    o = client.get("/api/overview").json()
    k = {x["id"]: x for x in o["kpis"]}
    assert k["total_records"]["value"] == int(daily["admissions"].sum()) == 15757
    assert k["avg_daily"]["value"] == pytest.approx(daily["admissions"].mean(), abs=1e-3)
    assert k["last_28"]["value"] == pytest.approx(daily.loc["2019-03-04":"2019-03-31", "admissions"].mean(), abs=1e-3)
    assert k["dedup_records"]["value"] == int(daily["admissions_dedup_mrd_doa"].sum()) == 14399
    assert k["dedup_records"]["provenance"] == "sensitivity" and k["total_records"]["provenance"] == "observed"
    assert k["forecast_avg"]["value"] == pytest.approx(direct_forecast["mean"].mean(), abs=1e-3)
    assert k["forecast_avg"]["provenance"] == "historical_forecast" and "not live" in k["forecast_avg"]["detail"]


def test_overview_trend_window_and_flags(client):
    for w in (30, 90, 365):
        t = client.get(f"/api/overview?window_days={w}").json()["trend"]
        assert len(t["points"]) == w and t["points"][-1]["date"] == "2019-03-31" and t["series"] == "raw"
        assert t["points"][-1]["potentially_incomplete"] is True
        assert sum(p["potentially_incomplete"] for p in t["points"]) == 1
    assert client.get("/api/overview?window_days=5").status_code == 422


def test_incomplete_day_is_only_ever_potentially_incomplete(client):
    for url in ENDPOINTS:
        for s in _strings(client.get(url).json()):
            for m in re.finditer(r"incomplete", s, re.I):
                assert re.search(r"potentially[ _-]$", s[max(0, m.start() - 12):m.start()].lower()), (url, s)


def test_alerts_cover_required_caveats(client):
    ids = {a["id"] for a in client.get("/api/overview").json()["alerts"]}
    assert {"historical", "potentially_incomplete_day", "repeat_rows", "model_bias"} <= ids


# ======================================================================== series
def test_series_are_separate_and_labelled(client, daily):
    sums = {}
    for key, col in (("raw", "admissions"), ("dedup_exact", "admissions_dedup_exact"), ("dedup_mrd", "admissions_dedup_mrd_doa")):
        r = client.get(f"/api/admissions/daily?series={key}").json()
        assert r["series"] == key and r["column"] == col and len(r["points"]) == 730
        assert r["is_scored_target"] is (key == "raw")
        sums[key] = sum(p["value"] for p in r["points"])
        assert sums[key] == int(daily[col].sum())
    assert sums == {"raw": 15757, "dedup_exact": 14872, "dedup_mrd": 14399}
    assert client.get("/api/admissions/daily?series=mixed").status_code == 422
    assert client.get("/api/admissions/daily?start=notadate").status_code == 422


# ====================================================================== PII guard
@pytest.mark.parametrize("bad", [{"MRD": 1}, {"x": {"SNO": 3}}, {"row_ref": "x"}, ["ok", "SNO 4693"], {"t": "MRD No. 234735"}])
def test_pii_guard_blocks(bad):
    with pytest.raises(PIIError):
        assert_no_pii(bad)


def test_pii_guard_allows_aggregate_language():
    assert_no_pii({"detail": "Rows repeating an MRD + admission date", "value": 15757})


def test_patient_level_sources_never_loaded():
    """No code string literal (docstrings excluded) references a patient-level source or column."""
    import ast
    literals = []
    for f in ("datastore.py", "main.py", "config.py", "forecast_runner.py"):
        tree = ast.parse((BACKEND / "app" / f).read_text(encoding="utf-8"))
        docs = {id(n.body[0].value) for n in ast.walk(tree)
                if isinstance(n, (ast.Module, ast.FunctionDef, ast.ClassDef)) and n.body
                and isinstance(n.body[0], ast.Expr) and isinstance(n.body[0].value, ast.Constant)}
        literals += [n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in docs]
    for name in ("HDHI Admission data", "HDHI Mortality", "duplicate_rows", "cleaning_log", "brought_dead"):
        assert not any(name in s for s in literals), name


# ====================================================================== runner safety
def test_runner_rejects_bad_cutoff():
    for bad in ("2019/04/01", "tomorrow", "", "2019-13-01"):
        with pytest.raises((FR.ForecastError, ValueError)):
            FR.run_forecast(bad)


def test_runner_refuses_modified_forecast_file(monkeypatch):
    FR.clear_cache()
    monkeypatch.setattr(C, "FROZEN_FORECAST_SHA256", "0" * 64)
    with pytest.raises(FR.ForecastError, match="differs from the frozen"):
        FR.run_forecast("2019-04-01")
    FR.clear_cache()


def test_runner_timeout_reported(monkeypatch):
    FR.clear_cache()

    def boom(*a, **k):
        raise subprocess.TimeoutExpired(cmd="python", timeout=1)
    monkeypatch.setattr(FR.subprocess, "run", boom)
    with pytest.raises(FR.ForecastError, match="timed out"):
        FR.run_forecast("2018-10-01")


@pytest.mark.parametrize("rows", [[], [{"date": "2019-04-01", "mean": 1, "lower_80": 2, "upper_80": 3}] * 14])
def test_runner_contract_validation(rows):
    with pytest.raises(FR.ForecastError):
        FR.validate_rows(rows, "2019-04-01")


def test_runner_writes_nothing_next_to_forecast_py():
    FR.clear_cache()
    FR.run_forecast("2019-04-01")
    assert not (ROOT / "__pycache__").exists()


# =========================================================== colours & integrity
def test_design_tokens_pass_contrast_and_cvd_checks():
    import color_check
    rows, failures = color_check.run()
    assert len(rows) >= 140 and failures == []


def test_forecast_py_and_protected_files_unchanged():
    from test_loaders import ORIGINAL_SHA256
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()  # noqa: E731
    assert sha(ROOT / "forecast.py") == C.FROZEN_FORECAST_SHA256
    for name, h in ORIGINAL_SHA256.items():
        assert sha(ROOT / name) == h, name
    for fx in ("step3_outputs_sha256.json", "src_and_step4_sha256.json", "holdout_outputs_sha256.json"):
        for path, h in json.loads((ROOT / "validation" / "fixtures" / fx).read_text()).items():
            assert sha(ROOT / path) == h, path
