"""Stage 1 redesign: /api/run (official 14-day forecast + descriptive context) and the new navigation.

Expected values are recomputed independently from forecast.py (called directly) and the saved data.
Run from the project root:
    PYTHONDONTWRITEBYTECODE=1 python -m pytest dashboard/backend/tests -v -p no:cacheprovider
"""
import hashlib
import importlib.util
import json
import os
import re
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

SRC = ROOT / "dashboard" / "frontend" / "src"


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def run(client):
    r = client.get("/api/run?as_of=2019-04-01")
    assert r.status_code == 200, r.text
    return r.json()


@pytest.fixture(scope="module")
def direct():
    os.environ["PS06_DATA_DIR"] = str(ROOT)
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location("submission_direct_run", ROOT / "forecast.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.forecast("2019-04-01")


@pytest.fixture(scope="module")
def daily():
    return pd.read_csv(ROOT / "outputs" / "daily_admissions.csv", parse_dates=["date"]).set_index("date")["admissions"]


@pytest.fixture(scope="module")
def festivals():
    f = pd.read_csv(ROOT / "festival_calendar_punjab.csv")
    f["date"] = pd.to_datetime(f["date"])
    return f


# ===================================================================== official output parity
def test_official_rows_equal_direct_forecast_py(run, direct):
    got = pd.DataFrame(run["official"])
    assert list(got.columns) == ["date", "mean", "lower_80", "upper_80"] and len(got) == 14
    assert list(got["date"]) == list(direct["date"])
    for col in ("mean", "lower_80", "upper_80"):
        assert np.allclose(got[col], direct[col], atol=6e-4), col          # API rounds to 3 decimals


def test_official_rows_equal_existing_historical_endpoint(client, run):
    hist = client.get("/api/forecast/historical").json()["forecast"]
    assert [{k: r[k] for k in ("date", "mean", "lower_80", "upper_80")} for r in hist] == run["official"]


def test_day_values_are_the_official_values_unchanged(run):
    for d, o in zip(run["days"], run["official"]):
        assert {k: d[k] for k in o} == o
    assert run["summary"]["total_mean"] == pytest.approx(sum(o["mean"] for o in run["official"]), abs=2e-3)


def test_no_actual_outcomes_for_forecast_dates(run):
    blob = json.dumps(run)
    assert not any(k in blob for k in ('"y"', '"actual', '"truth', '"observed"'))


def test_manifest(run):
    m = run["manifest"]
    assert run["mode"] == "demonstration" and "not a live hospital forecast" in run["label"]
    assert (m["as_of"], m["forecast_start"], m["forecast_end"], m["data_through"]) == ("2019-04-01", "2019-04-01", "2019-04-14", "2019-03-31")
    assert m["forecast_py_sha256"] == C.FROZEN_FORECAST_SHA256 and m["sha256_matches_frozen"] is True
    assert m["fallback_used"] is False and m["calendar_covered_until"] == "2019-12-28"
    assert "potentially incomplete" in m["data_through_note"]
    assert "different, current period" in m["period_note"] and "no influence" in m["period_note"]


# ===================================================================== inputs vs context
def test_weather_and_aqi_are_not_model_inputs(run):
    m = run["manifest"]
    assert {"Weather", "Air quality (AQI, pollutants)"} == set(m["not_model_inputs"])
    assert not any(re.search(r"weather|air|aqi|pollut", x, re.I) for x in m["model_inputs"])
    src = (ROOT / "forecast.py").read_text(encoding="utf-8")
    assert "weather_ludhiana_daily" not in src and "Pollution Data" not in src.split('"""', 2)[2]
    keys = set()

    def walk(o):
        if isinstance(o, dict):
            keys.update(o)
            [walk(v) for v in o.values()]
        elif isinstance(o, list):
            [walk(v) for v in o]
    walk(run)
    assert not keys & {"temperature", "temp_max_c", "aqi", "us_aqi", "pm25", "pm2_5", "precip_mm", "humidity"}


# ===================================================================== descriptive indicators
def test_historical_comparison_rule_recomputed(run, daily):
    cut = pd.Timestamp("2019-04-01")
    base = daily.loc[cut - pd.Timedelta(days=56): cut - pd.Timedelta(days=1)]
    assert len(base) == 56
    assert run["historical_comparison_rule"]["baseline_start"] == "2019-02-04"
    assert run["historical_comparison_rule"]["baseline_end"] == "2019-03-31"
    for d in run["days"]:
        t = pd.Timestamp(d["date"])
        v = base[base.index.dayofweek == t.dayofweek].to_numpy(dtype=float)
        p25, p75 = np.percentile(v, [25, 75])
        exp = "higher" if d["mean"] > p75 else "lower" if d["mean"] < p25 else "typical"
        hc = d["historical_comparison"]
        assert (hc["tier"], hc["n"]) == (exp, 8)
        assert hc["p25"] == pytest.approx(p25, abs=1e-3) and hc["p75"] == pytest.approx(p75, abs=1e-3)


def test_all_days_lower_reported_honestly(run):
    assert run["summary"]["historical_comparison_counts"] == {"higher": 0, "typical": 0, "lower": 14}
    assert all(d["historical_comparison"]["label"] == "Lower than recent historical baseline" for d in run["days"])
    r = run["historical_comparison_rule"]
    assert "No day in this forecast exceeds the higher-demand threshold" in r["no_higher_days_note"]
    assert "All 14 forecast days are below" in r["all_lower_note"]


def test_window_ranking_recomputed(run):
    means = np.array([o["mean"] for o in run["official"]])
    order = np.argsort(-means, kind="stable")
    assert [run["days"][i]["window_rank"] for i in order] == list(range(1, 15))
    top = [run["official"][i]["date"] for i in order[:3]]
    assert run["summary"]["busiest_days"] == top == ["2019-04-01", "2019-04-09", "2019-04-02"]
    assert [d["date"] for d in run["days"] if d["busiest_in_window"]] == sorted(top)
    assert run["summary"]["busiest"]["date"] == "2019-04-01" and run["summary"]["quietest"]["date"] == "2019-04-07"


def test_indicators_are_descriptive_not_capacity(run):
    for rule in (run["historical_comparison_rule"], run["window_ranking_rule"]):
        assert re.search(r"not a (hospital-)?capacity or staffing assessment", rule["descriptive_only"])
    blob = json.dumps(run).lower()
    for word in ("bed", "occupancy", "nurse"):
        assert not re.search(rf"\b{word}", blob), word
    for m in re.finditer(r"capacity|staffing", blob):
        assert "not a" in blob[max(0, m.start() - 30):m.start()], blob[m.start() - 40:m.end()]


# ===================================================================== calendar context
def test_holidays_and_adjacent_flags_from_official_calendar(run, festivals):
    hol = set(festivals["date"])
    for d in run["days"]:
        t = pd.Timestamp(d["date"])
        names = sorted(festivals.loc[festivals["date"] == t, "name"])
        assert sorted(h["name"] for h in d["holidays"]) == names
        assert d["day_before_holiday"] is ((t + pd.Timedelta(days=1)) in hol)
        assert d["day_after_holiday"] is ((t - pd.Timedelta(days=1)) in hol)
    flagged = {d["date"]: [h["name"] for h in d["holidays"]] for d in run["days"] if d["holidays"]}
    assert set(flagged) == {"2019-04-08", "2019-04-13", "2019-04-14"} and "Vaisakhi" in flagged["2019-04-14"]


def test_upcoming_holidays_next_30_days(run, festivals):
    cut = pd.Timestamp("2019-04-01")
    exp = festivals[(festivals.date >= cut) & (festivals.date < cut + pd.Timedelta(days=30))]
    got = run["upcoming_holidays"]
    assert sorted((h["date"], h["name"]) for h in got) == sorted(zip(exp.date.dt.strftime("%Y-%m-%d"), exp.name))
    assert all(h["in_forecast_window"] is (h["date"] <= "2019-04-14") for h in got)


def test_days_to_watch_only_busiest_and_holidays(run):
    kinds = {(w["kind"], w["date"]) for w in run["days_to_watch"]}
    assert kinds == {("busiest", d) for d in run["summary"]["busiest_days"]} | {
        ("holiday", d) for d in ("2019-04-08", "2019-04-13", "2019-04-14")}


# ===================================================================== reliability
def test_reliability_equals_saved_holdout(run):
    m = pd.read_csv(ROOT / "outputs" / "holdout" / "tables" / "metrics_primary_grid.csv", index_col=0).loc["M4 (forecast.py)"]
    r = run["reliability"]
    assert r["mae"] == pytest.approx(m.MAE, abs=1e-3) == pytest.approx(6.303, abs=1e-3)
    assert r["coverage80"] == pytest.approx(m.coverage80, abs=1e-3) == pytest.approx(0.842, abs=1e-3)
    assert r["bias"] == pytest.approx(m.bias, abs=1e-3) == pytest.approx(-4.581, abs=1e-3)
    assert (r["n_cutoffs"], r["n_points"]) == (19, 266)


# ===================================================================== contract & safety
@pytest.mark.parametrize("as_of", ["2019-03-01", "2026-11-24", "2019-04-02", "2019/04/01", "x", ""])
def test_only_the_demonstration_cutoff_is_served(client, as_of):
    r = client.get(f"/api/run?as_of={as_of}")
    assert r.status_code == 422 and "Demonstration mode" in r.json()["detail"]


def test_default_and_get_only(client, run):
    assert client.get("/api/run").json()["official"] == run["official"]
    assert client.post("/api/run").status_code == 405


def test_run_pii_free_and_read_only(client):
    files = [p for p in (ROOT / "outputs").rglob("*") if p.is_file()] + [ROOT / "forecast.py"]
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    for _ in range(2):
        r = client.get("/api/run")
        assert r.status_code == 200
        assert_no_pii(r.json())
    assert {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in files} == before
    assert before[ROOT / "forecast.py"] == C.FROZEN_FORECAST_SHA256
    assert not (ROOT / "__pycache__").exists()


def test_run_reports_runner_failure_without_inventing_values(client, monkeypatch):
    FR.clear_cache()

    def boom(*a, **k):
        raise FR.ForecastError("simulated")
    monkeypatch.setattr(FR, "run_forecast", boom)
    r = client.get("/api/run")
    assert r.status_code == 502 and "unavailable" in r.json()["detail"]
    FR.clear_cache()


# ===================================================================== navigation & wording
def test_navigation_groups_aliases_and_stages():
    router = (SRC / "lib" / "router.tsx").read_text(encoding="utf-8")
    def entry(path):
        return re.search(rf'\{{ path: "{re.escape(path)}".*?\}},', router, re.S).group(0)
    for p in ("/", "/forecast", "/conditions", "/calendar", "/insights"):
        assert 'group: "primary"' in entry(p), p
    for p in ("/performance", "/quality"):
        assert 'group: "secondary"' in entry(p) and "ready: true" in entry(p), p
    for p in ("/", "/forecast", "/conditions", "/calendar", "/insights"):   # all primary pages built (Stages 1-3)
        assert "ready: true" in entry(p), p
    assert '"/environment": "/conditions"' in router and '"/workload": "/forecast"' in router
    assert 'view === "retro" || view === "holdout"' in router and '"/performance"' in router
    assert not (SRC / "pages" / "ForecastStudio.tsx").exists()
    app_src = (SRC / "App.tsx").read_text(encoding="utf-8")
    for page in ("CommandCentre", "Forecast14", "ModelPerformance", "DataQuality"):
        assert f"<{page} />" in app_src, page


def test_model_performance_keeps_phase2_views():
    src = (SRC / "pages" / "ModelPerformance.tsx").read_text(encoding="utf-8")
    for part in ("function RetroView", "function HoldoutView", "function ModelExplainer", "<MissBreakdown", "<IntervalMeaning />",
                 "<BiasDisclosure />", 'id: "retro"', 'id: "holdout"'):
        assert part in src, part
    assert 'id: "historical"' not in src


def test_demonstration_labelling_and_no_weather_claims():
    parts = (SRC / "components" / "forecast" / "RunParts.tsx").read_text(encoding="utf-8")
    cc = (SRC / "pages" / "CommandCentre.tsx").read_text(encoding="utf-8")
    fc = (SRC / "pages" / "Forecast14.tsx").read_text(encoding="utf-8")
    assert "Historical hackathon demonstration – not a live hospital forecast" in parts
    assert "<RunBanner" in cc and "<RunBanner" in fc
    assert "Weather &amp; AQI: not model inputs" in parts
    assert "never feed into the April 2019 admission forecast" in re.sub(r"\s+", " ", cc)
    for src in (parts, cc, fc):
        flat = re.sub(r"\s+", " ", src).lower()
        for claim in ("because of the weather", "due to weather", "weather increases", "aqi increases", "pollution increases",
                      "weather-adjusted", "influenced by weather", "driven by air"):
            assert claim not in flat, claim
        for banned in ("staffing level", "beds needed", "bed occupancy", "nurses required"):
            assert banned not in flat, banned


def test_reliability_card_wording_mae_not_a_range():
    """MAE is reported as an average absolute error, never as a symmetric ± range or a guarantee."""
    parts = (SRC / "components" / "forecast" / "RunParts.tsx").read_text(encoding="utf-8")
    card = parts[parts.index("export function ReliabilityCard"):parts.index("// ====", parts.index("export function ReliabilityCard"))]
    assert '"Historical average absolute error"' in card
    assert "about ${fmt1(r.mae)} admissions/day (holdout MAE)" in card                # renders "about 6.3 admissions/day (holdout MAE)"
    assert "±${" not in card and "Typical daily error" not in card
    assert "${pct(r.coverage80)} of ${r.n_points} test days (target at least 80%)" in card   # 84.2% of 266
    assert "under-predicted by about ${Math.round(Math.abs(r.bias))} per day" in card    # bias -4.58 -> about 5
