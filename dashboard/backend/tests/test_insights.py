"""Stage 3: Prediction Insights page (client-facing). Uses only /api/model/explain and /api/run.

Run from the project root:
    PYTHONDONTWRITEBYTECODE=1 python -m pytest dashboard/backend/tests/test_insights.py -v -p no:cacheprovider
"""
import hashlib
import re
import sys
from pathlib import Path

import numpy as np
import pytest

BACKEND = Path(__file__).resolve().parents[1]
ROOT = BACKEND.parents[1]
sys.path.insert(0, str(BACKEND))

from fastapi.testclient import TestClient  # noqa: E402

from app import config as C  # noqa: E402
from app.main import app  # noqa: E402

SRC = ROOT / "dashboard" / "frontend" / "src"
PAGE = SRC / "pages" / "Insights.tsx"


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def flat():
    return re.sub(r"\s+", " ", PAGE.read_text(encoding="utf-8"))


def test_explain_api_supplies_every_value_the_page_shows(client):
    e = client.get("/api/model/explain").json()
    assert e["reproduces_forecast"] is True and e["fourier_harmonics"] == 2 and e["training"]["days"] == 730
    assert e["weekday"][0] == {"day": "Monday", "multiplier": 1.0} and len(e["weekday"]) == 7
    assert [t["term"] for t in e["calendar"]] == ["Holiday", "Day before a holiday", "Day after a holiday"]
    assert len(e["annual_curve"]) == 365 and np.mean([p["multiplier"] for p in e["annual_curve"]]) == pytest.approx(1.0, abs=1e-3)
    run = client.get("/api/run").json()
    assert all(d["lower_80"] <= d["mean"] <= d["upper_80"] for d in run["days"]) and len(run["days"]) == 14


def test_page_uses_existing_apis_and_components_only():
    src = PAGE.read_text(encoding="utf-8")
    assert set(re.findall(r"\buse[A-Z]\w+(?=\()", src)) == {"useModelExplain", "useForecastRun"}
    assert "WeekdayMultipliers" in src and "AnnualCurve" in src and "<RunBanner" in src
    assert not re.search(r"(?<![A-Za-z])fetch\(", src)            # no direct network calls (refetch() is fine)


def test_no_hard_coded_model_values(flat):
    """Multipliers, peaks and ranks are computed from the API, never typed into the page."""
    for v in ("0.60", "0.598", "0.77", "0.91", "0.906", "1.02", "1.025", "1.23", "0.86", "26 Dec", "23 May", "Monday (×", "Sunday (×"):
        assert v not in flat, v


def test_failures_handled_per_section(flat):
    assert flat.count("ex.isError") >= 3 and "run.isError" in flat
    assert "is unavailable right now" in flat and "The rest of this page and the forecast itself are unaffected" in flat


def test_client_safe_no_metrics_or_internal_links():
    metric = re.compile(r"\b(MAE|RMSE|WAPE|bias|coverage|validation|holdout|backtest|bootstrap|tuning|/performance|/quality|"
                        r"MlComparison|ReliabilityCard|IntervalMeaning)\b", re.I)
    src = PAGE.read_text(encoding="utf-8")
    assert [m.group(0) for m in metric.finditer(src)] == []
    charts = (SRC / "components" / "charts" / "StudioCharts.tsx").read_text(encoding="utf-8")
    for name in ("WeekdayMultipliers", "AnnualCurve"):
        s = charts.index(f"export function {name}")
        body = charts[s:charts.index("\nexport function", s + 1)]
        assert [m.group(0) for m in metric.finditer(body)] == [], name


def test_honest_wording(flat):
    low = flat.lower()
    for phrase in ("not causal explanations", "describes timing, not what drives admissions", "not proof that holidays cause",
                   "not a guarantee", "over many days", "Weather and air quality (AQI, pollutants) are not inputs",
                   "none met the improvement rule set in advance", "31 March 2019", "not an operational forecast for 2026",
                   "Historical demonstration · 1–14 Apr 2019"):
        assert phrase.lower() in low, phrase
    for bad in ("80% chance", "80% probability", "probability of falling", "guaranteed", "will fall inside",
                "weather affects", "aqi affects", "because of the weather", "staffing", "outpatient scheduling"):
        assert bad not in low, bad


def test_route_enabled_everywhere_and_forecast_py_unchanged():
    router = (SRC / "lib" / "router.tsx").read_text(encoding="utf-8")
    assert re.search(r'path: "/insights".*?ready: true', router, re.S)
    assert '"/insights": <Insights />' in (SRC / "App.tsx").read_text(encoding="utf-8")
    assert hashlib.sha256((ROOT / "forecast.py").read_bytes()).hexdigest() == C.FROZEN_FORECAST_SHA256
