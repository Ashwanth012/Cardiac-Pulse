"""Stage 3: Calendar page (client-facing). Data comes from the existing /api/run and /api/model/explain.

Run from the project root:
    PYTHONDONTWRITEBYTECODE=1 python -m pytest dashboard/backend/tests/test_calendar.py -v -p no:cacheprovider
"""
import re
import sys
from pathlib import Path

import pandas as pd
import pytest

BACKEND = Path(__file__).resolve().parents[1]
ROOT = BACKEND.parents[1]
sys.path.insert(0, str(BACKEND))

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402

PAGE = ROOT / "dashboard" / "frontend" / "src" / "pages" / "CalendarContext.tsx"


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def cal():
    f = pd.read_csv(ROOT / "festival_calendar_punjab.csv")
    f["date"] = pd.to_datetime(f["date"])
    return f


def test_forecast_days_carry_official_holiday_names_and_types(client, cal):
    run = client.get("/api/run").json()
    assert [d["date"] for d in run["days"]] == pd.date_range("2019-04-01", periods=14).strftime("%Y-%m-%d").tolist()
    for d in run["days"]:
        exp = cal[cal.date == pd.Timestamp(d["date"])].sort_values("name")
        assert [(h["name"], h["type"]) for h in d["holidays"]] == list(zip(exp.name, exp.type))
    flagged = {d["date"]: [h["type"] for h in d["holidays"]] for d in run["days"] if d["holidays"]}
    assert flagged == {"2019-04-08": ["state"], "2019-04-13": ["state"], "2019-04-14": ["state", "state"]}
    assert all(set(d) >= {"mean", "lower_80", "upper_80", "weekday", "day_before_holiday", "day_after_holiday"} for d in run["days"])


def test_next_30_days_are_real_calendar_entries(client, cal):
    up = client.get("/api/run").json()["upcoming_holidays"]
    exp = cal[(cal.date >= "2019-04-01") & (cal.date <= "2019-04-30")].sort_values(["date", "name"])
    assert [(h["date"], h["name"], h["type"]) for h in up] == list(zip(exp.date.dt.strftime("%Y-%m-%d"), exp.name, exp.type))
    assert [h["name"] for h in up if not h["in_forecast_window"]] == ["Mahavira's Birthday", "Good Friday"]


def test_holiday_explanation_uses_frozen_model_terms(client):
    terms = client.get("/api/model/explain").json()["calendar"]
    assert [t["term"] for t in terms] == ["Holiday", "Day before a holiday", "Day after a holiday"]
    assert all(0.5 < t["multiplier"] < 2 for t in terms)


def test_page_uses_existing_apis_only_and_handles_missing_entries():
    src = PAGE.read_text(encoding="utf-8")
    hooks = set(re.findall(r"\buse[A-Z]\w+(?=\()", src))
    assert hooks == {"useForecastRun", "useModelExplain"}, hooks
    flat = re.sub(r"\s+", " ", src)
    for phrase in ("No holiday listed", "No holidays are listed in the supplied calendar", "Type not listed",
                   "could not be loaded", "The forecast calendar is unavailable", "have no admission forecast"):
        assert phrase in flat, phrase


def test_page_is_client_safe_and_honest():
    src = PAGE.read_text(encoding="utf-8")
    flat = re.sub(r"\s+", " ", src)
    metric = re.compile(r"\b(MAE|RMSE|WAPE|bias|coverage|validation|holdout|backtest|bootstrap|/performance|/quality)\b", re.I)
    assert [m.group(0) for m in metric.finditer(src)] == []
    assert "not causes" in flat and "Weather and air quality are not part of the model" in flat
    for claim in ("weather affects", "aqi affects", "because of the weather", "causes more admissions", "2026"):
        assert claim not in flat.lower(), claim
    assert "<RunBanner" in src                                   # historical-demonstration banner shown


def test_route_enabled():
    router = (ROOT / "dashboard" / "frontend" / "src" / "lib" / "router.tsx").read_text(encoding="utf-8")
    assert re.search(r'path: "/calendar".*?ready: true', router, re.S)
    app_src = (ROOT / "dashboard" / "frontend" / "src" / "App.tsx").read_text(encoding="utf-8")
    assert '"/calendar": <CalendarContext />' in app_src
