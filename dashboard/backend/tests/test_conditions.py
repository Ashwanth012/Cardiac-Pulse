"""Stage 2: live Open-Meteo weather & air-quality forecasts (context only).

No test touches the network: the provider's fetch function and clock are injected.
Run from the project root:
    PYTHONDONTWRITEBYTECODE=1 python -m pytest dashboard/backend/tests -v -p no:cacheprovider
"""
import ast
import hashlib
import json
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path

import httpx
import pytest

BACKEND = Path(__file__).resolve().parents[1]
ROOT = BACKEND.parents[1]
sys.path.insert(0, str(BACKEND))

from fastapi.testclient import TestClient  # noqa: E402

from app import config as C  # noqa: E402
from app import forecast_runner as FR  # noqa: E402
from app import main as M  # noqa: E402
from app.datastore import assert_no_pii  # noqa: E402
from app.providers import open_meteo as OM  # noqa: E402

TZ = OM.TZ
NOW = datetime(2026, 11, 24, 9, 30, tzinfo=TZ)
SRC = ROOT / "dashboard" / "frontend" / "src"


# --------------------------------------------------------------------------- synthetic provider bodies
def weather_body(start="2026-11-23", days=16, units=None, **override):
    d0 = datetime.fromisoformat(start)
    times = [(d0 + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(days)]
    daily = {"time": times, "temperature_2m_max": [24.0 + i * 0.1 for i in range(days)],
             "temperature_2m_min": [12.0] * days, "temperature_2m_mean": [18.0] * days,
             "relative_humidity_2m_mean": [70] * days, "precipitation_sum": [0.0] * (days - 1) + [3.2],
             "precipitation_probability_max": [10] * days, "wind_speed_10m_max": [9.0] * days, "wind_gusts_10m_max": [20.0] * days}
    daily.update(override)
    u = {"time": "iso8601", "temperature_2m_max": "°C", "temperature_2m_min": "°C", "temperature_2m_mean": "°C",
         "relative_humidity_2m_mean": "%", "precipitation_sum": "mm", "precipitation_probability_max": "%",
         "wind_speed_10m_max": "km/h", "wind_gusts_10m_max": "km/h"}
    u.update(units or {})
    return {"latitude": 30.896, "longitude": 75.834, "elevation": 252.0, "daily_units": u, "daily": daily}


def air_body(start="2026-11-23", days=7, full_days=5, partial_hours=6):
    d0 = datetime.fromisoformat(start)
    times = [(d0 + timedelta(hours=h)).strftime("%Y-%m-%dT%H:%M") for h in range(24 * days)]
    n_ok = 24 * full_days + partial_hours

    def series(f):
        return [f(h) if h < n_ok else None for h in range(len(times))]
    hourly = {"time": times, "us_aqi": series(lambda h: 100 + (h % 24) * 2), "pm2_5": series(lambda h: 50.0 + h % 24),
              "pm10": series(lambda h: 80.0), "nitrogen_dioxide": series(lambda h: 20.0), "ozone": series(lambda h: 60.0),
              "carbon_monoxide": series(lambda h: 900.0), "sulphur_dioxide": series(lambda h: 15.0)}
    u = {"time": "iso8601", "us_aqi": "USAQI", **{k: "μg/m³" for k in hourly if k not in ("time", "us_aqi")}}
    return {"latitude": 30.9, "longitude": 75.9, "elevation": 252.0, "hourly_units": u, "hourly": hourly}


class FakeFetch:
    def __init__(self, weather=None, air=None, error=None):
        self.bodies = {OM.WEATHER_URL: weather or weather_body(), OM.AIR_URL: air or air_body()}
        self.error, self.calls = error, []

    def __call__(self, url, params):
        self.calls.append((url, dict(params)))
        if self.error:
            raise self.error
        return json.loads(json.dumps(self.bodies[url]))


class Clock:
    def __init__(self, t=NOW):
        self.t = t

    def __call__(self):
        return self.t


def provider(tmp_path, fetch=None, clock=None, **kw):
    return OM.OpenMeteo(fetch=fetch or FakeFetch(), now=clock or Clock(), cache_dir=tmp_path, **kw)


# =========================================================================== requests
def test_requests_carry_only_coordinates_timezone_and_variables(tmp_path):
    f = FakeFetch()
    p = provider(tmp_path, f)
    p.get("weather"), p.get("air_quality")
    for url, params in f.calls:
        assert url in (OM.WEATHER_URL, OM.AIR_URL) and url.startswith("https://")
        assert set(params) <= {"latitude", "longitude", "timezone", "forecast_days", "daily", "hourly"}
        assert (params["latitude"], params["longitude"], params["timezone"]) == (30.901, 75.8573, "Asia/Kolkata")
    assert f.calls[0][1]["forecast_days"] == 16 and f.calls[1][1]["forecast_days"] == 7


# =========================================================================== weather normalisation
def test_weather_window_starts_today_and_drops_past_days(tmp_path):
    w = provider(tmp_path).get("weather")
    assert w["status"] == "ok" and len(w["days"]) == 14
    assert w["days"][0]["date"] == "2026-11-24" and w["days"][-1]["date"] == "2026-12-07"
    assert all(d["available"] for d in w["days"])
    assert w["available_dates"] == {"first": "2026-11-24", "last": "2026-12-07", "count": 14}
    assert w["provider_dates"] == {"first": "2026-11-23", "last": "2026-12-08"}
    assert w["days"][0]["values"]["temperature_2m_max"] == pytest.approx(24.1)
    assert w["units"]["temperature_2m_max"] == "°C" and w["units"]["precipitation_sum"] == "mm"
    assert w["units"]["wind_speed_10m_max"] == "km/h" and w["units"]["relative_humidity_2m_mean"] == "%"


def test_short_weather_horizon_marks_missing_days_unavailable_never_filled(tmp_path):
    w = provider(tmp_path, FakeFetch(weather=weather_body(days=8))).get("weather")
    avail = [d for d in w["days"] if d["available"]]
    assert [d["date"] for d in avail][-1] == "2026-11-30" and len(avail) == 7
    for d in w["days"][7:]:
        assert d == {"date": d["date"], "available": False, "values": {}}


def test_unexpected_units_dropped_not_shown(tmp_path):
    w = provider(tmp_path, FakeFetch(weather=weather_body(units={"temperature_2m_max": "°F"}))).get("weather")
    assert "temperature_2m_max" not in w["units"] and all("temperature_2m_max" not in d["values"] for d in w["days"])
    assert any("unexpected unit" in i for i in w["issues"]) and "temperature_2m_min" in w["units"]


def test_invalid_and_out_of_range_values_become_missing(tmp_path):
    bad = weather_body(precipitation_sum=[-5.0, float("nan"), "x", 1e9] + [0.0] * 12,
                       relative_humidity_2m_mean=[70, 70, 140] + [70] * 13)
    w = provider(tmp_path, FakeFetch(weather=bad)).get("weather")
    d24, d25 = w["days"][0]["values"], w["days"][1]["values"]
    assert "precipitation_sum" not in d24 and "precipitation_sum" not in d25 and "relative_humidity_2m_mean" not in d25
    assert any("Precipitation: 4 invalid" in i for i in w["issues"]) and any("humidity: 1 invalid" in i for i in w["issues"])


def test_missing_variable_omitted_and_reported(tmp_path):
    body = weather_body()
    del body["daily"]["wind_gusts_10m_max"]
    w = provider(tmp_path, FakeFetch(weather=body)).get("weather")
    assert "wind_gusts_10m_max" not in w["units"] and any("not returned" in i for i in w["issues"])


@pytest.mark.parametrize("mutate", [
    lambda b: b.pop("daily"),
    lambda b: b["daily"].update(time=[]),
    lambda b: b["daily"].update(time=["24/11/2026"] * 16),
    lambda b: b["daily"].update(time=list(reversed(b["daily"]["time"]))),
    lambda b: [b["daily"].update({k: [None]}) for k in OM.WEATHER_VARS],
])
def test_unusable_structure_is_unavailable_without_cache(tmp_path, mutate):
    body = weather_body()
    mutate(body)
    w = provider(tmp_path, FakeFetch(weather=body)).get("weather")
    assert w["status"] == "unavailable" and w["error"] and w["retrieved_at"] is None
    assert len(w["days"]) == 14 and all(d["available"] is False and d["values"] == {} for d in w["days"])


# =========================================================================== air quality
def test_air_quality_daily_aggregation_and_horizon(tmp_path):
    a = provider(tmp_path).get("air_quality")
    by = {d["date"]: d for d in a["days"]}
    # synthetic: full hourly data 23-27 Nov, 6 hours on 28 Nov, nothing after
    for day in ("2026-11-24", "2026-11-25", "2026-11-26", "2026-11-27"):
        v = by[day]["values"]
        assert by[day]["status"] == "full" and by[day]["hours_with_data"] == 24
        assert v["us_aqi_max"] == 146 and by[day]["us_aqi_category"] == "Unhealthy for sensitive groups"
        assert v["pm2_5_mean"] == pytest.approx(61.5) and v["pm2_5_max"] == 73
        assert set(v) == {"us_aqi_max", "pm2_5_mean", "pm2_5_max", "pm10_mean", "pm10_max", "nitrogen_dioxide_mean",
                          "nitrogen_dioxide_max", "ozone_mean", "ozone_max", "carbon_monoxide_mean", "carbon_monoxide_max",
                          "sulphur_dioxide_mean", "sulphur_dioxide_max"}
    assert by["2026-11-28"]["status"] == "partial" and by["2026-11-28"]["hours_with_data"] == 6
    assert by["2026-11-28"]["values"] == {} and by["2026-11-28"]["available"] is False
    for day in ("2026-11-29", "2026-12-07"):
        assert by[day]["status"] == "none" and by[day]["values"] == {}
    assert a["available_dates"] == {"first": "2026-11-24", "last": "2026-11-27", "count": 4}
    assert a["units"]["us_aqi"] == "USAQI" and a["units"]["pm2_5"] == "μg/m³"
    assert a["labels"]["us_aqi"] == "US AQI" and a["aggregation"]["min_hours"] == 18


@pytest.mark.parametrize("v,cat", [(0, "Good"), (50, "Good"), (51, "Moderate"), (100, "Moderate"), (101, "Unhealthy for sensitive groups"),
                                   (151, "Unhealthy"), (201, "Very unhealthy"), (300, "Very unhealthy"), (301, "Hazardous"), (None, None)])
def test_us_aqi_categories(v, cat):
    assert OM.us_aqi_category(v) == cat


# =========================================================================== caching
def test_cache_reused_within_ttl_and_refreshed_after(tmp_path):
    f, clock = FakeFetch(), Clock()
    p = provider(tmp_path, f, clock)
    first = p.get("weather")
    second = p.get("weather")
    assert (first["cache_hit"], second["cache_hit"]) == (False, True) and len(f.calls) == 1
    assert second["retrieved_at"] == first["retrieved_at"] == "2026-11-24T09:30:00+05:30"
    clock.t = NOW + timedelta(minutes=59)
    assert p.get("weather")["cache_hit"] is True and len(f.calls) == 1
    clock.t = NOW + timedelta(minutes=61)
    r = p.get("weather")
    assert r["cache_hit"] is False and len(f.calls) == 2 and r["retrieved_at"] == "2026-11-24T10:31:00+05:30"


def test_cache_persists_across_instances_and_ignores_other_params(tmp_path, monkeypatch):
    provider(tmp_path).get("air_quality")
    assert (tmp_path / "open_meteo_air_quality.json").exists()
    f2 = FakeFetch()
    again = provider(tmp_path, f2).get("air_quality")
    assert again["cache_hit"] is True and f2.calls == []
    monkeypatch.setattr(OM, "AIR_DAYS_REQUESTED", 5)           # different request -> cached entry not reused
    f3 = FakeFetch()
    assert provider(tmp_path, f3).get("air_quality")["cache_hit"] is False and len(f3.calls) == 1


def test_corrupt_cache_file_is_ignored(tmp_path):
    (tmp_path / "open_meteo_weather.json").write_text("{not json", encoding="utf-8")
    f = FakeFetch()
    assert provider(tmp_path, f).get("weather")["status"] == "ok" and len(f.calls) == 1


def test_failure_serves_stale_then_unavailable(tmp_path):
    clock, f = Clock(), FakeFetch()
    p = provider(tmp_path, f, clock)
    p.get("weather")
    f.error = OM.ProviderError("HTTP 503")
    clock.t = NOW + timedelta(hours=3)
    s = p.get("weather")
    assert s["status"] == "stale" and s["error"] == "HTTP 503" and s["retrieved_at"] == "2026-11-24T09:30:00+05:30"
    assert s["age_minutes"] == pytest.approx(180)
    clock.t = NOW + timedelta(hours=25)
    u = p.get("weather")
    assert u["status"] == "unavailable" and u["error"] == "HTTP 503" and not any(d["available"] for d in u["days"])


def test_stale_cache_after_midnight_drops_the_past_day(tmp_path):
    clock, f = Clock(), FakeFetch(weather=weather_body(days=15))        # provider data run 23 Nov - 7 Dec
    p = provider(tmp_path, f, clock)
    p.get("weather")
    f.error = OM.ProviderError("timeout")
    clock.t = datetime(2026, 11, 25, 6, 0, tzinfo=TZ)
    s = p.get("weather")
    assert s["status"] == "stale" and s["days"][0]["date"] == "2026-11-25"
    assert "2026-11-24" not in [d["date"] for d in s["days"]]
    assert s["days"][-1]["date"] == "2026-12-08" and s["days"][-1]["available"] is False   # beyond the cached horizon


def test_invalid_refresh_does_not_replace_good_cache(tmp_path):
    clock, f = Clock(), FakeFetch()
    p = provider(tmp_path, f, clock)
    p.get("weather")
    f.bodies[OM.WEATHER_URL] = {"daily": {"time": []}}
    clock.t = NOW + timedelta(hours=2)
    s = p.get("weather")
    assert s["status"] == "stale" and "no 'daily.time'" in s["error"]
    assert json.loads((tmp_path / "open_meteo_weather.json").read_text(encoding="utf-8"))["body"]["daily"]["time"]


# =========================================================================== real fetch function (mocked transport)
def test_default_fetch_timeout_http_error_and_bad_json(monkeypatch):
    def timeout(*a, **k):
        raise httpx.ReadTimeout("slow")
    monkeypatch.setattr(OM.httpx, "get", timeout)
    with pytest.raises(OM.ProviderError, match="timed out"):
        OM.default_fetch(OM.WEATHER_URL, {})

    def conn(*a, **k):
        raise httpx.ConnectError("down")
    monkeypatch.setattr(OM.httpx, "get", conn)
    with pytest.raises(OM.ProviderError, match="network error"):
        OM.default_fetch(OM.WEATHER_URL, {})
    req = httpx.Request("GET", OM.WEATHER_URL)
    monkeypatch.setattr(OM.httpx, "get", lambda *a, **k: httpx.Response(400, json={"error": True, "reason": "bad days"}, request=req))
    with pytest.raises(OM.ProviderError, match="HTTP 400: bad days"):
        OM.default_fetch(OM.WEATHER_URL, {})
    monkeypatch.setattr(OM.httpx, "get", lambda *a, **k: httpx.Response(200, text="<html>", request=req))
    with pytest.raises(OM.ProviderError, match="not valid JSON"):
        OM.default_fetch(OM.WEATHER_URL, {})
    seen = {}

    def ok(url, params=None, timeout=None, headers=None):
        seen["timeout"] = timeout
        return httpx.Response(200, json={"daily": {}}, request=req)
    monkeypatch.setattr(OM.httpx, "get", ok)
    OM.default_fetch(OM.WEATHER_URL, {})
    assert seen["timeout"] == OM.TIMEOUT_S == 10.0


# =========================================================================== endpoint
@pytest.fixture()
def client_with(tmp_path, monkeypatch):
    def make(fetch):
        monkeypatch.setattr(M, "_provider", provider(tmp_path, fetch))
        return TestClient(M.app)
    return make


def test_endpoint_payload(client_with):
    with client_with(FakeFetch()) as c:
        r = c.get("/api/conditions")
    assert r.status_code == 200
    j = r.json()
    assert_no_pii(j)
    assert j["kind"] == "live_environmental_forecast" and j["used_by_admissions_model"] is False
    assert j["today"] == "2026-11-24" and j["window"] == {"start": "2026-11-24", "end": "2026-12-07", "days": 14}
    assert "unrelated to the 1–14 Apr 2019 historical admission demonstration" in j["separation_note"]
    assert j["aqi_scale"]["name"] == "US AQI" and "Not India's National AQI" in j["aqi_scale"]["note"]
    assert "Open-Meteo" in j["weather"]["attribution"]["text"] and "Copernicus" in j["air_quality"]["attribution"]["text"]
    assert j["weather"]["status"] == j["air_quality"]["status"] == "ok"
    assert j["cache_policy"] == {"ttl_minutes": 60, "stale_max_hours": 24, "timeout_seconds": 10.0}


def test_endpoint_degrades_gracefully_and_rest_of_dashboard_works(client_with):
    with client_with(FakeFetch(error=OM.ProviderError("network error (ConnectError)"))) as c:
        j = c.get("/api/conditions").json()
        assert j["weather"]["status"] == j["air_quality"]["status"] == "unavailable"
        assert j["weather"]["error"] == "network error (ConnectError)"
        for url in ("/api/run", "/api/forecast/historical", "/api/overview", "/api/quality/summary", "/api/evaluation/holdout"):
            assert c.get(url).status_code == 200, url


def test_endpoint_one_provider_down_other_up(client_with):
    class Half(FakeFetch):
        def __call__(self, url, params):
            if url == OM.AIR_URL:
                raise OM.ProviderError("HTTP 502")
            return super().__call__(url, params)
    with client_with(Half()) as c:
        j = c.get("/api/conditions").json()
    assert (j["weather"]["status"], j["air_quality"]["status"]) == ("ok", "unavailable")


def test_no_india_naqi_label_anywhere(client_with):
    with client_with(FakeFetch()) as c:
        blob = json.dumps(c.get("/api/conditions").json(), ensure_ascii=False)
    assert "NAQI" not in blob
    for m in re.finditer(r"National AQI", blob):
        assert "Not India's" in blob[max(0, m.start() - 20):m.start()]


# =========================================================================== separation from the admissions model
def test_conditions_never_runs_or_changes_the_admissions_forecast(client_with, monkeypatch):
    with client_with(FakeFetch()) as c:
        before = c.get("/api/run").json()

        def forbidden(*a, **k):
            raise AssertionError("conditions must not run the admissions model")
        monkeypatch.setattr(FR, "run_forecast", forbidden)
        assert c.get("/api/conditions").status_code == 200
        monkeypatch.undo()
    with TestClient(M.app) as c2:
        after = c2.get("/api/run").json()
    assert after["official"] == before["official"] and after["days"] == before["days"]
    assert after["manifest"]["forecast_start"] == "2019-04-01" and "Weather" in after["manifest"]["not_model_inputs"]


def test_admissions_code_does_not_import_providers():
    for f in ("run.py", "forecast_runner.py", "datastore.py", "quality.py", "config.py"):
        tree = ast.parse((BACKEND / "app" / f).read_text(encoding="utf-8"))
        names = [n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
        names += [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
        assert not any("provider" in x or "conditions" in x or "open_meteo" in x for x in names), f
    cond = (BACKEND / "app" / "conditions.py").read_text(encoding="utf-8")
    assert "forecast_runner" not in cond and "datastore" not in cond and "from .run" not in cond


def test_frozen_files_and_outputs_unchanged_and_cache_outside_project_outputs(client_with):
    files = [p for p in (ROOT / "outputs").rglob("*") if p.is_file()] + [ROOT / "forecast.py"]
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    with client_with(FakeFetch()) as c:
        c.get("/api/conditions")
    assert {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in files} == before
    assert before[ROOT / "forecast.py"] == C.FROZEN_FORECAST_SHA256 == "aa4809046af56166ceed6f3daab1521be093878a7d9c0ffaa9d93b784462c866"
    assert OM.CACHE_DIR == BACKEND / ".cache" and ROOT / "outputs" not in OM.CACHE_DIR.parents
    assert ".cache/" in (ROOT / "dashboard" / ".gitignore").read_text(encoding="utf-8")


# =========================================================================== frontend wording & route
def test_frontend_page_wording_states_and_route():
    page = re.sub(r"\s+", " ", (SRC / "pages" / "Conditions.tsx").read_text(encoding="utf-8"))
    charts = re.sub(r"\s+", " ", (SRC / "components" / "charts" / "EnvCharts.tsx").read_text(encoding="utf-8"))
    both = page + charts
    for phrase in ("US AQI", "Open-Meteo", "Retrieved", "not used by the admissions model", "1–14 Apr 2019",
                   "No forecast available", "unavailable", "stale", "Loading", "Copernicus"):
        assert phrase in both, phrase
    assert "NAQI" not in both and not re.search(r"(?<!Not India's )National AQI", both)
    for claim in ("caused", "will increase admissions", "drives admissions", "affects admissions", "admissions will rise"):
        assert claim not in both.lower(), claim
    router = (SRC / "lib" / "router.tsx").read_text(encoding="utf-8")
    assert re.search(r'path: "/conditions".*?ready: true', router, re.S)
    app_src = (SRC / "App.tsx").read_text(encoding="utf-8")
    assert '"/conditions": <Conditions />' in app_src
