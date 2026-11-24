"""Open-Meteo weather and air-quality FORECASTS for Ludhiana (Stage 2, context only).

* Requests go from this backend only and carry nothing but fixed coordinates, timezone and variable names.
* Responses are validated: expected units, numeric values in plausible ranges, consistent array lengths.
  Invalid values become missing; a variable with unexpected units is dropped (never shown with wrong units).
* Cache: a fresh response is reused for CACHE_TTL_MIN. If a refresh fails, the last good response is served as
  "stale" for up to STALE_MAX_HOURS (with its retrieval time); older than that -> "unavailable".
  Cached JSON lives in dashboard/backend/.cache (git-ignored), never in the project outputs.
* Dates before "today" (Asia/Kolkata) are dropped: an old forecast for a past day is not a forecast.
  Missing future dates are reported as unavailable and are NEVER filled (no historical observations, no
  interpolation, no climatology).

The admissions model does not use any of this data.
"""
from __future__ import annotations

import json
import math
import threading
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Callable
from zoneinfo import ZoneInfo

import httpx

# --------------------------------------------------------------------------- configuration
LOCATION = {"name": "Ludhiana, Punjab, India", "latitude": 30.901, "longitude": 75.8573, "timezone": "Asia/Kolkata"}
TZ = ZoneInfo(LOCATION["timezone"])
WINDOW_DAYS = 14
TIMEOUT_S = 10.0
CACHE_TTL_MIN = 60
STALE_MAX_HOURS = 24
CACHE_DIR = Path(__file__).resolve().parents[2] / ".cache"
MIN_HOURS_FOR_DAILY = 18          # an air-quality day is summarised only with >= 18 of 24 hourly values

WEATHER_URL = "https://api.open-meteo.com/v1/forecast"
WEATHER_DAYS_REQUESTED = 16       # provider maximum; lets us keep a full 14-day window starting today
AIR_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
AIR_DAYS_REQUESTED = 7            # provider maximum (CAMS global forecast itself runs about 5 days)

# variable -> (label, expected units, plausible range)
WEATHER_VARS = {
    "temperature_2m_max": ("Max temperature", {"°C"}, (-40, 60)),
    "temperature_2m_min": ("Min temperature", {"°C"}, (-40, 60)),
    "temperature_2m_mean": ("Mean temperature", {"°C"}, (-40, 60)),
    "relative_humidity_2m_mean": ("Mean relative humidity", {"%"}, (0, 100)),
    "precipitation_sum": ("Precipitation", {"mm"}, (0, 1000)),
    "precipitation_probability_max": ("Chance of precipitation (max)", {"%"}, (0, 100)),
    "wind_speed_10m_max": ("Max wind speed", {"km/h"}, (0, 400)),
    "wind_gusts_10m_max": ("Max wind gusts", {"km/h"}, (0, 400)),
}
MICRO = {"μg/m³", "µg/m³", "ug/m3"}
AIR_VARS = {
    "us_aqi": ("US AQI", {"USAQI"}, (0, 1000)),
    "pm2_5": ("PM2.5", MICRO, (0, 5000)),
    "pm10": ("PM10", MICRO, (0, 10000)),
    "nitrogen_dioxide": ("NO2", MICRO, (0, 5000)),
    "ozone": ("O3", MICRO, (0, 5000)),
    "carbon_monoxide": ("CO", MICRO, (0, 200000)),
    "sulphur_dioxide": ("SO2", MICRO, (0, 5000)),
}
US_AQI_CATEGORIES = [(50, "Good"), (100, "Moderate"), (150, "Unhealthy for sensitive groups"), (200, "Unhealthy"),
                     (300, "Very unhealthy"), (math.inf, "Hazardous")]

ATTRIBUTION = {
    "weather": {"text": "Weather forecast data by Open-Meteo.com", "licence": "CC BY 4.0",
                "url": "https://open-meteo.com/", "licence_url": "https://creativecommons.org/licenses/by/4.0/"},
    "air_quality": {"text": "Air-quality forecast data by Open-Meteo.com, based on the Copernicus Atmosphere Monitoring "
                            "Service (CAMS) global forecast. Contains modified Copernicus Atmosphere Monitoring Service information.",
                    "licence": "CC BY 4.0", "url": "https://open-meteo.com/en/docs/air-quality-api",
                    "licence_url": "https://creativecommons.org/licenses/by/4.0/"},
}


class ProviderError(RuntimeError):
    pass


def us_aqi_category(v: float | None) -> str | None:
    if v is None:
        return None
    return next(label for upper, label in US_AQI_CATEGORIES if v <= upper)


# --------------------------------------------------------------------------- fetching
def default_fetch(url: str, params: dict, timeout: float = TIMEOUT_S) -> dict:
    """GET JSON. Raises ProviderError on network errors, HTTP errors or non-JSON bodies."""
    try:
        r = httpx.get(url, params=params, timeout=timeout, headers={"Accept": "application/json"})
    except httpx.TimeoutException as e:
        raise ProviderError(f"request timed out after {timeout:.0f}s") from e
    except httpx.HTTPError as e:
        raise ProviderError(f"network error ({type(e).__name__})") from e
    if r.status_code != 200:
        reason = ""
        try:
            reason = str(r.json().get("reason", ""))[:160]
        except ValueError:
            pass
        raise ProviderError(f"HTTP {r.status_code}{': ' + reason if reason else ''}")
    try:
        body = r.json()
    except ValueError as e:
        raise ProviderError("response was not valid JSON") from e
    if not isinstance(body, dict) or body.get("error"):
        raise ProviderError(f"provider error: {str(body.get('reason', 'unknown'))[:160] if isinstance(body, dict) else 'bad body'}")
    return body


def weather_params() -> dict:
    return {"latitude": LOCATION["latitude"], "longitude": LOCATION["longitude"], "timezone": LOCATION["timezone"],
            "forecast_days": WEATHER_DAYS_REQUESTED, "daily": ",".join(WEATHER_VARS)}


def air_params() -> dict:
    return {"latitude": LOCATION["latitude"], "longitude": LOCATION["longitude"], "timezone": LOCATION["timezone"],
            "forecast_days": AIR_DAYS_REQUESTED, "hourly": ",".join(AIR_VARS)}


# --------------------------------------------------------------------------- validation
def _num(v, lo, hi):
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        return None, v is not None
    return (float(v), False) if lo <= v <= hi else (None, True)


def _validate(body: dict, block: str, units_key: str, spec: dict, time_fmt: str) -> tuple[list, dict, dict, list]:
    """-> (times, {var: values}, {var: unit}, issues). Raises ProviderError if the structure is unusable."""
    data, units = body.get(block), body.get(units_key) or {}
    if not isinstance(data, dict) or not isinstance(data.get("time"), list) or not data["time"]:
        raise ProviderError(f"response has no '{block}.time' array")
    times = []
    for t in data["time"]:
        try:
            times.append(datetime.strptime(t, time_fmt))
        except (TypeError, ValueError) as e:
            raise ProviderError(f"unparseable time value {t!r}") from e
    if times != sorted(times) or len(set(times)) != len(times):
        raise ProviderError("time values are not strictly increasing")
    out, unit_out, issues = {}, {}, []
    for var, (label, ok_units, (lo, hi)) in spec.items():
        if var not in data:
            issues.append(f"{label}: not returned by the provider")
            continue
        vals = data[var]
        if not isinstance(vals, list) or len(vals) != len(times):
            issues.append(f"{label}: array length does not match time; dropped")
            continue
        unit = units.get(var)
        if unit not in ok_units:
            issues.append(f"{label}: unexpected unit {unit!r}; dropped")
            continue
        clean, bad = [], 0
        for v in vals:
            x, invalid = _num(v, lo, hi)
            clean.append(x)
            bad += invalid
        if bad:
            issues.append(f"{label}: {bad} invalid or out-of-range value(s) treated as missing")
        out[var], unit_out[var] = clean, unit
    if not out:
        raise ProviderError("none of the requested variables were usable")
    return times, out, unit_out, issues


def _grid(body: dict) -> dict:
    return {k: body.get(k) for k in ("latitude", "longitude", "elevation") if isinstance(body.get(k), (int, float))}


def normalise_weather(body: dict, today: date) -> dict:
    times, vals, units, issues = _validate(body, "daily", "daily_units", WEATHER_VARS, "%Y-%m-%d")
    window = [today + timedelta(days=i) for i in range(WINDOW_DAYS)]
    by_date = {t.date(): i for i, t in enumerate(times)}
    days = []
    for d in window:
        i = by_date.get(d)
        values = {} if i is None else {v: vals[v][i] for v in vals if vals[v][i] is not None}
        days.append({"date": d.isoformat(), "available": bool(values), "values": values})
    return {"units": units, "labels": {v: WEATHER_VARS[v][0] for v in units}, "grid": _grid(body),
            "provider_dates": {"first": times[0].date().isoformat(), "last": times[-1].date().isoformat()},
            "days": days, "issues": issues}


def normalise_air(body: dict, today: date) -> dict:
    times, vals, units, issues = _validate(body, "hourly", "hourly_units", AIR_VARS, "%Y-%m-%dT%H:%M")
    window = [today + timedelta(days=i) for i in range(WINDOW_DAYS)]
    idx: dict[date, list[int]] = {}
    for i, t in enumerate(times):
        idx.setdefault(t.date(), []).append(i)
    days = []
    for d in window:
        rows = idx.get(d, [])
        per_var_hours = {v: sum(vals[v][i] is not None for i in rows) for v in vals}
        hours = max(per_var_hours.values(), default=0)
        values, hours_used = {}, {}
        for v in vals:
            xs = [vals[v][i] for i in rows if vals[v][i] is not None]
            if len(xs) >= MIN_HOURS_FOR_DAILY:
                values[f"{v}_max" if v == "us_aqi" else f"{v}_mean"] = max(xs) if v == "us_aqi" else sum(xs) / len(xs)
                if v != "us_aqi":
                    values[f"{v}_max"] = max(xs)
                hours_used[v] = len(xs)
        status = "full" if values else ("partial" if hours else "none")
        day = {"date": d.isoformat(), "available": bool(values), "status": status, "hours_with_data": hours,
               "values": values, "hours_used": hours_used}
        if "us_aqi_max" in values:
            day["us_aqi_category"] = us_aqi_category(values["us_aqi_max"])
        days.append(day)
    return {"units": units, "labels": {v: AIR_VARS[v][0] for v in units}, "grid": _grid(body),
            "provider_dates": {"first": times[0].date().isoformat(), "last": times[-1].date().isoformat()},
            "aggregation": {"us_aqi": "daily maximum of the hourly US AQI",
                            "pollutants": "daily mean (and maximum) of hourly concentrations",
                            "min_hours": MIN_HOURS_FOR_DAILY},
            "days": days, "issues": issues}


# --------------------------------------------------------------------------- cache + provider
@dataclass
class Source:
    key: str
    url: str
    params: Callable[[], dict]
    normalise: Callable[[dict, date], dict]


SOURCES = {"weather": Source("weather", WEATHER_URL, weather_params, normalise_weather),
           "air_quality": Source("air_quality", AIR_URL, air_params, normalise_air)}


class OpenMeteo:
    """Fetch + validate + cache. `fetch` and `now` are injectable for tests (no network in tests)."""

    def __init__(self, fetch: Callable = default_fetch, now: Callable[[], datetime] = lambda: datetime.now(TZ),
                 cache_dir: Path | None = CACHE_DIR, ttl_min: int = CACHE_TTL_MIN, stale_max_h: int = STALE_MAX_HOURS):
        self.fetch, self.now, self.cache_dir = fetch, now, cache_dir
        self.ttl, self.stale_max = timedelta(minutes=ttl_min), timedelta(hours=stale_max_h)
        self._mem: dict[str, dict] = {}
        self._locks = {k: threading.Lock() for k in SOURCES}
        self.fetch_count = {k: 0 for k in SOURCES}

    # ---- persistence (best effort; a broken cache file is ignored, never fatal)
    def _path(self, key):
        return None if self.cache_dir is None else self.cache_dir / f"open_meteo_{key}.json"

    def _load(self, key):
        if key in self._mem:
            return self._mem[key]
        p = self._path(key)
        if p and p.exists():
            try:
                entry = json.loads(p.read_text(encoding="utf-8"))
                entry["retrieved_at"] = datetime.fromisoformat(entry["retrieved_at"])
                if entry.get("params") == SOURCES[key].params() and isinstance(entry.get("body"), dict):
                    self._mem[key] = entry
                    return entry
            except (ValueError, KeyError, TypeError, OSError):
                pass
        return None

    def _save(self, key, entry):
        self._mem[key] = entry
        p = self._path(key)
        if p:
            try:
                p.parent.mkdir(parents=True, exist_ok=True)
                tmp = p.with_suffix(".tmp")
                tmp.write_text(json.dumps({**entry, "retrieved_at": entry["retrieved_at"].isoformat()}), encoding="utf-8")
                tmp.replace(p)
            except OSError:
                pass

    # ---- public
    def get(self, key: str) -> dict:
        src = SOURCES[key]
        with self._locks[key]:
            now = self.now()
            today = now.date()
            cached = self._load(key)
            if cached and now - cached["retrieved_at"] < self.ttl:
                return self._result(src, cached, today, now, "ok", cache_hit=True)
            error = None
            try:
                self.fetch_count[key] += 1
                body = self.fetch(src.url, src.params())
                src.normalise(body, today)                         # validate before accepting into the cache
                entry = {"retrieved_at": now, "params": src.params(), "body": body}
                self._save(key, entry)
                return self._result(src, entry, today, now, "ok", cache_hit=False)
            except (ProviderError, ValueError, TypeError, KeyError) as e:
                error = str(e) if isinstance(e, ProviderError) else f"invalid response ({type(e).__name__})"
            if cached and now - cached["retrieved_at"] < self.stale_max:
                return self._result(src, cached, today, now, "stale", cache_hit=True, error=error)
            return {"status": "unavailable", "error": error, "cache_hit": False, "retrieved_at": None, "age_minutes": None,
                    "attribution": ATTRIBUTION[key], "available_dates": {"first": None, "last": None, "count": 0},
                    "units": {}, "labels": {}, "grid": {}, "provider_dates": None, "issues": [],
                    "days": [{"date": (today + timedelta(days=i)).isoformat(), "available": False, "values": {}}
                             for i in range(WINDOW_DAYS)]}

    def _result(self, src, entry, today, now, status, cache_hit, error=None):
        norm = src.normalise(entry["body"], today)
        avail = [d["date"] for d in norm["days"] if d["available"]]
        return {"status": status, "error": error, "cache_hit": cache_hit,
                "retrieved_at": entry["retrieved_at"].isoformat(timespec="seconds"),
                "age_minutes": round((now - entry["retrieved_at"]).total_seconds() / 60, 1),
                "attribution": ATTRIBUTION[src.key],
                "available_dates": {"first": avail[0] if avail else None, "last": avail[-1] if avail else None, "count": len(avail)},
                **norm}
