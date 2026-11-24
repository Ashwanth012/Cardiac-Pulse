"""Stage 2 - GET /api/conditions: live weather and air-quality forecasts for Ludhiana (context only).

Separation from the admissions forecast is deliberate and tested:
  * this module never imports or calls the forecast runner, run.py or the DataStore;
  * the payload covers the CURRENT period (today .. today+13, Asia/Kolkata), not the 1-14 Apr 2019 demonstration;
  * the frozen admissions model does not use weather or air quality, and nothing here feeds into it.
"""
from __future__ import annotations

from datetime import timedelta
from typing import Callable

from fastapi import APIRouter

from .providers import open_meteo as OM

SEPARATION_NOTE = ("Live forecasts for the current period, shown as context only. The admissions model does not use weather "
                   "or air quality, and these values are unrelated to the 1–14 Apr 2019 historical admission demonstration.")


def build_payload(provider: OM.OpenMeteo) -> dict:
    today = provider.now().date()
    weather = provider.get("weather")
    air = provider.get("air_quality")
    return {
        "kind": "live_environmental_forecast",
        "label": "Live weather & air-quality forecast – current period, context only",
        "separation_note": SEPARATION_NOTE,
        "used_by_admissions_model": False,
        "location": OM.LOCATION,
        "today": today.isoformat(),
        "window": {"start": today.isoformat(), "end": (today + timedelta(days=OM.WINDOW_DAYS - 1)).isoformat(),
                   "days": OM.WINDOW_DAYS},
        "aqi_scale": {"name": "US AQI", "note": "United States EPA Air Quality Index as supplied by Open-Meteo. Not India's "
                                                "National AQI; the two scales differ and are not interchangeable.",
                      "categories": [{"upper": None if u == float("inf") else u, "label": lab} for u, lab in OM.US_AQI_CATEGORIES]},
        "cache_policy": {"ttl_minutes": OM.CACHE_TTL_MIN, "stale_max_hours": OM.STALE_MAX_HOURS, "timeout_seconds": OM.TIMEOUT_S},
        "weather": weather,
        "air_quality": air,
    }


def build_router(safe: Callable, provider_factory: Callable[[], OM.OpenMeteo]) -> APIRouter:
    router = APIRouter(prefix="/api", tags=["live conditions"])

    @router.get("/conditions")
    def conditions():
        """Live Open-Meteo forecasts (cached). Provider failures degrade to 'stale' or 'unavailable', never to fake data."""
        return safe(build_payload(provider_factory()))

    return router
