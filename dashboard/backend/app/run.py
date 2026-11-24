"""Stage 1 - the forecast "run": the official 14-day output of forecast.py plus descriptive context.

Everything numeric about admissions comes from the frozen forecast.py (via forecast_runner, unchanged).
This module only adds labels derived from those rows and from data before the cut-off:
  * historical comparison  - the approved fixed rule: each day's forecast mean vs the 25th/75th percentile of
                             the same weekday's observed raw records in the 8 weeks before the cut-off;
  * forecast-period ranking - the 14 forecast means ranked against each other (busiest days of this window);
  * calendar context        - holidays from the official festival calendar, plus day-before / day-after flags
                              (the same flags the model uses);
  * reliability             - read from the saved Jan-Mar 2019 M4 holdout evaluation (previously viewed; not used to
                              select the newer candidate models).
Both demand indicators are descriptive. They are not capacity, bed or staffing assessments.

Mode: DEMONSTRATION only. The admissions data end on 2019-03-31, so the only genuine "next 14 days" is
1-14 Apr 2019 (cut-off 2019-04-01). Weather or air-quality forecasts for the current period (Stage 2) are a
different period and never feed into this run.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Query

from . import config as C

DEMO_AS_OF = C.HISTORICAL_CUTOFF                     # "2019-04-01"
H = 14
BASELINE_WEEKS = 8
BUSIEST_N = 3
TIER_LABEL = {"higher": "Higher than recent historical baseline",
              "typical": "Within recent historical range",
              "lower": "Lower than recent historical baseline"}
MODEL_INPUTS = ["Day of the week", "Public holidays and festivals (incl. day before / day after)",
                "Time of year (annual seasonality)", "Past daily admission records (before the cut-off)"]
NOT_MODEL_INPUTS = ["Weather", "Air quality (AQI, pollutants)"]


def _iso(t) -> str:
    return pd.Timestamp(t).strftime("%Y-%m-%d")


def build_run(store, fc: dict, as_of: str) -> dict:
    """Pure: forecast rows (from the runner) + DataStore -> run payload. Never alters the official rows."""
    cut = pd.Timestamp(as_of)
    rows = fc["rows"]
    if len(rows) != H or [r["date"] for r in rows] != [_iso(cut + pd.Timedelta(days=i)) for i in range(H)]:
        raise ValueError("forecast rows do not match the official 14-day contract")
    official = [{"date": r["date"], "mean": float(r["mean"]), "lower_80": float(r["lower_80"]), "upper_80": float(r["upper_80"])}
                for r in rows]

    fest = store.festivals
    hol_days = set(fest["date"])
    end = cut + pd.Timedelta(days=H - 1)

    def holidays_on(t):
        g = fest[fest["date"] == t]
        return [{"name": n, "type": ty} for n, ty in sorted(zip(g["name"], g["type"]))]

    # historical comparison baseline: same weekday, the 8 weeks (56 days) before the cut-off, raw records
    obs = store.daily["admissions"]
    base = obs.loc[cut - pd.Timedelta(days=7 * BASELINE_WEEKS): cut - pd.Timedelta(days=1)]
    if len(base) != 7 * BASELINE_WEEKS:
        raise ValueError("the 8 weeks before the cut-off are not fully observed")
    ref = {}
    for wd in range(7):
        v = base[base.index.dayofweek == wd].to_numpy(dtype=float)
        p25, p75 = np.percentile(v, [25, 75])
        ref[wd] = {"p25": float(p25), "p75": float(p75), "n": int(len(v)),
                   "includes_potentially_incomplete_day": bool(any(_iso(t) == C.POTENTIALLY_INCOMPLETE_DAY
                                                                   for t in base[base.index.dayofweek == wd].index))}

    means = np.array([r["mean"] for r in official])
    order = np.argsort(-means, kind="stable")
    rank = np.empty(H, dtype=int)
    rank[order] = np.arange(1, H + 1)
    window_avg = float(means.mean())

    days = []
    for i, r in enumerate(official):
        t = pd.Timestamp(r["date"])
        b = ref[t.dayofweek]
        tier = "higher" if r["mean"] > b["p75"] else "lower" if r["mean"] < b["p25"] else "typical"
        days.append({
            **r,
            "weekday": t.day_name(),
            "holidays": holidays_on(t),
            "day_before_holiday": (t + pd.Timedelta(days=1)) in hol_days,
            "day_after_holiday": (t - pd.Timedelta(days=1)) in hol_days,
            "historical_comparison": {"tier": tier, "label": TIER_LABEL[tier], **b},
            "window_rank": int(rank[i]),
            "busiest_in_window": bool(rank[i] <= BUSIEST_N),
            "vs_window_average_pct": float(100 * (r["mean"] / window_avg - 1)),
        })

    tiers = [d["historical_comparison"]["tier"] for d in days]
    counts = {k: tiers.count(k) for k in ("higher", "typical", "lower")}
    busiest = sorted((d for d in days if d["busiest_in_window"]), key=lambda d: d["window_rank"])
    recent28 = float(obs.loc[cut - pd.Timedelta(days=28): cut - pd.Timedelta(days=1)].mean())

    # days to watch: busiest days of the window and calendar days (no weather/AQI in Stage 1)
    watch = [{"date": d["date"], "kind": "busiest", "title": f"Busiest day #{d['window_rank']} in this forecast",
              "detail": f"{d['weekday']}: {d['mean']:.1f} expected records (range {d['lower_80']:.1f}-{d['upper_80']:.1f})."}
             for d in busiest]
    for d in days:
        if d["holidays"]:
            watch.append({"date": d["date"], "kind": "holiday", "title": ", ".join(h["name"] for h in d["holidays"]),
                          "detail": f"{d['weekday']}: holiday in the official calendar; the model applies its fitted holiday effect. "
                                    f"{d['mean']:.1f} expected records."})

    up_end = cut + pd.Timedelta(days=30)
    upcoming = fest[(fest["date"] >= cut) & (fest["date"] < up_end)].sort_values(["date", "name"])
    m = store.holdout_m4
    hold = store.holdout
    return {
        "mode": "demonstration",
        "label": "Historical hackathon demonstration – not a live hospital forecast",
        "manifest": {
            "as_of": as_of,
            "forecast_start": official[0]["date"], "forecast_end": official[-1]["date"],
            "data_through": C.DATA_END,
            "data_through_note": "31 Mar 2019 is the last day in the file and is potentially incomplete.",
            "calendar_source": "festival_calendar_punjab.csv (official hackathon dataset)",
            "calendar_covered_until": _iso(fest["date"].max()),
            "model": "M4 Poisson GLM (frozen forecast.py)",
            "forecast_py_sha256": fc["forecast_py_sha256"],
            "sha256_matches_frozen": fc["forecast_py_sha256"] == C.FROZEN_FORECAST_SHA256,
            "fallback_used": bool(fc["fallback_used"]),
            "model_inputs": MODEL_INPUTS,
            "not_model_inputs": NOT_MODEL_INPUTS,
            "period_note": "This admission forecast covers 1–14 Apr 2019. Weather or air-quality forecasts shown elsewhere in "
                           "the dashboard cover a different, current period and had no influence on it.",
        },
        "official": official,
        "days": days,
        "summary": {
            "total_mean": float(means.sum()), "average_mean": window_avg,
            "busiest": {"date": days[int(order[0])]["date"], "mean": float(means[order[0]])},
            "quietest": {"date": days[int(order[-1])]["date"], "mean": float(means[order[-1]])},
            "recent_28_day_mean": recent28,
            "historical_comparison_counts": counts,
            "busiest_days": [d["date"] for d in busiest],
        },
        "historical_comparison_rule": {
            "description": "Each day's forecast mean compared with the same weekday's observed raw records in the 8 weeks "
                           "before the cut-off: above the 75th percentile = higher, below the 25th = lower, otherwise within range.",
            "baseline_start": _iso(base.index.min()), "baseline_end": _iso(base.index.max()),
            "no_higher_days_note": (None if counts["higher"] else
                                    "No day in this forecast exceeds the higher-demand threshold of the recent historical baseline."),
            "all_lower_note": (None if counts["lower"] < H else
                               "All 14 forecast days are below the recent same-weekday baseline. The frozen model follows the "
                               "2017–18 seasonal pattern and has no recent-level term, while early 2019 ran higher."),
            "descriptive_only": "Descriptive comparison only; not a hospital-capacity or staffing assessment.",
        },
        "window_ranking_rule": {
            "description": f"The 14 forecast means ranked against each other; the top {BUSIEST_N} are the busiest days of this "
                           "forecast period.",
            "descriptive_only": "Relative to this forecast period only; not a capacity or staffing assessment.",
        },
        "days_to_watch": sorted(watch, key=lambda w: (w["date"], w["kind"])),
        "upcoming_holidays": [{"date": _iso(t), "name": n, "type": ty, "in_forecast_window": bool(t <= end)}
                              for t, n, ty in zip(upcoming["date"], upcoming["name"], upcoming["type"])],
        "reliability": {
            "source": "Jan–Mar 2019, previously viewed in the M4 holdout context (saved evaluation, 19 cut-offs, 266 forecast-days); "
                      "not used to select the newer candidate models",
            "mae": float(m["MAE"]), "coverage80": float(m["coverage80"]), "bias": float(m["bias"]),
            "n_cutoffs": int(m["n_cutoffs"]), "n_points": int(m["n_points"]),
            "ratio_vs_baseline": float(m["MAE_ratio_vs_B0"]), "ratio_ci": hold["ratio_ci"]["primary"],
        },
    }


def build_router(get_store: Callable, safe: Callable, run_forecast: Callable) -> APIRouter:
    router = APIRouter(prefix="/api", tags=["forecast run"])

    @router.get("/run")
    def run(as_of: str = Query(DEMO_AS_OF)):
        """The official 14-day forecast with descriptive context. Demonstration mode: as_of must be 2019-04-01."""
        if as_of != DEMO_AS_OF:
            raise HTTPException(422, f"Demonstration mode: admissions data end on {C.DATA_END}, so the only supported "
                                     f"as_of is {DEMO_AS_OF}. Earlier cut-offs are in the retrospective explorer.")
        try:
            fc = run_forecast(as_of)
        except Exception as e:  # noqa: BLE001
            raise HTTPException(502, f"official forecast unavailable: {type(e).__name__}") from e
        return safe(build_run(get_store(), fc, as_of))

    return router
