"""PS-06 Cardiac Admissions Intelligence - API (Phase 1: foundation + Command Centre).

Read-only. Aggregate data only. Every response passes the PII guard.
Run from dashboard/backend:   python -m uvicorn app.main:app --port 8000
"""
from __future__ import annotations

import math
import re
import threading
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from . import config as C
from . import forecast_runner as FR
from . import quality as Q
from . import run as R
from . import conditions as COND
from . import ml as ML
from .providers import open_meteo as OM
from .datastore import SERIES, DataStore, PIIError, assert_no_pii, round_floats

store: DataStore | None = None


def _warm():
    try:
        FR.run_forecast(C.HISTORICAL_CUTOFF)
    except Exception:  # noqa: BLE001 - warm-up failures surface on first request
        pass


@asynccontextmanager
async def lifespan(_app):
    global store
    store = DataStore()
    threading.Thread(target=_warm, daemon=True).start()
    yield


app = FastAPI(title="PS-06 Cardiac Admissions Intelligence API", version="0.1.0", lifespan=lifespan,
              docs_url="/api/docs", openapi_url="/api/openapi.json")
app.add_middleware(CORSMiddleware, allow_origins=C.ALLOWED_ORIGINS, allow_methods=["GET"], allow_headers=["*"])


def _store() -> DataStore:
    global store
    if store is None:
        store = DataStore()
    return store


def safe(payload):
    payload = round_floats(payload)
    try:
        return assert_no_pii(payload)
    except PIIError as e:  # pragma: no cover - guarded by tests
        raise HTTPException(500, f"response blocked by PII guard: {e}") from e


def _redact(msg: str) -> str:
    return re.sub(r"([A-Za-z]:\\|/)[^\s'\"()]+", "<path>", msg)


def _historical_forecast():
    try:
        r = FR.run_forecast(C.HISTORICAL_CUTOFF)
    except FR.ForecastError as e:
        raise HTTPException(502, f"historical forecast unavailable: {e}") from e
    return r


# ====================================================================== endpoints
@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/meta")
def meta():
    s = _store()
    return safe({
        "data": {"start": C.DATA_START, "end": C.DATA_END, "days": int(len(s.daily)),
                 "potentially_incomplete": s.incomplete_day_context()},
        "historical_forecast": {
            "cutoff": C.HISTORICAL_CUTOFF, "horizon_start": C.HISTORICAL_CUTOFF,
            "horizon_end": (pd.Timestamp(C.HISTORICAL_CUTOFF) + pd.Timedelta(days=13)).strftime("%Y-%m-%d"),
            "label": "Historical forecast - not a live forecast (data end 31 Mar 2019)"},
        "model": {
            "name": "M4 Poisson GLM", "id": "M4_glm_Nall_K2_T0_L0", "file": "forecast.py",
            "sha256": FR.forecast_sha256(), "frozen": FR.forecast_sha256() == C.FROZEN_FORECAST_SHA256,
            "features": ["weekday", "holiday", "holiday eve", "day after holiday", "annual cycle (2 harmonics)"],
            "not_used": ["weather", "air quality", "recent-level adjustment", "trend"],
            "interval": {"level": "80%", "q_lo": float(s.interval_m4.q_lo), "q_hi": float(s.interval_m4.q_hi),
                         "calibration_origins": int(s.interval_m4.pool),
                         "meaning": "80% prediction interval (target coverage, as defined by HARNESS.md); its bounds are "
                                    "the central 85% of past residuals, calibrated to reach at least 80% coverage",
                         **s.interval_evidence},
            "validation": {k: float(v) for k, v in s.validation_m4.items()},
            "holdout": {k: float(v) for k, v in s.holdout_m4.items()},
        },
        "series": {k: {"column": v[0], "label": v[1]} for k, v in SERIES.items()},
    })


@app.get("/api/admissions/daily")
def admissions_daily(series: str = Query("raw"), start: str | None = None, end: str | None = None):
    if series not in SERIES:
        raise HTTPException(422, f"series must be one of {sorted(SERIES)}")
    try:
        for d in (start, end):
            if d is not None:
                pd.Timestamp(d)
    except ValueError as e:
        raise HTTPException(422, "dates must be YYYY-MM-DD") from e
    return safe(_store().series(series, start, end))


@app.get("/api/forecast/historical")
def forecast_historical(context_days: int = Query(42, ge=7, le=180)):
    """Historical 14-day forecast from cut-off 2019-04-01. Contains NO actual outcomes for forecast dates."""
    s = _store()
    r = _historical_forecast()
    cutoff = pd.Timestamp(C.HISTORICAL_CUTOFF)
    hol = s.holidays_between(cutoff, cutoff + pd.Timedelta(days=13))
    ctx = s.daily.loc[cutoff - pd.Timedelta(days=context_days): cutoff - pd.Timedelta(days=1)]
    t28 = s.trailing_28
    return safe({
        "kind": "historical_forecast",
        "label": "Historical forecast - cut-off 1 Apr 2019, forecasts 1-14 Apr 2019 (not live)",
        "cutoff": C.HISTORICAL_CUTOFF,
        "fallback_used": r["fallback_used"],
        "fallback_messages": [_redact(m) for m in r["fallback_messages"]],
        "forecast": [{**row, "weekday": pd.Timestamp(row["date"]).day_name(),
                      "holidays": hol.get(row["date"], [])} for row in r["rows"]],
        "observed_before_cutoff": [
            {"date": d.strftime("%Y-%m-%d"), "admissions": int(v), "trailing_28d_mean": float(t28[d]),
             "potentially_incomplete": d.strftime("%Y-%m-%d") == C.POTENTIALLY_INCOMPLETE_DAY}
            for d, v in ctx["admissions"].items()],
    })


@app.get("/api/overview")
def overview(window_days: int = Query(90, ge=14, le=730)):
    s = _store()
    d = s.daily
    end = pd.Timestamp(C.DATA_END)
    last28 = d.loc[end - pd.Timedelta(days=27): end, "admissions"]
    raw_total = int(d["admissions"].sum())
    r = _historical_forecast()
    f = pd.DataFrame(r["rows"])
    fc_mean, fc_lo, fc_hi = float(f["mean"].mean()), float(f["lower_80"].mean()), float(f["upper_80"].mean())
    peak = f.loc[f["mean"].idxmax()]
    hol = s.holidays_between(C.HISTORICAL_CUTOFF, f["date"].iloc[-1])
    inc = s.incomplete_day_context()
    dm = s.dup_month
    mar = float(dm.at["2019-03", "mrd_doa_extra_pct"])
    dev_rate = s.dev_repeat_rate_pct()
    ho = s.holdout_m4
    jm18 = float(d.loc["2018-01-01":"2018-03-31", "admissions"].mean())
    jm19 = float(d.loc["2019-01-01":"2019-03-31", "admissions"].mean())
    wk = s.weekday_means()
    win = d.loc[end - pd.Timedelta(days=window_days - 1): end]
    t28 = s.trailing_28

    kpis = [
        {"id": "total_records", "label": "Admission records", "value": raw_total, "unit": "records",
         "detail": f"Raw rows, {C.DATA_START} to {C.DATA_END}", "provenance": "observed", "series": "raw"},
        {"id": "avg_daily", "label": "Average daily admissions", "value": float(d["admissions"].mean()),
         "unit": "per day", "detail": f"All {len(d)} days, raw records", "provenance": "observed", "series": "raw"},
        {"id": "last_28", "label": "Last 28 days", "value": float(last28.mean()), "unit": "per day",
         "detail": "4-31 Mar 2019, raw records; includes potentially incomplete 31 Mar",
         "delta_vs_avg_pct": float(100 * (last28.mean() / d["admissions"].mean() - 1)),
         "provenance": "observed", "series": "raw"},
        {"id": "forecast_avg", "label": "Historical forecast average", "value": fc_mean, "unit": "per day",
         "detail": "1-14 Apr 2019 (cut-off 1 Apr 2019) - not live", "provenance": "historical_forecast"},
        {"id": "forecast_range", "label": "Typical 80% daily range", "value": [fc_lo, fc_hi], "unit": "per day",
         "detail": "Average of the daily 80% bounds, 1-14 Apr 2019", "provenance": "historical_forecast"},
        {"id": "dedup_records", "label": "Records after de-duplication",
         "value": int(d["admissions_dedup_mrd_doa"].sum()), "unit": "records",
         "detail": f"One per MRD + admission date; {raw_total - int(d['admissions_dedup_mrd_doa'].sum()):,} repeat rows. "
                   "Sensitivity view - not the scored target", "provenance": "sensitivity", "series": "dedup_mrd"},
    ]
    alerts = [
        {"id": "historical", "severity": "info", "title": "Historical data only",
         "detail": "Data end on 31 Mar 2019. The forecast shown is a historical forecast for 1-14 Apr 2019, not a live forecast."},
        {"id": "potentially_incomplete_day", "severity": "warning", "title": "31 Mar 2019 potentially incomplete",
         "detail": f"{inc['admissions']} records vs a median of {inc['median_previous_28_days']:.0f} in the previous 28 days. "
                   "It is the last day in the file; kept unchanged and flagged."},
        {"id": "repeat_rows", "severity": "warning", "title": f"Repeat-row rate {mar:.1f}% in Mar 2019",
         "detail": f"Rows repeating an MRD + admission date, vs {dev_rate:.1f}% on average in Apr 2017-Dec 2018. "
                   "Raw counts (the forecast target) include them."},
        {"id": "model_bias", "severity": "warning", "title": "Known under-forecast in early 2019",
         "detail": "In Jan-Mar 2019 (the period used once to evaluate the model; previously viewed) actual admissions ran "
                   "above the model's forecasts. Read forecasts alongside the recent level and use the upper end of the "
                   "range as the cautious planning reference."},
    ]
    if hol:
        alerts.append({"id": "holidays", "severity": "info", "title": "Holidays in the forecast window",
                       "detail": "; ".join(f"{pd.Timestamp(k).strftime('%d %b')}: {', '.join(v)}" for k, v in hol.items())})
    insights = [
        {"id": "weekday", "title": "Strong weekday rhythm",
         "detail": f"Mondays average {wk['Monday']:.1f} records, Sundays {wk['Sunday']:.1f} (all days, raw).",
         "provenance": "observed"},
        {"id": "year_on_year", "title": "Higher level in early 2019",
         "detail": f"Jan-Mar 2019 averaged {jm19:.1f}/day vs {jm18:.1f}/day in Jan-Mar 2018 "
                   f"({100 * (jm19 / jm18 - 1):+.0f}%, raw records; part of the rise is repeat rows).",
         "provenance": "observed"},
        {"id": "forecast_vs_recent", "title": "Forecast sits below the recent level",
         "detail": f"Historical forecast average {fc_mean:.1f}/day vs {last28.mean():.1f}/day over the last 28 days. "
                   "The model follows the 2017-18 seasonal pattern and has no recent-level term.",
         "provenance": "historical_forecast"},
        {"id": "peak_day", "title": "Highest forecast day",
         "detail": f"{pd.Timestamp(peak['date']).strftime('%a %d %b %Y')}: {peak['mean']:.1f} "
                   f"(80% range {peak['lower_80']:.0f}-{peak['upper_80']:.0f}).",
         "provenance": "historical_forecast"},
    ]
    trend = [{"date": dt.strftime("%Y-%m-%d"), "admissions": int(v),
              "trailing_28d_mean": None if pd.isna(t28[dt]) else float(t28[dt]),
              "low_count_flag": bool(win.at[dt, "low_count_flag"]),
              "potentially_incomplete": dt.strftime("%Y-%m-%d") == C.POTENTIALLY_INCOMPLETE_DAY}
             for dt, v in win["admissions"].items()]
    return safe({"kpis": kpis, "alerts": alerts, "insights": insights,
                 "trend": {"window_days": window_days, "end": C.DATA_END, "series": "raw", "points": trend},
                 "forecast_fallback_used": r["fallback_used"]})


# ============================================================ Phase 2: Forecasting Studio
def _validate_retro_cutoff(cutoff: str) -> pd.Timestamp:
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", cutoff or ""):
        raise HTTPException(422, "cutoff must be YYYY-MM-DD")
    try:
        c = pd.Timestamp(cutoff)
    except ValueError as e:
        raise HTTPException(422, "cutoff is not a valid date") from e
    lo, hi = pd.Timestamp(C.RETRO_MIN_CUTOFF), pd.Timestamp(C.RETRO_MAX_CUTOFF)
    if c < lo or c > hi:
        raise HTTPException(422, f"Retrospective cut-offs must be between {C.RETRO_MIN_CUTOFF} and {C.RETRO_MAX_CUTOFF} "
                                 "(development period). Forecasts reaching the Jan-Mar 2019 holdout cannot be explored.")
    if c + pd.Timedelta(days=13) >= pd.Timestamp(C.HOLDOUT_START):   # defensive; unreachable with the bounds above
        raise HTTPException(422, "forecast window would reach the 2019 holdout")
    return c


@app.get("/api/forecast/retrospective")
def forecast_retrospective(cutoff: str = Query(...), context_days: int = Query(42, ge=7, le=180)):
    """Development-period forecast from the frozen model; actual outcomes are returned SEPARATELY for comparison."""
    c = _validate_retro_cutoff(cutoff)
    s = _store()
    try:
        r = FR.run_forecast(c.strftime("%Y-%m-%d"))
    except FR.ForecastError as e:
        raise HTTPException(502, f"retrospective forecast unavailable: {e}") from e
    window = pd.date_range(c, periods=14, freq="D")
    actual = s.daily.loc[window, "admissions"].astype(float).to_numpy()
    actual_dedup = s.daily.loc[window, "admissions_dedup_mrd_doa"].astype(float).to_numpy()
    f = pd.DataFrame(r["rows"])
    err = f["mean"].to_numpy() - actual
    inside = (actual >= f["lower_80"].to_numpy()) & (actual <= f["upper_80"].to_numpy())
    base = s.seasonal_naive(c)
    base_mae = float(abs(pd.Series(base).to_numpy() - actual).mean())
    hol = s.holidays_between(window[0], window[-1])
    ctx = s.daily.loc[c - pd.Timedelta(days=context_days): c - pd.Timedelta(days=1), "admissions"]
    day = lambda d: d.strftime("%Y-%m-%d")  # noqa: E731
    return safe({
        "kind": "retrospective_forecast",
        "label": "Retrospective development-period analysis - frozen model, already selected; not used for tuning",
        "cutoff": day(c),
        "allowed_range": {"min": C.RETRO_MIN_CUTOFF, "max": C.RETRO_MAX_CUTOFF},
        "fallback_used": r["fallback_used"],
        "forecast": [{**row, "weekday": pd.Timestamp(row["date"]).day_name(), "holidays": hol.get(row["date"], [])}
                     for row in r["rows"]],
        "observed_before_cutoff": [{"date": day(d), "admissions": int(v)} for d, v in ctx.items()],
        "actual_after_cutoff": {"series": "raw", "label": "Actual raw records (revealed retrospectively)",
                                "points": [{"date": day(d), "value": int(v)} for d, v in zip(window, actual)]},
        "actual_after_cutoff_dedup": {"series": "dedup_mrd", "label": "Actual de-duplicated records (sensitivity only)",
                                      "points": [{"date": day(d), "value": int(v)} for d, v in zip(window, actual_dedup)]},
        "baseline": {"name": "Seasonal naive (harness baseline)",
                     "points": [{"date": day(d), "value": v} for d, v in zip(window, base)]},
        "metrics": {"MAE": float(abs(err).mean()), "RMSE": float((err ** 2).mean() ** 0.5),
                    "WAPE": float(abs(err).sum() / actual.sum()), "bias": float(err.mean()),
                    "coverage80": float(inside.mean()), "days_inside": int(inside.sum()),
                    "width80": float((f["upper_80"] - f["lower_80"]).mean()),
                    "MAE_baseline": base_mae,
                    "MAE_ratio_vs_baseline": float(abs(err).mean() / base_mae) if base_mae else None,
                    "scored_against": "raw records"},
    })


@app.get("/api/evaluation/holdout")
def evaluation_holdout():
    """Saved Jan-Mar 2019 M4 holdout evaluation (Steps 5.3/5.4; previously viewed). Read from files only; nothing is recomputed."""
    h = _store().holdout
    return safe({
        "kind": "holdout_evaluation",
        "label": "Saved Jan-Mar 2019 holdout evaluation - frozen forecast.py, evaluated once; read-only",
        "frozen_sha256_at_evaluation": h["forecast_py_sha256"],
        "sha256_matches_current": h["forecast_py_sha256"] == FR.forecast_sha256(),
        **{k: v for k, v in h.items() if k != "forecast_py_sha256"},
        "interval_evidence": _store().interval_evidence,
        "notes": ["Primary set: 19 cut-offs on the Step 4 grid (2019-01-04 to 2019-03-17), 266 forecast-days.",
                  "Baseline: seasonal naive - same weekday in the last observed week (harness definition).",
                  "Bias = mean(forecast - actual); negative means under-forecasting.",
                  "Holdout results must not be used for further model selection."],
    })


@app.get("/api/model/explain")
def model_explain(cutoff: str = Query(C.HISTORICAL_CUTOFF)):
    """Fitted multipliers of the frozen M4 model for a cut-off (descriptive associations, not causes)."""
    if cutoff != C.HISTORICAL_CUTOFF:
        _validate_retro_cutoff(cutoff)
    try:
        e = FR.run_explain(cutoff)
    except FR.ForecastError as err:
        raise HTTPException(502, f"model explanation unavailable: {err}") from err
    b = e["coef"]
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    weekday = [{"day": days[0], "multiplier": 1.0}] + [{"day": days[i], "multiplier": math.exp(b[i])} for i in range(1, 7)]
    calendar = [{"term": t, "multiplier": math.exp(b[i])} for t, i in
                (("Holiday", 7), ("Day before a holiday", 8), ("Day after a holiday", 9))]
    curve = []
    if e["fourier"]:
        vals = []
        for doy in range(1, 366):
            x = sum(b[10 + 2 * (k - 1)] * math.sin(2 * math.pi * k * doy / 365.25)
                    + b[11 + 2 * (k - 1)] * math.cos(2 * math.pi * k * doy / 365.25) for k in range(1, e["fourier"] + 1))
            vals.append(math.exp(x))
        avg = sum(vals) / len(vals)
        start = pd.Timestamp("2019-01-01")
        curve = [{"day_of_year": i + 1, "label": (start + pd.Timedelta(days=i)).strftime("%d %b"), "multiplier": v / avg}
                 for i, v in enumerate(vals)]
    return safe({
        "cutoff": cutoff, "training": {"start": e["train_start"], "end": e["train_end"], "days": e["train_days"]},
        "ridge": e["ridge"], "fourier_harmonics": e["fourier"],
        "weekday": weekday, "calendar": calendar, "annual_curve": curve, "reproduces_forecast": True,
        "note": "Multipliers are the frozen model's fitted associations in this hospital's records before the cut-off "
                "(Monday, non-holiday = 1). They describe patterns, not causes. Weather and air quality are not in the model.",
    })


# ============================================================ Phase 3A: data quality
app.include_router(Q.build_router(_store, safe))

# ============================================================ Stage 1: forecast run (official 14 days + context)
app.include_router(R.build_router(_store, safe, lambda cutoff: FR.run_forecast(cutoff)))

# ============================================================ Stage 2: live weather & air quality (context only)
_provider: OM.OpenMeteo | None = None


def _open_meteo() -> OM.OpenMeteo:
    global _provider
    if _provider is None:
        _provider = OM.OpenMeteo()
    return _provider


app.include_router(COND.build_router(safe, lambda: _open_meteo()))

# ============================================================ Stage A: multi-variable model comparison (internal)
app.include_router(ML.build_router(safe))
