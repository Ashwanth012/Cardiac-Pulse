"""Phase 3A - Data Quality & Monitoring (read-only, aggregate data only).

Sources (all saved Step 2 outputs, never modified):
  * outputs/daily_admissions.csv and outputs/duplicates_by_month.csv - via the DataStore whitelist.
  * outputs/clean_daily.csv - environment columns and their imputed flags only (whitelisted; no admission
    or mortality columns are read).
  * outputs/cleaning_log.csv - guarded: the per-record reference column is never read (`usecols`), and entries
    from the admission file are reduced to counts per rule immediately. Their values and reasons (which can
    contain per-record details) are never returned; only the pollution-station entries are listed.

Nothing here runs the forecasting model, and every response passes the PII guard via `safe`.
"""
from __future__ import annotations

from functools import cached_property
from typing import Callable

import numpy as np
import pandas as pd
from fastapi import APIRouter

from . import config as C

# --------------------------------------------------------------------------- whitelists
LOG_COUNT_COLUMNS = ["source", "rule", "action"]
LOG_POLLUTION_COLUMNS = ["source", "date", "column", "raw_value", "new_value", "action", "rule", "reason"]

WEATHER = [("temp_mean_c", "Mean temperature", "°C"), ("temp_max_c", "Max temperature", "°C"),
           ("temp_min_c", "Min temperature", "°C"), ("precip_mm", "Precipitation", "mm"),
           ("rel_humidity_mean_pct", "Relative humidity", "%"), ("wind_mean_kmh", "Mean wind", "km/h"),
           ("wind_max_kmh", "Max wind", "km/h")]
POLLUTION = [("aqi", "AQI", "index"), ("pm25", "PM2.5", "µg/m³"), ("pm10", "PM10", "µg/m³"), ("no2", "NO2", "µg/m³"),
             ("nh3", "NH3", "µg/m³"), ("so2", "SO2", "µg/m³"), ("co", "CO", "as reported"), ("ozone", "Ozone", "µg/m³")]
STATION = [("station_tmax_c", "Station max temperature", "°C"), ("station_tmin_c", "Station min temperature", "°C"),
           ("station_humidity_pct", "Station humidity", "%")]
GROUPS = [("weather", "Weather (Open-Meteo, primary source)", WEATHER),
          ("air_quality", "Air quality (pollution station)", POLLUTION),
          ("station_weather", "Station weather (cross-check only)", STATION)]
CLEAN_DAILY_COLUMNS = (["date"] + [v for _, _, vs in GROUPS for v, _, _ in vs]
                       + [f"{v}_imputed" for v, _, _ in POLLUTION])

# Fixed, generic rule descriptions (no per-record values).
RULES = {
    "A1": "Discharge date inconsistent or malformed (analysis-only field)",
    "A2": "Discharge date does not match the recorded length of stay (analysis-only field)",
    "A3": "Age outside the expected adult range (analysis-only field)",
    "A4": "Intensive-care days inconsistent with the total stay (analysis-only field)",
    "A5": "Yes/no field holds an unexpected value (analysis-only field)",
    "A6": "Daily record count unusually low (below 40% of the 29-day rolling median)",
    "P1": "Non-numeric or blank value: corrected only if unambiguous, otherwise set missing",
    "P2": "NO2 at the instrument cap or implausible; the AQI it drove set missing",
    "P3": "Daily average outside the day's reported min-max range (or min above max)",
    "P4": "AQI differs from the highest pollutant average",
    "P5": "Robust spike against the 15-day median (kept, flagged)",
    "P6": "Short gap (3 days or fewer) filled by linear interpolation",
}
ACTIONS = {"corrected": "value changed to a confidently recovered value",
           "set_missing": "invalid value replaced by a missing value",
           "imputed": "missing value filled by linear interpolation (gaps of 3 days or fewer)",
           "flagged": "value kept unchanged and only flagged"}

DEFINITIONS = {
    "exact": "Exact repeat: identical to an earlier record in every field except the serial number.",
    "mrd_date": "MRD + admission-date repeat: same medical-record number and same admission date as an earlier record "
           "(includes exact repeats).",
}
OPD_ANOMALY_MONTHS = ["2017-11", "2017-12"]
PERIODS = [("Apr-Dec 2017", "2017-04-01", "2017-12-31"), ("Jan-Dec 2018", "2018-01-01", "2018-12-31"),
           ("Jan-Mar 2019", "2019-01-01", "2019-03-31")]
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def _iso(d) -> str:
    return pd.Timestamp(d).strftime("%Y-%m-%d")


def _str_or_none(v):
    return None if v is None or (isinstance(v, float) and np.isnan(v)) else str(v)


class QualityData:
    """Read-only aggregates for the Data Quality page. Built once per DataStore."""

    def __init__(self, store):
        self.s = store
        self._out = store._root / "outputs"

    # ----------------------------------------------------------------- loaders
    @cached_property
    def _log_counts(self) -> pd.DataFrame:
        return pd.read_csv(self._out / "cleaning_log.csv", usecols=LOG_COUNT_COLUMNS)

    @cached_property
    def _pollution_log(self) -> pd.DataFrame:
        log = pd.read_csv(self._out / "cleaning_log.csv", usecols=LOG_POLLUTION_COLUMNS, dtype=str)
        return log[log["source"] == "pollution"].drop(columns="source").reset_index(drop=True)

    @cached_property
    def _env(self) -> pd.DataFrame:
        e = pd.read_csv(self._out / "clean_daily.csv", usecols=CLEAN_DAILY_COLUMNS, parse_dates=["date"])
        return e.set_index("date").sort_index().loc[C.DATA_START:C.DATA_END]

    # ----------------------------------------------------------------- payloads
    @cached_property
    def summary(self) -> dict:
        d = self.s.daily
        raw, ex, mrd = (int(d[c].sum()) for c in ("admissions", "admissions_dedup_exact", "admissions_dedup_mrd_doa"))
        log = self._log_counts
        low = d[d["low_count_flag"]]
        hol = self.s.holidays_between(C.DATA_START, C.DATA_END)
        by_rule = []
        for (src, rule), g in log.groupby(["source", "rule"], sort=True):
            entry = {"source": src, "rule": rule, "description": RULES.get(rule, f"Rule {rule} (see Step 2 report)"),
                     "entries": int(len(g)), "actions": {a: int(n) for a, n in g["action"].value_counts().sort_index().items()}}
            if src == "pollution":
                entry["fields"] = sorted(self._pollution_log.loc[self._pollution_log.rule == rule, "column"].unique().tolist())
            by_rule.append(entry)
        pol = log[log.source == "pollution"]["action"].value_counts()
        return {
            "period": {"start": C.DATA_START, "end": C.DATA_END, "days": int(len(d))},
            "scored_target": "raw",
            "records": {"raw": raw, "dedup_exact": ex, "dedup_mrd": mrd,
                        "exact_extra": raw - ex, "mrd_extra": raw - mrd, "near_extra": int(d["extra_near_copies"].sum()),
                        "exact_extra_rate": (raw - ex) / raw, "mrd_extra_rate": (raw - mrd) / raw},
            "days": {
                "zero_count": int(d["zero_count_day"].sum()),
                "low_count": int(len(low)),
                "potentially_incomplete": self.s.incomplete_day_context(),
                "low_count_days": [{"date": _iso(t), "weekday": t.day_name(), "admissions": int(v),
                                    "holidays": hol.get(_iso(t), []),
                                    "potentially_incomplete": _iso(t) == C.POTENTIALLY_INCOMPLETE_DAY}
                                   for t, v in low["admissions"].items()],
            },
            "environment_cells": {a: int(pol.get(a, 0)) for a in ("corrected", "set_missing", "imputed", "flagged")},
            "cleaning": {
                "total": int(len(log)),
                "by_source_action": [{"source": s, "action": a, "entries": int(n)}
                                     for (s, a), n in log.groupby(["source", "action"]).size().items()],
                "by_rule": by_rule,
                "actions": ACTIONS,
                "admissions_note": "Admission-file entries are flags only: no admission record was removed or changed, "
                                   "and the flagged fields (other than the daily-count check) are not used by the forecast. "
                                   "They are shown here as counts per rule only.",
            },
        }

    @cached_property
    def duplicates(self) -> dict:
        d = self.s.daily

        def rates(frame) -> dict:
            rows = int(frame["admissions"].sum())
            ex = int((frame["admissions"] - frame["admissions_dedup_exact"]).sum())
            mrd = int((frame["admissions"] - frame["admissions_dedup_mrd_doa"]).sum())
            return {"rows": rows, "exact_extra": ex, "mrd_extra": mrd,
                    "exact_rate": ex / rows if rows else None, "mrd_rate": mrd / rows if rows else None}

        monthly = [{"month": m, "near_extra": int(r.near_extra), **rates(d.loc[m])} for m, r in self.s.dup_month.iterrows()]
        weekday = [{"weekday": WEEKDAYS[i], **rates(g)} for i, g in d.groupby(d.index.dayofweek)]
        return {
            "provenance": "sensitivity",
            "scored_series": "raw",
            "definitions": DEFINITIONS,
            "note": "Repeats are identified approximately and are not removed: the forecasting target remains the raw "
                    "record count. De-duplicated series are sensitivity views only.",
            "monthly": monthly,
            "weekday": weekday,
            "periods": [{"label": lab, "start": a, "end": b, **rates(d.loc[a:b])} for lab, a, b in PERIODS],
            "daily": [{"date": _iso(t), "raw": int(r.admissions), "dedup_exact": int(r.admissions_dedup_exact),
                       "dedup_mrd": int(r.admissions_dedup_mrd_doa),
                       "extra_exact": int(r.admissions - r.admissions_dedup_exact),
                       "extra_mrd": int(r.admissions - r.admissions_dedup_mrd_doa),
                       "low_count_flag": bool(r.low_count_flag),
                       "potentially_incomplete": _iso(t) == C.POTENTIALLY_INCOMPLETE_DAY}
                      for t, r in d.iterrows()],
        }

    @cached_property
    def missing(self) -> dict:
        e = self._env
        months = pd.period_range(C.DATA_START, C.DATA_END, freq="M")
        key = e.index.to_period("M")
        groups = []
        for gid, glabel, vars_ in GROUPS:
            rows = []
            for col, label, unit in vars_:
                miss = e[col].isna()
                imp = e[f"{col}_imputed"].astype(bool) if f"{col}_imputed" in e else pd.Series(False, index=e.index)
                by_m = [{"month": str(m), "days": int((key == m).sum()), "missing": int(miss[key == m].sum()),
                         "imputed": int(imp[key == m].sum())} for m in months]
                rows.append({"variable": col, "label": label, "unit": unit, "missing": int(miss.sum()),
                             "imputed": int(imp.sum()), "months": by_m})
            groups.append({"group": gid, "label": glabel, "variables": rows})
        d = self.s.daily
        return {
            "provenance": "observed",
            "period": {"start": C.DATA_START, "end": C.DATA_END, "days": int(len(e))},
            "months": [str(m) for m in months],
            "admissions": {"missing_days": int(d["admissions"].isna().sum()), "zero_count_days": int(d["zero_count_day"].sum())},
            "groups": groups,
            "notes": ["Imputed = a missing pollution value filled by linear interpolation across a gap of 3 days or fewer.",
                      "Station weather is a cross-check only; Open-Meteo is the primary weather source.",
                      "Pollution values are as reported by the station (the AQI is a sub-index scale, not a concentration)."],
        }

    @cached_property
    def pollution_log(self) -> dict:
        p = self._pollution_log.sort_values(["date", "rule", "column"], kind="stable")
        rows = [{"date": r.date, "field": r.column, "original_value": _str_or_none(r.raw_value),
                 "cleaned_value": _str_or_none(r.new_value), "action": r.action, "rule": r.rule,
                 "rule_description": RULES.get(r.rule, ""), "reason": r.reason}
                for r in p.itertuples(index=False)]
        return {"provenance": "observed", "source": "pollution station file (Step 2 cleaning log)",
                "entries": len(rows), "actions": ACTIONS, "rows": rows}

    @cached_property
    def case_mix(self) -> dict:
        d = self.s.daily
        m = d[["admissions", "admissions_emergency", "admissions_opd"]].resample("MS").sum()
        months = [{"month": t.strftime("%Y-%m"), "records": int(r.admissions), "emergency": int(r.admissions_emergency),
                   "opd": int(r.admissions_opd), "opd_share": r.admissions_opd / r.admissions,
                   "anomaly": t.strftime("%Y-%m") in OPD_ANOMALY_MONTHS} for t, r in m.iterrows()]
        dev_other = [x["opd_share"] for x in months if x["month"] < "2019-01" and not x["anomaly"]]
        tot = d[["admissions", "admissions_emergency", "admissions_opd"]].sum()
        return {
            "provenance": "observed",
            "months": months,
            "overall": {"emergency_share": tot.admissions_emergency / tot.admissions, "opd_share": tot.admissions_opd / tot.admissions},
            "other_development_months_opd_share_range": [min(dev_other), max(dev_other)],
            "anomaly_months": OPD_ANOMALY_MONTHS,
            "note": "The Emergency/OPD split is unreliable around Nov-Dec 2017, when the OPD-coded share rose sharply while "
                    "emergency counts stayed flat (Step 3). It is shown for context only and is not a forecasting target.",
        }


def build_router(get_store: Callable, safe: Callable) -> APIRouter:
    """Read-only GET endpoints under /api/quality. `get_store` returns the shared DataStore."""
    router = APIRouter(prefix="/api/quality", tags=["data quality"])
    holder: dict = {}

    def q() -> QualityData:
        s = get_store()
        if holder.get("store") is not s:
            holder.update(store=s, data=QualityData(s))
        return holder["data"]

    @router.get("/summary")
    def summary():
        return safe(q().summary)

    @router.get("/duplicates")
    def duplicates():
        return safe(q().duplicates)

    @router.get("/missing")
    def missing():
        return safe(q().missing)

    @router.get("/pollution-log")
    def pollution_log():
        return safe(q().pollution_log)

    @router.get("/case-mix")
    def case_mix():
        return safe(q().case_mix)

    return router
