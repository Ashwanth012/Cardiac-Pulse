"""Multi-variable features for direct 14-day forecasting (Stage A experiments).

One row per (origin, horizon h=1..14); the origin is the forecast cut-off and the target date is origin + h - 1.
Leakage rules (tested in test_stage_a.py):
  * history features use ONLY admissions on days < origin;
  * "env_cut" features (deployable) use ONLY environment observed on days < origin (7-day means/sums),
    i.e. what a harness submission can see - the harness files stop the day before the cut-off;
  * "env_or" features (ORACLE) are the ACTUAL environment on the target date. They are an upper bound on what
    perfect weather/AQI forecasts could add, never a deployable result;
  * calendar features come from the official festival calendar, which is complete by design (known in advance).
The target is the RAW daily admission count (repeats included), as scored by the harness.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

H = 14
ENV_VARS = {  # column -> aggregation over the 7 days before the cut-off
    "temp_mean_c": "mean", "rel_humidity_mean_pct": "mean", "precip_mm": "sum", "wind_mean_kmh": "mean",
    "aqi": "mean", "pm25": "mean", "pm10": "mean", "no2": "mean", "co": "mean",
}
HIST = ["lvl7", "lvl14", "lvl28", "lvl56", "wd4", "wd8", "last1", "trend7_56"]
CAL = ["h", "dow", "weekend", "holiday", "hol_eve", "hol_after", "doy_s1", "doy_c1", "doy_s2", "doy_c2"]
ENV_CUT = [f"cut_{v}" for v in ENV_VARS]
ENV_OR = [f"or_{v}" for v in ENV_VARS]
FEATURE_SETS = {"H": HIST + CAL, "HE": HIST + CAL + ENV_CUT, "HX": HIST + CAL + ENV_OR}


def calendar_flags(festivals: pd.DataFrame, start, end) -> pd.DataFrame:
    """holiday / day before / day after, from the official calendar (same definition as forecast.py)."""
    idx = pd.date_range(pd.Timestamp(start) - pd.Timedelta(days=1), pd.Timestamp(end) + pd.Timedelta(days=1))
    hol = pd.Series(idx.isin(pd.DatetimeIndex(pd.to_datetime(festivals["date"]).unique())), index=idx)
    return pd.DataFrame({"holiday": hol, "hol_eve": hol.shift(-1, fill_value=False),
                         "hol_after": hol.shift(1, fill_value=False)}).loc[start:end].astype(int)


def build_rows(y: pd.Series, env: pd.DataFrame | None, festivals: pd.DataFrame, origins, min_history=56) -> pd.DataFrame:
    """Feature rows for every origin in `origins` and h = 1..14.

    y: daily raw admissions indexed by date (only days < origin are ever read for a given origin).
    env: daily environment indexed by date (same rule for env_cut; env_or reads the target date).
    """
    y = y.sort_index().astype(float)
    first = y.index.min()
    cal = calendar_flags(festivals, first, max(origins) + pd.Timedelta(days=H))
    rows = []
    for o in pd.DatetimeIndex(origins):
        past = y.loc[: o - pd.Timedelta(days=1)]
        if len(past) < min_history or past.index.max() != o - pd.Timedelta(days=1):
            continue
        last56 = past.iloc[-56:]
        base = {"origin": o, "lvl7": past.iloc[-7:].mean(), "lvl14": past.iloc[-14:].mean(),
                "lvl28": past.iloc[-28:].mean(), "lvl56": last56.mean(), "last1": past.iloc[-1]}
        base["trend7_56"] = (base["lvl7"] + 1) / (base["lvl56"] + 1)
        wd = {d: last56[last56.index.dayofweek == d] for d in range(7)}
        if env is not None:
            e7 = env.loc[o - pd.Timedelta(days=7): o - pd.Timedelta(days=1)]
            for v, agg in ENV_VARS.items():
                base[f"cut_{v}"] = float(getattr(e7[v], agg)()) if len(e7) else np.nan
        for h in range(1, H + 1):
            t = o + pd.Timedelta(days=h - 1)
            w = wd[t.dayofweek]
            r = dict(base, date=t, h=h, dow=t.dayofweek, weekend=int(t.dayofweek >= 5),
                     wd4=w.iloc[-4:].mean(), wd8=w.mean(), **cal.loc[t].to_dict())
            doy = 2 * np.pi * t.dayofyear / 365.25
            r.update(doy_s1=np.sin(doy), doy_c1=np.cos(doy), doy_s2=np.sin(2 * doy), doy_c2=np.cos(2 * doy))
            if env is not None:
                for v in ENV_VARS:
                    r[f"or_{v}"] = float(env.at[t, v]) if t in env.index else np.nan
            r["y"] = float(y.at[t]) if t in y.index else np.nan
            rows.append(r)
    return pd.DataFrame(rows)
