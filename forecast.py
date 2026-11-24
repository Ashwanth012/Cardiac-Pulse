"""PS-06 submission: 14-day cardiac admission forecast with 80% prediction intervals.

Contract (HARNESS.md):  forecast(cutoff_date: str) -> pandas.DataFrame with exactly 14 rows and
columns date, mean, lower_80, upper_80 for cutoff_date .. cutoff_date + 13 days.

Model: M4_glm_Nall_K2_T0_L0, selected in Step 4 rolling-origin validation, reproduced exactly:
  * target   raw daily admission rows by D.O.A (duplicates kept), all history before the cut-off
  * features intercept, 6 weekday dummies (Mon = reference), holiday, holiday_eve, holiday_after,
             annual Fourier terms k = 1, 2 (dropped if the history spans < 360 days)
  * fit      ridge-penalised Poisson IRLS, ridge 1.0 (intercept unpenalised), <= 50 iterations, tol 1e-8
Intervals (frozen on Step 4 tuning cut-offs): M4's own back-forecasts at origins on the grid
2017-07-01 + 4k days whose whole 14-day window ended before the cut-off (origin <= cutoff - 14), the
most recent 90; residual log((y+1)/(mean+1)) per horizon bucket 1-3 / 4-7 / 8-14; quantiles
0.075 / 0.925 applied as (mean+1)*exp(q) - 1, clipped at 0 and forced to contain the mean.

Data use: ONLY "HDHI Admission data.csv" (rows dated before the cut-off) and the festival calendar,
read from PS06_DATA_DIR. Weather and pollution files are never opened. No files are written.
Dependencies: pandas, numpy and the Python standard library only.
Fallbacks (each reported on stderr): M4 -> recent-level x weekday model -> seasonal naive -> constant;
empirical intervals -> seasonal-naive residual band -> fixed relative band.
"""
from __future__ import annotations

import os
import re
import sys

import numpy as np
import pandas as pd

H = 14
ADMISSION_FILE = "HDHI Admission data.csv"
FESTIVAL_FILE = "festival_calendar_punjab.csv"

# ---- frozen M4 configuration (Step 4: M4_glm_Nall_K2_T0_L0)
M4_FOURIER = 2
M4_RIDGE = 1.0
M4_MIN_SPAN_FOR_FOURIER = 360
IRLS_ITERS = 50
IRLS_TOL = 1e-8

# ---- frozen interval configuration (Step 4 tuning: interval_settings_chosen.csv)
Q_LO, Q_HI = 0.075, 0.925
POOL = 90
MIN_POOL = 10              # Step 4 backtest required >= 10 origins
MIN_POOL_DEGRADED = 3      # below MIN_POOL, still usable with a stderr warning
BUCKETS = ((1, 3), (4, 7), (8, 14))
GRID_ANCHOR = pd.Timestamp("2017-07-01")
GRID_STEP_DAYS = 4
CAL_MIN = pd.Timestamp("2017-01-01")   # backtest calendar span
CAL_MAX = pd.Timestamp("2019-12-31")

_DATE_RE = r"^\s*(\d{1,2})\s*[/\-.]\s*(\d{1,2})\s*[/\-.]\s*(\d{4})\s*$"


def _warn(msg: str) -> None:
    print(f"[forecast.py] {msg}", file=sys.stderr)


# ============================================================================ reading
def _read_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, encoding="utf-8-sig", keep_default_na=False)


def _norm(name) -> str:
    return re.sub(r"\s+", " ", str(name).replace("﻿", "")).strip().upper()


def _col(df: pd.DataFrame, *names, required=True):
    lookup = {_norm(c): c for c in df.columns}
    for n in names:
        if _norm(n) in lookup:
            return lookup[_norm(n)]
    if required:
        raise KeyError(f"none of {names} in {list(df.columns)}")
    return None


def _make_date(year, month, day) -> pd.Series:
    ok = year.notna() & month.between(1, 12) & day.between(1, 31)
    txt = pd.Series(pd.NA, index=year.index, dtype="object")
    txt[ok] = (year[ok].astype(int).astype(str).str.zfill(4) + "-" + month[ok].astype(int).astype(str).str.zfill(2)
               + "-" + day[ok].astype(int).astype(str).str.zfill(2))
    return pd.to_datetime(txt, format="%Y-%m-%d", errors="coerce")


def parse_admission_dates(doa: pd.Series, month_year: pd.Series | None) -> pd.Series:
    """Resolve per-row mixed M/D/YYYY vs DD/MM/YYYY admission dates (same rules as src/loaders.py).

    1. keep the reading that falls in the `month year` month (both -> day == month, same date)
    2. only one reading is a real date -> use it
    3. both readings identical -> use it
    4. two different readings -> the one nearest the rolling median of resolved neighbours (<= 45 days)
    otherwise NaT (row ignored).
    """
    text = doa.astype(str).fillna("")
    parts = text.str.extract(_DATE_RE).apply(pd.to_numeric, errors="coerce")
    p1, p2, yr = parts[0], parts[1], parts[2]
    mdy, dmy = _make_date(yr, p1, p2), _make_date(yr, p2, p1)
    date = pd.Series(pd.NaT, index=doa.index, dtype="datetime64[ns]")
    done = pd.Series(False, index=doa.index)
    if month_year is not None:
        mtxt = month_year.astype(str).str.strip().str.title()
        my = pd.to_datetime(mtxt, format="%b-%y", errors="coerce")
        my = my.fillna(pd.to_datetime(mtxt, format="%b-%Y", errors="coerce")).dt.to_period("M")
        m_ok = mdy.notna() & my.notna() & (mdy.dt.to_period("M") == my)
        d_ok = dmy.notna() & my.notna() & (dmy.dt.to_period("M") == my)
        date[m_ok] = mdy[m_ok]
        only_d = d_ok & ~m_ok
        date[only_d] = dmy[only_d]
        done = m_ok | d_ok
    todo = ~done
    one_m = todo & mdy.notna() & dmy.isna()
    one_d = todo & dmy.notna() & mdy.isna()
    same = todo & mdy.notna() & dmy.notna() & (mdy == dmy)
    date[one_m], date[one_d], date[same] = mdy[one_m], dmy[one_d], mdy[same]
    amb = todo & mdy.notna() & dmy.notna() & (mdy != dmy)
    if amb.any() and date.notna().any():
        epoch = pd.Timestamp("1970-01-01")
        days = (date - epoch).dt.days.astype(float)
        ref = days.rolling(21, center=True, min_periods=1).median().ffill().bfill()
        dm = ((mdy - epoch).dt.days - ref).abs()
        dd = ((dmy - epoch).dt.days - ref).abs()
        pm = amb & (dm < dd) & (dm <= 45)
        pd_ = amb & (dd < dm) & (dd <= 45)
        date[pm], date[pd_] = mdy[pm], dmy[pd_]
    return date


def load_counts(data_dir: str, cutoff: pd.Timestamp) -> pd.Series:
    """Raw admission rows per day for days < cutoff (internal gaps = 0; nothing after the last observed day)."""
    raw = _read_csv(os.path.join(data_dir, ADMISSION_FILE))
    doa = parse_admission_dates(raw[_col(raw, "D.O.A")], raw[_col(raw, "month year", required=False)]
                                if _col(raw, "month year", required=False) else None)
    n_bad = int(doa.isna().sum())
    if n_bad:
        _warn(f"{n_bad} admission rows with unresolvable D.O.A ignored")
    doa = doa[doa.notna() & (doa < cutoff)]           # never use rows on/after the cut-off
    if doa.empty:
        raise ValueError("no admission rows before the cut-off")
    days = pd.date_range(doa.min(), doa.max(), freq="D")
    y = doa.value_counts().reindex(days, fill_value=0).astype(float).sort_index()
    gap = (cutoff - pd.Timedelta(days=1) - y.index.max()).days
    if gap > 0:
        _warn(f"history ends {gap} day(s) before the cut-off; those days are treated as unobserved")
    return y


def load_calendar(data_dir: str, cutoff: pd.Timestamp, start: pd.Timestamp) -> pd.DataFrame:
    """holiday / holiday_eve / holiday_after for every day of the history and the forecast window."""
    fest = _read_csv(os.path.join(data_dir, FESTIVAL_FILE))
    d = pd.to_datetime(fest[_col(fest, "date")].str.strip(), format="%Y-%m-%d", errors="coerce").dropna()
    lo = min(CAL_MIN, start, d.min() if len(d) else CAL_MIN)
    hi = max(CAL_MAX, cutoff + pd.Timedelta(days=H), d.max() if len(d) else CAL_MAX)
    idx = pd.date_range(lo, hi, freq="D")
    hol = pd.Series(idx.isin(pd.DatetimeIndex(d.unique())), index=idx)
    return pd.DataFrame({"holiday": hol, "holiday_eve": hol.shift(-1, fill_value=False),
                         "holiday_after": hol.shift(1, fill_value=False)}, index=idx)


# ============================================================================= model M4
def poisson_irls(X: np.ndarray, y: np.ndarray, ridge: float = M4_RIDGE, iters: int = IRLS_ITERS) -> np.ndarray:
    """Ridge-penalised Poisson regression by IRLS (column 0 = unpenalised intercept). Identical to Step 4."""
    b = np.zeros(X.shape[1])
    b[0] = np.log(max(y.mean(), 1e-3))
    P = np.eye(X.shape[1]) * ridge
    P[0, 0] = 0.0
    for _ in range(iters):
        eta = np.clip(X @ b, -20, 20)
        mu = np.exp(eta)
        z = eta + (y - mu) / mu
        XtW = X.T * mu
        b_new = np.linalg.solve(XtW @ X + P, XtW @ z)
        if np.max(np.abs(b_new - b)) < IRLS_TOL:
            return b_new
        b = b_new
    return b


def design(dates: pd.DatetimeIndex, cal: pd.DataFrame, fourier: int) -> np.ndarray:
    cols = [np.ones(len(dates))]
    dow = dates.dayofweek
    cols += [(dow == d).astype(float) for d in range(1, 7)]
    c = cal.reindex(dates)
    cols += [c[k].to_numpy(dtype=float) for k in ("holiday", "holiday_eve", "holiday_after")]
    doy = dates.dayofyear.to_numpy()
    for k in range(1, fourier + 1):
        cols += [np.sin(2 * np.pi * k * doy / 365.25), np.cos(2 * np.pi * k * doy / 365.25)]
    X = np.column_stack(cols)
    if not np.isfinite(X).all():
        raise ValueError("non-finite design matrix (calendar does not cover these dates)")
    return X


def m4_forecast(y: pd.Series, cal: pd.DataFrame, cutoff: pd.Timestamp) -> np.ndarray:
    """M4_glm_Nall_K2_T0_L0: all history, weekday + holiday/eve/after + Fourier K=2, no trend, no level rescale."""
    if len(y) and y.index.max() >= cutoff:
        raise AssertionError("M4 was given data on/after its cut-off")
    fourier = M4_FOURIER if (y.index.max() - y.index.min()).days >= M4_MIN_SPAN_FOR_FOURIER else 0
    b = poisson_irls(design(y.index, cal, fourier), y.to_numpy(dtype=float))
    dates = pd.date_range(cutoff, periods=H, freq="D")
    out = np.exp(np.clip(design(dates, cal, fourier) @ b, -20, 20))
    return np.maximum(out, 0.0)


# ======================================================================= fallback means
def weekday_level_forecast(y: pd.Series, cutoff: pd.Timestamp, level_window=56, wday_window=364) -> np.ndarray:
    w = y.iloc[-wday_window:]
    by = w.groupby(w.index.dayofweek).mean().reindex(range(7))
    f = (by / by.mean()).to_numpy(dtype=float)
    f = np.where(np.isfinite(f) & (f > 0), f, 1.0)
    recent = y.iloc[-level_window:]
    level = float((recent / f[recent.index.dayofweek]).mean())
    return level * f[pd.date_range(cutoff, periods=H, freq="D").dayofweek]


def seasonal_naive_forecast(y: pd.Series, cutoff: pd.Timestamp) -> np.ndarray:
    start = cutoff - pd.Timedelta(days=7)
    src = [start + pd.Timedelta(days=(d - start).days % 7) for d in pd.date_range(cutoff, periods=H, freq="D")]
    return y.reindex(src).to_numpy(dtype=float)


# ========================================================================== intervals
def calibration_origins(cutoff: pd.Timestamp, last_observed: pd.Timestamp) -> pd.DatetimeIndex:
    """Grid origins 2017-07-01 + 4k whose whole 14-day window ended before the cut-off and is observed."""
    end = min(cutoff - pd.Timedelta(days=H), last_observed - pd.Timedelta(days=H - 1))
    if end < GRID_ANCHOR:
        return pd.DatetimeIndex([])
    grid = pd.date_range(GRID_ANCHOR, end, freq=f"{GRID_STEP_DAYS}D")
    return grid[-POOL:]


def bucket_index(h: int) -> int:
    for i, (a, b) in enumerate(BUCKETS):
        if a <= h <= b:
            return i
    raise ValueError(h)


def calibration_residuals(y: pd.Series, cal: pd.DataFrame, cutoff: pd.Timestamp, model=None) -> pd.DataFrame:
    """M4 back-forecasts at each calibration origin, fitted only on data before that origin."""
    model = model or m4_forecast
    rows = []
    for o in calibration_origins(cutoff, y.index.max()):
        hist = y[y.index < o]
        if len(hist) < 28:
            continue
        mean = np.maximum(np.asarray(model(hist, cal, o), dtype=float), 0.0)
        truth = y.reindex(pd.date_range(o, periods=H, freq="D")).to_numpy(dtype=float)
        if not (np.isfinite(mean).all() and np.isfinite(truth).all()):
            continue
        for h in range(1, H + 1):
            rows.append((o, h, bucket_index(h), np.log((truth[h - 1] + 1) / (mean[h - 1] + 1))))
    return pd.DataFrame(rows, columns=["origin", "h", "bucket", "r"])


def empirical_interval(mean: np.ndarray, res: pd.DataFrame):
    q = res.groupby("bucket")["r"].quantile([Q_LO, Q_HI]).unstack()
    b = np.array([bucket_index(h) for h in range(1, H + 1)])
    ql, qh = q.loc[b, Q_LO].to_numpy(), q.loc[b, Q_HI].to_numpy()
    lo = np.maximum(np.minimum((mean + 1) * np.exp(ql) - 1, mean), 0.0)
    hi = np.maximum((mean + 1) * np.exp(qh) - 1, mean)
    return lo, hi


def fallback_interval(mean: np.ndarray, y: pd.Series):
    """Band from recent seasonal-naive (lag-7) log-ratio residuals; fixed +-50% if history is too short."""
    recent = y.iloc[-91:]
    r = np.log((recent + 1) / (recent.shift(7) + 1)).dropna()
    if len(r) >= 28:
        ql, qh = np.quantile(r, [Q_LO, Q_HI])
    else:
        ql, qh = np.log(0.5), np.log(1.5)
    lo = np.maximum(np.minimum((mean + 1) * np.exp(ql) - 1, mean), 0.0)
    hi = np.maximum((mean + 1) * np.exp(qh) - 1, mean)
    return lo, hi


# ============================================================================ assembly
def _parse_cutoff(cutoff_date) -> pd.Timestamp:
    s = str(cutoff_date).strip()
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
        raise ValueError(f"cutoff_date must be 'YYYY-MM-DD', got {cutoff_date!r}")
    return pd.Timestamp(pd.to_datetime(s, format="%Y-%m-%d"))


def _frame(cutoff: pd.Timestamp, mean, lo, hi) -> pd.DataFrame:
    dates = pd.date_range(cutoff, periods=H, freq="D")
    mean = np.asarray(mean, dtype=float)
    lo = np.minimum(np.asarray(lo, dtype=float), mean)
    hi = np.maximum(np.asarray(hi, dtype=float), mean)
    return pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "mean": mean, "lower_80": lo, "upper_80": hi})


def _valid(df) -> bool:
    try:
        v = df[["mean", "lower_80", "upper_80"]].to_numpy(dtype=float)
        return (len(df) == H and list(df.columns) == ["date", "mean", "lower_80", "upper_80"]
                and np.isfinite(v).all() and (v[:, 1] <= v[:, 0]).all() and (v[:, 0] <= v[:, 2]).all())
    except Exception:
        return False


def forecast_from_history(y: pd.Series, cal: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    """Pure core: history (days < cutoff) + calendar -> 14-row forecast frame, with fallbacks."""
    y = y[y.index < cutoff]
    mean, used = None, None
    for name, fn in (("M4", lambda: m4_forecast(y, cal, cutoff)),
                     ("weekday-level fallback", lambda: weekday_level_forecast(y, cutoff)),
                     ("seasonal-naive fallback", lambda: seasonal_naive_forecast(y, cutoff))):
        try:
            m = np.asarray(fn(), dtype=float)
            if m.shape == (H,) and np.isfinite(m).all():
                mean, used = np.maximum(m, 0.0), name
                break
            _warn(f"FALLBACK: {name} returned non-finite or mis-shaped output")
        except Exception as e:  # noqa: BLE001
            _warn(f"FALLBACK: {name} failed: {type(e).__name__}: {e}")
    if mean is None:
        raise RuntimeError("all mean models failed")
    if used != "M4":
        _warn(f"FALLBACK: mean from {used}")

    lo = hi = None
    if used == "M4":
        try:
            res = calibration_residuals(y, cal, cutoff)
            n = res["origin"].nunique() if len(res) else 0
            if n >= MIN_POOL_DEGRADED and res["bucket"].nunique() == len(BUCKETS):
                if n < MIN_POOL:
                    _warn(f"interval pool has only {n} origins (< {MIN_POOL}); intervals less reliable")
                lo, hi = empirical_interval(mean, res)
            else:
                _warn(f"FALLBACK: interval pool too small ({n} origins)")
        except Exception as e:  # noqa: BLE001
            _warn(f"FALLBACK: empirical interval failed: {type(e).__name__}: {e}")
    if lo is None or not (np.isfinite(lo).all() and np.isfinite(hi).all()):
        _warn("FALLBACK: using seasonal-naive residual band for the interval")
        lo, hi = fallback_interval(mean, y)
    return _frame(cutoff, mean, lo, hi)


def _constant_frame(cutoff: pd.Timestamp) -> pd.DataFrame:
    """Last resort when no data can be read: DEV-period average level (Apr 2017 - Dec 2018) and spread."""
    _warn("FALLBACK: constant forecast (no usable history)")
    return _frame(cutoff, np.full(H, 21.0), np.full(H, 11.0), np.full(H, 33.0))


def forecast(cutoff_date: str) -> pd.DataFrame:
    """cutoff_date is 'YYYY-MM-DD'. Returns 14 rows: date, mean, lower_80, upper_80 for cutoff_date .. +13 days."""
    cutoff = _parse_cutoff(cutoff_date)
    data_dir = os.environ.get("PS06_DATA_DIR", os.getcwd())
    try:
        y = load_counts(data_dir, cutoff)
        try:
            cal = load_calendar(data_dir, cutoff, y.index.min())
        except Exception as e:  # noqa: BLE001
            _warn(f"FALLBACK: festival calendar unreadable ({type(e).__name__}: {e}); no holidays assumed")
            idx = pd.date_range(min(CAL_MIN, y.index.min()), max(CAL_MAX, cutoff + pd.Timedelta(days=H)), freq="D")
            cal = pd.DataFrame(False, index=idx, columns=["holiday", "holiday_eve", "holiday_after"])
        out = forecast_from_history(y, cal, cutoff)
    except Exception as e:  # noqa: BLE001
        _warn(f"FALLBACK: forecasting failed ({type(e).__name__}: {e})")
        out = _constant_frame(cutoff)
    if not _valid(out):
        _warn("FALLBACK: output failed validation")
        out = _constant_frame(cutoff)
    return out
