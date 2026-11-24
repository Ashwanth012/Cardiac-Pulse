"""Step 4 forecasting models - pandas + numpy ONLY (eligible for the scored forecast.py).

Every model is a function  model(ctx, **params) -> np.ndarray of 14 means  where `ctx`
holds ONLY information available before the cut-off:

* ctx.y        daily raw admission counts, index = every day < cutoff (float)
* ctx.env      daily environment (weather + cleaned pollution), index = days < cutoff
* ctx.cal      calendar features for ANY date (festival calendar is complete in the
               harness, so future holiday flags are legitimately known)
* ctx.cutoff   first forecast day;  ctx.dates = cutoff .. cutoff + 13

No model may read anything else. Tests perturb post-cutoff data and assert that
forecasts do not change.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

H = 14


@dataclass
class Context:
    y: pd.Series
    env: pd.DataFrame
    cal: pd.DataFrame
    cutoff: pd.Timestamp

    @property
    def dates(self) -> pd.DatetimeIndex:
        return pd.date_range(self.cutoff, periods=H, freq="D")

    def check(self):
        assert self.y.index.max() < self.cutoff, "history reaches the cut-off"
        assert len(self.env) == 0 or self.env.index.max() < self.cutoff, "environment reaches the cut-off"


# ======================================================================= baselines
def seasonal_naive(ctx: Context) -> np.ndarray:
    """HARNESS baseline: value of the same weekday in the last observed week.

    Last observed week = cutoff-7 .. cutoff-1. Day d maps to cutoff-7 + ((d - (cutoff-7)) mod 7).
    """
    start = ctx.cutoff - pd.Timedelta(days=7)
    src = [start + pd.Timedelta(days=(d - start).days % 7) for d in ctx.dates]
    return ctx.y.reindex(src).to_numpy(dtype=float)


def rolling_mean(ctx: Context, window: int = 28) -> np.ndarray:
    return np.full(H, float(ctx.y.iloc[-window:].mean()))


# ============================================================ weekday / level models
def weekday_factors(y: pd.Series, window: int) -> np.ndarray:
    """Multiplicative weekday factors (Mon..Sun) from the last `window` days (a multiple of 7)."""
    w = y.iloc[-window:]
    by = w.groupby(w.index.dayofweek).mean().reindex(range(7))
    f = (by / by.mean()).to_numpy(dtype=float)
    return np.where(np.isfinite(f) & (f > 0), f, 1.0)


def calendar_multipliers(y: pd.Series, cal: pd.DataFrame, wf: np.ndarray, lookback: int = 365,
                         shrink: float = 5.0) -> dict:
    """Holiday / eve / after multipliers estimated from history only (log-ratio mean, shrunk to 1)."""
    hist = y.iloc[-lookback:]
    base = y.rolling(28, min_periods=14).mean().shift(1)          # trailing level, no look-ahead
    expected = base.reindex(hist.index) * wf[hist.index.dayofweek]
    out = {}
    for col in ("holiday", "holiday_eve", "holiday_after"):
        days = hist.index[cal.reindex(hist.index)[col].to_numpy(dtype=bool)]
        r = np.log((hist[days] + 0.5) / (expected[days] + 0.5)).dropna()
        out[col] = float(np.exp(r.sum() / (len(r) + shrink))) if len(r) else 1.0
    return out


def calendar_factor(dates, cal: pd.DataFrame, mult: dict) -> np.ndarray:
    c = cal.reindex(dates)
    f = np.ones(len(dates))
    for col, m in mult.items():
        f = f * np.where(c[col].to_numpy(dtype=bool), m, 1.0)
    return f


def weekday_level(ctx: Context, level_window: int = 28, wday_window: int = 91, calendar: bool = False) -> np.ndarray:
    """Recent level x weekday profile (x calendar multipliers)."""
    y = ctx.y
    wf = weekday_factors(y, wday_window)
    recent = y.iloc[-level_window:]
    adj = wf[recent.index.dayofweek]
    if calendar:
        mult = calendar_multipliers(y, ctx.cal, wf)
        adj = adj * calendar_factor(recent.index, ctx.cal, mult)
    level = float((recent / adj).mean())
    out = level * wf[ctx.dates.dayofweek]
    if calendar:
        out = out * calendar_factor(ctx.dates, ctx.cal, mult)
    return out


def year_ago(ctx: Context, level_window: int = 28, wday_window: int = 91) -> np.ndarray:
    """Last year's seasonal shape scaled to the current level; falls back to weekday_level."""
    y = ctx.y
    if (ctx.cutoff - y.index.min()).days < 364 + level_window + 7:
        return weekday_level(ctx, level_window, wday_window)
    wf = weekday_factors(y, wday_window)
    smooth = y.rolling(7, center=True, min_periods=4).mean()
    now = y.iloc[-level_window:].mean()
    then = y.reindex(y.index[-level_window:] - pd.Timedelta(days=364)).mean()
    ratio = now / then if then > 0 else 1.0
    shape = smooth.reindex(ctx.dates - pd.Timedelta(days=364)).to_numpy(dtype=float)
    shape = np.where(np.isfinite(shape), shape, now)
    return shape * ratio * wf[ctx.dates.dayofweek]


# ================================================================ Poisson GLM (numpy)
def poisson_irls(X: np.ndarray, y: np.ndarray, ridge: float = 1.0, iters: int = 50) -> np.ndarray:
    """Ridge-penalised Poisson regression by IRLS (column 0 = unpenalised intercept)."""
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
        if np.max(np.abs(b_new - b)) < 1e-8:
            return b_new
        b = b_new
    return b


def _design(dates: pd.DatetimeIndex, cal: pd.DataFrame, t0: pd.Timestamp, fourier: int, trend: bool,
            extra: np.ndarray | None = None) -> np.ndarray:
    cols = [np.ones(len(dates))]
    dow = dates.dayofweek
    cols += [(dow == d).astype(float) for d in range(1, 7)]
    c = cal.reindex(dates)
    cols += [c[k].to_numpy(dtype=float) for k in ("holiday", "holiday_eve", "holiday_after")]
    doy = dates.dayofyear.to_numpy()
    for k in range(1, fourier + 1):
        cols += [np.sin(2 * np.pi * k * doy / 365.25), np.cos(2 * np.pi * k * doy / 365.25)]
    if trend:
        cols.append(((dates - t0).days.to_numpy() / 365.25))
    X = np.column_stack(cols)
    if extra is not None:
        X = np.column_stack([X, extra])
    return X


def glm_calendar(ctx: Context, train_days: int | None = 364, fourier: int = 0, trend: bool = False,
                 level_window: int = 0, ridge: float = 1.0, env_cols=(), env_lag: int = 14,
                 oracle_env: pd.DataFrame | None = None) -> np.ndarray:
    """Poisson GLM: weekday + holiday/eve/after (+ annual Fourier, + trend, + environment).

    level_window > 0 rescales the forecast so the fitted values match the observed
    total of the last `level_window` days (recent-level correction).
    env_cols use the 7-day mean ending `env_lag` days before each target day. With
    env_lag >= 14 every value is known at the cut-off (deployable). `oracle_env` is an
    analysis-only switch that supplies future observations (NOT deployable).
    """
    y = ctx.y if train_days is None else ctx.y.iloc[-train_days:]
    if fourier and (y.index.max() - y.index.min()).days < 360:
        fourier = 0                       # cannot estimate an annual cycle from < 1 year
    t0 = y.index.min()
    tr_extra = fc_extra = None
    if env_cols:
        src = oracle_env if oracle_env is not None else ctx.env
        if oracle_env is None:
            # deployable: a 7-day mean ending env_lag >= 14 days before any forecast day ends
            # on or before cutoff-1, so it only uses observed history
            assert env_lag >= H, "env_lag < 14 would need post-cut-off observations"
        full = pd.date_range(src.index.min(), ctx.dates[-1], freq="D")
        base = src[list(env_cols)].reindex(full)          # deployable: future rows are NaN
        ma = base.rolling(7, min_periods=5).mean().shift(env_lag)
        mu_, sd_ = ma.reindex(y.index).mean(), ma.reindex(y.index).std().replace(0, 1)
        z = (ma - mu_) / sd_
        tr_extra = z.reindex(y.index).to_numpy()
        fc_extra = z.reindex(ctx.dates).to_numpy()
        keep = np.isfinite(tr_extra).all(axis=1)
        y = y[keep]
        tr_extra = tr_extra[keep]
        fc_extra = np.where(np.isfinite(fc_extra), fc_extra, 0.0)
    X = _design(y.index, ctx.cal, t0, fourier, trend, tr_extra)
    b = poisson_irls(X, y.to_numpy(dtype=float), ridge=ridge)
    Xf = _design(ctx.dates, ctx.cal, t0, fourier, trend, fc_extra)
    out = np.exp(np.clip(Xf @ b, -20, 20))
    if level_window:
        fitted = np.exp(np.clip(X[-level_window:] @ b, -20, 20))
        obs = y.to_numpy(dtype=float)[-level_window:]
        out = out * (obs.sum() / max(fitted.sum(), 1e-6))
    return out


# ===================================================== Holt-Winters (log, weekly season)
def _hw_run(x: np.ndarray, a: float, b: float, g: float, phi: float, m: int = 7):
    level = x[:m].mean()
    trend = 0.0
    season = list(x[:m] - level)
    sse = 0.0
    for t in range(m, len(x)):
        s = season[t - m]
        pred = level + phi * trend + s
        e = x[t] - pred
        sse += e * e
        new_level = a * (x[t] - s) + (1 - a) * (level + phi * trend)
        trend = b * (new_level - level) + (1 - b) * phi * trend
        season.append(g * (x[t] - new_level) + (1 - g) * s)
        level = new_level
    return sse, level, trend, season


def holt_winters(ctx: Context, train_days: int | None = 364) -> np.ndarray:
    """Additive Holt-Winters on log1p counts, weekly season, damped trend; grid-fitted on one-step SSE."""
    y = ctx.y if train_days is None else ctx.y.iloc[-train_days:]
    x = np.log1p(y.to_numpy(dtype=float))
    best = None
    for a in (0.05, 0.1, 0.2, 0.3):
        for b, phi in ((0.0, 1.0), (0.02, 0.9), (0.1, 0.9)):
            for g in (0.05, 0.1, 0.2):
                r = _hw_run(x, a, b, g, phi)
                if best is None or r[0] < best[0][0]:
                    best = (r, phi)
    (_, level, trend, season), phi = best
    n = len(x)
    out = []
    for h in range(1, H + 1):
        damp = sum(phi ** i for i in range(1, h + 1))
        out.append(level + damp * trend + season[n - 7 + (h - 1) % 7])
    return np.expm1(np.array(out))


# =========================================================== model registry helpers
def build_calendar(festival_daily: pd.DataFrame) -> pd.DataFrame:
    """holiday / holiday_eve / holiday_after flags for every date in the festival calendar span."""
    idx = pd.date_range(festival_daily.index.min(), festival_daily.index.max(), freq="D")
    hol = festival_daily["holiday_any"].reindex(idx, fill_value=False).astype(bool)
    return pd.DataFrame({"holiday": hol, "holiday_eve": hol.shift(-1, fill_value=False),
                         "holiday_after": hol.shift(1, fill_value=False)}, index=idx)
