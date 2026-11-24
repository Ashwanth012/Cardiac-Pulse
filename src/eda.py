"""Step 3 EDA helpers: lag features, adjusted Poisson associations, multiple testing.

Analysis-only module (may use statsmodels / scipy). The scored forecast.py will NOT
import this file.

Conventions
* DEV period = 2017-04-01 .. 2018-12-31. All inferential statistics use DEV only;
  Jan-Mar 2019 is the forecasting holdout and is never used to select anything.
* lag k of x at day t  = x(t - k)                (past exposure, never future)
* ma0-k of x at day t  = mean(x(t-k) .. x(t))    (includes the same day)
* Associations are rate ratios (RR) from a log-link Poisson GLM with
  Newey-West (HAC, 7 lags) standard errors, which are robust to the observed
  over-dispersion and day-to-day autocorrelation. RR is per inter-quartile range
  (IQR) of the exposure measured on DEV days.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

DEV_START = pd.Timestamp("2017-04-01")
DEV_END = pd.Timestamp("2018-12-31")
HOLDOUT_START = pd.Timestamp("2019-01-01")
HAC_LAGS = 7
SPLINE_DF_PER_YEAR = 4

EXPOSURES = ["temp_mean_c", "temp_min_c", "temp_max_c", "rel_humidity_mean_pct", "precip_mm",
             "wind_mean_kmh", "aqi", "pm25", "pm10", "no2", "so2", "co", "ozone", "nh3"]
LAGS = [0, 1, 2, 3, 7, 14]
MOVING = [2, 7, 14]  # ma0-2, ma0-7, ma0-14


# ------------------------------------------------------------------ data prep
def load_inputs(outputs_dir: str):
    daily = pd.read_csv(f"{outputs_dir}/clean_daily.csv", index_col="date", parse_dates=["date"])
    env = pd.read_csv(f"{outputs_dir}/environment_daily_extended.csv", index_col="date", parse_dates=["date"])
    return daily, env


def lagged(series: pd.Series, k: int) -> pd.Series:
    """x(t-k) on a complete daily index (raises if the index has gaps)."""
    _require_daily(series.index)
    return series.shift(k)


def moving_avg(series: pd.Series, k: int) -> pd.Series:
    """mean of x(t-k)..x(t); NaN unless all k+1 days are present."""
    _require_daily(series.index)
    return series.rolling(k + 1, min_periods=k + 1).mean()


def _require_daily(idx):
    full = pd.date_range(idx.min(), idx.max(), freq="D")
    if len(full) != len(idx) or not (full == idx).all():
        raise ValueError("series index must be a complete, sorted daily index")


def exposure_features(env: pd.DataFrame, exposures=EXPOSURES, lags=LAGS, moving=MOVING) -> pd.DataFrame:
    """All lag and moving-average variants, computed on the extended env table."""
    cols = {}
    for x in exposures:
        for k in lags:
            cols[f"{x}__lag{k}"] = lagged(env[x], k)
        for k in moving:
            cols[f"{x}__ma0_{k}"] = moving_avg(env[x], k)
    return pd.DataFrame(cols, index=env.index)


def model_frame(daily: pd.DataFrame, env: pd.DataFrame, start=DEV_START, end=DEV_END) -> pd.DataFrame:
    """DEV-period frame with target, calendar covariates and all exposure variants."""
    f = daily.loc[start:end].copy()
    f["t"] = (f.index - start).days.astype(float)
    f["weekday"] = f.index.dayofweek
    f["holiday"] = f["holiday_any"].astype(int)
    f["major"] = f["holiday_major_festival"].astype(int)
    hol = daily["holiday_any"].astype(int)
    f["holiday_eve"] = hol.shift(-1).reindex(f.index).fillna(0).astype(int)   # calendar is known ahead
    f["holiday_after"] = hol.shift(1).reindex(f.index).fillna(0).astype(int)
    f = f.join(exposure_features(env), how="left")
    if f.index.max() > DEV_END and end <= DEV_END:
        raise AssertionError("holdout leaked into DEV frame")
    return f


def spline_df(frame: pd.DataFrame, per_year: float = SPLINE_DF_PER_YEAR) -> int:
    years = len(frame) / 365.25
    return max(3, int(round(per_year * years)))


# ------------------------------------------------------------------ modelling
def base_formula(target="admissions", season_df: int | None = None, weekday=True, holiday=True) -> str:
    terms = []
    if weekday:
        terms.append("C(weekday)")
    if holiday:
        terms.append("holiday")
    if season_df:
        terms.append(f"cr(t, df={season_df}, constraints='center')")
    return f"{target} ~ " + (" + ".join(terms) if terms else "1")


def fit_poisson(formula: str, data: pd.DataFrame):
    model = smf.glm(formula, data=data, family=sm.families.Poisson())
    return model.fit(cov_type="HAC", cov_kwds={"maxlags": HAC_LAGS})


def dispersion(formula: str, data: pd.DataFrame) -> float:
    r = smf.glm(formula, data=data, family=sm.families.Poisson()).fit()
    return float(r.pearson_chi2 / r.df_resid)


def rr_per_unit(result, term: str, unit: float) -> dict:
    """Rate ratio for a `unit` increase in `term`, with 95% CI and p-value."""
    b, se = result.params[term], result.bse[term]
    return dict(rr=float(np.exp(b * unit)), lo=float(np.exp((b - 1.96 * se) * unit)),
                hi=float(np.exp((b + 1.96 * se) * unit)), p=float(result.pvalues[term]))


def association(frame: pd.DataFrame, col: str, base: str, extra: str = "", unit: float | None = None) -> dict:
    """Fit `base + extra + col` on rows where every used column is present."""
    unit = unit if unit is not None else iqr(frame[col])
    formula = base + (f" + {extra}" if extra else "") + f" + Q('{col}')"
    quoted = re.findall(r"Q\('([^']+)'\)", formula)
    data = frame.dropna(subset=list(dict.fromkeys(quoted)))
    res = fit_poisson(formula, data)
    out = rr_per_unit(res, f"Q('{col}')", unit)
    out.update(unit=unit, n=int(res.nobs))
    return out


def iqr(x: pd.Series) -> float:
    q = x.dropna().quantile([0.25, 0.75])
    return float(q.iloc[1] - q.iloc[0])


def bh_qvalues(p) -> np.ndarray:
    """Benjamini-Hochberg false-discovery-rate adjusted p-values."""
    p = np.asarray(p, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(q, 1.0)
    return out


def acf(x: pd.Series, nlags: int = 28) -> pd.Series:
    x = x.dropna().to_numpy(dtype=float)
    x = x - x.mean()
    denom = (x * x).sum()
    return pd.Series([1.0] + [float((x[k:] * x[:-k]).sum() / denom) for k in range(1, nlags + 1)],
                     index=range(nlags + 1))


def vif(frame: pd.DataFrame, cols) -> pd.Series:
    """Variance inflation factors of `cols` (complete rows)."""
    X = frame[cols].dropna()
    X = (X - X.mean()) / X.std()
    out = {}
    for c in cols:
        others = X.drop(columns=c)
        beta, *_ = np.linalg.lstsq(np.c_[np.ones(len(X)), others], X[c], rcond=None)
        resid = X[c] - np.c_[np.ones(len(X)), others] @ beta
        r2 = 1 - resid.var() / X[c].var()
        out[c] = 1 / (1 - r2) if r2 < 1 else np.inf
    return pd.Series(out)
