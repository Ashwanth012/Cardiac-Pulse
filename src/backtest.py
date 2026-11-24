"""Step 4 rolling-origin backtesting that mimics the PS-06 harness.

Harness emulation (HARNESS.md):
* For cut-off c the harness supplies admission / pollution / weather rows dated < c and the
  complete festival calendar. `DataStore.context(c)` returns exactly that information:
  - y   : raw admission rows per resolved D.O.A, days < c (days with no rows -> 0)
  - env : Open-Meteo weather + pollution cleaned by the Step 2 rules applied to the
          TRUNCATED pollution file (so interpolation can never use post-cut-off values)
  - cal : holiday / eve / after flags from the complete festival calendar
* Truth for scoring is read separately and never passed to a model.

Prediction intervals (empirical, horizon-bucketed, leakage-free):
  for cut-off c, take this model's own past forecasts whose whole 14-day window ended
  before c (origin <= c-14), most recent `pool` origins; residual r = log((y+1)/(mean+1));
  interval = (mean+1)*exp(q_lo..q_hi of r in the horizon bucket) - 1, then forced to
  contain the mean. At runtime the scored file can rebuild this pool with an internal
  backtest on its own history - the same computation.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from . import daily as D
from . import forecasting as F
from . import loaders as L

DEV_START = pd.Timestamp("2017-04-01")
DEV_END = pd.Timestamp("2018-12-31")
BUCKETS = ((1, 3), (4, 7), (8, 14))


class DataStore:
    """Raw data loaded once; `context(c)` reproduces what the harness gives at cut-off c."""

    def __init__(self, root: str):
        adm, _ = L.load_admissions(root)
        days = pd.date_range(adm["doa"].min(), adm["doa"].max(), freq="D")
        self.counts = adm.groupby("doa").size().reindex(days, fill_value=0).astype(float)
        w = L.load_weather(root).set_index("date")
        self.weather = w[["temp_max_c", "temp_min_c", "temp_mean_c", "precip_mm", "rel_humidity_mean_pct",
                          "wind_max_kmh", "wind_mean_kmh"]].apply(pd.to_numeric)
        self.pollution_raw = L.load_pollution(root)
        fest = L.load_festivals(root)
        span = pd.date_range("2017-01-01", "2019-12-31", freq="D")
        self.cal = F.build_calendar(D.festival_daily(fest, span))
        self._env = {}
        self._full_env = None

    def env_at(self, cutoff: pd.Timestamp) -> pd.DataFrame:
        if cutoff not in self._env:
            pol = self.pollution_raw[self.pollution_raw["date"] < cutoff]
            clean = D.clean_pollution(pol, [])
            poll = clean[["aqi", "pm25", "pm10", "no2", "so2", "co", "ozone", "nh3"]]
            self._env[cutoff] = self.weather[self.weather.index < cutoff].join(poll, how="left")
        return self._env[cutoff]

    def full_env(self) -> pd.DataFrame:
        """ANALYSIS-ONLY (oracle experiments): environment including post-cut-off days."""
        if self._full_env is None:
            clean = D.clean_pollution(self.pollution_raw, [])
            self._full_env = self.weather.join(clean[["aqi", "pm25", "pm10", "no2", "so2", "co", "ozone", "nh3"]])
        return self._full_env

    def context(self, cutoff, with_env: bool = False) -> F.Context:
        cutoff = pd.Timestamp(cutoff)
        y = self.counts[self.counts.index < cutoff]
        env = self.env_at(cutoff) if with_env else self.weather.iloc[0:0]
        ctx = F.Context(y=y, env=env, cal=self.cal, cutoff=cutoff)
        ctx.check()
        return ctx

    def truth(self, cutoff) -> np.ndarray:
        return self.counts.reindex(pd.date_range(cutoff, periods=F.H)).to_numpy(dtype=float)


def make_cutoffs(start="2017-07-01", end="2018-12-18", step=4) -> pd.DatetimeIndex:
    """Grid cut-offs whose 14-day window lies inside DEV (no cut-off after 2018-12-18; on the default grid the last is 2018-12-15)."""
    end = min(pd.Timestamp(end), DEV_END - pd.Timedelta(days=F.H - 1))
    return pd.date_range(start, end, freq=f"{step}D")


def run_backtest(store: DataStore, models: dict, cutoffs, with_env=()) -> pd.DataFrame:
    """models: name -> callable(ctx) returning 14 means. Returns one row per (model, cutoff, h)."""
    rows = []
    for c in cutoffs:
        ctx = store.context(c)
        ctx_env = None
        truth = store.truth(c)
        if truth[-1] != truth[-1] or pd.Timestamp(c) + pd.Timedelta(days=F.H - 1) > DEV_END:
            raise AssertionError(f"cut-off {c} forecast window leaves DEV")
        for name, fn in models.items():
            use = ctx
            if name in with_env:
                ctx_env = ctx_env or store.context(c, with_env=True)
                use = ctx_env
            t0 = time.perf_counter()
            mean = np.asarray(fn(use), dtype=float)
            dt = time.perf_counter() - t0
            if mean.shape != (F.H,) or not np.isfinite(mean).all():
                raise ValueError(f"{name} at {c}: bad forecast {mean}")
            mean = np.maximum(mean, 0.0)
            for h in range(F.H):
                rows.append((name, c, h + 1, c + pd.Timedelta(days=h), truth[h], mean[h], dt))
    return pd.DataFrame(rows, columns=["model", "cutoff", "h", "date", "y", "mean", "seconds"])


# ================================================================== intervals
def bucket_of(h):
    for i, (a, b) in enumerate(BUCKETS):
        if a <= h <= b:
            return i
    raise ValueError(h)


def add_intervals(fc: pd.DataFrame, q_lo=0.10, q_hi=0.90, pool=45, min_pool=10) -> pd.DataFrame:
    """Empirical horizon-bucket intervals from each model's own past (completed) forecasts."""
    out = []
    for name, g in fc.groupby("model", sort=False):
        g = g.copy()
        g["bucket"] = g["h"].map(bucket_of)
        g["r"] = np.log((g["y"] + 1) / (g["mean"] + 1))
        cuts = np.sort(g["cutoff"].unique())
        lo = np.full(len(g), np.nan)
        hi = np.full(len(g), np.nan)
        npool = np.zeros(len(g), dtype=int)
        for c in cuts:
            c = pd.Timestamp(c)
            done = [pd.Timestamp(o) for o in cuts if pd.Timestamp(o) <= c - pd.Timedelta(days=F.H)]
            done = done[-pool:]
            mask = (g["cutoff"] == c).to_numpy()
            npool[mask] = len(done)
            if len(done) < min_pool:
                continue
            past = g[g["cutoff"].isin(done)]
            q = past.groupby("bucket")["r"].quantile([q_lo, q_hi]).unstack()
            sub = g[mask]
            ql = q.loc[sub["bucket"], q_lo].to_numpy()
            qh = q.loc[sub["bucket"], q_hi].to_numpy()
            m = sub["mean"].to_numpy()
            lo[mask] = np.maximum(np.minimum((m + 1) * np.exp(ql) - 1, m), 0.0)
            hi[mask] = np.maximum((m + 1) * np.exp(qh) - 1, m)
        g["lo"], g["hi"], g["pool"] = lo, hi, npool
        out.append(g.drop(columns=["r"]))
    return pd.concat(out, ignore_index=True)


# ==================================================================== metrics
def metrics(df: pd.DataFrame) -> dict:
    e = df["mean"] - df["y"]
    out = {
        "MAE": float(e.abs().mean()),
        "RMSE": float(np.sqrt((e ** 2).mean())),
        "WAPE": float(e.abs().sum() / df["y"].sum()),
        "bias": float(e.mean()),
        "MAE_h1_7": float(e[df["h"] <= 7].abs().mean()),
        "MAE_h8_14": float(e[df["h"] > 7].abs().mean()),
        "n_points": int(len(df)),
        "n_cutoffs": int(df["cutoff"].nunique()),
    }
    if "lo" in df:
        ok = df["lo"].notna()
        d = df[ok]
        out["coverage80"] = float(((d["y"] >= d["lo"]) & (d["y"] <= d["hi"])).mean()) if len(d) else np.nan
        out["width80"] = float((d["hi"] - d["lo"]).mean()) if len(d) else np.nan
        out["interval_points"] = int(len(d))
    return out


def summarise(fc: pd.DataFrame, by=("model",)) -> pd.DataFrame:
    rows = []
    for key, g in fc.groupby(list(by), sort=False):
        key = key if isinstance(key, tuple) else (key,)
        rows.append(dict(zip(by, key), **metrics(g)))
    return pd.DataFrame(rows)


def per_cutoff(fc: pd.DataFrame) -> pd.DataFrame:
    g = fc.assign(ae=(fc["mean"] - fc["y"]).abs(),
                  inside=((fc["y"] >= fc.get("lo", np.nan)) & (fc["y"] <= fc.get("hi", np.nan))))
    return g.groupby(["model", "cutoff"]).agg(MAE=("ae", "mean"), coverage=("inside", "mean")).reset_index()


# ============================================================ harness score
def error_points(ratio):
    """EXACT HARNESS rule: 25 * clip(1 - ratio/2, 0, 1)."""
    return 25.0 * np.clip(1 - np.asarray(ratio) / 2, 0, 1)


def coverage_points(cov):
    """EXACT HARNESS rule for the coverage share: 15 at >= 80%, linear to 0 at 40%."""
    return 15.0 * np.clip((np.asarray(cov) - 0.40) / 0.40, 0, 1)


def width_penalty_factor(width, baseline_width):
    """APPROXIMATION. HARNESS.md: 'intervals wider than twice the baseline interval lose points in
    proportion' but does not define the baseline interval. Assumed: factor = min(1, 2*w_base / w)."""
    return np.minimum(1.0, 2 * np.asarray(baseline_width) / np.asarray(width))


def simulate_scores(fc: pd.DataFrame, model: str, baseline: str = "B0_seasonal_naive", n_cutoffs=4,
                    draws=5000, seed=0, cutoffs=None) -> pd.DataFrame:
    """Score distribution over random draws of `n_cutoffs` cut-offs (harness uses ~4).

    error ratio is computed two ways because HARNESS.md does not say whether MAE is pooled
    over the 56 values or averaged per cut-off: 'pooled' (MAE over 56 / baseline MAE over 56)
    and 'per_cutoff' (mean of per-cut-off ratios).
    """
    rng = np.random.default_rng(seed)
    pc = fc[fc["model"].isin([model, baseline])]
    if cutoffs is not None:
        pc = pc[pc["cutoff"].isin(cutoffs)]
    pc = pc[pc["lo"].notna()] if "lo" in pc else pc
    cuts = np.array(sorted(set(pc[pc.model == model].cutoff) & set(pc[pc.model == baseline].cutoff)))
    m = pc[pc.model == model].set_index(["cutoff", "h"]).sort_index()
    b = pc[pc.model == baseline].set_index(["cutoff", "h"]).sort_index()
    out = []
    for _ in range(draws):
        pick = rng.choice(cuts, n_cutoffs, replace=False)
        mm, bb = m.loc[pick], b.loc[pick]
        ae_m = (mm["mean"] - mm["y"]).abs()
        ae_b = (bb["mean"] - bb["y"]).abs()
        r_pool = ae_m.mean() / ae_b.mean()
        r_cut = np.mean([ae_m.loc[c].mean() / max(ae_b.loc[c].mean(), 1e-9) for c in pick])
        cov = ((mm["y"] >= mm["lo"]) & (mm["y"] <= mm["hi"])).mean()
        wf = width_penalty_factor((mm["hi"] - mm["lo"]).mean(), (bb["hi"] - bb["lo"]).mean())
        out.append(dict(ratio_pooled=r_pool, ratio_per_cutoff=r_cut, coverage=cov, width_factor=wf,
                        err_pts_pooled=float(error_points(r_pool)), err_pts_per_cutoff=float(error_points(r_cut)),
                        cov_pts=float(coverage_points(cov) * wf)))
    s = pd.DataFrame(out)
    s["total_pooled"] = s["err_pts_pooled"] + s["cov_pts"]
    return s


def block_bootstrap_ratio(fc: pd.DataFrame, model: str, ref: str, cutoffs, block=4, draws=4000, seed=1):
    """CI for MAE(model)/MAE(ref) resampling blocks of consecutive cut-offs (windows overlap)."""
    rng = np.random.default_rng(seed)
    cutoffs = np.array(sorted(cutoffs))
    pc = per_cutoff(fc[fc["cutoff"].isin(cutoffs) & fc["model"].isin([model, ref])])
    a = pc[pc.model == model].set_index("cutoff")["MAE"].reindex(cutoffs).to_numpy()
    r = pc[pc.model == ref].set_index("cutoff")["MAE"].reindex(cutoffs).to_numpy()
    n = len(cutoffs)
    nb = int(np.ceil(n / block))
    stats_ = []
    for _ in range(draws):
        starts = rng.integers(0, n - block + 1, nb)
        idx = np.concatenate([np.arange(s, s + block) for s in starts])[:n]
        stats_.append(a[idx].sum() / r[idx].sum())
    point = a.sum() / r.sum()
    lo, hi = np.percentile(stats_, [2.5, 97.5])
    return float(point), float(lo), float(hi)
