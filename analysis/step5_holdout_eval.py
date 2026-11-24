"""Step 5.3: one-time evaluation of the FROZEN forecast.py on the Jan-Mar 2019 holdout.

Run from the project root:
    PYTHONDONTWRITEBYTECODE=1 python analysis/step5_holdout_eval.py

Nothing is tuned here. forecast.py is executed end-to-end, exactly as the harness would run it:
for every cut-off a fresh folder holds the admission / pollution / weather files truncated to days
< cut-off (discharge columns blanked for in-patients) plus the complete festival calendar and glossary;
a new Python process imports forecast.py with PS06_DATA_DIR set to that folder and calls
forecast(cutoff) once.

Cut-off sets
* PRIMARY  - Step 4 methodology: the 4-day grid anchored at 2017-07-01, cut-offs in 2019 whose
             14-day window ends by 2019-03-31 -> 2019-01-04 .. 2019-03-17 (19 cut-offs).
* DAILY    - sensitivity: every day 2019-01-01 .. 2019-03-18 (77 cut-offs; the last one scores
             2019-03-31, which Step 2 flagged as possibly incomplete).
Baseline: the exact harness seasonal naive (same weekday in the last observed week).
Outputs: outputs/holdout/{holdout_report.md, tables/*.csv, figures/*.png}
"""
import hashlib
import itertools
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from io import StringIO

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src import backtest as B  # noqa: E402
from src import forecasting as F  # noqa: E402
from src import loaders as L  # noqa: E402

OUT = os.path.join(ROOT, "outputs", "holdout")
TAB = os.path.join(OUT, "tables")
FIG = os.path.join(OUT, "figures")
SCRATCH = os.environ.get("PS06_SCRATCH", tempfile.gettempdir())
FORECAST_PY = os.path.join(ROOT, "forecast.py")
FROZEN_SHA256 = "aa4809046af56166ceed6f3daab1521be093878a7d9c0ffaa9d93b784462c866"
H = 14
SUSPECT_DAY = pd.Timestamp("2019-03-31")      # Step 2: last day in file, possibly incomplete


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


class Harness:
    """Builds HARNESS.md-style truncated folders and runs forecast.py in a fresh process."""

    def __init__(self):
        self.raw = L.read_raw_csv(os.path.join(ROOT, L.ADMISSION_FILE))
        self.adm, _ = L.load_admissions(raw=self.raw)
        self.pol_raw = L.read_raw_csv(os.path.join(ROOT, L.POLLUTION_FILE))
        self.pol_date = L.load_pollution(ROOT)["date"].to_numpy()
        self.w_raw = L.read_raw_csv(os.path.join(ROOT, L.WEATHER_FILE))
        days = pd.date_range(self.adm["doa"].min(), self.adm["doa"].max())
        self.counts = self.adm.groupby("doa").size().reindex(days, fill_value=0).astype(float)

    def folder(self, path, cutoff):
        keep = (self.adm["doa"] < cutoff).to_numpy()
        t = self.raw[keep].copy()
        inpat = (self.adm.loc[keep, "dod"].isna() | (self.adm.loc[keep, "dod"] >= cutoff)).to_numpy()
        for c in ["D.O.D", "DURATION OF STAY", "duration of intensive unit stay", "OUTCOME"]:
            t.loc[inpat, c] = ""
        t.to_csv(os.path.join(path, L.ADMISSION_FILE), index=False, encoding="utf-8-sig")
        self.pol_raw[self.pol_date < cutoff].to_csv(os.path.join(path, L.POLLUTION_FILE), index=False, encoding="utf-8-sig")
        self.w_raw[pd.to_datetime(self.w_raw["date"]) < cutoff].to_csv(os.path.join(path, L.WEATHER_FILE), index=False)
        for f in (L.FESTIVAL_FILE, L.GLOSSARY_FILE, L.MORTALITY_FILE):
            shutil.copy(os.path.join(ROOT, f), path)
        # leakage audit of the folder itself
        check = L.load_admissions(path)[0]
        assert check["doa"].max() < cutoff, "folder contains admissions on/after the cut-off"
        return dict(folder_rows=len(t), folder_last_doa=check["doa"].max(), inpatients_blanked=int(inpat.sum()))

    def run(self, cutoff):
        cutoff = pd.Timestamp(cutoff)
        with tempfile.TemporaryDirectory(dir=SCRATCH) as d:
            audit = self.folder(d, cutoff)
            code = ("import importlib.util, sys\n"
                    f"spec = importlib.util.spec_from_file_location('submission', r'{FORECAST_PY}')\n"
                    "m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)\n"
                    f"df = m.forecast('{cutoff.date()}')\n"
                    "sys.stdout.write(df.to_json(orient='split', double_precision=15))\n")
            env = dict(os.environ, PS06_DATA_DIR=d, PYTHONDONTWRITEBYTECODE="1")
            before = sorted(os.listdir(d))
            t0 = time.perf_counter()
            p = subprocess.run([sys.executable, "-c", code], cwd=d, env=env, capture_output=True, text=True, timeout=900)
            dt = time.perf_counter() - t0
            if p.returncode != 0:
                raise RuntimeError(p.stderr)
            assert sorted(os.listdir(d)) == before, "forecast.py wrote files"
        df = pd.read_json(StringIO(p.stdout), orient="split", dtype={"date": str})
        audit.update(seconds=dt, stderr=p.stderr.strip())
        return df, audit


def baseline_forecast(counts, cutoff):
    ctx = F.Context(y=counts[counts.index < cutoff], env=pd.DataFrame(), cal=pd.DataFrame(), cutoff=cutoff)
    return F.seasonal_naive(ctx)


def baseline_interval(counts, cutoff, mean, q=0.10, pool=45):
    """APPROXIMATION of the unstated harness 'baseline interval' (same as Step 4): seasonal-naive residual
    quantiles 10/90 from the 45 most recent completed grid origins before the cut-off."""
    origins = pd.date_range("2017-07-01", cutoff - pd.Timedelta(days=H), freq="4D")[-pool:]
    rs = []
    for o in origins:
        m = baseline_forecast(counts, o)
        y = counts.reindex(pd.date_range(o, periods=H)).to_numpy()
        for h in range(1, H + 1):
            rs.append((B.bucket_of(h), np.log((y[h - 1] + 1) / (m[h - 1] + 1))))
    r = pd.DataFrame(rs, columns=["bucket", "r"])
    qq = r.groupby("bucket")["r"].quantile([q, 1 - q]).unstack()
    b = [B.bucket_of(h) for h in range(1, H + 1)]
    lo = np.maximum(np.minimum((mean + 1) * np.exp(qq.loc[b, q].to_numpy()) - 1, mean), 0)
    hi = np.maximum((mean + 1) * np.exp(qq.loc[b, 1 - q].to_numpy()) - 1, mean)
    return lo, hi


def evaluate(hz, cutoffs, label):
    rows, audits = [], []
    for c in cutoffs:
        c = pd.Timestamp(c)
        df, audit = hz.run(c)
        truth = hz.counts.reindex(pd.date_range(c, periods=H)).to_numpy()
        assert np.isfinite(truth).all()
        b0 = baseline_forecast(hz.counts, c)
        b_lo, b_hi = baseline_interval(hz.counts, c, b0)
        audits.append(dict(set=label, cutoff=c.date(), **audit))
        for h in range(H):
            d = c + pd.Timedelta(days=h)
            assert df["date"].iloc[h] == d.strftime("%Y-%m-%d")
            rows.append(dict(set=label, cutoff=c, h=h + 1, date=d, y=truth[h], mean=df["mean"].iloc[h],
                             lo=df["lower_80"].iloc[h], hi=df["upper_80"].iloc[h], b0=b0[h], b0_lo=b_lo[h], b0_hi=b_hi[h]))
    return pd.DataFrame(rows), pd.DataFrame(audits)


def metrics(d, mean="mean", lo="lo", hi="hi"):
    e = d[mean] - d["y"]
    out = dict(MAE=e.abs().mean(), RMSE=np.sqrt((e ** 2).mean()), WAPE=e.abs().sum() / d["y"].sum(), bias=e.mean(),
               MAE_h1_7=e[d.h <= 7].abs().mean(), MAE_h8_14=e[d.h > 7].abs().mean())
    if lo:
        inside = (d["y"] >= d[lo]) & (d["y"] <= d[hi])
        out.update(coverage80=inside.mean(), coverage80_h1_7=inside[d.h <= 7].mean(),
                   coverage80_h8_14=inside[d.h > 7].mean(), width80=(d[hi] - d[lo]).mean())
    out.update(n_cutoffs=d["cutoff"].nunique(), n_points=len(d))
    return out


def compare(d):
    m4 = metrics(d)
    b0 = metrics(d, "b0", "b0_lo", "b0_hi")
    tab = pd.DataFrame({"M4 (forecast.py)": m4, "Seasonal naive (harness baseline)": b0}).T
    tab["MAE_ratio_vs_B0"] = tab["MAE"] / b0["MAE"]
    tab.loc["Seasonal naive (harness baseline)", "MAE_ratio_vs_B0"] = 1.0
    return tab


def block_ci(d, block, draws=4000, seed=1):
    pc = d.assign(a=(d["mean"] - d.y).abs(), b=(d.b0 - d.y).abs()).groupby("cutoff")[["a", "b"]].sum().sort_index()
    a, b = pc["a"].to_numpy(), pc["b"].to_numpy()
    n = len(pc)
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(draws):
        st = rng.integers(0, n - block + 1, int(np.ceil(n / block)))
        idx = np.concatenate([np.arange(s, s + block) for s in st])[:n]
        stats.append(a[idx].sum() / b[idx].sum())
    return a.sum() / b.sum(), *np.percentile(stats, [2.5, 97.5])


def harness_scores(d):
    """Every combination of 4 cut-offs: EXACT error/coverage rules; width penalty APPROXIMATED."""
    cuts = sorted(d.cutoff.unique())
    g = {c: x for c, x in d.groupby("cutoff")}
    rows = []
    for combo in itertools.combinations(cuts, 4):
        x = pd.concat([g[c] for c in combo])
        ae, be = (x["mean"] - x.y).abs(), (x.b0 - x.y).abs()
        r_pool = ae.mean() / be.mean()
        r_cut = np.mean([(g[c]["mean"] - g[c].y).abs().mean() / (g[c].b0 - g[c].y).abs().mean() for c in combo])
        cov = ((x.y >= x.lo) & (x.y <= x.hi)).mean()
        wf = min(1.0, 2 * (x.b0_hi - x.b0_lo).mean() / (x.hi - x.lo).mean())
        cov_pts = 15 * np.clip((cov - 0.4) / 0.4, 0, 1) * wf
        rows.append(dict(ratio_pooled=r_pool, ratio_per_cutoff=r_cut, coverage=cov, width_factor=wf,
                         err_pts_pooled=25 * np.clip(1 - r_pool / 2, 0, 1),
                         err_pts_per_cutoff=25 * np.clip(1 - r_cut / 2, 0, 1), cov_pts=cov_pts))
    s = pd.DataFrame(rows)
    s["total_pooled"] = s.err_pts_pooled + s.cov_pts
    s["total_per_cutoff"] = s.err_pts_per_cutoff + s.cov_pts
    return s


def main():
    assert sha(FORECAST_PY) == FROZEN_SHA256, "forecast.py differs from the frozen version"
    os.makedirs(TAB, exist_ok=True)
    os.makedirs(FIG, exist_ok=True)
    t0 = time.perf_counter()
    hz = Harness()
    grid = pd.date_range("2017-07-01", "2019-03-18", freq="4D")
    primary = grid[grid >= "2019-01-01"]
    daily = pd.date_range("2019-01-01", "2019-03-18")
    P, Pa = evaluate(hz, primary, "primary_grid")
    D, Da = evaluate(hz, daily, "daily_sensitivity")
    assert sha(FORECAST_PY) == FROZEN_SHA256
    # grid cut-offs are a subset of the daily run: forecasts must be identical (determinism)
    chk = D[D.cutoff.isin(primary)].set_index(["cutoff", "h"])[["mean", "lo", "hi"]]
    assert np.allclose(chk.to_numpy(), P.set_index(["cutoff", "h"])[["mean", "lo", "hi"]].to_numpy(), atol=1e-12)

    R = dict(P=P, D=D, audits=pd.concat([Pa, Da]), elapsed=time.perf_counter() - t0)
    R["primary"] = compare(P)
    R["daily"] = compare(D)
    R["daily_excl_0331"] = compare(D[D.cutoff + pd.Timedelta(days=H - 1) < SUSPECT_DAY])
    R["ci_primary"] = block_ci(P, block=4)
    R["ci_daily"] = block_ci(D, block=14)
    per_c = P.assign(ae=(P["mean"] - P.y).abs(), be=(P.b0 - P.y).abs(), inside=(P.y >= P.lo) & (P.y <= P.hi),
                     width=P.hi - P.lo).groupby("cutoff").agg(MAE=("ae", "mean"), MAE_B0=("be", "mean"),
                                                              coverage80=("inside", "mean"), width80=("width", "mean"))
    per_c["ratio"] = per_c.MAE / per_c.MAE_B0
    R["per_cutoff"] = per_c
    R["by_h"] = P.assign(ae=(P["mean"] - P.y).abs(), be=(P.b0 - P.y).abs(), inside=(P.y >= P.lo) & (P.y <= P.hi)) \
        .groupby("h").agg(MAE=("ae", "mean"), MAE_B0=("be", "mean"), coverage80=("inside", "mean"))
    month = {1: "Jan 2019", 2: "Feb 2019", 3: "Mar 2019"}
    bym = []
    for m, g in D.groupby(D.cutoff.dt.month):
        c = compare(g)
        bym.append(dict(cutoff_month=month[m], n_cutoffs=g.cutoff.nunique(), MAE=c.iloc[0].MAE, MAE_B0=c.iloc[1].MAE,
                        ratio=c.iloc[0].MAE_ratio_vs_B0, coverage80=c.iloc[0].coverage80, width80=c.iloc[0].width80))
    R["by_month"] = pd.DataFrame(bym).set_index("cutoff_month")
    R["scores"] = harness_scores(P)
    # earlier (non-holdout) results for context - read from approved Step 4 tables
    st4 = os.path.join(ROOT, "outputs", "backtest", "tables")
    val = pd.read_csv(os.path.join(st4, "validation_family_winners.csv"), index_col=0)
    jm = pd.read_csv(os.path.join(st4, "jan_mar_2018_analogue.csv")).set_index("model")
    tune = pd.read_csv(os.path.join(st4, "tuning_all_configs.csv")).set_index("model")
    R["context"] = pd.DataFrame([
        dict(stage="Step 4 tuning (2017-07..2018-06; configs chosen here)", cutoffs=int(tune.loc["M4_glm_Nall_K2_T0_L0", "n_cutoffs"]),
             MAE=tune.loc["M4_glm_Nall_K2_T0_L0", "MAE"], ratio_vs_B0=tune.loc["M4_glm_Nall_K2_T0_L0", "ratio_vs_B0"], coverage80=np.nan),
        dict(stage="Step 4 Jan-Mar 2018 analogue (subset of tuning)", cutoffs=int(jm.loc["M4_glm_Nall_K2_T0_L0", "n_cutoffs"]),
             MAE=jm.loc["M4_glm_Nall_K2_T0_L0", "MAE"], ratio_vs_B0=jm.loc["M4_glm_Nall_K2_T0_L0", "ratio_vs_B0"], coverage80=np.nan),
        dict(stage="Step 4 validation (2018-07..2018-12; model chosen here)", cutoffs=int(val.loc["M4", "n_cutoffs"]),
             MAE=val.loc["M4", "MAE"], ratio_vs_B0=val.loc["M4", "ratio_vs_B0"], coverage80=val.loc["M4", "coverage80"]),
        dict(stage="Step 5.3 HOLDOUT Jan-Mar 2019 (primary grid, frozen forecast.py)", cutoffs=int(R["primary"].iloc[0].n_cutoffs),
             MAE=R["primary"].iloc[0].MAE, ratio_vs_B0=R["primary"].iloc[0].MAE_ratio_vs_B0, coverage80=R["primary"].iloc[0].coverage80),
    ]).set_index("stage")
    R["level"] = dict(holdout_mean=P.drop_duplicates("date").y.mean(),
                      prior28_mean=hz.counts["2018-12-04":"2018-12-31"].mean(),
                      forecast_mean=P["mean"].mean(), truth_mean=P.y.mean())
    return R


def save(R):
    T = lambda n: os.path.join(TAB, n)  # noqa: E731
    R["P"].to_csv(T("forecasts_primary_grid.csv"), index=False)
    R["D"].to_csv(T("forecasts_daily_sensitivity.csv"), index=False)
    R["audits"].to_csv(T("leakage_and_runtime_audit.csv"), index=False)
    R["primary"].to_csv(T("metrics_primary_grid.csv"))
    R["daily"].to_csv(T("metrics_daily_sensitivity.csv"))
    R["daily_excl_0331"].to_csv(T("metrics_daily_excluding_2019-03-31.csv"))
    R["per_cutoff"].to_csv(T("per_cutoff_primary.csv"))
    R["by_h"].to_csv(T("by_horizon_primary.csv"))
    R["by_month"].to_csv(T("by_cutoff_month_daily.csv"))
    R["scores"].describe(percentiles=[.1, .5, .9]).to_csv(T("harness_score_all_4cutoff_combinations.csv"))
    R["context"].to_csv(T("context_earlier_stages.csv"))
    json.dump(dict(ci_primary=R["ci_primary"], ci_daily=R["ci_daily"], forecast_py_sha256=FROZEN_SHA256,
                   elapsed_s=R["elapsed"]), open(T("summary.json"), "w"), indent=1, default=float)
    return sorted(os.listdir(TAB))


if __name__ == "__main__":
    import warnings
    warnings.simplefilter("ignore")
    os.makedirs(SCRATCH, exist_ok=True)
    R = main()
    tables = save(R)
    import step5_report as REP  # noqa: E402
    figs = REP.figures(R, FIG)
    REP.write(R, tables, figs, os.path.join(OUT, "holdout_report.md"))
    pd.set_option("display.width", 220)
    print(f"done in {R['elapsed']:.0f}s")
    print(R["primary"].round(3).to_string())
    print("CI primary", np.round(R["ci_primary"], 3), "CI daily", np.round(R["ci_daily"], 3))
