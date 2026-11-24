"""Stage A: train and compare multi-variable models against the frozen M4 benchmark.

PRE-REGISTERED SELECTION RULE (written before any result was computed):
  1. Candidates: {LGBM, XGB, RF, GLMH} x {H (history+calendar), HE (+ environment known at the cut-off)}.
     HX (oracle environment on the target date) is reported as an UPPER BOUND only and can never be selected.
  2. Hyper-parameters are fixed in this file (no search). Each candidate's interval quantile is chosen on the TUNING
     cut-offs only: the narrowest of {0.10/0.90, 0.075/0.925, 0.05/0.95} with tuning coverage >= 80%.
  3. On the VALIDATION cut-offs (2018-07-01 .. 2018-12-15), the candidate with the lowest MAE is the challenger.
     It replaces M4 only if (a) its block-bootstrap 95% CI of the MAE ratio vs M4 lies entirely below 1 and
     (b) its validation 80%-interval coverage is >= 80%. Otherwise M4 stays.
  4. Jan-Mar 2019 was already viewed for M4 (Step 5). It is reported ONCE for the selected model, labelled
     "previously viewed", and is never used to choose between candidates.

Data: outputs/clean_daily.csv (raw target, cleaned Open-Meteo weather + station pollution) and the official
festival calendar. Nothing in outputs/ is written; results go to experiments/multivar/results/.
Run from the project root:  PYTHONDONTWRITEBYTECODE=1 python experiments/multivar/run_stage_a.py
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src import backtest as B  # noqa: E402  (existing Step 4 metric / interval / bootstrap code, unchanged)
from features import ENV_VARS, FEATURE_SETS, H, build_rows  # noqa: E402

OUT = Path(__file__).resolve().parent / "results"
TUNE_END, VAL_START = pd.Timestamp("2018-06-17"), pd.Timestamp("2018-07-01")
DEV_CUTOFFS = B.make_cutoffs()                                             # identical grid to Step 4
HOLD_CUTOFFS = pd.date_range("2019-01-04", "2019-03-17", freq="4D")        # identical grid to Step 5 primary
Q_OPTIONS = [(0.10, 0.90), (0.075, 0.925), (0.05, 0.95)]
SEED = 7
warnings.filterwarnings("ignore")


# =============================================================================== models
def glm_design(df, cols, stats=None):
    """Numpy-only Poisson GLM design: log history levels, weekday dummies, calendar, harmonics, standardised env."""
    X = [np.ones(len(df))]
    names = ["const"]
    for c in ("lvl7", "lvl28", "wd4", "wd8"):
        X.append(np.log1p(df[c].to_numpy())); names.append(f"log_{c}")
    X.append(np.log(df["trend7_56"].to_numpy())); names.append("log_trend")
    for d in range(1, 7):
        X.append((df["dow"] == d).to_numpy(float)); names.append(f"dow{d}")
    for c in ("holiday", "hol_eve", "hol_after", "doy_s1", "doy_c1", "doy_s2", "doy_c2"):
        X.append(df[c].to_numpy(float)); names.append(c)
    X.append(df["h"].to_numpy(float) / H); names.append("h")
    env = [c for c in cols if c.startswith(("cut_", "or_"))]
    if env:
        E = df[env].to_numpy(float)
        if stats is None:
            stats = (np.nanmean(E, 0), np.nanstd(E, 0) + 1e-9)               # fitted on TRAINING rows only
        E = np.where(np.isnan(E), stats[0], E)
        X += list(((E - stats[0]) / stats[1]).T); names += env
    return np.column_stack(X), stats


def poisson_irls(X, y, ridge=1.0, iters=50):
    beta = np.zeros(X.shape[1]); beta[0] = np.log(y.mean() + 1e-9)
    P = ridge * np.eye(X.shape[1]); P[0, 0] = 0
    for _ in range(iters):
        mu = np.exp(np.clip(X @ beta, -20, 20))
        z = X @ beta + (y - mu) / mu
        W = mu
        new = np.linalg.solve(X.T @ (W[:, None] * X) + P, X.T @ (W * z))
        if np.max(np.abs(new - beta)) < 1e-8:
            beta = new
            break
        beta = new
    return beta


def fit_predict(kind, train, test, cols):
    ytr = train["y"].to_numpy()
    if kind == "GLMH":
        Xtr, st = glm_design(train, cols)
        Xte, _ = glm_design(test, cols, st)
        return np.exp(Xte @ poisson_irls(Xtr, ytr))
    Xtr, Xte = train[cols].to_numpy(float), test[cols].to_numpy(float)
    if kind == "LGBM":
        import lightgbm as lgb
        m = lgb.LGBMRegressor(objective="poisson", n_estimators=300, learning_rate=0.05, num_leaves=15, min_child_samples=30,
                              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=SEED, n_jobs=1, verbose=-1)
    elif kind == "XGB":
        import xgboost as xgb
        m = xgb.XGBRegressor(objective="count:poisson", n_estimators=300, learning_rate=0.05, max_depth=4, subsample=0.8,
                             colsample_bytree=0.8, random_state=SEED, n_jobs=1)
    elif kind == "RF":
        from sklearn.ensemble import RandomForestRegressor
        Xtr = np.nan_to_num(Xtr, nan=0.0); Xte = np.nan_to_num(Xte, nan=0.0)
        m = RandomForestRegressor(n_estimators=300, min_samples_leaf=5, max_features=0.5, random_state=SEED, n_jobs=1)
    else:
        raise ValueError(kind)
    m.fit(Xtr, ytr)
    return m.predict(Xte)


CONFIGS = [(k, fs) for k in ("GLMH", "LGBM", "XGB", "RF") for fs in ("H", "HE", "HX")]


def run_cutoff(rows, c, configs):
    train = rows[(rows["date"] < c) & rows["y"].notna()]
    test = rows[rows["origin"] == c].sort_values("h")
    if len(test) != H:
        raise AssertionError(f"cut-off {c.date()}: {len(test)} prediction rows")
    assert train["date"].max() < c and (train["origin"] < c).all()           # no leakage
    out = []
    for kind, fs in configs:
        t0 = time.perf_counter()
        pred = np.maximum(fit_predict(kind, train, test, FEATURE_SETS[fs]), 0.0)
        dt = time.perf_counter() - t0
        for h, (d, yv, p) in enumerate(zip(test["date"], test["y"], pred), start=1):
            out.append((f"{kind}_{fs}", c, h, d, yv, float(p), dt))
    return out


def add_q_intervals(fc, q):
    return B.add_intervals(fc, q_lo=q[0], q_hi=q[1], pool=90, min_pool=10)


def split(df):
    return df[df.cutoff <= TUNE_END], df[df.cutoff >= VAL_START]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    d = pd.read_csv(ROOT / "outputs" / "clean_daily.csv", parse_dates=["date"]).set_index("date")
    fest = pd.read_csv(ROOT / "festival_calendar_punjab.csv")
    y, env = d["admissions"], d[list(ENV_VARS)]
    origins = pd.date_range(d.index.min() + pd.Timedelta(days=56), HOLD_CUTOFFS.max())
    rows = build_rows(y, env, fest, origins)
    print(f"feature rows: {len(rows)} ({rows.origin.nunique()} origins) in {time.time() - t0:.0f}s", flush=True)

    # ------------------------------------------------------------------ development grid (tuning + validation)
    res = Parallel(n_jobs=14)(delayed(run_cutoff)(rows, c, CONFIGS) for c in DEV_CUTOFFS)
    fc = pd.DataFrame([r for part in res for r in part], columns=["model", "cutoff", "h", "date", "y", "mean", "seconds"])
    print(f"dev grid done: {fc.cutoff.nunique()} cut-offs x {fc.model.nunique()} configs in {time.time() - t0:.0f}s", flush=True)

    # benchmark rows from the saved Step 4 archive (read-only): M4 and the harness baseline
    arc = pd.read_csv(ROOT / "outputs" / "backtest" / "forecasts_selected_models.csv.gz", parse_dates=["cutoff", "date"])
    bench = arc[arc.model.isin(["M4_glm_Nall_K2_T0_L0", "B0_seasonal_naive"])][["model", "cutoff", "h", "date", "y", "mean"]]
    bench = bench.assign(seconds=np.nan)
    assert np.allclose(bench.merge(fc[fc.model == "GLMH_H"], on=["cutoff", "h"]).eval("y_x - y_y").abs(), 0)  # same truth

    # ------------------------------------------------------------------ intervals: q chosen on TUNING only
    allfc, qchoice = [], {}
    for name, g in pd.concat([fc, bench]).groupby("model", sort=False):
        if name == "M4_glm_Nall_K2_T0_L0":
            q = (0.075, 0.925)                                                     # M4's frozen setting
        elif name == "B0_seasonal_naive":
            q = (0.10, 0.90)
        else:
            q = Q_OPTIONS[-1]
            for cand in Q_OPTIONS:
                tune = split(add_q_intervals(g, cand))[0]
                if B.metrics(tune)["coverage80"] >= 0.80:
                    q = cand
                    break
        qchoice[name] = q
        allfc.append(add_q_intervals(g, q))
    allfc = pd.concat(allfc, ignore_index=True)
    allfc.to_csv(OUT / "forecasts_dev_grid.csv.gz", index=False)

    tune, val = split(allfc)
    rows_out = []
    for name in allfc.model.unique():
        mt, mv = B.metrics(tune[tune.model == name]), B.metrics(val[val.model == name])
        rows_out.append({"model": name, "feature_set": name.split("_")[-1] if "_" in name else "",
                         "q": f"{qchoice[name][0]}/{qchoice[name][1]}",
                         **{f"tune_{k}": mt[k] for k in ("MAE", "coverage80")},
                         **{f"val_{k}": mv[k] for k in ("MAE", "RMSE", "WAPE", "bias", "coverage80", "width80", "n_cutoffs", "n_points")},
                         "sec_per_fit": float(allfc[allfc.model == name].seconds.mean())})
    summ = pd.DataFrame(rows_out)
    m4_val = summ.set_index("model").loc["M4_glm_Nall_K2_T0_L0", "val_MAE"]
    b0_val = summ.set_index("model").loc["B0_seasonal_naive", "val_MAE"]
    summ["val_ratio_vs_M4"] = summ.val_MAE / m4_val
    summ["val_ratio_vs_B0"] = summ.val_MAE / b0_val
    vc = val.cutoff.unique()
    cis = {}
    for name in summ.model:
        if name in ("M4_glm_Nall_K2_T0_L0",):
            continue
        cis[name] = B.block_bootstrap_ratio(val, name, "M4_glm_Nall_K2_T0_L0", vc)
    summ["ci_vs_M4"] = summ.model.map(lambda n: cis.get(n))
    summ = summ.sort_values("val_MAE")
    summ.to_csv(OUT / "comparison_validation.csv", index=False)

    # ------------------------------------------------------------------ pre-registered selection
    eligible = summ[summ.model.str.match(r"^(GLMH|LGBM|XGB|RF)_(H|HE)$")]
    challenger = eligible.iloc[0]
    ci = cis[challenger.model]
    lo_hi = (ci[1], ci[2]) if isinstance(ci, (tuple, list)) and len(ci) == 3 else tuple(ci)[-2:]
    adopt = bool(lo_hi[1] < 1.0 and challenger.val_coverage80 >= 0.80)
    decision = {"challenger": challenger.model, "challenger_val_MAE": float(challenger.val_MAE),
                "M4_val_MAE": float(m4_val), "ratio_vs_M4": float(challenger.val_ratio_vs_M4),
                "ratio_ci_95": [float(lo_hi[0]), float(lo_hi[1])], "challenger_val_coverage80": float(challenger.val_coverage80),
                "selected": challenger.model if adopt else "M4_glm_Nall_K2_T0_L0", "rule_passed": adopt,
                "q": qchoice[challenger.model]}
    # oracle upper bound: best HX vs the same model family's H
    orc = {}
    for k in ("GLMH", "LGBM", "XGB", "RF"):
        s = summ.set_index("model")
        orc[k] = {"H": float(s.loc[f"{k}_H", "val_MAE"]), "HE": float(s.loc[f"{k}_HE", "val_MAE"]), "HX_oracle": float(s.loc[f"{k}_HX", "val_MAE"])}
    decision["environment_effect_val_MAE"] = orc
    print(json.dumps(decision, indent=1), flush=True)
    print(summ[["model", "q", "tune_MAE", "tune_coverage80", "val_MAE", "val_ratio_vs_M4", "ci_vs_M4", "val_RMSE", "val_WAPE",
                "val_bias", "val_coverage80", "val_width80", "sec_per_fit"]].round(3).to_string(index=False), flush=True)

    # ------------------------------------------------------------------ previously-viewed Jan-Mar 2019: selected model only
    hold = None
    if adopt:
        kind, fs = decision["selected"].split("_")
        hres = Parallel(n_jobs=14)(delayed(run_cutoff)(rows, c, [(kind, fs)]) for c in HOLD_CUTOFFS)
        hfc = pd.DataFrame([r for part in hres for r in part], columns=["model", "cutoff", "h", "date", "y", "mean", "seconds"])
        both = pd.concat([allfc[allfc.model == decision["selected"]][hfc.columns], hfc])
        hfc_i = add_q_intervals(both, qchoice[decision["selected"]])
        hfc_i = hfc_i[hfc_i.cutoff.isin(HOLD_CUTOFFS)]
        hfc_i.to_csv(OUT / "forecasts_jan_mar_2019_selected.csv", index=False)
        saved = pd.read_csv(ROOT / "outputs" / "holdout" / "tables" / "forecasts_primary_grid.csv", parse_dates=["cutoff", "date"])
        m4h = B.metrics(saved.assign(model="M4").rename(columns={}))
        sh = B.metrics(hfc_i)
        hold = {"label": "Jan-Mar 2019, previously viewed (Step 5) - NOT an untouched holdout for this model",
                "selected": sh, "M4_saved": m4h, "ratio_vs_M4": sh["MAE"] / m4h["MAE"]}
        print(json.dumps(hold, indent=1, default=float), flush=True)
    (OUT / "decision.json").write_text(json.dumps({"decision": decision, "jan_mar_2019_previously_viewed": hold,
                                                   "runtime_s": round(time.time() - t0)}, indent=1, default=float))
    print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
