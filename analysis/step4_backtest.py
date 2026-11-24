"""Step 4: rolling-origin backtest of candidate 14-day forecasting models.

Run from the project root:
    PYTHONDONTWRITEBYTECODE=1 python analysis/step4_backtest.py

Design (all within DEV = 2017-04-01 .. 2018-12-31; Jan-Mar 2019 never touched):
* Cut-offs every 4 days (so every weekday is a cut-off day), 2017-07-01 .. 2018-12-15 (latest grid date whose window ends inside DEV).
* TUNING cut-offs  : 2017-07-01 .. 2018-06-14 (windows end by 2018-06-27) - choose configs.
* VALIDATION cut-offs: 2018-07-04 .. 2018-12-15 - compare the chosen configs once.
* Intervals: empirical horizon-bucket quantiles from each model's own completed past
  forecasts (src/backtest.add_intervals). Quantile level chosen on TUNING only.
Outputs: outputs/backtest/{tables,figures,backtest_report.md, forecasts.csv.gz}
"""
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src import backtest as B  # noqa: E402
from src import forecasting as F  # noqa: E402

OUT = os.path.join(ROOT, "outputs", "backtest")
TAB = os.path.join(OUT, "tables")
FIG = os.path.join(OUT, "figures")
TUNE_END = pd.Timestamp("2018-06-17")
VAL_START = pd.Timestamp("2018-07-01")
BASE = "B0_seasonal_naive"
COMPLEXITY = {"B0": 0, "B1": 1, "M1": 2, "M2": 3, "M3": 4, "M5": 5, "M4": 6, "M6": 7, "E": 8, "X": 9}
FAMILY = {"B0": "Seasonal naive (harness baseline)", "B1": "Recent rolling average",
          "M1": "Weekday-aware (recent level x weekday profile)", "M2": "Recent level + weekday + calendar",
          "M3": "Seasonal: year-ago shape x current level", "M4": "Trend/seasonal Poisson GLM + calendar (+level)",
          "M5": "Holt-Winters (log, weekly season, damped trend)",
          "M6": "Ensemble: equal-weight mean of the M2 and M4 winners (pre-specified, untuned)",
          "E": "Environmental GLM (deployable, lag >= 14)",
          "X": "Environmental GLM ORACLE (uses future observations - not deployable)"}


def registry():
    m = {"B0_seasonal_naive": F.seasonal_naive}
    for w in (7, 14, 28, 56):
        m[f"B1_mean_w{w}"] = lambda c, w=w: F.rolling_mean(c, w)
    for L in (7, 14, 28, 56, 84, 112):        # 84/112 added after 56 sat on the grid edge
        for W in (56, 91, 182, 364):
            m[f"M1_wk_L{L}_W{W}"] = lambda c, L=L, W=W: F.weekday_level(c, L, W)
            m[f"M2_wkcal_L{L}_W{W}"] = lambda c, L=L, W=W: F.weekday_level(c, L, W, calendar=True)
    for L in (14, 28, 56):
        m[f"M3_yearago_L{L}"] = lambda c, L=L: F.year_ago(c, L, 91)
    for N in (182, 364, None):
        for K in (0, 2):
            for T in (False, True):
                for Lw in (0, 14, 28):
                    m[f"M4_glm_N{N or 'all'}_K{K}_T{int(T)}_L{Lw}"] = (
                        lambda c, N=N, K=K, T=T, Lw=Lw: F.glm_calendar(c, N, K, T, Lw))
    for N in (182, 364, None):
        m[f"M5_hw_N{N or 'all'}"] = lambda c, N=N: F.holt_winters(c, N)
    return m


def env_registry(store, glm_name):
    p = glm_name.split("_")
    N = None if p[2] == "Nall" else int(p[2][1:])
    K, T, Lw = int(p[3][1:]), bool(int(p[4][1:])), int(p[5][1:])
    sets = {"co": ("co",), "pm25": ("pm25",), "temp": ("temp_mean_c",), "co+pm25+temp": ("co", "pm25", "temp_mean_c")}
    m, oracle = {}, store.full_env()
    for tag, cols in sets.items():
        m[f"E_{tag}_lag14"] = lambda c, cols=cols: F.glm_calendar(c, N, K, T, Lw, env_cols=cols, env_lag=14)
        m[f"E_{tag}_lag21"] = lambda c, cols=cols: F.glm_calendar(c, N, K, T, Lw, env_cols=cols, env_lag=21)
        m[f"X_{tag}_oracle_lag0"] = lambda c, cols=cols: F.glm_calendar(c, N, K, T, Lw, env_cols=cols, env_lag=0,
                                                                          oracle_env=oracle)
    m[glm_name] = registry()[glm_name]          # same base, re-run for a paired comparison
    return m


def split(df):
    return df[df.cutoff <= TUNE_END], df[df.cutoff >= VAL_START]


def season_of(c):
    return {12: "Winter (Dec-Feb)", 1: "Winter (Dec-Feb)", 2: "Winter (Dec-Feb)", 3: "Spring (Mar-May)",
            4: "Spring (Mar-May)", 5: "Spring (Mar-May)", 6: "Monsoon (Jun-Sep)", 7: "Monsoon (Jun-Sep)",
            8: "Monsoon (Jun-Sep)", 9: "Monsoon (Jun-Sep)", 10: "Post-monsoon (Oct-Nov)",
            11: "Post-monsoon (Oct-Nov)"}[pd.Timestamp(c).month]


def main():
    warnings.simplefilter("ignore")
    os.makedirs(TAB, exist_ok=True)
    t_all = time.perf_counter()
    store = B.DataStore(ROOT)
    cutoffs = B.make_cutoffs()
    tune_c = cutoffs[cutoffs <= TUNE_END]
    val_c = cutoffs[cutoffs >= VAL_START]

    # ---------------- stage 1: every non-environment configuration on every cut-off
    reg = registry()
    fc = B.run_backtest(store, reg, cutoffs)
    fc["family"] = fc["model"].str.split("_").str[0]
    tune, val = split(fc)
    tune_sum = B.summarise(tune).assign(family=lambda d: d.model.str.split("_").str[0])
    base_mae_t = tune_sum.set_index("model").loc[BASE, "MAE"]
    tune_sum["ratio_vs_B0"] = tune_sum["MAE"] / base_mae_t
    winners = tune_sum.sort_values("MAE").groupby("family").head(1).set_index("family")["model"]
    winners = winners.reindex([f for f in COMPLEXITY if f in winners.index])

    # ---------------- stage 1b: pre-specified ensemble of the M2 and M4 winners (no weights tuned)
    a = fc[fc.model == winners["M2"]].set_index(["cutoff", "h"])
    b = fc[fc.model == winners["M4"]].set_index(["cutoff", "h"])
    ens = a.copy()
    ens["mean"] = (a["mean"] + b["mean"]) / 2
    ens["seconds"] = a["seconds"] + b["seconds"]
    ens = ens.reset_index().assign(model="M6_ens_M2_M4", family="M6")
    fc = pd.concat([fc, ens[fc.columns]], ignore_index=True)
    tune, val = split(fc)
    m6 = B.summarise(tune[tune.model == "M6_ens_M2_M4"]).assign(family="M6")
    m6["ratio_vs_B0"] = m6["MAE"] / base_mae_t
    tune_sum = pd.concat([tune_sum, m6], ignore_index=True)
    winners["M6"] = "M6_ens_M2_M4"
    winners = winners.reindex([f for f in COMPLEXITY if f in winners.index])

    # ---------------- stage 2: environmental experiments on top of the tuned GLM
    env_reg = env_registry(store, winners["M4"])
    efc = B.run_backtest(store, env_reg, cutoffs, with_env=tuple(env_reg))
    efc["family"] = efc["model"].str.split("_").str[0]
    etune, eval_ = split(efc)
    env_tune = B.summarise(etune)
    env_winner = env_tune[env_tune.model.str.startswith("E_")].sort_values("MAE").iloc[0]["model"]
    winners["E"] = env_winner

    # ---------------- intervals: quantile level and pool size chosen on TUNING only
    sel = pd.concat([fc[fc.model.isin(winners.values)], efc[efc.model == env_winner]], ignore_index=True)
    calib = []
    for q in (0.10, 0.075, 0.05):
        for pool in (30, 45, 90):
            iv = B.add_intervals(sel, q, 1 - q, pool=pool)
            t_, v_ = split(iv)
            for name, g in t_.groupby("model"):
                mt = B.metrics(g)
                calib.append(dict(model=name, q_lo=q, q_hi=1 - q, pool=pool, tune_coverage=mt["coverage80"],
                                  tune_width=mt["width80"], tune_points=mt["interval_points"]))
    calib = pd.DataFrame(calib)

    def pick_level(g):  # narrowest setting whose TUNING coverage >= 0.80; else the highest coverage
        ok = g[g.tune_coverage >= 0.80]
        return (ok.sort_values("tune_width").iloc[0] if len(ok) else g.sort_values("tune_coverage").iloc[-1])
    chosen_iv = calib.groupby("model").apply(pick_level).reset_index(drop=True)
    parts = []
    for r in chosen_iv.itertuples():
        parts.append(B.add_intervals(sel[sel.model == r.model], r.q_lo, r.q_hi, pool=int(r.pool)))
    ivs = pd.concat(parts, ignore_index=True)
    # same-rule reference intervals for B0 at the fixed 10/90 level (for the width approximation)
    base_ref = B.add_intervals(fc[fc.model == BASE], 0.10, 0.90, pool=45).assign(model="B0_ref_interval_10_90")
    ivs_t, ivs_v = split(ivs)

    # ---------------- validation comparison of family winners
    pts = pd.concat([fc, efc[efc.model == env_winner]], ignore_index=True)   # point forecasts, one copy each
    val_rows = []
    for fam, name in winners.items():
        g = ivs_v[ivs_v.model == name]
        mt = B.metrics(g)
        pt, lo, hi = B.block_bootstrap_ratio(pts, name, BASE, val_c)
        mt.update(family=fam, model=name, description=FAMILY[fam], ratio_vs_B0=pt, ratio_lo=lo, ratio_hi=hi,
                  tune_ratio_vs_B0=float(tune_sum.set_index("model").reindex([name])["ratio_vs_B0"].iloc[0])
                  if name in set(tune_sum.model) else float(env_tune.set_index("model").loc[name, "MAE"] / base_mae_t))
        val_rows.append(mt)
    val_tab = pd.DataFrame(val_rows).set_index("family")

    # simplest model within noise of the best (validation)
    best = val_tab["MAE"].idxmin()
    within = []
    for fam in val_tab.index:
        pt, lo, hi = B.block_bootstrap_ratio(pts, val_tab.loc[fam, "model"], val_tab.loc[best, "model"], val_c)
        within.append(dict(family=fam, ratio_vs_best=pt, lo=lo, hi=hi, within_noise=lo <= 1.0))
    within = pd.DataFrame(within).set_index("family")
    eligible = [f for f in within.index if within.loc[f, "within_noise"] and f not in ("X",) and f != "B0"]
    recommended = min(eligible, key=lambda f: COMPLEXITY[f]) if eligible else best

    # ---------------- stability: season and level, all cut-offs (tuning numbers optimistic)
    allw = pd.concat([fc[fc.model.isin(winners.values)], efc[efc.model == env_winner]])
    allw = allw.assign(season=allw.cutoff.map(season_of), period=np.where(allw.cutoff <= TUNE_END, "tuning", "validation"))
    lvl = store.counts.rolling(28).mean().shift(1)
    allw["level"] = allw.cutoff.map(lvl)
    allw["level_band"] = pd.qcut(allw["level"], 3, labels=["low level", "mid level", "high level"])
    by_season = B.summarise(allw, by=("model", "season"))[["model", "season", "MAE", "WAPE", "n_cutoffs"]]
    by_level = B.summarise(allw, by=("model", "level_band"))[["model", "level_band", "MAE", "WAPE", "n_cutoffs"]]
    by_season["ratio_vs_B0"] = by_season.apply(lambda r: r.MAE / by_season[(by_season.model == BASE) & (by_season.season == r.season)].MAE.iloc[0], axis=1)
    by_level["ratio_vs_B0"] = by_level.apply(lambda r: r.MAE / by_level[(by_level.model == BASE) & (by_level.level_band == r.level_band)].MAE.iloc[0], axis=1)

    # closest analogue of the hidden Jan-Mar 2019 cut-offs: Jan-Mar 2018 (part of TUNING, so optimistic)
    jm = allw[(allw.cutoff >= "2018-01-01") & (allw.cutoff <= "2018-03-18")]
    jm_tab = B.summarise(jm)[["model", "MAE", "WAPE", "MAE_h1_7", "MAE_h8_14", "n_cutoffs"]]
    jm_tab["ratio_vs_B0"] = jm_tab["MAE"] / jm_tab.set_index("model").loc[BASE, "MAE"]
    jm_c = sorted(jm.cutoff.unique())
    jm_ci = []
    for fam in ("M1", "M2", "M4", "M6", "E"):
        for ref_fam in ("B0", "M1"):
            if fam == ref_fam:
                continue
            pt, lo, hi = B.block_bootstrap_ratio(pts, winners[fam], winners[ref_fam], jm_c)
            jm_ci.append(dict(model=fam, reference=ref_fam, ratio=pt, lo=lo, hi=hi))
    jm_ci = pd.DataFrame(jm_ci)

    hor = allw[allw.cutoff >= VAL_START].assign(ae=lambda d: (d["mean"] - d["y"]).abs()).groupby(["model", "h"])["ae"].mean().unstack(0)

    # ---------------- environment: paired comparison against the same GLM base
    env_cmp = []
    for name in env_reg:
        for per, part in (("tuning", etune), ("validation", eval_)):
            g = part[part.model == name]
            mt = B.metrics(g)
            env_cmp.append(dict(model=name, period=per, MAE=mt["MAE"], MAE_h1_7=mt["MAE_h1_7"], MAE_h8_14=mt["MAE_h8_14"]))
    env_cmp = pd.DataFrame(env_cmp)
    base_mae = env_cmp[env_cmp.model == winners["M4"]].set_index("period")["MAE"]
    env_cmp["ratio_vs_same_GLM"] = env_cmp.apply(lambda r: r.MAE / base_mae[r.period], axis=1)
    env_ci = []
    for name in env_reg:
        if name == winners["M4"]:
            continue
        pt, lo, hi = B.block_bootstrap_ratio(efc, name, winners["M4"], val_c)
        env_ci.append(dict(model=name, val_ratio_vs_GLM=pt, lo=lo, hi=hi))
    env_ci = pd.DataFrame(env_ci)

    # ---------------- interval diagnostics on validation
    iv_rows = []
    for name, g in ivs_v.groupby("model"):
        for bkt, gg in g.groupby(g["h"].map(B.bucket_of)):
            mt = B.metrics(gg)
            iv_rows.append(dict(model=name, horizon=f"h{B.BUCKETS[bkt][0]}-{B.BUCKETS[bkt][1]}", coverage80=mt["coverage80"], width80=mt["width80"]))
    iv_bucket = pd.DataFrame(iv_rows)
    base_w_val = B.metrics(base_ref[base_ref.cutoff >= VAL_START])["width80"]

    # ---------------- simulated harness score on validation cut-offs
    score_rows = []
    sim_store = {}
    ref = base_ref.assign(model=BASE)
    ref_v = ref[ref.cutoff >= VAL_START]
    for fam, name in winners.items():
        if name == BASE:
            s = B.simulate_scores(ref_v, BASE, BASE)   # baseline scored against itself (ratio = 1)
        else:
            s = B.simulate_scores(pd.concat([ivs_v[ivs_v.model == name], ref_v]), name, BASE)
        sim_store[fam] = s
        score_rows.append(dict(family=fam, model=name,
                               mean_ratio_pooled=s.ratio_pooled.mean(), mean_ratio_per_cutoff=s.ratio_per_cutoff.mean(),
                               err_pts_mean=s.err_pts_pooled.mean(), err_pts_p10=s.err_pts_pooled.quantile(.1),
                               err_pts_p90=s.err_pts_pooled.quantile(.9), cov_pts_mean=s.cov_pts.mean(),
                               p_cov_below_80=(s.coverage < 0.8).mean(), width_factor_mean=s.width_factor.mean(),
                               total_mean=s.total_pooled.mean(), total_p10=s.total_pooled.quantile(.1),
                               total_p90=s.total_pooled.quantile(.9)))
    scores = pd.DataFrame(score_rows).set_index("family")

    # ---------------- runtime
    rt = pd.concat([fc, efc]).groupby("model")["seconds"].agg(["mean", "max"]).loc[list(winners.values)]
    rt["family"] = [k for k in winners.index]
    pool_used = chosen_iv.set_index("model")["pool"]
    rt["internal_backtest_seconds_est"] = rt["mean"] * rt.index.map(lambda m: pool_used.get(m, 45))
    results = dict(store=store, cutoffs=cutoffs, tune_c=tune_c, val_c=val_c, fc=fc, efc=efc, tune_sum=tune_sum,
                   winners=winners, env_tune=env_tune, calib=calib, chosen_iv=chosen_iv, ivs=ivs, base_ref=base_ref,
                   val_tab=val_tab, within=within, best=best, recommended=recommended, by_season=by_season,
                   by_level=by_level, jm_tab=jm_tab, jm_ci=jm_ci, hor=hor, env_cmp=env_cmp, env_ci=env_ci, iv_bucket=iv_bucket,
                   base_w_val=base_w_val, scores=scores, sims=sim_store, runtime=rt,
                   elapsed=time.perf_counter() - t_all)
    return results


def save(R):
    T = lambda n: os.path.join(TAB, n)  # noqa: E731
    R["tune_sum"].sort_values("MAE").to_csv(T("tuning_all_configs.csv"), index=False)
    R["env_tune"].to_csv(T("tuning_environment_configs.csv"), index=False)
    R["val_tab"].to_csv(T("validation_family_winners.csv"))
    R["within"].to_csv(T("validation_vs_best.csv"))
    R["calib"].to_csv(T("interval_calibration_tuning.csv"), index=False)
    R["chosen_iv"].to_csv(T("interval_settings_chosen.csv"), index=False)
    R["iv_bucket"].to_csv(T("interval_by_horizon_validation.csv"), index=False)
    R["by_season"].to_csv(T("stability_by_season.csv"), index=False)
    R["by_level"].to_csv(T("stability_by_level.csv"), index=False)
    R["jm_tab"].to_csv(T("jan_mar_2018_analogue.csv"), index=False)
    R["jm_ci"].to_csv(T("jan_mar_2018_bootstrap.csv"), index=False)
    R["hor"].to_csv(T("mae_by_horizon_validation.csv"))
    R["env_cmp"].to_csv(T("environment_vs_same_glm.csv"), index=False)
    R["env_ci"].to_csv(T("environment_bootstrap_validation.csv"), index=False)
    R["scores"].to_csv(T("simulated_harness_score_validation.csv"))
    R["runtime"].to_csv(T("runtime_seconds.csv"))
    pd.Series(R["winners"]).to_csv(T("family_winners.csv"))
    keep = pd.concat([R["ivs"], R["base_ref"]], ignore_index=True)
    keep.to_csv(os.path.join(OUT, "forecasts_selected_models.csv.gz"), index=False)
    return sorted(os.listdir(TAB))


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    R = main()
    tables = save(R)
    import step4_report as REP  # noqa: E402
    figures = REP.figures(R, FIG)
    REP.write(R, tables, figures, os.path.join(OUT, "backtest_report.md"))
    print(f"done in {R['elapsed']:.0f}s; tables {len(tables)}, figures {len(figures)}")
    print(R["val_tab"][["model", "MAE", "ratio_vs_B0", "ratio_lo", "ratio_hi", "coverage80", "width80"]].round(3).to_string())
    print("best", R["best"], "recommended", R["recommended"])
