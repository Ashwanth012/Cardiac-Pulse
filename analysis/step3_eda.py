"""Step 3: exploratory data analysis.

Run from the project root:
    PYTHONDONTWRITEBYTECODE=1 python analysis/step3_eda.py

Inputs : outputs/clean_daily.csv, outputs/environment_daily_extended.csv (read-only)
Outputs: outputs/eda/  (tables/*.csv, figures/*.png, eda_report.md)

Inference uses DEV = 2017-04-01..2018-12-31 only. Jan-Mar 2019 appears only in
descriptive plots, shaded and labelled as the holdout.
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src import eda as E  # noqa: E402

OUT = os.path.join(ROOT, "outputs", "eda")
TAB = os.path.join(OUT, "tables")
FIG = os.path.join(OUT, "figures")

KEY = ["temp_mean_c", "temp_min_c", "rel_humidity_mean_pct", "aqi", "pm25", "pm10", "no2"]
LABEL = {"temp_mean_c": "Mean temperature (C)", "temp_min_c": "Min temperature (C)",
         "temp_max_c": "Max temperature (C)", "rel_humidity_mean_pct": "Relative humidity (%)",
         "precip_mm": "Precipitation (mm)", "wind_mean_kmh": "Mean wind (km/h)", "aqi": "AQI",
         "pm25": "PM2.5 (index)", "pm10": "PM10 (index)", "no2": "NO2 (index)", "so2": "SO2 (index)",
         "co": "CO (index)", "ozone": "Ozone (index)", "nh3": "NH3 (index)"}
SEASON = {12: "Winter (Dec-Feb)", 1: "Winter (Dec-Feb)", 2: "Winter (Dec-Feb)",
          3: "Summer (Mar-May)", 4: "Summer (Mar-May)", 5: "Summer (Mar-May)",
          6: "Monsoon (Jun-Sep)", 7: "Monsoon (Jun-Sep)", 8: "Monsoon (Jun-Sep)", 9: "Monsoon (Jun-Sep)",
          10: "Post-monsoon (Oct-Nov)", 11: "Post-monsoon (Oct-Nov)"}
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
# Influence window: the 2017 smog season, which coincides with an unexplained surge in OPD-coded
# admissions (OPD share 46-51% in Nov-Dec 2017 vs 23-37% in every other DEV month).
ANOMALY = (pd.Timestamp("2017-10-15"), pd.Timestamp("2018-01-15"))


def mean_ci(x):
    x = pd.Series(x).dropna()
    m, se = x.mean(), x.std(ddof=1) / np.sqrt(len(x))
    t = stats.t.ppf(0.975, len(x) - 1)
    return m, m - t * se, m + t * se, len(x)


# ============================================================== 1. patterns
def patterns(daily, frame):
    y = daily["admissions"]
    dev = daily.loc[E.DEV_START:E.DEV_END]
    res = {}
    m = daily[["admissions", "admissions_emergency", "admissions_opd",
               "admissions_dedup_exact", "admissions_dedup_mrd_doa"]].resample("MS").mean()
    m["period"] = np.where(m.index >= E.HOLDOUT_START, "holdout (descriptive only)", "DEV")
    m.index = m.index.strftime("%Y-%m")
    res["monthly"] = m.round(2)

    s = dev.groupby(dev.index.month.map(SEASON))["admissions"].apply(lambda v: pd.Series(mean_ci(v), index=["mean", "lo", "hi", "days"])).unstack()
    res["season"] = s.loc[["Winter (Dec-Feb)", "Summer (Mar-May)", "Monsoon (Jun-Sep)", "Post-monsoon (Oct-Nov)"]].round(2)
    kw = stats.kruskal(*[g.values for _, g in dev.groupby(dev.index.month.map(SEASON))["admissions"]])
    res["season_kruskal_p"] = kw.pvalue

    # same months, two years (Apr-Dec): raw vs de-duplicated
    rows = []
    for col in ["admissions", "admissions_dedup_exact", "admissions_dedup_mrd_doa", "admissions_emergency", "admissions_opd"]:
        a = daily.loc["2017-04-01":"2017-12-31", col].mean()
        b = daily.loc["2018-04-01":"2018-12-31", col].mean()
        rows.append(dict(series=col, apr_dec_2017=a, apr_dec_2018=b, change_pct=100 * (b / a - 1)))
    res["yoy"] = pd.DataFrame(rows).set_index("series").round(2)

    # trend (per year) adjusted for weekday, holidays and annual Fourier terms
    tr = []
    f = frame.copy()
    f["s1"], f["c1"] = np.sin(2 * np.pi * f["t"] / 365.25), np.cos(2 * np.pi * f["t"] / 365.25)
    f["s2"], f["c2"] = np.sin(4 * np.pi * f["t"] / 365.25), np.cos(4 * np.pi * f["t"] / 365.25)
    keep = ~f.index.to_series().between(ANOMALY[0], ANOMALY[1]).to_numpy()
    for col in ["admissions", "admissions_dedup_exact", "admissions_dedup_mrd_doa", "admissions_emergency", "admissions_opd"]:
        r = E.fit_poisson(f"{col} ~ C(weekday) + holiday + s1 + c1 + s2 + c2 + t", f)
        tr.append(dict(series=col, window="full DEV", **E.rr_per_unit(r, "t", 365.25)))
        r = E.fit_poisson(f"{col} ~ C(weekday) + holiday + s1 + c1 + s2 + c2 + t", f[keep])
        tr.append(dict(series=col, window="excl. 2017-10-15..2018-01-15", **E.rr_per_unit(r, "t", 365.25)))
    t = pd.DataFrame(tr).set_index(["series", "window"])
    t[["pct_per_year", "lo_pct", "hi_pct"]] = 100 * (t[["rr", "lo", "hi"]] - 1)
    res["trend"] = t.round(4)

    base = E.base_formula(season_df=E.spline_df(frame))
    r = E.fit_poisson(base, frame)
    res["base_residual"] = frame["admissions"] / r.fittedvalues
    res["acf"] = pd.DataFrame({"raw_counts": E.acf(dev["admissions"]),
                               "pearson_resid_base_model": E.acf((frame["admissions"] - r.fittedvalues) / np.sqrt(r.fittedvalues))}).round(4)
    res["dispersion_base"] = E.dispersion(base, frame)
    res["base_df"] = E.spline_df(frame)
    return res


# =========================================================== 2/3. weekday & type
def weekday_type(daily, frame):
    dev = daily.loc[E.DEV_START:E.DEV_END]
    rows = []
    for col, name in [("admissions", "total"), ("admissions_emergency", "emergency"), ("admissions_opd", "opd")]:
        for d in range(7):
            m, lo, hi, n = mean_ci(dev.loc[dev.index.dayofweek == d, col])
            rows.append(dict(series=name, weekday=DAYS[d], mean=m, lo=lo, hi=hi, days=n))
    wk = pd.DataFrame(rows)

    # adjusted rate ratios vs Monday (season spline + holiday)
    df_ = E.spline_df(frame)
    rr = []
    for col, name in [("admissions", "total"), ("admissions_emergency", "emergency"), ("admissions_opd", "opd")]:
        r = E.fit_poisson(f"{col} ~ C(weekday) + holiday + cr(t, df={df_}, constraints='center')", frame)
        for d in range(1, 7):
            rr.append(dict(series=name, weekday=DAYS[d], **E.rr_per_unit(r, f"C(weekday)[T.{d}]", 1)))
        frame2 = frame.assign(weekend=(frame["weekday"] >= 5).astype(int))
        r2 = E.fit_poisson(f"{col} ~ weekend + holiday + cr(t, df={df_}, constraints='center')", frame2)
        rr.append(dict(series=name, weekday="Weekend vs weekday", **E.rr_per_unit(r2, "weekend", 1)))
    kw = {name: stats.kruskal(*[g.values for _, g in dev.groupby(dev.index.dayofweek)[col]]).pvalue
          for col, name in [("admissions", "total"), ("admissions_emergency", "emergency"), ("admissions_opd", "opd")]}

    share = dev[["admissions_emergency", "admissions_opd"]].resample("MS").sum()
    share["opd_share_pct"] = 100 * share["admissions_opd"] / share.sum(axis=1)
    season_ratio = {}
    for col in ["admissions_emergency", "admissions_opd"]:
        g = dev.groupby(dev.index.month.map(SEASON))[col].mean()
        season_ratio[col] = g["Winter (Dec-Feb)"] / g["Monsoon (Jun-Sep)"]
    return dict(weekday=wk.round(3), weekday_rr=pd.DataFrame(rr).round(4), weekday_kruskal=kw,
                opd_share=share.round(2), winter_vs_monsoon=season_ratio)


# =============================================== 4/5. environment & lags
def environment(frame):
    df_ = E.spline_df(frame)
    base = E.base_formula(season_df=df_)
    # crude correlations (confounded by season/weekday)
    corr = []
    for x in E.EXPOSURES:
        c = f"{x}__lag0"
        sp = stats.spearmanr(frame["admissions"], frame[c], nan_policy="omit")
        corr.append(dict(exposure=x, spearman_rho=sp.statistic, p=sp.pvalue))
    corr = pd.DataFrame(corr).set_index("exposure")

    # adjusted lag-specific and moving-average associations
    rows = []
    for x in E.EXPOSURES:
        unit = E.iqr(frame[f"{x}__lag0"])
        variants = [f"lag{k}" for k in E.LAGS] + [f"ma0_{k}" for k in E.MOVING]
        for v in variants:
            r = E.association(frame, f"{x}__{v}", base, unit=unit)
            rows.append(dict(exposure=x, window=v, **r))
    lag = pd.DataFrame(rows)
    lag["q_bh"] = E.bh_qvalues(lag["p"])
    lag["pct_per_iqr"] = 100 * (lag["rr"] - 1)
    return dict(crude=corr.round(4), lags=lag.round(5), base=base, df=df_)


def temperature_bins(frame):
    """Non-linear check: quintiles of 7-day mean temperature, season-adjusted."""
    df_ = E.spline_df(frame)
    f = frame.dropna(subset=["temp_mean_c__ma0_7"]).copy()
    f["tq"] = pd.qcut(f["temp_mean_c__ma0_7"], 5, labels=["Q1 coldest", "Q2", "Q3", "Q4", "Q5 hottest"])
    r = E.fit_poisson(f"admissions ~ C(weekday) + holiday + cr(t, df={df_}, constraints='center') + C(tq, Treatment('Q3'))", f)
    rows = []
    edges = f.groupby("tq", observed=True)["temp_mean_c__ma0_7"].agg(["min", "max"])
    for q in ["Q1 coldest", "Q2", "Q3", "Q4", "Q5 hottest"]:
        if q == "Q3":
            rows.append(dict(bin=q, rr=1.0, lo=1.0, hi=1.0, p=np.nan))
        else:
            rows.append(dict(bin=q, **E.rr_per_unit(r, f"C(tq, Treatment('Q3'))[T.{q}]", 1)))
    out = pd.DataFrame(rows).set_index("bin").join(edges.rename(columns={"min": "temp_from", "max": "temp_to"}))
    return out.round(4)


# ================================================ 6. festivals and holidays
def festivals(daily, frame):
    df_ = E.spline_df(frame)
    specs = {
        "any holiday (day of)": "holiday",
        "major festival (day of, extra to any holiday)": "major",
        "day before a holiday": "holiday_eve",
        "day after a holiday": "holiday_after",
    }
    # one joint model: all four calendar terms adjusted for each other
    r = E.fit_poisson(f"admissions ~ C(weekday) + cr(t, df={df_}, constraints='center') + holiday + major + holiday_eve + holiday_after", frame)
    joint = pd.DataFrame([dict(effect=label, **E.rr_per_unit(r, term, 1), days=int(frame[term].sum()))
                          for label, term in specs.items()]).set_index("effect")
    for col, name in [("admissions_emergency", "emergency"), ("admissions_opd", "opd")]:
        r = E.fit_poisson(f"{col} ~ C(weekday) + cr(t, df={df_}, constraints='center') + holiday + major + holiday_eve + holiday_after", frame)
        joint[f"{name}_rr"] = [float(np.exp(r.params[t])) for t in specs.values()]
        joint[f"{name}_p"] = [float(r.pvalues[t]) for t in specs.values()]

    # event study around major festivals: observed / expected (no-holiday model)
    r = E.fit_poisson(f"admissions ~ C(weekday) + cr(t, df={df_}, constraints='center')", frame)
    ratio = frame["admissions"] / r.fittedvalues
    fest = daily.loc[E.DEV_START:E.DEV_END]
    dates = list(fest.index[fest["holiday_major_festival"]])
    events = []
    for d in dates:
        if not events or (d - events[-1]).days > 3:
            events.append(d)
    rel = []
    for ev in events:
        for k in range(-7, 15):
            day = ev + pd.Timedelta(days=k)
            if day in ratio.index:
                rel.append(dict(event=ev.date().isoformat(), name=fest.at[ev, "holiday_names"], rel_day=k, ratio=ratio[day]))
    rel = pd.DataFrame(rel)
    es = rel.groupby("rel_day")["ratio"].apply(lambda v: pd.Series(mean_ci(v), index=["mean", "lo", "hi", "n"])).unstack()

    # Diwali specifically (smog overlaps)
    diwali = rel[rel["name"].str.contains("Diwali")].pivot(index="rel_day", columns="event", values="ratio")
    return dict(joint=joint.round(4), event_study=es.round(4), events=pd.Series(events), diwali=diwali.round(3))


# ================================================== 7. overlapping effects
def overlap(frame):
    df_ = E.spline_df(frame)
    cols = ["temp_mean_c__ma0_7", "rel_humidity_mean_pct__ma0_7", "pm25__ma0_7", "pm10__ma0_7", "no2__ma0_7",
            "co__ma0_7", "aqi__ma0_7"]
    f = frame.dropna(subset=cols).copy()
    f["month_sin"] = np.sin(2 * np.pi * f.index.dayofyear / 365.25)
    f["month_cos"] = np.cos(2 * np.pi * f.index.dayofyear / 365.25)
    corr = f[cols + ["major", "holiday", "month_cos"]].corr(method="spearman").round(3)

    v = E.vif(f, ["temp_mean_c__ma0_7", "rel_humidity_mean_pct__ma0_7", "pm25__ma0_7", "no2__ma0_7", "month_cos", "month_sin"]).round(2)

    # confounding ladder: how the RR changes as adjustments are added
    ladder = {
        "M0 crude": "admissions ~ 1",
        "M1 + weekday + holidays": "admissions ~ C(weekday) + holiday",
        f"M2 + season/trend spline (df={df_})": f"admissions ~ C(weekday) + holiday + cr(t, df={df_}, constraints='center')",
    }
    co_adjust = {
        "temp_mean_c__ma0_7": "Q('pm25__ma0_7') + Q('rel_humidity_mean_pct__ma0_7')",
        "rel_humidity_mean_pct__ma0_7": "Q('temp_mean_c__ma0_7') + Q('pm25__ma0_7')",
        "pm25__ma0_7": "Q('temp_mean_c__ma0_7') + Q('rel_humidity_mean_pct__ma0_7')",
        "pm10__ma0_7": "Q('temp_mean_c__ma0_7') + Q('rel_humidity_mean_pct__ma0_7')",
        "no2__ma0_7": "Q('temp_mean_c__ma0_7') + Q('rel_humidity_mean_pct__ma0_7')",
        "co__ma0_7": "Q('temp_mean_c__ma0_7') + Q('rel_humidity_mean_pct__ma0_7')",
        "aqi__ma0_7": "Q('temp_mean_c__ma0_7') + Q('rel_humidity_mean_pct__ma0_7')",
    }
    # two-pollutant check: PM2.5 and CO together
    two = []
    for x, other in [("pm25__ma0_7", "co__ma0_7"), ("co__ma0_7", "pm25__ma0_7")]:
        two.append(dict(exposure=x, adjusted_for=other, **E.association(
            f, x, f"admissions ~ C(weekday) + holiday + cr(t, df={df_}, constraints='center')",
            f"Q('{other}') + Q('temp_mean_c__ma0_7')", unit=E.iqr(f[x]))))
    rows = []
    for x, extra in co_adjust.items():
        unit = E.iqr(f[x])
        for name, base in ladder.items():
            rows.append(dict(exposure=x, model=name, **E.association(f, x, base, unit=unit)))
        rows.append(dict(exposure=x, model="M3 + other environment", **E.association(f, x, list(ladder.values())[-1], extra, unit=unit)))
    lad = pd.DataFrame(rows)

    # sensitivity to seasonal spline flexibility
    sens = []
    for x in ["temp_mean_c__ma0_7", "pm25__ma0_7", "co__ma0_7", "rel_humidity_mean_pct__ma0_7"]:
        unit = E.iqr(f[x])
        for per_year in [2, 3, 4, 6, 8, 12]:
            d = E.spline_df(f, per_year)
            sens.append(dict(exposure=x, df_per_year=per_year, total_df=d,
                             **E.association(f, x, f"admissions ~ C(weekday) + holiday + cr(t, df={d}, constraints='center')", unit=unit)))
    # Emergency vs OPD (OPD = partial negative control) and de-duplicated target
    sub = []
    for target in ["admissions", "admissions_emergency", "admissions_opd", "admissions_dedup_mrd_doa"]:
        base = E.base_formula(target, season_df=df_)
        for x in ["temp_mean_c__ma0_7", "pm25__ma0_7", "co__ma0_7", "rel_humidity_mean_pct__ma0_7", "no2__ma0_7"]:
            sub.append(dict(target=target, exposure=x, **E.association(f, x, base, unit=E.iqr(f[x]))))
    # influence: drop the 2017 smog / OPD-coding-anomaly window and refit
    keep = ~f.index.to_series().between(ANOMALY[0], ANOMALY[1]).to_numpy()
    infl = []
    for target in ["admissions", "admissions_emergency", "admissions_opd"]:
        base = E.base_formula(target, season_df=df_)
        for x in ["pm25__ma0_7", "co__ma0_7", "temp_mean_c__ma0_7"]:
            unit = E.iqr(f[x])
            infl.append(dict(target=target, exposure=x, window="full DEV", **E.association(f, x, base, unit=unit)))
            infl.append(dict(target=target, exposure=x, window="excl. 2017-10-15..2018-01-15",
                             **E.association(f[keep], x, base, unit=unit)))
    return dict(corr=corr, vif=v, ladder=lad.round(5), spline_sensitivity=pd.DataFrame(sens).round(5),
                by_target=pd.DataFrame(sub).round(5), two_pollutant=pd.DataFrame(two).round(5),
                influence=pd.DataFrame(infl).round(5))


# ======================================================= 8. duplicates & trend
def duplicates(daily):
    dev = daily.loc[E.DEV_START:E.DEV_END]
    m = dev.resample("MS").sum()
    m["dup_extra_pct"] = 100 * (m["admissions"] - m["admissions_dedup_mrd_doa"]) / m["admissions"]
    m["exact_extra_pct"] = 100 * m["extra_exact_copies"] / m["admissions"]
    env_m = dev[["pm25", "temp_mean_c"]].resample("MS").mean()
    rho_pm = stats.spearmanr(m["dup_extra_pct"], env_m["pm25"])
    rho_t = stats.spearmanr(m["dup_extra_pct"], env_m["temp_mean_c"])
    # daily: is duplication associated with weekday?
    wk = dev.groupby(dev.index.dayofweek).apply(lambda g: 100 * (g["admissions"] - g["admissions_dedup_mrd_doa"]).sum() / g["admissions"].sum())
    wk.index = DAYS
    return dict(monthly=m[["admissions", "admissions_dedup_mrd_doa", "dup_extra_pct", "exact_extra_pct"]].round(2),
                rho_pm25=(rho_pm.statistic, rho_pm.pvalue), rho_temp=(rho_t.statistic, rho_t.pvalue),
                weekday_dup_pct=wk.round(2))


def run():
    daily, env = E.load_inputs(os.path.join(ROOT, "outputs"))
    frame = E.model_frame(daily, env)
    R = dict(daily=daily, env=env, frame=frame)
    R["patterns"] = patterns(daily, frame)
    R["weekday"] = weekday_type(daily, frame)
    R["env"] = environment(frame)
    R["tbins"] = temperature_bins(frame)
    R["fest"] = festivals(daily, frame)
    R["overlap"] = overlap(frame)
    R["dups"] = duplicates(daily)
    return R


def save_tables(R):
    os.makedirs(TAB, exist_ok=True)
    p, w, e, f, o, d = R["patterns"], R["weekday"], R["env"], R["fest"], R["overlap"], R["dups"]
    tables = {
        "monthly_means.csv": p["monthly"], "season_means.csv": p["season"], "yoy_apr_dec.csv": p["yoy"],
        "trend_per_year.csv": p["trend"], "autocorrelation.csv": p["acf"],
        "weekday_means.csv": w["weekday"], "weekday_rate_ratios.csv": w["weekday_rr"],
        "opd_share_monthly.csv": w["opd_share"], "crude_correlations.csv": e["crude"],
        "lag_associations.csv": e["lags"], "temperature_quintiles.csv": R["tbins"],
        "festival_effects.csv": f["joint"], "festival_event_study.csv": f["event_study"],
        "diwali_windows.csv": f["diwali"], "exposure_correlation.csv": o["corr"], "vif.csv": o["vif"].rename("vif"),
        "confounding_ladder.csv": o["ladder"], "season_adjustment_sensitivity.csv": o["spline_sensitivity"],
        "associations_by_target.csv": o["by_target"], "two_pollutant.csv": o["two_pollutant"],
        "influence_2017_smog_window.csv": o["influence"],
        "duplicates_monthly.csv": d["monthly"], "duplicates_by_weekday.csv": d["weekday_dup_pct"].rename("repeat_pct"),
    }
    for name, t in tables.items():
        t.to_csv(os.path.join(TAB, name))
    return sorted(tables)


def _md(df, fmt="{:.3f}"):
    df = df.copy()
    head = "| " + " | ".join([str(df.index.name or "")] + [str(c) for c in df.columns]) + " |"
    lines = [head, "|" + "---|" * (len(df.columns) + 1)]
    for i, row in df.iterrows():
        cells = [fmt.format(v) if isinstance(v, (float, np.floating)) and not pd.isna(v) else ("" if pd.isna(v) else str(v)) for v in row]
        lines.append("| " + " | ".join([str(i)] + cells) + " |")
    return "\n".join(lines)


def _p(p):
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


def _rr(row, unit_label=None):
    return f"{100 * (row['rr'] - 1):+.1f}% (95% CI {100 * (row['lo'] - 1):+.1f} to {100 * (row['hi'] - 1):+.1f}; {_p(row['p'])})"


def build_report(R, tables, figures):
    p, w, e, f, o, d = R["patterns"], R["weekday"], R["env"], R["fest"], R["overlap"], R["dups"]
    lags = e["lags"]
    get = lambda x, win: lags[(lags.exposure == x) & (lags.window == win)].iloc[0]  # noqa: E731
    lad = o["ladder"].set_index(["exposure", "model"])
    m2 = [m for m in o["ladder"]["model"].unique() if m.startswith("M2")][0]
    bt = o["by_target"].set_index(["target", "exposure"])
    sens = o["spline_sensitivity"]
    two = o["two_pollutant"].set_index("exposure")
    tr = p["trend"].xs("full DEV", level="window")
    trx = p["trend"].xs("excl. 2017-10-15..2018-01-15", level="window")
    inf = o["influence"].set_index(["target", "exposure", "window"])
    FULL, EXCL = "full DEV", "excl. 2017-10-15..2018-01-15"
    ipct = lambda t, x, wdw: f"{100 * (inf.loc[(t, x, wdw), 'rr'] - 1):+.1f}% ({_p(inf.loc[(t, x, wdw), 'p'])})"  # noqa: E731
    dev = R["daily"].loc[E.DEV_START:E.DEV_END]
    wrr = w["weekday_rr"].set_index(["series", "weekday"])
    sig = lags[lags["q_bh"] < 0.05].sort_values("p")
    nominal = lags[lags["p"] < 0.05]
    lagtab = lags.assign(**{"% per IQR": 100 * (lags.rr - 1), "CI low %": 100 * (lags.lo - 1), "CI high %": 100 * (lags.hi - 1)})
    lagtab = lagtab[["exposure", "window", "unit", "% per IQR", "CI low %", "CI high %", "p", "q_bh"]]
    pm_sens = sens[sens.exposure == "pm25__ma0_7"].set_index("df_per_year")
    t_sens = sens[sens.exposure == "temp_mean_c__ma0_7"].set_index("df_per_year")
    co_sens = sens[sens.exposure == "co__ma0_7"].set_index("df_per_year")
    fj = f["joint"]
    es = f["event_study"]
    dw = f["diwali"]
    mon = d["monthly"]

    lad_tab = o["ladder"].assign(pct=100 * (o["ladder"].rr - 1), lo_pct=100 * (o["ladder"].lo - 1),
                                 hi_pct=100 * (o["ladder"].hi - 1)).pivot(index="exposure", columns="model", values="pct")
    bt_tab = o["by_target"].assign(pct=100 * (o["by_target"].rr - 1)).pivot(index="exposure", columns="target", values="pct")
    bt_p = o["by_target"].pivot(index="exposure", columns="target", values="p")

    txt = f"""# PS-06 exploratory data analysis (Build step 3)

Generated by `analysis/step3_eda.py` (figures: `analysis/step3_figures.py`, helpers: `src/eda.py`).
Inputs are the Step 2 outputs `clean_daily.csv` and `environment_daily_extended.csv`, read only.

## 0. How to read this report

* **Target**: raw daily admission rows (duplicates kept), as agreed.
* **Inference period (DEV)**: 2017-04-01 .. 2018-12-31 ({len(R['frame'])} days). **Jan-Mar 2019 is never used** for any
  statistic below; it appears only in descriptive time-series figures, shaded grey.
* **Associations are not causal effects.** This is one hospital, observational daily data, with exposures measured at
  a city monitor. "Associated with" means "co-varies with after the stated adjustments", nothing more.
* **Adjustment model** (unless stated): Poisson log-link GLM with weekday dummies, a holiday indicator and a natural
  cubic regression spline of time with {p['base_df']} df (4 per year, centred so the design matrix is full rank) to absorb
  season and trend. Standard errors are Newey-West
  (HAC, 7 lags), robust to over-dispersion (Pearson dispersion of the base model = {p['dispersion_base']:.2f}) and
  autocorrelation. Effects are % change in admissions per inter-quartile range (IQR) of the exposure.
* **Multiple testing**: {len(lags)} exposure x delay tests; Benjamini-Hochberg q-values are reported. With this many tests,
  about {0.05 * len(lags):.0f} nominal p < 0.05 results are expected by chance alone; {len(nominal)} were observed.
* **Pollution units**: the pollutant columns appear to be AQI sub-indices (Step 2 report), so IQRs are in index points.

## 1. Strongest findings (summary)

1. **Weekday is the largest, most certain pattern.** Season-adjusted, Sunday admissions are
   {100 * (1 - wrr.loc[('total', 'Sun'), 'rr']):.0f}% below Monday (RR {wrr.loc[('total', 'Sun'), 'rr']:.2f}, 95% CI
   {wrr.loc[('total', 'Sun'), 'lo']:.2f}-{wrr.loc[('total', 'Sun'), 'hi']:.2f}). The effect is driven by planned OPD admissions
   (Sunday RR {wrr.loc[('opd', 'Sun'), 'rr']:.2f}) far more than emergencies (Sunday RR {wrr.loc[('emergency', 'Sun'), 'rr']:.2f}).
2. **Strong winter seasonality** (Kruskal-Wallis across seasons p = {p['season_kruskal_p']:.1e}): winter (Dec-Feb) averages
   {p['season'].loc['Winter (Dec-Feb)', 'mean']:.1f}/day vs {p['season'].loc['Summer (Mar-May)', 'mean']:.1f}/day in summer (Mar-May).
3. **Temperature cannot be separated from season with two years of data.** Crude: a 1-IQR warmer week goes with
   {100 * (lad.loc[('temp_mean_c__ma0_7', 'M0 crude'), 'rr'] - 1):+.1f}% admissions (colder = busier). With the primary season spline the sign
   flips to {_rr(lad.loc[('temp_mean_c__ma0_7', m2)])}, and across spline flexibilities it ranges from
   {100 * (t_sens['rr'].min() - 1):+.0f}% to {100 * (t_sens['rr'].max() - 1):+.0f}%, significant only at the primary setting. The
   same adjusted model gives planned OPD admissions {ipct('admissions_opd', 'temp_mean_c__ma0_7', FULL)}, which temperature
   cannot plausibly cause. Temperature and "winter-ness" correlate at rho = {o['corr'].loc['temp_mean_c__ma0_7', 'month_cos']:.2f}
   (VIF {o['vif']['temp_mean_c__ma0_7']:.0f}). Conclusion: the adjusted temperature estimates reflect how the spline carves up the
   seasonal curve, not a temperature effect; "cold" and "winter" are not separable here. No heat signal either: the hottest
   season-adjusted quintile is {100 * (R['tbins'].loc['Q5 hottest', 'rr'] - 1):+.0f}% (p = {R['tbins'].loc['Q5 hottest', 'p']:.2f}).
4. **No reliable evidence of a short-term pollution effect.**
   * **CO** gives the only pollutant association that survives FDR correction
     ({', '.join(f"{r.window} {100 * (r.rr - 1):+.1f}% per same-day IQR, q = {r.q_bh:.3f}" for r in sig[sig.exposure == 'co'].itertuples())};
     {_rr(lad.loc[('co__ma0_7', m2)])} per IQR of the 7-day mean) and is stable across
     spline flexibilities ({100 * (co_sens['rr'].min() - 1):+.1f}% to {100 * (co_sens['rr'].max() - 1):+.1f}%). **But it is strongest in planned OPD
     admissions** ({ipct('admissions_opd', 'co__ma0_7', FULL)}) and not significant in emergencies
     ({ipct('admissions_emergency', 'co__ma0_7', FULL)}). Air pollution cannot bring forward a scheduled admission within a week,
     so this pattern points to confounding - CO is probably tracking something that also drives planned activity
     (e.g. traffic and working-day patterns) - rather than a cardiac effect.
   * **PM2.5** (7-day mean): {_rr(lad.loc[('pm25__ma0_7', m2)])} per IQR of the 7-day mean
     ({lad.loc[('pm25__ma0_7', m2), 'unit']:.0f} index points); BH q = {get('pm25', 'ma0_7')['q_bh']:.2f} in the delay screen.
     It is in emergencies ({ipct('admissions_emergency', 'pm25__ma0_7', FULL)}) rather than OPD
     ({ipct('admissions_opd', 'pm25__ma0_7', FULL)}), which is the direction a real effect would show, but it weakens with
     temperature/humidity adjustment ({_rr(lad.loc[('pm25__ma0_7', 'M3 + other environment')])}), disappears when CO is
     adjusted for ({_rr(two.loc['pm25__ma0_7'])}), and disappears without the 2017 smog window
     ({ipct('admissions', 'pm25__ma0_7', EXCL)}).
   * **Influence check.** Removing 2017-10-15..2018-01-15 (the 2017 smog season, also when OPD-coded admissions surged -
     finding 6): PM2.5 {ipct('admissions', 'pm25__ma0_7', FULL)} -> {ipct('admissions', 'pm25__ma0_7', EXCL)};
     CO {ipct('admissions', 'co__ma0_7', FULL)} -> {ipct('admissions', 'co__ma0_7', EXCL)}.
   * Conclusion: a small real PM2.5 effect is **not ruled out** - it would need the large 2017 episode for statistical
     power - but nothing here establishes one, and the CO signal behaves like confounding.
5. **Holidays: a modest dip, not statistically significant at 5%.** Any holiday {_rr(fj.loc['any holiday (day of)'])};
   day before {_rr(fj.loc['day before a holiday'])}; day after {_rr(fj.loc['day after a holiday'])}. The dip is larger
   in OPD (day-of RR {fj.loc['any holiday (day of)', 'opd_rr']:.2f}) than emergency ({fj.loc['any holiday (day of)', 'emergency_rr']:.2f}), i.e. a
   service-availability effect. Diwali days are far below expectation (observed/expected {dw.loc[0].iloc[0]:.2f} in 2017,
   {dw.loc[0].iloc[1]:.2f} in 2018) with no consistent rebound within 14 days (n = 2). No post-festival surge is visible.
6. **No reliable growth trend; two data artefacts shape the apparent one.**
   * Duplicates: Apr-Dec 2018 vs Apr-Dec 2017, raw {p['yoy'].loc['admissions', 'change_pct']:+.1f}% but de-duplicated
     {p['yoy'].loc['admissions_dedup_mrd_doa', 'change_pct']:+.1f}%. Season-adjusted trend: raw {tr.loc['admissions', 'pct_per_year']:+.1f}%/yr
     (CI {tr.loc['admissions', 'lo_pct']:+.1f} to {tr.loc['admissions', 'hi_pct']:+.1f}), de-duplicated
     {tr.loc['admissions_dedup_mrd_doa', 'pct_per_year']:+.1f}%/yr (CI {tr.loc['admissions_dedup_mrd_doa', 'lo_pct']:+.1f} to {tr.loc['admissions_dedup_mrd_doa', 'hi_pct']:+.1f}).
   * Emergency/OPD coding: OPD share was {w['opd_share'].loc['2017-11-01', 'opd_share_pct']:.0f}% in Nov 2017 and
     {w['opd_share'].loc['2017-12-01', 'opd_share_pct']:.0f}% in Dec 2017 vs {w['opd_share'].drop(pd.to_datetime(['2017-11-01', '2017-12-01']))['opd_share_pct'].min():.0f}-{w['opd_share'].drop(pd.to_datetime(['2017-11-01', '2017-12-01']))['opd_share_pct'].max():.0f}%
     in every other DEV month, while emergency counts stayed flat (12-14/day). The full-period trends (emergency
     {tr.loc['admissions_emergency', 'pct_per_year']:+.1f}%/yr, OPD {tr.loc['admissions_opd', 'pct_per_year']:+.1f}%/yr) shrink to
     {trx.loc['admissions_emergency', 'pct_per_year']:+.1f}%/yr (p = {trx.loc['admissions_emergency', 'p']:.2f}) and
     {trx.loc['admissions_opd', 'pct_per_year']:+.1f}%/yr (p = {trx.loc['admissions_opd', 'p']:.2f}) without that window. The E/O split is
     therefore unreliable around Nov-Dec 2017 and should not be used as a modelling target.
7. **Strong short-term persistence.** After weekday, holiday and season are removed, residual autocorrelation is
   {p['acf'].loc[1, 'pearson_resid_base_model']:.2f} at lag 1 and {p['acf'].loc[2, 'pearson_resid_base_model']:.2f} at lag 2 (95% white-noise
   limit +-{1.96 / np.sqrt(len(R['frame'])):.2f}). The recent level carries information the calendar does not - relevant for the forecast.

## 2. Daily, weekly, monthly and seasonal patterns

![daily](figures/fig01_daily_series.png)

Seasons (DEV, mean admissions/day with 95% CI):
{_md(p['season'], '{:.2f}')}

Monthly means (holdout months shown for description only):
{_md(p['monthly'], '{:.2f}')}

Season-adjusted log-linear trend (weekday + holiday + two annual harmonics + linear time; % per year), on the full
DEV period and excluding the 2017-10-15..2018-01-15 influence window:
{_md(p['trend'][['pct_per_year', 'lo_pct', 'hi_pct', 'p']], '{:.2f}')}

![acf](figures/fig10_autocorrelation.png)

## 3. Weekday and weekend

![weekday](figures/fig02_weekday.png)

Season-adjusted rate ratios vs Monday (DEV):
{_md(w['weekday_rr'].set_index(['series', 'weekday']), '{:.3f}')}

Kruskal-Wallis across weekdays: total p = {w['weekday_kruskal']['total']:.1e}, emergency p = {w['weekday_kruskal']['emergency']:.1e},
OPD p = {w['weekday_kruskal']['opd']:.1e}.

## 4. Emergency vs OPD

![eo](figures/fig04_emergency_vs_opd.png)

* Emergency admissions are {100 * dev['admissions_emergency'].sum() / dev['admissions'].sum():.0f}% of DEV admissions. Winter/monsoon ratio:
  emergency {w['winter_vs_monsoon']['admissions_emergency']:.2f}, OPD {w['winter_vs_monsoon']['admissions_opd']:.2f}. The OPD ratio is driven by the
  Nov-Dec 2017 surge (OPD share {w['opd_share'].loc['2017-11-01', 'opd_share_pct']:.0f}% and {w['opd_share'].loc['2017-12-01', 'opd_share_pct']:.0f}%), not a repeatable
  seasonal pattern: Nov-Dec 2018 OPD share was {w['opd_share'].loc['2018-11-01', 'opd_share_pct']:.0f}% and {w['opd_share'].loc['2018-12-01', 'opd_share_pct']:.0f}%.
* In Nov-Dec 2017 OPD roughly doubled (to {p['monthly'].loc['2017-11', 'admissions_opd']:.1f} and {p['monthly'].loc['2017-12', 'admissions_opd']:.1f}/day) while
  emergency stayed flat ({p['monthly'].loc['2017-11', 'admissions_emergency']:.1f} and {p['monthly'].loc['2017-12', 'admissions_emergency']:.1f}/day), and OPD dropped
  back in Jan 2018 just as emergency jumped to {p['monthly'].loc['2018-01', 'admissions_emergency']:.1f}/day. This could be a classification
  change or a genuine planned-admission campaign; the data cannot tell which.
* Consequence: the E/O split is useful for weekday/holiday structure (very clear) but not as a clean negative control for
  environmental effects, and not as a forecasting target.

## 5. Environment: crude vs adjusted, and delays

![env](figures/fig05_environment_timeseries.png)

Crude Spearman correlations with daily admissions (same day, DEV) - **confounded by season and weekday**:
{_md(e['crude'], '{:.3f}')}

Adjusted associations at delays 0, 1, 2, 3, 7, 14 days and 0-2 / 0-7 / 0-14 day moving averages. In this delay
screen every window is scaled by the **same-day** IQR so delays are directly comparable; sections 1 and 7 scale 7-day
means by the IQR of the 7-day mean, which is smaller, so the same association reads slightly lower there.

![lags](figures/fig06_lag_associations.png)

Associations with nominal p < 0.05 ({len(nominal)} of {len(lags)}; ~{0.05 * len(lags):.0f} expected by chance):
{_md(lagtab[lagtab.p < 0.05].set_index(['exposure', 'window']), '{:.3f}')}

Full table: `tables/lag_associations.csv`.

Temperature, non-linear check (season-adjusted quintiles of 7-day mean temperature vs the middle quintile):

![tbins](figures/fig12_temperature_quintiles.png)

{_md(R['tbins'], '{:.3f}')}

No quintile differs significantly from the middle one (hottest {100 * (R['tbins'].loc['Q5 hottest', 'rr'] - 1):+.0f}%, p = {R['tbins'].loc['Q5 hottest', 'p']:.2f};
coldest {100 * (R['tbins'].loc['Q1 coldest', 'rr'] - 1):+.0f}%, p = {R['tbins'].loc['Q1 coldest', 'p']:.2f}). Neither a cold-spell nor a heat-spell excess is visible once
season is adjusted for.

## 6. Festivals and holidays

Joint model (weekday + season spline + all four calendar terms; DEV):
{_md(fj, '{:.3f}')}

![fest](figures/fig09_festival_event_study.png)

* Major festivals: {int(es['n'].max())} events in DEV. Observed/expected is below 1 on the day ({es.loc[0, 'mean']:.2f}) and stays
  slightly low for about 5 days; no reliable post-festival surge is visible.
* The holiday dip is larger for planned (OPD) admissions than for emergencies (OPD columns above), suggesting a
  service-availability effect rather than a health effect; none of the calendar terms reaches p < 0.05 in this joint model.
* Diwali is the clearest single event but n = 2, and it overlaps with the smog season, so its "effect" cannot be separated
  from pollution or from the staffing pattern around the festival.

## 7. Overlapping effects

![overlap](figures/fig11_exposure_overlap.png)

Variance inflation factors (7-day means plus annual harmonics):
{_md(o['vif'].rename('VIF').to_frame(), '{:.1f}')}

How each association changes as confounders are added (% per IQR):

![ladder](figures/fig07_confounding_ladder.png)

{_md(lad_tab, '{:+.2f}')}

Sensitivity to the flexibility of the season adjustment:

![sens](figures/fig08_season_adjustment_sensitivity.png)

By target (% per IQR; p-values below). Planned OPD admissions serve as a **negative control**: a short-term
environmental effect should appear in emergencies, not in scheduled admissions. Temperature and CO fail this check:
{_md(bt_tab, '{:+.2f}')}

{_md(bt_p, '{:.3f}')}

Two-pollutant model (each adjusted for the other and for temperature):
{_md(o['two_pollutant'].set_index('exposure')[['adjusted_for', 'rr', 'lo', 'hi', 'p']], '{:.3f}')}

Influence of the 2017 smog season / OPD-coding window (rate ratio per IQR):
{_md(o['influence'].set_index(['target', 'exposure', 'window'])[['rr', 'lo', 'hi', 'p', 'n']], '{:.3f}')}

Interpretation:
* **Season vs temperature**: inseparable here (rho {o['corr'].loc['temp_mean_c__ma0_7', 'month_cos']:.2f}). The crude cold-weather
  association reverses once season is adjusted for, the adjusted estimate depends on spline flexibility, and it also
  appears in planned OPD admissions - all signs that it is the residual of the seasonal fit, not temperature.
* **Pollution vs season**: CO is stable across season adjustments, but the negative control fails (strongest in OPD).
  PM2.5 shrinks with temperature/humidity adjustment, vanishes with CO adjustment, and vanishes without the 2017 smog
  episode (influence table above).
* **Festivals vs smog**: Diwali sits inside the Oct-Nov smog rise; only 2 occurrences, so no separation is possible.
* **Duplicates vs season**: the monthly repeat-row share correlates with monthly temperature (rho = {d['rho_temp'][0]:.2f},
  p = {d['rho_temp'][1]:.3f}) because duplication rose through the second half of 2018 into winter; part of the raw winter-2018
  peak is therefore a data artefact. It is unrelated to PM2.5 (rho = {d['rho_pm25'][0]:.2f}, p = {d['rho_pm25'][1]:.2f}) and flat across
  weekdays ({d['weekday_dup_pct'].min():.1f}-{d['weekday_dup_pct'].max():.1f}%).

## 8. Duplicate frequency and the apparent trend

![dups](figures/fig13_duplicates_and_trend.png)

Apr-Dec comparison, 2017 vs 2018 (mean/day):
{_md(p['yoy'], '{:.2f}')}

* The raw series suggests mild growth; the de-duplicated series suggests mild decline. The sign of the "trend" depends on
  a data-entry artefact, so **no growth trend should be extrapolated** from the raw series.
* Because the harness scores raw rows, the forecasting model must follow the **recent raw level** (which includes the
  current duplication rate) rather than a long-run trend.

## 9. Limitations and uncertainty

* **Two years, one hospital.** Each calendar month is observed twice in DEV at most (Jan-Mar only once, in 2018). Seasonal
  shape, trend and any environment effect that is itself seasonal are confounded.
* **Season adjustment is a modelling choice.** Temperature estimates change sign with spline flexibility; PM2.5 is
  significant only at some settings. No single specification is "correct".
* **Multiple testing.** Of {len(lags)} delay tests, {len(sig)} survive FDR 5%:
  {'; '.join(f"{r.exposure} {r.window} ({100 * (r.rr - 1):+.2f}% per IQR {r.unit:g}, q = {r.q_bh:.3f})" for r in sig.itertuples())}.
  The precipitation result is a very small same-day effect (fewer admissions on rainy days, plausibly access-related).
  All other nominal results should be treated as hypotheses.
* **Exposure measurement.** One city monitor, units probably sub-indices, short gaps interpolated (10 cells). Weather is a
  reanalysis product (Open-Meteo), not a hospital-site station.
* **Outcome measurement.** Raw counts include 5.6-8.6% duplicate rows with a time-varying rate; admissions are
  hospital-level and include referrals from outside Ludhiana, whose exposure differs.
* **Influential episode.** The PM2.5 association depends on 2017-10-15..2018-01-15, a window that also contains an
  unexplained Emergency/OPD classification shift. A pollution effect is "not ruled out", not "shown".
* **Earlier draft corrected.** A first run of this analysis used a season spline that was collinear with the intercept
  (rank-deficient design, effectively one df fewer). It was replaced with a centred spline (full rank, verified by a
  test); temperature results changed materially, pollution and calendar results modestly. Only the corrected results are
  reported here.
* **Correlation, not causation.** No causal claim is made.

## 10. Implications for Step 4 (not decisions)

* Weekday structure (especially Sunday) is certain and should be in every candidate model. Holiday dips are plausible
  (consistent direction, OPD-driven) but imprecise; their value should be tested in backtests rather than assumed.
* Recent level matters (residual autocorrelation, time-varying duplication): models should adapt to the last few weeks.
* Environmental information known at the cut-off (recent 7-day CO / PM2.5) is at best a weak candidate predictor whose
  evidence rests on one episode; it must earn its place in rolling-origin backtests on DEV cut-offs, and the default
  expectation is that it will not.
* Temperature should not be expected to add much beyond seasonality.

## Files

Tables (`outputs/eda/tables/`): {', '.join(tables)}

Figures (`outputs/eda/figures/`): {', '.join(figures)}
"""
    return txt


if __name__ == "__main__":
    import warnings
    warnings.simplefilter("ignore")
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import step3_figures as F  # noqa: E402
    R = run()
    tables = save_tables(R)
    figures = F.make_all(R, FIG, LABEL)
    with open(os.path.join(OUT, "eda_report.md"), "w", encoding="utf-8") as fh:
        fh.write(build_report(R, tables, figures))
    for sub in ["tables", "figures"]:
        print(sub, len(os.listdir(os.path.join(OUT, sub))))
    print("report", os.path.getsize(os.path.join(OUT, "eda_report.md")), "bytes")
