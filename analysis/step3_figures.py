"""Static figures for the Step 3 EDA (matplotlib, PNG).

Palette: the validated reference palette (categorical slots 1-3 only, blue / orange /
aqua, which pass the all-pairs colour-vision check), muted ink for axes, one axis
per chart (no dual axes), direct labels where a legend would be ambiguous.
Every figure's numbers are also saved as CSV in outputs/eda/tables/.
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE, NEUTRAL = "#e1e0d9", "#c3c2b7", "#fcfcfb", "#f0efec"
RED = "#e34948"
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.labelsize": 9, "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "legend.fontsize": 8, "font.size": 9, "lines.linewidth": 2,
})


def _save(fig, path, note=None):
    if note:
        # place the note below everything already drawn (tick labels, x labels, legends)
        fig.canvas.draw()
        bottom = min(a.get_tightbbox().transformed(fig.transFigure.inverted()).y0 for a in fig.axes if a.get_visible())
        fig.text(0.01, bottom - 0.02, note, fontsize=7, color=MUTED, ha="left", va="top")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _holdout(ax):
    ax.axvspan(pd.Timestamp("2019-01-01"), pd.Timestamp("2019-03-31"), color=NEUTRAL, zorder=0)
    ax.text(pd.Timestamp("2019-02-14"), ax.get_ylim()[1], "holdout\n(descriptive only)", ha="center",
            va="top", fontsize=7, color=MUTED)


def _date_axis(ax):
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))


def fig_daily(daily, path):
    y = daily["admissions"]
    fig, ax = plt.subplots(figsize=(11, 3.8))
    ax.plot(y.index, y, color=BLUE, alpha=0.25, linewidth=0.8)
    ax.plot(y.index, y.rolling(28, center=True, min_periods=14).mean(), color=BLUE)
    ax.set_ylim(0, y.max() * 1.08)
    _holdout(ax)
    _date_axis(ax)
    ax.text(y.index[40], 44, "daily count (thin)  ·  28-day centred mean (thick)", fontsize=8, color=INK2)
    ax.set_ylabel("Admissions per day (raw rows)")
    ax.set_title("Daily cardiac admissions, Apr 2017 - Mar 2019")
    _save(fig, path, "Source: HDHI Admission data.csv, raw row counts by D.O.A (duplicates not removed).")


def fig_weekday(wk, path):
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharey=False)
    for ax, (name, col) in zip(axes, [("total", BLUE), ("emergency", ORANGE), ("opd", AQUA)]):
        w = wk[wk["series"] == name].set_index("weekday").loc[DAYS]
        ax.bar(range(7), w["mean"], color=col, width=0.62)
        ax.errorbar(range(7), w["mean"], yerr=[w["mean"] - w["lo"], w["hi"] - w["mean"]], fmt="none",
                    ecolor=INK2, elinewidth=1, capsize=2)
        ax.set_xticks(range(7), DAYS)
        ax.set_title({"total": "All admissions", "emergency": "Emergency", "opd": "OPD (planned)"}[name])
        ax.set_ylim(0, w["hi"].max() * 1.15)
        ax.grid(axis="x", visible=False)
        for i, v in enumerate(w["mean"]):
            if i in (0, 6):
                ax.text(i, v / 2, f"{v:.1f}", ha="center", color="white", fontsize=8, fontweight="bold")
    axes[0].set_ylabel("Mean admissions per day")
    _save(fig, path, "DEV period Apr 2017-Dec 2018; bars = mean, whiskers = 95% CI.")


def fig_monthly_dedup(monthly, path):
    m = monthly.copy()
    m.index = pd.to_datetime(m.index + "-01")
    fig, ax = plt.subplots(figsize=(11, 3.8))
    for col, c, lab in [("admissions", BLUE, "raw rows (target)"),
                        ("admissions_dedup_exact", ORANGE, "exact duplicates removed"),
                        ("admissions_dedup_mrd_doa", AQUA, "one row per MRD + D.O.A")]:
        ax.plot(m.index, m[col], color=c, marker="o", markersize=4)
        ax.text(m.index[-1] + pd.Timedelta(days=8), m[col].iloc[-1], lab, color=INK2, fontsize=8, va="center")
    ax.set_ylim(0, m["admissions"].max() * 1.15)
    _holdout(ax)
    _date_axis(ax)
    ax.set_xlim(m.index[0] - pd.Timedelta(days=15), m.index[-1] + pd.Timedelta(days=150))
    ax.set_ylabel("Mean admissions per day")
    ax.set_title("Monthly level: raw target vs de-duplicated (sensitivity) series")
    _save(fig, path, "The gap between the lines is duplicate rows; it is near zero in 2017 and grows through 2018.")


def fig_emergency_opd(monthly, share, path):
    m = monthly.copy()
    m.index = pd.to_datetime(m.index + "-01")
    fig, axes = plt.subplots(2, 1, figsize=(11, 5.6), sharex=True, gridspec_kw={"height_ratios": [3, 2]})
    ax = axes[0]
    for col, c, lab in [("admissions_emergency", ORANGE, "Emergency"), ("admissions_opd", AQUA, "OPD")]:
        ax.plot(m.index, m[col], color=c, marker="o", markersize=4)
        ax.text(m.index[-1] + pd.Timedelta(days=8), m[col].iloc[-1], lab, color=INK2, fontsize=8, va="center")
    ax.set_ylim(0, m["admissions_emergency"].max() * 1.15)
    _holdout(ax)
    ax.set_ylabel("Mean per day")
    ax.set_title("Emergency vs OPD admissions (monthly mean per day)")
    ax2 = axes[1]
    s = share.copy()
    ax2.plot(s.index, s["opd_share_pct"], color=AQUA, marker="o", markersize=4)
    ax2.set_ylim(0, 60)
    ax2.set_ylabel("OPD share (%)")
    ax2.set_title("OPD share of admissions (DEV period)")
    _date_axis(ax2)
    ax2.set_xlim(m.index[0] - pd.Timedelta(days=15), m.index[-1] + pd.Timedelta(days=120))
    _save(fig, path)


def fig_environment(daily, path):
    d = daily.loc[:"2019-03-31"]
    w = d[["admissions", "temp_mean_c", "pm25", "co", "rel_humidity_mean_pct"]].rolling(7, center=True, min_periods=4).mean()
    panels = [("admissions", "Admissions / day", BLUE), ("temp_mean_c", "Mean temp (C)", ORANGE),
              ("pm25", "PM2.5 (index)", INK2), ("co", "CO (index)", INK2), ("rel_humidity_mean_pct", "Humidity (%)", AQUA)]
    fig, axes = plt.subplots(len(panels), 1, figsize=(11, 8.5), sharex=True)
    for ax, (col, lab, c) in zip(axes, panels):
        ax.plot(w.index, w[col], color=c, linewidth=1.6)
        ax.set_ylabel(lab, fontsize=8)
        ax.set_ylim(min(0, w[col].min()), w[col].max() * 1.1)
        ax.axvspan(pd.Timestamp("2019-01-01"), pd.Timestamp("2019-03-31"), color=NEUTRAL, zorder=0)
        for yr in (2017, 2018):
            ax.axvspan(pd.Timestamp(f"{yr}-10-15"), pd.Timestamp(f"{yr}-11-30"), color="#fde7dc", zorder=0)
    axes[0].set_title("Admissions and environment, 7-day centred means (shaded orange: mid-Oct to Nov smog season)")
    _date_axis(axes[-1])
    _save(fig, path, "Each panel has its own axis; no panel is plotted against another's scale. Grey = holdout.")


def fig_lags(lags, path, exposures, labels):
    order = [f"lag{k}" for k in [0, 1, 2, 3, 7, 14]] + ["ma0_2", "ma0_7", "ma0_14"]
    names = ["0", "1", "2", "3", "7", "14", "0-2", "0-7", "0-14"]
    n = len(exposures)
    ncol = 4
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(12, 3.0 * nrow), sharey=True)
    axes = axes.ravel()
    for ax, x in zip(axes, exposures):
        t = lags[lags["exposure"] == x].set_index("window").loc[order]
        pct, lo, hi = 100 * (t["rr"] - 1), 100 * (t["lo"] - 1), 100 * (t["hi"] - 1)
        col = [BLUE if w.startswith("lag") else ORANGE for w in order]
        ax.axhline(0, color=AXIS, linewidth=1)
        for i in range(len(order)):
            ax.plot([i, i], [lo.iloc[i], hi.iloc[i]], color=col[i], linewidth=1.5)
            ax.plot(i, pct.iloc[i], "o", color=col[i], markersize=5,
                    markeredgecolor=SURFACE, markeredgewidth=1)
            if t["q_bh"].iloc[i] < 0.05:
                ax.text(i, hi.iloc[i] + 1, "*", ha="center", color=INK, fontsize=11)
        ax.set_xticks(range(len(order)), names, fontsize=7)
        ax.set_title(f"{labels[x]}\nper IQR = {t['unit'].iloc[0]:.3g}", fontsize=9)
        ax.set_ylim(-50, 55)
    for ax in axes[n:]:
        ax.set_visible(False)
    for ax in axes[::ncol]:
        ax.set_ylabel("% change in admissions")
    fig.suptitle("Season-adjusted association by delay (blue: single-day lag, orange: moving average)",
                 x=0.01, ha="left", fontsize=11, fontweight="bold", color=INK)
    fig.tight_layout()
    _save(fig, path, "Poisson GLM, weekday + holiday + natural spline of time (df 7); HAC 95% CI. "
                     "* = survives Benjamini-Hochberg FDR 5% over all 126 tests. DEV period only.")


def fig_ladder(ladder, path, labels):
    exps = list(dict.fromkeys(ladder["exposure"]))
    models = list(dict.fromkeys(ladder["model"]))
    colors = [MUTED, INK2, BLUE, ORANGE]
    fig, ax = plt.subplots(figsize=(10, 0.75 * len(exps) + 1.5))
    for j, mname in enumerate(models):
        t = ladder[ladder["model"] == mname].set_index("exposure").loc[exps]
        yy = np.arange(len(exps)) + (j - 1.5) * 0.17
        ax.hlines(yy, 100 * (t["lo"] - 1), 100 * (t["hi"] - 1), color=colors[j], linewidth=1.5)
        ax.plot(100 * (t["rr"] - 1), yy, "o", color=colors[j], markersize=5, label=mname,
                markeredgecolor=SURFACE, markeredgewidth=1)
    ax.axvline(0, color=AXIS)
    ax.set_yticks(range(len(exps)), [labels[e.split("__")[0]] + " (7-day mean)" for e in exps])
    ax.invert_yaxis()
    ax.set_xlabel("% change in admissions per IQR (95% CI)")
    ax.set_title("How each association changes as confounders are added")
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.12), ncol=4)
    ax.grid(axis="y", visible=False)
    _save(fig, path, "Temperature reverses sign once season is adjusted for. PM2.5 and CO stay positive here, but see the "
                     "OPD negative control and the 2017 smog-window influence check in the report.")


def fig_spline_sensitivity(sens, path, labels):
    exps = list(dict.fromkeys(sens["exposure"]))
    fig, axes = plt.subplots(1, len(exps), figsize=(12, 3.2), sharey=True)
    for ax, x in zip(axes, exps):
        t = sens[sens["exposure"] == x]
        ax.fill_between(t["df_per_year"], 100 * (t["lo"] - 1), 100 * (t["hi"] - 1), color=BLUE, alpha=0.15, linewidth=0)
        ax.plot(t["df_per_year"], 100 * (t["rr"] - 1), color=BLUE, marker="o", markersize=4)
        ax.axhline(0, color=AXIS)
        ax.axvline(4, color=MUTED, linestyle=":", linewidth=1)
        ax.set_title(labels[x.split("__")[0]] + " (7-day mean)", fontsize=9)
        ax.set_xlabel("season spline df per year")
    axes[0].set_ylabel("% change per IQR")
    axes[0].text(4.2, axes[0].get_ylim()[1] * 0.9, "primary", fontsize=7, color=MUTED)
    fig.suptitle("Sensitivity of each association to how flexibly season is modelled", x=0.01, ha="left",
                 fontsize=11, fontweight="bold", color=INK)
    fig.tight_layout()
    _save(fig, path, "A stable estimate across df supports a real short-term association; a swinging one means "
                     "the exposure cannot be separated from season.")


def fig_festival(es, diwali, path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.6), sharey=True)
    ax = axes[0]
    ax.fill_between(es.index, es["lo"], es["hi"], color=BLUE, alpha=0.15, linewidth=0)
    ax.plot(es.index, es["mean"], color=BLUE, marker="o", markersize=3.5)
    ax.axhline(1, color=AXIS)
    ax.axvline(0, color=MUTED, linestyle=":")
    ax.set_title(f"All major festivals (n = {int(es['n'].max())} events)")
    ax.set_xlabel("days relative to festival")
    ax.set_ylabel("observed / expected admissions")
    ax = axes[1]
    for c, ev in zip([ORANGE, AQUA], diwali.columns):
        ax.plot(diwali.index, diwali[ev], color=c, marker="o", markersize=3.5, label=f"Diwali {ev[:4]}")
    ax.axhline(1, color=AXIS)
    ax.axvline(0, color=MUTED, linestyle=":")
    ax.legend(loc="lower right")
    ax.set_title("Diwali 2017 and 2018 (n = 2)")
    ax.set_xlabel("days relative to Diwali")
    _save(fig, path, "Expected = Poisson model with weekday + season spline only (no holiday terms); DEV period.")


def fig_acf(acf, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.2), sharey=True)
    lim = 1.96 / np.sqrt(640)
    for ax, (col, title, c) in zip(axes, [("raw_counts", "Raw daily counts", BLUE),
                                          ("pearson_resid_base_model", "Residuals after weekday + holiday + season", ORANGE)]):
        a = acf[col].iloc[1:]
        ax.bar(a.index, a, color=c, width=0.6)
        ax.axhspan(-lim, lim, color=NEUTRAL, zorder=0)
        ax.axhline(0, color=AXIS)
        ax.set_title(title)
        ax.set_xlabel("lag (days)")
        ax.set_xticks([1, 7, 14, 21, 28])
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("autocorrelation")
    _save(fig, path, "Grey band = approximate 95% limits for white noise (n = 640).")


def fig_corr(corr, path, labels):
    names = {c: (labels.get(c.split("__")[0], c) if "__" in c else c) for c in corr.columns}
    names.update({"major": "Major festival", "holiday": "Holiday", "month_cos": "Winter-ness (cos day of year)"})
    fig, ax = plt.subplots(figsize=(7.5, 6.2))
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("div", ["#1c5cab", "#86b6ef", NEUTRAL, "#f2a3a2", "#b7302f"])
    im = ax.imshow(corr.values, cmap=cmap, vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)), [names[c] for c in corr.columns], rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(corr)), [names[c] for c in corr.index], fontsize=8)
    for i in range(len(corr)):
        for j in range(len(corr)):
            v = corr.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7, color="white" if abs(v) > 0.6 else INK)
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.04, label="Spearman correlation")
    ax.set_title("Overlap between exposures, festivals and season (7-day means)")
    _save(fig, path, "DEV period. Temperature and season are almost the same variable (|rho| > 0.9).")


def fig_tbins(tb, path):
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    x = np.arange(len(tb))
    ax.axhline(0, color=AXIS)
    ax.vlines(x, 100 * (tb["lo"] - 1), 100 * (tb["hi"] - 1), color=BLUE, linewidth=1.5)
    ax.plot(x, 100 * (tb["rr"] - 1), "o", color=BLUE, markersize=6)
    ax.set_xticks(x, [f"{b}\n{lo:.0f}-{hi:.0f} C" for b, lo, hi in zip(tb.index, tb["temp_from"], tb["temp_to"])], fontsize=8)
    ax.set_ylabel("% vs middle quintile")
    ax.set_title("7-day mean temperature quintiles, season-adjusted")
    ax.grid(axis="x", visible=False)
    _save(fig, path, "Reference = Q3. Because season is adjusted for, quintiles compare unusually warm/cold spells, "
                     "not winter vs summer.")


def fig_dups(m, path):
    t = m.copy()
    fig, axes = plt.subplots(2, 1, figsize=(11, 5.4), sharex=True, gridspec_kw={"height_ratios": [3, 2]})
    ax = axes[0]
    for col, c, lab in [("admissions", BLUE, "raw rows"), ("admissions_dedup_mrd_doa", AQUA, "one row per MRD + D.O.A")]:
        ax.plot(t.index, t[col] / t.index.days_in_month, color=c, marker="o", markersize=4)
        ax.text(t.index[-1] + pd.Timedelta(days=8), (t[col] / t.index.days_in_month).iloc[-1], lab, fontsize=8,
                color=INK2, va="center")
    ax.set_ylim(0, 30)
    ax.set_ylabel("Mean per day")
    ax.set_title("Repeat rows inflate the raw level from Oct 2017 onward (DEV period)")
    ax2 = axes[1]
    ax2.bar(t.index, t["dup_extra_pct"], width=20, color=ORANGE)
    ax2.set_ylabel("repeat rows (%)")
    ax2.set_title("Share of rows that repeat an MRD + D.O.A")
    ax2.grid(axis="x", visible=False)
    _date_axis(ax2)
    ax2.set_xlim(t.index[0] - pd.Timedelta(days=20), t.index[-1] + pd.Timedelta(days=140))
    _save(fig, path)


def make_all(R, fig_dir, labels):
    os.makedirs(fig_dir, exist_ok=True)
    p = lambda n: os.path.join(fig_dir, n)  # noqa: E731
    fig_daily(R["daily"], p("fig01_daily_series.png"))
    fig_weekday(R["weekday"]["weekday"], p("fig02_weekday.png"))
    fig_monthly_dedup(R["patterns"]["monthly"], p("fig03_monthly_raw_vs_dedup.png"))
    fig_emergency_opd(R["patterns"]["monthly"], R["weekday"]["opd_share"], p("fig04_emergency_vs_opd.png"))
    fig_environment(R["daily"], p("fig05_environment_timeseries.png"))
    fig_lags(R["env"]["lags"], p("fig06_lag_associations.png"),
             ["temp_mean_c", "temp_min_c", "rel_humidity_mean_pct", "precip_mm", "aqi", "pm25", "pm10", "co",
              "no2", "so2", "ozone", "nh3"], labels)
    fig_ladder(R["overlap"]["ladder"], p("fig07_confounding_ladder.png"), labels)
    fig_spline_sensitivity(R["overlap"]["spline_sensitivity"], p("fig08_season_adjustment_sensitivity.png"), labels)
    fig_festival(R["fest"]["event_study"], R["fest"]["diwali"], p("fig09_festival_event_study.png"))
    fig_acf(R["patterns"]["acf"], p("fig10_autocorrelation.png"))
    fig_corr(R["overlap"]["corr"], p("fig11_exposure_overlap.png"), labels)
    fig_tbins(R["tbins"], p("fig12_temperature_quintiles.png"))
    fig_dups(R["dups"]["monthly"], p("fig13_duplicates_and_trend.png"))
    return sorted(os.listdir(fig_dir))
