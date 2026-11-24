"""Step 5.4: finalise outputs/holdout/holdout_report.md with the 2019 under-forecasting diagnosis.

Run from the project root, AFTER analysis/step5_holdout_eval.py:
    PYTHONDONTWRITEBYTECODE=1 python analysis/step5_4_finalize_report.py

Read-only with respect to the model and all existing evaluation files: it reads the saved holdout forecasts
and earlier-stage tables, and writes ONLY
  * outputs/holdout/holdout_report.md                  (sections 1-5 kept verbatim, sections 6+ rewritten)
  * outputs/holdout/tables/diagnosis_by_month.csv      (new)
  * outputs/holdout/tables/diagnosis_by_cutoff.csv     (new)
  * outputs/holdout/figures/fig04_level_diagnosis.png  (new)
No model is fitted and no alternative model is evaluated. Idempotent: re-running rebuilds the same report.
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from step3_figures import AQUA, BLUE, INK2, MUTED, NEUTRAL, ORANGE, _date_axis, _save  # noqa: E402

HO = os.path.join(ROOT, "outputs", "holdout")
REPORT = os.path.join(HO, "holdout_report.md")
MARK = "<!-- finalized: build step 5.4 -->"


def _md(df, fmt="{:.2f}"):
    head = "| " + " | ".join([str(df.index.name or "")] + [str(c) for c in df.columns]) + " |"
    lines = [head, "|" + "---|" * (len(df.columns) + 1)]
    for i, row in df.iterrows():
        cells = [fmt.format(v) if isinstance(v, (float, np.floating)) and not pd.isna(v) else ("" if pd.isna(v) else str(v)) for v in row]
        lines.append("| " + " | ".join([str(i)] + cells) + " |")
    return "\n".join(lines)


def load():
    P = pd.read_csv(os.path.join(HO, "tables", "forecasts_primary_grid.csv"), parse_dates=["cutoff", "date"])
    d = pd.read_csv(os.path.join(ROOT, "outputs", "daily_admissions.csv"), index_col=0, parse_dates=True)
    dup = pd.read_csv(os.path.join(ROOT, "outputs", "duplicates_by_month.csv"), index_col=0)
    ctx = pd.read_csv(os.path.join(HO, "tables", "context_earlier_stages.csv"), index_col=0)
    met = pd.read_csv(os.path.join(HO, "tables", "metrics_primary_grid.csv"), index_col=0)
    return P, d, dup, ctx, met


def diagnose(P, d, dup):
    P = P.copy()
    P["y_dedup"] = d["admissions_dedup_mrd_doa"].reindex(P["date"]).to_numpy()
    P["month"] = P["date"].dt.strftime("%Y-%m")
    bm = P.groupby("month").agg(forecast=("mean", "mean"), actual_raw=("y", "mean"), actual_dedup=("y_dedup", "mean"))
    bm["gap_raw"] = bm.actual_raw - bm.forecast
    bm["gap_dedup"] = bm.actual_dedup - bm.forecast
    bm["gap_from_duplicates"] = bm.actual_raw - bm.actual_dedup
    bm["repeat_rows_pct"] = dup.reindex(bm.index)["mrd_doa_extra_pct"].to_numpy()
    prev = [(pd.Period(m) - 12).strftime("%Y-%m") for m in bm.index]
    mon = d.resample("MS").mean()
    mon.index = mon.index.strftime("%Y-%m")
    bm["raw_same_month_2018"] = mon.reindex(prev)["admissions"].to_numpy()
    bm["dedup_same_month_2018"] = mon.reindex(prev)["admissions_dedup_mrd_doa"].to_numpy()
    bm.index.name = "target month"

    rows = []
    for c, g in P.groupby("cutoff"):
        hist = d["admissions"][d.index < c]
        rows.append(dict(cutoff=c.date(), history_mean_all=hist.mean(), level_last_28d=hist.iloc[-28:].mean(),
                         forecast_mean=g["mean"].mean(), actual_raw=g.y.mean(), actual_dedup=g.y_dedup.mean(),
                         gap_recent_level_minus_forecast=hist.iloc[-28:].mean() - g["mean"].mean(),
                         under_forecast_days=int((g["mean"] < g.y).sum())))
    bc = pd.DataFrame(rows).set_index("cutoff")

    e = P["mean"] - P["y"]
    s = dict(bias=e.mean(), mse=(e ** 2).mean(), bias_sq=e.mean() ** 2, under_share=(e < 0).mean(),
             f_mean=P["mean"].mean(), y_raw=P.y.mean(), y_dedup=P.y_dedup.mean(),
             n_cut_all_under=int((bc.forecast_mean < bc.actual_raw).sum()), n_cut=len(bc))
    per = lambda a, b: d.loc[a:b]  # noqa: E731
    s["jm18_raw"], s["jm19_raw"] = per("2018-01-01", "2018-03-31").admissions.mean(), per("2019-01-01", "2019-03-31").admissions.mean()
    s["jm18_dd"] = per("2018-01-01", "2018-03-31").admissions_dedup_mrd_doa.mean()
    s["jm19_dd"] = per("2019-01-01", "2019-03-31").admissions_dedup_mrd_doa.mean()
    s["dev_raw"] = per("2017-04-01", "2018-12-31").admissions.mean()
    s["dev_dup_pct"] = 100 * (1 - per("2017-04-01", "2018-12-31").admissions_dedup_mrd_doa.sum()
                              / per("2017-04-01", "2018-12-31").admissions.sum())
    return P, bm, bc, s


def figure(P, d, dup, path):
    fig, axes = plt.subplots(2, 1, figsize=(11, 6.2), sharex=True, gridspec_kw={"height_ratios": [3, 2]})
    ax = axes[0]
    w = d.loc["2017-10-01":"2019-03-31"]
    ax.plot(w.index, w.admissions.rolling(28, center=True, min_periods=14).mean(), color=BLUE, label="actual, raw rows (target)")
    ax.plot(w.index, w.admissions_dedup_mrd_doa.rolling(28, center=True, min_periods=14).mean(), color=AQUA,
            label="actual, one row per MRD + D.O.A")
    fm = P.groupby("date")["mean"].mean()
    ax.plot(fm.index, fm.rolling(28, center=True, min_periods=7).mean(), color=ORANGE, label="M4 forecasts (holdout, 28-day mean)")
    ax.axvspan(pd.Timestamp("2019-01-01"), pd.Timestamp("2019-03-31"), color=NEUTRAL, zorder=0)
    ax.set_ylim(10, 32)
    ax.set_ylabel("admissions per day (28-day centred mean)")
    ax.set_title("2019: recent level above the fitted seasonal curve; Feb-Mar gap mostly repeat rows")
    ax.legend(loc="lower left", fontsize=8)
    ax2 = axes[1]
    m = dup.loc["2017-10":"2019-03", "mrd_doa_extra_pct"]
    ax2.bar(pd.to_datetime(m.index + "-15"), m.values, width=20, color=MUTED)
    ax2.set_ylabel("repeat rows (%)")
    ax2.set_title("Share of admission rows repeating an MRD + D.O.A")
    ax2.grid(axis="x", visible=False)
    _date_axis(ax2)
    _save(fig, path, "Grey band = holdout. Descriptive decomposition only: the model forecasts raw rows; no alternative model is evaluated.")


def build(base, bm, bc, s, ctx, met):
    m4 = met.loc["M4 (forecast.py)"]
    jan, feb, mar = bm.iloc[0], bm.iloc[1], bm.iloc[2]
    hdr, rest = base.split("## 1. Method", 1)
    core = "## 1. Method" + rest.split("\n## 6.", 1)[0].rstrip() + "\n"
    header = f"""# PS-06 final holdout evaluation - Jan-Mar 2019 (Build steps 5.3-5.4)
{MARK}

**This is a one-time evaluation of the frozen submission.** `forecast.py` (SHA-256 `aa4809046af5...`, verified
before and after the run) was not modified, and nothing was tuned on these results. Model, interval quantiles, pool,
features and fallbacks are exactly as approved in Steps 4 and 5.2. Sections 1-5 were generated by
`analysis/step5_holdout_eval.py` (Step 5.3); sections 6-9 were added by `analysis/step5_4_finalize_report.py`
(Step 5.4) from the saved holdout forecasts, without fitting or evaluating any model.

**Headline.** On 19 untouched Jan-Mar 2019 cut-offs the frozen M4 forecast has MAE {m4.MAE:.2f} vs {met.iloc[1].MAE:.2f} for the
harness baseline (ratio {m4.MAE_ratio_vs_B0:.2f}, 95% CI 0.80-1.03) and 80% interval coverage {100 * m4.coverage80:.1f}%. It beats the
baseline, but by much less than in validation (0.70), because it systematically **under-forecast** 2019
(mean error {s['bias']:+.2f} admissions/day) - explained in section 6.

"""
    diag = f"""
## 6. Diagnosis: why M4 under-forecast Jan-Mar 2019

### 6.1 The symptom
* Mean error (forecast - actual) {s['bias']:+.2f} admissions/day; {100 * s['under_share']:.0f}% of the 266 forecast-days were below the actual;
  the 14-day mean forecast was below the 14-day mean actual at {s['n_cut_all_under']} of {s['n_cut']} cut-offs.
* Systematic bias accounts for {100 * s['bias_sq'] / s['mse']:.0f}% of the mean squared error (bias^2 {s['bias_sq']:.1f} of MSE {s['mse']:.1f});
  the rest is day-to-day noise.
* Coverage stayed at {100 * m4.coverage80:.1f}% only because the frozen intervals were conservative (92% in validation): the
  intervals absorbed a shift they were not designed for.

### 6.2 Historical seasonal pattern vs recent admission level
M4 (`M4_glm_Nall_K2_T0_L0`) is a Poisson GLM fitted to **all** history with weekday, holiday and two annual harmonics,
**no trend term and no recent-level adjustment** (`L0`). Its forecast level is therefore the 2017-18 average
(history mean {bc.history_mean_all.iloc[0]:.1f}-{bc.history_mean_all.iloc[-1]:.1f}/day) shaped by an annual curve learned essentially from a single
previous winter: Jan-Mar 2018 averaged {s['jm18_raw']:.1f}/day, falling from {jan.raw_same_month_2018:.1f}/day in January to
{mar.raw_same_month_2018:.1f}/day in March.

The forecast reproduced that shape - falling from {bc.forecast_mean.iloc[0]:.1f}/day (first cut-off) to {bc.forecast_mean.iloc[-1]:.1f}/day
(last cut-off) - while the 2019 level stayed high. The 28 days before each cut-off averaged
{bc.level_last_28d.min():.1f}-{bc.level_last_28d.max():.1f}/day, {bc.gap_recent_level_minus_forecast.min():.1f}-{bc.gap_recent_level_minus_forecast.max():.1f} above the forecast
(mean gap {bc.gap_recent_level_minus_forecast.mean():.1f}). That recent level was observable at every cut-off, but M4 by design does not use it.

Per cut-off (`tables/diagnosis_by_cutoff.csv`):
{_md(bc, '{:.2f}')}

Jan-Mar averages: {s['jm18_raw']:.1f}/day in 2018 vs {s['jm19_raw']:.1f}/day in 2019 (+{100 * (s['jm19_raw'] / s['jm18_raw'] - 1):.0f}%) in raw rows.
Validation (Jul-Dec 2018) did not expose this: in that period the observed level happened to track the fitted curve.

### 6.3 The changing duplicate-record rate
The target is raw admission rows, which include repeat rows (same MRD + D.O.A) at a rate that varies over time
(DEV average {s['dev_dup_pct']:.1f}% of rows). Comparing the forecast with raw and with de-duplicated actuals separates the two
causes - **descriptively only**; the model forecasts raw rows, as agreed, and is not re-scored on de-duplicated data:

{_md(bm, '{:.2f}')}

* **January 2019 - a genuine rise.** Repeat rows were low ({jan.repeat_rows_pct:.1f}%), yet de-duplicated admissions were
  {jan.actual_dedup:.1f}/day vs {jan.dedup_same_month_2018:.1f}/day in Jan 2018. Of January's {jan.gap_raw:.1f}/day gap, {jan.gap_dedup:.1f}/day
  ({100 * jan.gap_dedup / jan.gap_raw:.0f}%) remains against de-duplicated actuals, i.e. is extra patients rather than repeat rows.
* **February-March 2019 - mostly duplication.** Repeat rows rose to {feb.repeat_rows_pct:.1f}% and {mar.repeat_rows_pct:.1f}% (vs {s['dev_dup_pct']:.1f}% on average in
  the training history), adding {feb.gap_from_duplicates:.1f} and {mar.gap_from_duplicates:.1f}/day to the raw actuals. Against de-duplicated actuals the
  forecast is {abs(feb.gap_dedup):.1f}/day {'low' if feb.gap_dedup > 0 else 'high'} in February and {abs(mar.gap_dedup):.1f}/day {'low' if mar.gap_dedup > 0 else 'high'} in March,
  compared with {feb.gap_raw:.1f} and {mar.gap_raw:.1f}/day low against raw actuals.
* Over all scored days: forecast {s['f_mean']:.1f}/day, raw actual {s['y_raw']:.1f}, de-duplicated actual {s['y_dedup']:.1f}. Roughly
  {s['y_raw'] - s['y_dedup']:.1f} of the {s['y_raw'] - s['f_mean']:.1f}/day shortfall coincides with repeat rows.

Implications:
* The harness scores raw rows, so a rise in duplication is indistinguishable from a rise in admissions and counts as
  forecast error. A model that tracks the recent raw level would partly absorb such a rise; M4 cannot.
* Duplication is a data-entry process, not a clinical one; its rate is not predictable from weather, season or
  calendar. Any forecast of raw rows carries this extra, irreducible uncertainty.
* For hospital planning, de-duplicated admissions (one row per patient-admission) are the more meaningful quantity;
  the scored target and the planning quantity should be presented separately.

![diagnosis](figures/fig04_level_diagnosis.png)

## 7. Tuning, validation and holdout results are not interchangeable

{_md(ctx, '{:.3f}')}

| Stage | Period | What it was used for | How to read it |
|---|---|---|---|
| Tuning | cut-offs 2017-07-01 .. 2018-06-14 | Choosing each family's configuration, interval quantiles and pool size | In-sample for those choices; optimistic |
| Jan-Mar 2018 analogue | subset of tuning | Context for the winter scoring window | Part of tuning; optimistic; < 1 year of history then |
| Validation | cut-offs 2018-07-04 .. 2018-12-15 | Choosing M4 among the families | Selection was made on it; mildly optimistic |
| **Holdout** | cut-offs 2019-01-04 .. 2019-03-17 | **Nothing - evaluation only** | **The only unbiased estimate of future performance** |

The drop from 0.70 (validation) to 0.90 (holdout) is the expected direction of an out-of-sample test, amplified by a
level shift and a duplication rise that the validation period did not contain.

## 8. Why the holdout must not be reused for model selection

* Its value is that **no decision has been made with it**. If any model, configuration, interval setting or fallback
  were now changed because of these results - for example adding a recent-level adjustment because 2019 ran high -
  the holdout would become a tuning set and the 0.90 would no longer estimate future performance. Any re-score of a
  changed model on the same period would be optimistically biased.
* The diagnosis above is descriptive. It explains the observed error; it does not show that an alternative would have
  done better, and no alternative was run on the holdout.
* If the team nevertheless changes the model, the change must be justified on DEV evidence alone (Steps 3-4 already
  flagged recent-level persistence and duplication drift), the holdout results above must be reported as the frozen
  M4 result, and any new holdout number must be labelled as post-selection (not an independent estimate).
* The hidden harness cut-offs fall in the same Jan-Mar 2019 period, so this holdout is the closest available preview of
  the scored result; that is a reason to keep it clean, not to optimise against it.

## 9. Limitations

* **Small sample.** 19 primary cut-offs with overlapping windows cover 86 distinct days; the MAE-ratio CI (0.80-1.03)
  includes 1. Month-level splits (section 4) rest on 18-31 overlapping cut-offs each.
* **Single regime change.** The diagnosis rests on one winter. Whether 2019's higher level and duplication persist
  cannot be known from these data.
* **Duplication.** Repeat rows are identified from MRD + D.O.A (Step 2); some may be genuine same-day re-admissions,
  so the de-duplicated series is an approximation of unique admissions.
* **2019-03-31** may be incomplete; it is excluded from the primary set and appears only in the daily sensitivity run.
* **Harness unknowns.** The harness's truncation method, library versions, baseline-interval definition and MAE-ratio
  pooling are not published; section 5 uses the Step 4 approximations.
* **Score spread.** The harness uses ~4 unknown cut-offs; a single draw of 4 can land anywhere in the section 5 range.
* **Association, not causation.** Nothing here identifies why admissions or duplication rose in 2019.

## Files

Tables (`outputs/holdout/tables/`): by_cutoff_month_daily.csv, by_horizon_primary.csv, context_earlier_stages.csv,
diagnosis_by_cutoff.csv (5.4), diagnosis_by_month.csv (5.4), forecasts_daily_sensitivity.csv, forecasts_primary_grid.csv,
harness_score_all_4cutoff_combinations.csv, leakage_and_runtime_audit.csv, metrics_daily_excluding_2019-03-31.csv,
metrics_daily_sensitivity.csv, metrics_primary_grid.csv, per_cutoff_primary.csv, summary.json

Figures (`outputs/holdout/figures/`): fig01_holdout_mae_by_cutoff.png, fig02_holdout_example_forecasts.png,
fig03_holdout_by_horizon.png, fig04_level_diagnosis.png (5.4)
"""
    return header + core + diag


def main():
    P, d, dup, ctx, met = load()
    P, bm, bc, s = diagnose(P, d, dup)
    bm.to_csv(os.path.join(HO, "tables", "diagnosis_by_month.csv"))
    bc.to_csv(os.path.join(HO, "tables", "diagnosis_by_cutoff.csv"))
    figure(P, d, dup, os.path.join(HO, "figures", "fig04_level_diagnosis.png"))
    base = open(REPORT, encoding="utf-8").read()
    with open(REPORT, "w", encoding="utf-8") as fh:
        fh.write(build(base, bm, bc, s, ctx, met))
    print(bm.round(2).to_string())


if __name__ == "__main__":
    main()
