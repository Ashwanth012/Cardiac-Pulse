"""Figures and markdown report for the Step 4 backtest (reads the results dict from step4_backtest.main)."""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from step3_figures import AXIS, BLUE, INK, INK2, MUTED, NEUTRAL, ORANGE, AQUA, SURFACE, _save  # noqa: E402,F401

BASE = "B0_seasonal_naive"


def _fam_label(fam, R):
    return f"{fam}  {R['winners'][fam]}"


def figures(R, fig_dir):
    os.makedirs(fig_dir, exist_ok=True)
    p = lambda n: os.path.join(fig_dir, n)  # noqa: E731
    vt, rec, best = R["val_tab"], R["recommended"], R["best"]

    # 1. validation MAE ratio vs harness baseline, per family winner
    order = vt.sort_values("ratio_vs_B0", ascending=False).index
    fig, ax = plt.subplots(figsize=(9, 0.5 * len(order) + 1.4))
    for i, fam in enumerate(order):
        r = vt.loc[fam]
        c = ORANGE if fam == rec else (BLUE if fam == best else MUTED)
        ax.hlines(i, r.ratio_lo, r.ratio_hi, color=c, linewidth=2)
        ax.plot(r.ratio_vs_B0, i, "o", color=c, markersize=7, markeredgecolor=SURFACE)
        ax.text(r.ratio_hi + 0.01, i, f"{r.ratio_vs_B0:.2f}", va="center", fontsize=8, color=INK2)
    ax.axvline(1, color=AXIS)
    ax.set_yticks(range(len(order)), [_fam_label(f, R) for f in order], fontsize=8)
    ax.set_xlabel("validation MAE / seasonal-naive MAE (95% block-bootstrap CI)")
    ax.set_title("Validation error relative to the harness baseline (lower is better)")
    ax.grid(axis="y", visible=False)
    ax.text(0.99, 0.02, f"orange = recommended ({rec}), blue = lowest MAE ({best})", transform=ax.transAxes,
            ha="right", fontsize=7, color=MUTED)
    _save(fig, p("fig01_validation_ratio.png"), f"Validation cut-offs {R['val_c'].min().date()}..{R['val_c'].max().date()} (every 4 days). Configs were chosen on tuning cut-offs only.")

    # 2. MAE by horizon
    hor = R["hor"]
    fig, ax = plt.subplots(figsize=(9, 3.6))
    for name, c, lab in [(BASE, MUTED, "seasonal naive (harness)"), (R["winners"][rec], ORANGE, f"recommended {rec}"),
                         (R["winners"][best], BLUE, f"lowest MAE {best}")]:
        if name in hor:
            ax.plot(hor.index, hor[name], color=c, marker="o", markersize=4, label=lab)
    ax.set_xlabel("forecast horizon (days after cut-off; 1 = cut-off day)")
    ax.set_ylabel("MAE (admissions)")
    ax.set_ylim(0, None)
    ax.set_xticks(range(1, 15))
    ax.legend(loc="lower right")
    ax.set_title("Error by horizon (validation)")
    _save(fig, p("fig02_mae_by_horizon.png"), "The baseline gets worse in week 2 because it repeats one noisy week twice.")

    # 3. per cut-off MAE through time
    fc = pd.concat([R["fc"], R["efc"]])
    pc = fc[fc.model.isin([BASE, R["winners"][rec], R["winners"][best]])].assign(ae=lambda d: (d["mean"] - d["y"]).abs())
    pc = pc.groupby(["model", "cutoff"])["ae"].mean().unstack(0)
    fig, ax = plt.subplots(figsize=(11, 3.8))
    for name, c, lab in [(BASE, MUTED, "seasonal naive"), (R["winners"][rec], ORANGE, rec), (R["winners"][best], BLUE, best)]:
        if name in pc:
            ax.plot(pc.index, pc[name], color=c, linewidth=1.4 if name == BASE else 2, label=lab)
    ax.axvspan(pd.Timestamp("2018-07-01"), pc.index.max(), color=NEUTRAL, zorder=0)
    ax.text(pd.Timestamp("2018-07-05"), ax.get_ylim()[1] * 0.95, "validation", fontsize=8, color=MUTED, va="top")
    ax.set_ylabel("14-day MAE")
    ax.set_title("Error at every cut-off (tuning period white, validation grey)")
    ax.legend(loc="upper right", ncol=3)
    _save(fig, p("fig03_mae_by_cutoff.png"))

    # 4. coverage vs width
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for fam in vt.index:
        r = vt.loc[fam]
        c = ORANGE if fam == rec else (BLUE if fam == best else INK2)
        ax.plot(r.width80, r.coverage80, "o", color=c, markersize=8, markeredgecolor=SURFACE)
        ax.annotate(fam, (r.width80, r.coverage80), xytext=(6, 4), textcoords="offset points", fontsize=8, color=INK2)
    ax.axhline(0.80, color=MUTED, linestyle=":")
    ax.axvline(2 * R["base_w_val"], color=MUTED, linestyle="--")
    ax.text(2 * R["base_w_val"], 0.79, " 2x baseline width\n (approximate cap)", fontsize=7, color=MUTED, va="top")
    ax.set_xlim(0, 2.2 * R["base_w_val"])
    ax.set_ylim(0.7, 1.0)
    ax.set_xlabel("mean 80% interval width (admissions)")
    ax.set_ylabel("validation coverage")
    ax.set_title("Interval coverage vs width (validation)")
    _save(fig, p("fig04_interval_coverage_width.png"), "Dotted: 80% target. Dashed: twice the baseline's interval width under our approximation of the harness rule.")

    # 5. environment vs same GLM
    ec = R["env_ci"].copy()
    ec = ec.sort_values("val_ratio_vs_GLM")
    fig, ax = plt.subplots(figsize=(9, 0.4 * len(ec) + 1.4))
    for i, r in enumerate(ec.itertuples()):
        c = MUTED if r.model.startswith("X_") else BLUE
        ax.hlines(i, r.lo, r.hi, color=c, linewidth=2)
        ax.plot(r.val_ratio_vs_GLM, i, "o", color=c, markeredgecolor=SURFACE, markersize=6)
    ax.axvline(1, color=AXIS)
    ax.set_yticks(range(len(ec)), ec.model, fontsize=8)
    ax.set_xlabel("validation MAE / MAE of the same GLM without environment (95% CI)")
    ax.set_title("Do environmental features improve the forecast? (blue: deployable, grey: oracle)")
    ax.grid(axis="y", visible=False)
    _save(fig, p("fig05_environment_effect.png"), "Oracle (X_) models use the actual future observations - an upper bound that cannot be deployed.")

    # 6. example forecasts
    ivs = R["ivs"]
    name = R["winners"][rec]
    g = ivs[(ivs.model == name) & (ivs.cutoff >= "2018-07-01")]
    picks = sorted(g.cutoff.unique())
    picks = [picks[2], picks[len(picks) // 2], picks[-1]]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), sharey=True)
    for ax, c in zip(axes, picks):
        s = g[g.cutoff == c]
        hist = R["fc"][(R["fc"].model == BASE)]
        ax.fill_between(s.date, s.lo, s.hi, color=ORANGE, alpha=0.2, linewidth=0)
        ax.plot(s.date, s["mean"], color=ORANGE, label="forecast")
        ax.plot(s.date, s.y, "o", color=INK, markersize=3.5, label="actual")
        ax.set_title(f"cut-off {pd.Timestamp(c).date()}", fontsize=9)
        ax.tick_params(axis="x", rotation=45)
    axes[0].set_ylabel("admissions")
    axes[0].legend(loc="upper left")
    _save(fig, p("fig06_example_forecasts.png"), f"{name}: mean and 80% interval, validation cut-offs.")

    # 7. simulated harness score distribution
    sims = R["sims"]
    fams = [f for f in vt.sort_values("ratio_vs_B0").index if f in sims]
    fig, ax = plt.subplots(figsize=(9, 0.45 * len(fams) + 1.4))
    for i, f in enumerate(fams):
        s = sims[f]["total_pooled"]
        q = s.quantile([0.05, 0.25, 0.5, 0.75, 0.95])
        c = ORANGE if f == rec else (BLUE if f == best else MUTED)
        ax.hlines(i, q[0.05], q[0.95], color=c, linewidth=1)
        ax.hlines(i, q[0.25], q[0.75], color=c, linewidth=5)
        ax.plot(q[0.5], i, "|", color=INK, markersize=12)
    ax.set_yticks(range(len(fams)), fams)
    ax.invert_yaxis()
    ax.set_xlabel("simulated forecast score out of 40 (error + coverage; 4 random validation cut-offs)")
    ax.set_title("Simulated harness score (5000 draws)")
    ax.grid(axis="y", visible=False)
    _save(fig, p("fig07_simulated_score.png"), "Thick bar = middle 50%, thin = 90%. Width penalty uses an approximation (see report).")
    return sorted(os.listdir(fig_dir))


def _md(df, fmt="{:.3f}"):
    head = "| " + " | ".join([str(df.index.name or "")] + [str(c) for c in df.columns]) + " |"
    lines = [head, "|" + "---|" * (len(df.columns) + 1)]
    for i, row in df.iterrows():
        cells = [fmt.format(v) if isinstance(v, (float, np.floating)) and not pd.isna(v) else ("" if pd.isna(v) else str(v)) for v in row]
        lines.append("| " + " | ".join([str(i)] + cells) + " |")
    return "\n".join(lines)


def write(R, tables, figs, path):
    vt, w, rec, best = R["val_tab"], R["winners"], R["recommended"], R["best"]
    ts = R["tune_sum"]
    env_ci = R["env_ci"]
    deploy = env_ci[env_ci.model.str.startswith("E_")]
    oracle = env_ci[env_ci.model.str.startswith("X_")]
    env_helps = bool((deploy["hi"] < 1).any())
    oracle_helps = bool((oracle["hi"] < 1).any())
    civ = R["chosen_iv"].set_index("model")
    sc = R["scores"]
    jm = R["jm_tab"].set_index("model")
    recm, bestm = w[rec], w[best]
    vcols = ["model", "MAE", "RMSE", "WAPE", "MAE_h1_7", "MAE_h8_14", "ratio_vs_B0", "ratio_lo", "ratio_hi",
             "tune_ratio_vs_B0", "coverage80", "width80"]
    tune_top = ts.sort_values("MAE").groupby("family").head(3)[["family", "model", "MAE", "ratio_vs_B0", "MAE_h1_7", "MAE_h8_14"]]
    season = R["by_season"].pivot(index="model", columns="season", values="ratio_vs_B0")
    level = R["by_level"].pivot(index="model", columns="level_band", values="ratio_vs_B0")
    env_tab = R["env_cmp"].pivot(index="model", columns="period", values="ratio_vs_same_GLM").join(env_ci.set_index("model"))
    rt = R["runtime"]

    # computed statements (no hand-written claims)
    # compare only families ranked in both periods (E is tuned in a separate stage, so it is excluded here)
    order_t = list(ts[ts.model.isin(w.values)].sort_values("MAE")["family"])
    order_v = [f for f in vt.sort_values("MAE").index if f in order_t]
    rho = pd.Series(range(len(order_t)), index=order_t).corr(pd.Series(range(len(order_v)), index=order_v)
                                                             .reindex(order_t), method="spearman")
    top_t, top_v = set(order_t[:4]), set(order_v[:4])
    consistency = (f"Among the {len(order_t)} families ranked in both periods - tuning: {' < '.join(order_t)}; "
                   f"validation: {' < '.join(order_v)} (Spearman rank correlation {rho:.2f}). The four best are "
                   f"{'the same set' if top_t == top_v else 'different sets'} in both periods "
                   f"(tuning {', '.join(sorted(top_t))}; validation {', '.join(sorted(top_v))}) and their order within "
                   f"that group changes, consistent with differences inside the group being noise.")
    worst = season.drop(index=BASE).idxmax(axis=1)
    not_winter = [m for m, s in worst.items() if s != "Winter (Dec-Feb)"]
    all_beat = bool((season.drop(index=BASE) < 1).all().all() and (level.drop(index=BASE) < 1).all().all())
    season_note = (f"Winter is the hardest season (highest ratio) for every model"
                   f"{' except ' + ', '.join(not_winter) + ' (worst: ' + ', '.join(worst[not_winter]) + ')' if not_winter else ''}. "
                   f"{'Every' if all_beat else 'Not every'} candidate beats the baseline in every season and every level band.")
    jmc = R["jm_ci"].set_index(["model", "reference"])
    better = [m for m in ("M4", "M6", "E", "M2") if (m, "M1") in jmc.index and jmc.loc[(m, "M1"), "hi"] < 1]
    jm_note = (f"Against M1 on these cut-offs, {', '.join(better)} {'has' if len(better) == 1 else 'have'} a 95% CI entirely "
               f"below 1." if better else "No model's advantage over M1 on these cut-offs is outside bootstrap noise "
               "(every CI vs M1 includes 1).")

    txt = f"""# PS-06 forecasting model development and rolling-origin backtest (Build step 4)

Generated by `analysis/step4_backtest.py` (models: `src/forecasting.py`, harness emulation and metrics:
`src/backtest.py`, report: `analysis/step4_report.py`). Run time {R['elapsed'] / 60:.1f} min.

## 1. Set-up

**Target**: raw daily admission rows by resolved D.O.A (duplicates kept), exactly as the harness counts them.

**Harness emulation.** For each cut-off c a model receives only (`DataStore.context`):
admission counts for days < c; weather for days < c; pollution for days < c, cleaned with the Step 2 rules applied to
the *truncated* file (so short-gap interpolation can never borrow a post-cut-off value); and the complete festival
calendar (HARNESS.md: it is supplied complete). Truth is read separately for scoring. Tests perturb post-cut-off data
and confirm forecasts do not change.

**Cut-offs.** Every 4 days (so every weekday occurs as a cut-off day), {len(R['cutoffs'])} in total, all with the full
14-day window inside DEV (first {R['cutoffs'].min().date()}, last {R['cutoffs'].max().date()}). **Jan-Mar 2019 is not used anywhere.**
* **Tuning** ({len(R['tune_c'])} cut-offs, {R['tune_c'].min().date()}..{R['tune_c'].max().date()}): every configuration is run; the best configuration of
  each family, the interval quantile level and the residual pool size are chosen here.
* **Validation** ({len(R['val_c'])} cut-offs, {R['val_c'].min().date()}..{R['val_c'].max().date()}): the chosen configurations are compared once.

**Baseline - exact HARNESS.md definition.** "The value of the same weekday in the last observed week." Implementation
(`forecasting.seasonal_naive`): the last observed week is c-7..c-1; forecast day d takes the value on
c-7 + ((d - (c-7)) mod 7) days, i.e. day c and c+7 both take the value of c-7, day c+6 and c+13 the value of c-1.

**Metrics.** MAE, RMSE, WAPE (sum|error| / sum actual), MAE for horizons 1-7 and 8-14, 80% interval coverage and
mean width. Uncertainty of MAE ratios: block bootstrap over consecutive cut-offs (blocks of 4 = 16 days, because
14-day windows from neighbouring cut-offs overlap).

**Exact harness rules vs approximations.**
| Item | Status |
|---|---|
| Error points `25 * clip(1 - ratio/2, 0, 1)` | exact (from HARNESS.md) |
| Coverage points: 15 at >= 80%, linear to 0 at 40% | exact |
| Whether MAE ratio is pooled over all 56 values or averaged per cut-off | **unknown** - both reported |
| "Baseline interval" used for the width penalty | **not defined in HARNESS.md** - approximated as the seasonal-naive forecast with the same empirical 10-90% residual method; penalty factor approximated as min(1, 2 x baseline width / width) |

## 2. Candidate families and tuning

| Family | Model | Configurations searched |
|---|---|---|
| B0 | Seasonal naive (harness baseline) | 1 |
| B1 | Recent rolling average | window 7/14/28/56 |
| M1 | Recent level x weekday profile | level window 7..112, weekday window 56/91/182/364 |
| M2 | M1 + holiday / eve / after multipliers estimated from history | same grid |
| M3 | Year-ago 7-day shape x current level (falls back to M1 with < 1 year) | level window 14/28/56 |
| M4 | Poisson GLM (numpy IRLS): weekday + holiday/eve/after (+ annual Fourier K=2, + trend, + recent-level rescale) | training 182/364/all x K 0/2 x trend x level rescale 0/14/28 |
| M5 | Holt-Winters on log counts, weekly season, damped trend (grid-fitted on one-step error) | training 182/364/all |
| M6 | Equal-weight mean of the M2 and M4 winners (defined before validation, no weights tuned) | 1 |
| E | M4 winner + 7-day mean of CO / PM2.5 / temperature / all three, lagged 14 or 21 days (known at cut-off) | 8 |
| X | Same, but with **same-day future observations (oracle, not deployable)** - upper bound on what environment could add | 4 |

Top configurations per family on the tuning cut-offs (MAE ratio vs seasonal naive):
{_md(tune_top.set_index('family'), '{:.3f}')}

Family winners: {', '.join(f'{k} = `{v}`' for k, v in w.items())}.

## 3. Validation results (the comparison that counts)

{_md(vt[vcols], '{:.3f}')}

![ratio](figures/fig01_validation_ratio.png)

**Lowest validation MAE**: {best} (`{bestm}`), MAE {vt.loc[best, 'MAE']:.2f} = {vt.loc[best, 'ratio_vs_B0']:.2f} x baseline
(95% CI {vt.loc[best, 'ratio_lo']:.2f}-{vt.loc[best, 'ratio_hi']:.2f}).

Comparison with the best model (ratio of MAEs, block-bootstrap 95% CI):
{_md(R['within'], '{:.3f}')}

**Recommended for Step 5: {rec} (`{recm}`)** - the simplest family whose validation MAE is statistically
indistinguishable from the best (CI of the ratio includes 1). Its metrics on validation: MAE {vt.loc[rec, 'MAE']:.2f},
RMSE {vt.loc[rec, 'RMSE']:.2f}, WAPE {100 * vt.loc[rec, 'WAPE']:.1f}%, MAE h1-7 {vt.loc[rec, 'MAE_h1_7']:.2f}, h8-14
{vt.loc[rec, 'MAE_h8_14']:.2f}; ratio to baseline {vt.loc[rec, 'ratio_vs_B0']:.2f} (95% CI {vt.loc[rec, 'ratio_lo']:.2f}-{vt.loc[rec, 'ratio_hi']:.2f});
80% coverage {100 * vt.loc[rec, 'coverage80']:.1f}% with mean width {vt.loc[rec, 'width80']:.1f} admissions.

**Consistency between periods.** {consistency}
Tuning ratios are worse overall because the tuning period contains the volatile Jan 2018 peak and the start of duplication.

**Caveat for the hidden Jan-Mar window.** On the Jan-Mar 2018 cut-offs (closest analogue; part of tuning, n = {int(jm.loc[BASE, 'n_cutoffs'])}),
the GLM and ensemble did better than the recommended model:
{_md(R['jm_ci'].set_index(['model', 'reference']), '{:.3f}')}
{jm_note} The recommendation above follows the pre-specified "simplest within noise on validation" rule; the
choice between {rec} and the GLM-based models for the Jan-Mar scoring window is left as an explicit Step 5 decision.

## 4. Horizons

![h](figures/fig02_mae_by_horizon.png)

MAE by horizon on validation (`tables/mae_by_horizon_validation.csv`). The recommended model is flat across the 14
days (h1-7 {vt.loc[rec, 'MAE_h1_7']:.2f} vs h8-14 {vt.loc[rec, 'MAE_h8_14']:.2f}); the baseline degrades in week 2
(h1-7 {vt.loc['B0', 'MAE_h1_7']:.2f} vs h8-14 {vt.loc['B0', 'MAE_h8_14']:.2f}).

## 5. Stability across seasons and admission levels

All cut-offs (tuning + validation; winners were selected on tuning, so tuning-period numbers are optimistic).
MAE ratio vs seasonal naive:
{_md(season, '{:.3f}')}

By the admission level of the 28 days before the cut-off (terciles):
{_md(level, '{:.3f}')}

Jan-Mar 2018 cut-offs (closest analogue of the hidden Jan-Mar 2019 cut-offs; only ~9-12 months of history then):
{_md(jm[['MAE', 'WAPE', 'MAE_h1_7', 'MAE_h8_14', 'ratio_vs_B0', 'n_cutoffs']], '{:.3f}')}

![cut](figures/fig03_mae_by_cutoff.png)

{season_note}

## 6. Environmental features (experimental)

Paired comparison against the identical GLM without environment (ratio < 1 = environment helps):
{_md(env_tab, '{:.3f}')}

![env](figures/fig05_environment_effect.png)

* **Deployable environmental features {'do' if env_helps else 'do not'} improve the forecast**: no deployable variant has a
  validation CI entirely below 1{'' if not env_helps else ' (see table)'}. Tuning-period gains (e.g. PM2.5 lag 14:
  {env_tab.loc['E_pm25_lag14', 'tuning']:.3f}) did not carry over to validation ({env_tab.loc['E_pm25_lag14', 'validation']:.3f}).
* **Even perfect knowledge of the future environment {'helps' if oracle_helps else 'does not help'}**: the oracle variants,
  which use the actual same-day future observations, are within noise of the plain GLM. This is the strongest available
  evidence that weather and air quality add no forecasting value beyond weekday, calendar and recent level here -
  consistent with Step 3 (no reliable short-term environmental association).
* Environmental features therefore stay in the analysis and the what-if planning view, not in the scored forecast.

## 7. Prediction intervals

**Construction** (`backtest.add_intervals`, pandas/numpy only):
1. For cut-off c, take the model's own past forecasts whose entire 14-day window ended before c (origin <= c-14);
   keep the most recent `pool` origins.
2. Residual r = log((actual+1) / (forecast+1)) in three horizon buckets (days 1-3, 4-7, 8-14).
3. Interval = (forecast+1) x exp(q_lo, q_hi of r in the bucket) - 1, clipped at 0 and forced to contain the forecast.
4. Quantile level and pool size were chosen on **tuning** cut-offs only: the narrowest setting with tuning coverage >= 80%
   (or the highest coverage if none reached 80%). Intervals were never narrowed to meet the approximate width cap.

At runtime the scored file must rebuild the pool itself by back-forecasting at `pool` internal origins from its own
history; runtime cost estimates are in section 9.

Chosen settings and tuning coverage:
{_md(civ[['q_lo', 'q_hi', 'pool', 'tune_coverage', 'tune_width']], '{:.3f}')}

Validation coverage and width by horizon:
{_md(R['iv_bucket'].set_index(['model', 'horizon']), '{:.3f}')}

![iv](figures/fig04_interval_coverage_width.png)

Baseline interval width on validation (approximation): {R['base_w_val']:.1f}; twice that = {2 * R['base_w_val']:.1f}.
All candidate intervals are far narrower than the approximate cap. Validation coverage by model:
{', '.join(f"{f} {100 * vt.loc[f, 'coverage80']:.0f}%" for f in vt.index)}
({'all' if (vt['coverage80'] >= 0.80).all() else 'not all'} at or above 80%). Coverage is conservative, most of all for the
GLM-based models (M4/M6/E, ~92%), whose tuning-chosen pool of 90 origins reaches back into the volatile winter of
2017-18. Under-coverage costs far more than mild over-coverage (coverage points are capped at 80%), so conservative
intervals are accepted; the width cost is visible in the table (M1 is the narrowest calibrated interval).

![ex](figures/fig06_example_forecasts.png)

## 8. Simulated harness score (validation cut-offs, 4 random cut-offs per draw, 5000 draws)

{_md(sc.drop(columns=['model']), '{:.2f}')}

![score](figures/fig07_simulated_score.png)

Error points use the exact rule; the width factor uses the approximation in section 1. The baseline scores 12.5 error
points by definition.

## 9. Runtime (per forecast call, this machine)

{_md(rt, '{:.4f}')}

`internal_backtest_seconds_est` = time to rebuild the interval residual pool at runtime (pool x one fit), excluding
data loading (~1 s). Every candidate is far below the 15-minute limit, so the number of internal origins is not
runtime-constrained; it is set by calibration (section 7).

## 10. Limitations

* Validation covers Jul-Dec 2018 ({len(R['val_c'])} overlapping cut-offs, about 24 independent fortnights). The hidden
  cut-offs are Jan-Mar 2019, a season observed only once before (Jan-Mar 2018, with < 1 year of history).
* Differences between the top weekday/GLM models are within bootstrap noise; the ranking among them is not reliable.
* The raw target includes a time-varying duplicate rate (Step 2/3); models that track the recent level absorb it, but a
  sudden change in duplication during the hidden window would appear as forecast error.
* The width penalty and the error-ratio pooling are approximations of unstated harness details.

## Files

Tables (`outputs/backtest/tables/`): {', '.join(tables)}

Figures (`outputs/backtest/figures/`): {', '.join(figs)}

Forecast archive: `outputs/backtest/forecasts_selected_models.csv.gz` (every selected model x cut-off x horizon with
mean, interval and truth).
"""
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(txt)
