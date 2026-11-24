import clsx from "clsx";
import { ChevronLeft, ChevronRight, FlaskConical, History, LineChart as LineIcon, Lock, Sigma } from "lucide-react";
import { useEffect, useId, useState, type ReactNode } from "react";
import {
  AnnualCurve, DiagnosisChart, HorizonCoverageChart, HorizonErrorChart, PerCutoffRatioChart, RetroForecastChart, WeekdayMultipliers,
} from "../components/charts/StudioCharts";
import { Badge, ProvenanceBadge } from "../components/ui/Badge";
import { Panel } from "../components/ui/Panel";
import { ErrorState, Loading } from "../components/ui/States";
import { BiasDisclosure, CoverageHistory, IntervalMeaning, Note, Stat } from "../components/forecast/ForecastNotes";
import { MlComparisonPanel } from "../components/forecast/MlComparison";
import { ReliabilityCard } from "../components/forecast/RunParts";
import { useForecastRun, useHoldout, useModelExplain, useRetroForecast } from "../lib/api";
import { fmt1, fmtDate, fmtDayMonth, fmtLong } from "../lib/format";
import { hrefFor } from "../lib/router";
import type { HoldoutMetricSet, IntervalEvidence } from "../lib/types";

type View = "retro" | "holdout";
const RETRO_MIN = "2017-07-01";
const RETRO_MAX = "2018-12-15";
const pct = (v: number) => `${(v * 100).toFixed(1)}%`;
const signed = (v: number) => `${v > 0 ? "+" : ""}${fmt1(v)}`;
const addDays = (iso: string, n: number) => {
  const d = new Date(`${iso}T00:00:00Z`);
  d.setUTCDate(d.getUTCDate() + n);
  return d.toISOString().slice(0, 10);
};
const clampRetro = (iso: string) => (iso < RETRO_MIN ? RETRO_MIN : iso > RETRO_MAX ? RETRO_MAX : iso);

/** Where the holdout misses fell (from the saved holdout forecasts, via the API). */
function MissBreakdown({ ev }: { ev: IntervalEvidence }) {
  const m = ev.holdout_misses;
  const parts = [
    { key: "below", label: "Below lower bound", value: m.below_lower, cls: "bg-observed" },
    { key: "inside", label: "Inside the interval", value: m.inside, cls: "bg-forecast/60" },
    { key: "above", label: "Above upper bound", value: m.above_upper, cls: "bg-amber" },
  ];
  return (
    <div className="mt-4 border-t border-border/15 pt-4">
      <p className="text-[12.5px] font-semibold text-text">Where the {m.points} holdout forecast-days fell</p>
      <div className="mt-2 flex h-3 w-full overflow-hidden rounded-full bg-surface-2" role="img"
        aria-label={parts.map((x) => `${x.label} ${pct(x.value)}`).join(", ")}>
        {parts.map((x) => <span key={x.key} className={x.cls} style={{ width: `${x.value * 100}%` }} />)}
      </div>
      <ul className="mt-2 flex flex-wrap gap-x-5 gap-y-1 text-[12px] text-text-2">
        {parts.map((x) => (
          <li key={x.key} className="flex items-center gap-1.5">
            <span aria-hidden className={clsx("h-2.5 w-2.5 rounded-sm", x.cls)} />{x.label}: <span className="tabular font-semibold text-text">{pct(x.value)}</span>
          </li>
        ))}
      </ul>
      <p className="mt-2 text-[12px] leading-5 text-text-2">
        The bounds are the central 85% of past errors: by construction, 7.5% of those past errors lay below the lower bound and 7.5% above
        the upper bound, so a well-centred band would miss roughly evenly on both sides. In 2019 almost all misses were above the band,
        reflecting the under-forecast. Coverage by stage (target at least 80%):
      </p>
      <div className="mt-2"><CoverageHistory ev={ev} compact /></div>
    </div>
  );
}

// ============================================================== model explainer
function ModelExplainer() {
  const ex = useModelExplain();
  return (
    <Panel title="How the M4 model forecasts" className="mb-6"
      subtitle="A Poisson regression on daily raw records. These multipliers are the frozen model's own fitted values for cut-off 1 Apr 2019 and reproduce its forecast exactly."
      actions={<><Badge tone="forecast" icon={Sigma}>Frozen forecast.py</Badge><Badge tone="neutral" icon={Lock}>No weather · no air quality</Badge></>}>
      {ex.isError ? <ErrorState title="Model explanation unavailable" message={(ex.error as Error).message} onRetry={() => ex.refetch()} />
        : !ex.data ? <Loading label="Reading the frozen model" /> : (
          <>
            <div className="grid grid-cols-1 gap-5 lg:grid-cols-3">
              <div>
                <h3 className="text-[13px] font-semibold text-text">1 · Weekday effect</h3>
                <p className="mb-2 text-[12px] text-text-2">Multiplier vs Monday. Sunday sees about {Math.round((1 - ex.data.weekday[6].multiplier) * 100)}% fewer records.</p>
                <WeekdayMultipliers data={ex.data} />
              </div>
              <div>
                <h3 className="text-[13px] font-semibold text-text">2 · Holidays</h3>
                <p className="mb-3 text-[12px] text-text-2">Punjab festival and holiday calendar (known in advance).</p>
                <ul className="space-y-2.5">
                  {ex.data.calendar.map((t) => (
                    <li key={t.term} className="rounded-xl bg-surface-2/60 px-3.5 py-2.5 ring-1 ring-inset ring-border/15">
                      <div className="flex items-baseline justify-between">
                        <span className="text-[12.5px] font-medium text-text">{t.term}</span>
                        <span className="tabular text-[15px] font-semibold text-text">×{t.multiplier.toFixed(2)}</span>
                      </div>
                      <div aria-hidden className="mt-2 h-1.5 rounded-full bg-surface">
                        <div className="h-1.5 rounded-full bg-forecast/70" style={{ width: `${Math.min(100, (t.multiplier / 1.2) * 100)}%` }} />
                      </div>
                      <p className="mt-1 text-[11.5px] text-text-2">
                        {t.multiplier < 1 ? `${Math.round((1 - t.multiplier) * 100)}% fewer records` : `${Math.round((t.multiplier - 1) * 100)}% more records`} than a comparable day
                      </p>
                    </li>
                  ))}
                </ul>
              </div>
              <div>
                <h3 className="text-[13px] font-semibold text-text">3 · Annual seasonality</h3>
                <p className="mb-2 text-[12px] text-text-2">Two annual harmonics; shaded = forecast window (1–14 Apr).</p>
                <AnnualCurve data={ex.data} />
              </div>
            </div>
            <p className="mt-4 text-[12px] leading-5 text-text-3">
              Trained on {ex.data.training.days} days ({fmtDate(ex.data.training.start)} – {fmtDate(ex.data.training.end)}), ridge penalty {ex.data.ridge}.
              {" "}{ex.data.note}
            </p>
          </>
        )}
    </Panel>
  );
}

// ====================================================================== views
function RetroView() {
  const [draft, setDraft] = useState("2018-10-02");
  const [cutoff, setCutoff] = useState("2018-10-02");
  const [showDedup, setShowDedup] = useState(false);
  const [showBaseline, setShowBaseline] = useState(true);
  const inputId = useId();
  const valid = /^\d{4}-\d{2}-\d{2}$/.test(draft) && draft >= RETRO_MIN && draft <= RETRO_MAX;
  useEffect(() => {
    if (valid) setCutoff(draft);
  }, [draft, valid]);
  const q = useRetroForecast(cutoff);
  const m = q.data?.metrics;
  const presets = [
    { label: "Winter 2017–18", v: "2018-01-10" }, { label: "Spring 2018", v: "2018-04-20" },
    { label: "Monsoon 2018", v: "2018-08-01" }, { label: "Post-monsoon 2018", v: "2018-11-05" }, { label: "Latest allowed", v: RETRO_MAX },
  ];
  return (
    <div className="space-y-6">
      <Panel title="Choose a development-period cut-off"
        subtitle={`Any date from ${fmtDate(RETRO_MIN)} to ${fmtDate(RETRO_MAX)}. Later cut-offs would reach the Jan–Mar 2019 holdout and are blocked.`}
        actions={<Badge tone="amber" icon={FlaskConical}>Retrospective · development period</Badge>}>
        <div className="flex flex-wrap items-end gap-3">
          <div>
            <label htmlFor={inputId} className="block text-[12px] font-medium text-text-2">Cut-off date</label>
            <div className="mt-1 flex items-center gap-1.5">
              <button type="button" aria-label="One week earlier" onClick={() => setDraft(clampRetro(addDays(cutoff, -7)))}
                className="grid h-9 w-9 place-items-center rounded-lg bg-surface-2 text-text-2 ring-1 ring-inset ring-border/20 hover:text-text"><ChevronLeft className="h-4 w-4" /></button>
              <input id={inputId} type="date" min={RETRO_MIN} max={RETRO_MAX} value={draft} onChange={(e) => setDraft(e.target.value)}
                aria-invalid={!valid} aria-describedby={`${inputId}-hint`}
                className="h-9 rounded-lg bg-surface-2 px-3 text-[13px] font-medium text-text ring-1 ring-inset ring-border/25 focus:ring-observed/60 [color-scheme:inherit]" />
              <button type="button" aria-label="One week later" onClick={() => setDraft(clampRetro(addDays(cutoff, 7)))}
                className="grid h-9 w-9 place-items-center rounded-lg bg-surface-2 text-text-2 ring-1 ring-inset ring-border/20 hover:text-text"><ChevronRight className="h-4 w-4" /></button>
            </div>
          </div>
          <div className="flex flex-wrap gap-1.5" role="group" aria-label="Example cut-offs">
            {presets.map((p) => (
              <button key={p.v} type="button" onClick={() => setDraft(p.v)} aria-pressed={cutoff === p.v}
                className={clsx("rounded-lg px-2.5 py-1.5 text-[12px] font-medium ring-1 ring-inset transition-colors",
                  cutoff === p.v ? "bg-surface text-text ring-observed/40" : "bg-surface-2/70 text-text-2 ring-border/20 hover:text-text")}>
                {p.label}
              </button>
            ))}
          </div>
        </div>
        <p id={`${inputId}-hint`} className={clsx("mt-2 text-[12px]", valid ? "text-text-3" : "font-semibold text-amber-ink")}>
          {valid ? `Showing cut-off ${fmtLong(cutoff)} · forecast window ${fmtDayMonth(cutoff)} – ${fmtDate(addDays(cutoff, 13))}.`
            : `Choose a date between ${fmtDate(RETRO_MIN)} and ${fmtDate(RETRO_MAX)}; the 2019 holdout cannot be explored.`}
        </p>
      </Panel>

      {q.isError ? <ErrorState title="Retrospective forecast unavailable" message={(q.error as Error).message} onRetry={() => q.refetch()} />
        : !q.data ? <Loading label="Running the frozen model for this cut-off" /> : (
          <>
            <div className="grid grid-cols-2 gap-3 md:grid-cols-3 xl:grid-cols-6">
              <Stat label="MAE (M4)" value={fmt1(m!.MAE)} sub="records/day vs raw actuals" />
              <Stat label="MAE (baseline)" value={fmt1(m!.MAE_baseline)} sub="seasonal naive" />
              <Stat label="Ratio vs baseline" value={m!.MAE_ratio_vs_baseline?.toFixed(2) ?? "–"} sub="below 1 = better" />
              <Stat label="Bias" value={signed(m!.bias)} sub="mean forecast − actual" tone={Math.abs(m!.bias) > 3 ? "warn" : "neutral"} />
              <Stat label="Inside 80% interval" value={`${m!.days_inside}/14`} sub={pct(m!.coverage80)} />
              <Stat label="WAPE" value={pct(m!.WAPE)} sub={`RMSE ${fmt1(m!.RMSE)}`} />
            </div>
            <Panel title={`Retrospective forecast · cut-off ${fmtDate(q.data.cutoff)}`}
              subtitle="Forecast made with data before the cut-off only; actual raw records for the window are revealed afterwards for comparison. The model was already selected – this view is not used for tuning."
              actions={<>
                <Badge tone="amber" icon={FlaskConical}>Retrospective</Badge>
                <label className="inline-flex cursor-pointer items-center gap-1.5 text-[12px] font-medium text-text-2">
                  <input type="checkbox" className="accent-[rgb(var(--c-observed))]" checked={showBaseline} onChange={(e) => setShowBaseline(e.target.checked)} />
                  Baseline
                </label>
                <label className="inline-flex cursor-pointer items-center gap-1.5 text-[12px] font-medium text-text-2">
                  <input type="checkbox" className="accent-[rgb(var(--c-dedup))]" checked={showDedup} onChange={(e) => setShowDedup(e.target.checked)} />
                  De-duplicated actuals (sensitivity)
                </label>
              </>}>
              <RetroForecastChart data={q.data} showDedup={showDedup} showBaseline={showBaseline} />
              {q.isFetching && <p role="status" className="mt-2 text-[12px] text-text-3">Updating…</p>}
            </Panel>
            <Note title="How to read this view">
              Metrics are scored against <strong className="text-text">raw records</strong>, the forecast target. De-duplicated actuals are an optional
              sensitivity overlay and are never combined with raw counts. Individual 14-day windows are noisy; the holdout tab summarises
              performance over many cut-offs.
            </Note>
          </>
        )}
    </div>
  );
}

function MetricCompare({ label, model, baseline, fmt, lowerBetter = true, note }: {
  label: string; model: number; baseline: number; fmt: (v: number) => string; lowerBetter?: boolean; note?: string;
}) {
  const better = lowerBetter ? model < baseline : model > baseline;
  return (
    <div className="rounded-xl bg-surface-2/60 px-4 py-3 ring-1 ring-inset ring-border/15">
      <p className="text-[12px] font-medium text-text-2">{label}</p>
      <div className="mt-1 flex items-baseline gap-2">
        <span className="tabular text-[22px] font-semibold text-text">{fmt(model)}</span>
        <span className="text-[11.5px] text-text-2">M4</span>
      </div>
      <p className="tabular text-[12px] text-text-2">Baseline {fmt(baseline)} · <span className={better ? "font-semibold text-ok-ink" : "font-semibold text-amber-ink"}>{better ? "better" : "worse"}</span></p>
      {note && <p className="mt-1 text-[11.5px] text-text-3">{note}</p>}
    </div>
  );
}

function HoldoutView() {
  const q = useHoldout();
  if (q.isError) return <ErrorState title="Holdout results unavailable" message={(q.error as Error).message} onRetry={() => q.refetch()} />;
  if (!q.data) return <Loading label="Loading saved holdout results" />;
  const d = q.data;
  const p = d.metrics.primary;
  const [ratio, lo, hi] = d.ratio_ci.primary;
  const sets: [string, string, { model: HoldoutMetricSet; baseline: HoldoutMetricSet }][] = [
    ["Primary (Step 4 grid)", "19 cut-offs, windows end by 30 Mar", d.metrics.primary],
    ["Every daily cut-off", "77 cut-offs; includes potentially incomplete 31 Mar", d.metrics.daily],
    ["Daily, excluding 31 Mar windows", "sensitivity", d.metrics.daily_excl_last_day],
  ];
  return (
    <div className="space-y-6">
      <Note icon={Lock} title="Saved, read-only evaluation">
        These numbers come from the saved Jan–Mar 2019 evaluation of the frozen forecast.py (Step 5). This period is
        <strong className="text-text"> previously viewed in the M4 holdout context</strong> – it is not an untouched holdout – and it was not used
        to select the newer candidate models. The numbers are read from saved files;
        nothing on this page reruns or tunes the model. {d.sha256_matches_current ? "The forecast.py evaluated then is byte-identical to the one serving forecasts now." : ""}
      </Note>
      <div className="grid grid-cols-2 gap-3 md:grid-cols-3 xl:grid-cols-6">
        <MetricCompare label="MAE" model={p.model.MAE} baseline={p.baseline.MAE} fmt={fmt1} />
        <MetricCompare label="RMSE" model={p.model.RMSE} baseline={p.baseline.RMSE} fmt={fmt1} />
        <MetricCompare label="WAPE" model={p.model.WAPE} baseline={p.baseline.WAPE} fmt={pct} />
        <Stat label="MAE ratio vs baseline" value={ratio.toFixed(2)} sub={`95% CI ${lo.toFixed(2)}–${hi.toFixed(2)} (includes 1)`} />
        <Stat label="80% interval coverage" value={pct(p.model.coverage80)}
          sub={`target ≥80% · misses ${pct(d.interval_evidence.holdout_misses.above_upper)} above, ${pct(d.interval_evidence.holdout_misses.below_lower)} below`} />
        <Stat label="Bias (forecast − actual)" value={signed(p.model.bias)} sub="records/day · under-forecast" tone="warn" />
      </div>

      <div className="grid grid-cols-1 gap-6 xl:grid-cols-2">
        <Panel title="Error by forecast day" subtitle="Mean absolute error across the 19 holdout cut-offs" actions={<ProvenanceBadge kind="evaluation" />}>
          <HorizonErrorChart data={d} />
          <p className="mt-2 text-[12px] text-text-2">Days 1–7: MAE {fmt1(p.model.MAE_h1_7)} (baseline {fmt1(p.baseline.MAE_h1_7)}) · days 8–14: {fmt1(p.model.MAE_h8_14)} (baseline {fmt1(p.baseline.MAE_h8_14)}).</p>
        </Panel>
        <Panel title="Interval coverage by forecast day" subtitle="Share of actual days inside the 80% interval (19 values per bar)" actions={<ProvenanceBadge kind="evaluation" />}>
          <HorizonCoverageChart data={d} />
          <p className="mt-2 text-[12px] text-text-2">Days 1–7: {pct(p.model.coverage80_h1_7)} · days 8–14: {pct(p.model.coverage80_h8_14)}. With only 19 values per day, single bars are noisy.</p>
          <MissBreakdown ev={d.interval_evidence} />
        </Panel>
      </div>

      <div className="grid grid-cols-1 gap-6 xl:grid-cols-2">
        <Panel title="Each holdout cut-off" subtitle="MAE ratio vs baseline (below 1 = M4 better; amber = worse than baseline)" actions={<ProvenanceBadge kind="evaluation" />}>
          <PerCutoffRatioChart data={d} />
        </Panel>
        <Panel title="By cut-off month" subtitle="All daily cut-offs in each month (sensitivity set)" actions={<ProvenanceBadge kind="evaluation" />}>
          <div className="overflow-x-auto">
            <table className="w-full min-w-[460px] text-[13px]">
              <caption className="sr-only">Holdout performance by cut-off month</caption>
              <thead><tr className="text-left text-[11.5px] font-semibold uppercase tracking-wide text-text-3">
                {["Month", "Cut-offs", "MAE", "Baseline", "Ratio", "Coverage"].map((h) => <th key={h} scope="col" className="border-b border-border/15 px-3 py-2">{h}</th>)}
              </tr></thead>
              <tbody>
                {d.by_month.map((r) => (
                  <tr key={r.cutoff_month}>
                    <td className="border-b border-border/10 px-3 py-2 font-medium text-text">{r.cutoff_month}</td>
                    <td className="tabular border-b border-border/10 px-3 py-2 text-text-2">{r.n_cutoffs}</td>
                    <td className="tabular border-b border-border/10 px-3 py-2 text-text">{fmt1(r.MAE)}</td>
                    <td className="tabular border-b border-border/10 px-3 py-2 text-text-2">{fmt1(r.MAE_B0)}</td>
                    <td className="tabular border-b border-border/10 px-3 py-2 font-semibold text-text">{r.ratio.toFixed(2)}</td>
                    <td className="tabular border-b border-border/10 px-3 py-2 text-text-2">{pct(r.coverage80)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="mt-3 text-[12px] text-text-3">The holdout covers only three months, so this is the only seasonal split available.</p>
        </Panel>
      </div>

      <Panel title="Why the model under-forecast 2019" subtitle="Monthly average of the 14-day windows: forecast vs actual raw and de-duplicated records (shown separately)"
        actions={<ProvenanceBadge kind="evaluation" />}>
        <DiagnosisChart data={d} />
        <div className="mt-4 grid grid-cols-1 gap-3 md:grid-cols-3">
          {d.diagnosis_by_month.map((m) => (
            <div key={m.month} className="rounded-xl bg-surface-2/60 px-3.5 py-3 text-[12px] leading-5 text-text-2 ring-1 ring-inset ring-border/15">
              <p className="font-semibold text-text">{new Date(`${m.month}-01T00:00:00`).toLocaleDateString("en-GB", { month: "long", year: "numeric" })}</p>
              Gap vs raw <span className="tabular font-semibold text-text">{fmt1(m.gap_raw)}</span>/day · vs de-duplicated{" "}
              <span className="tabular font-semibold text-text">{signed(m.gap_dedup)}</span> · repeat rows {fmt1(m.repeat_rows_pct)}%
            </div>
          ))}
        </div>
        <p className="mt-3 text-[12px] leading-5 text-text-3">
          January's gap is mostly genuine extra patients; February–March's is mostly repeat rows. This is a descriptive decomposition – the model
          forecasts raw records, and no alternative model was evaluated on the holdout.
        </p>
      </Panel>

      <div className="grid grid-cols-1 gap-6 xl:grid-cols-2">
        <Panel title="Tuning, validation and holdout are different evidence" subtitle="The Jan–Mar 2019 row is previously viewed (M4 holdout context); it was not used to select newer models">
          <div className="overflow-x-auto">
            <table className="w-full min-w-[480px] text-[13px]">
              <caption className="sr-only">Model performance at each development stage</caption>
              <thead><tr className="text-left text-[11.5px] font-semibold uppercase tracking-wide text-text-3">
                {["Stage", "Cut-offs", "MAE", "Ratio", "Coverage"].map((h) => <th key={h} scope="col" className="border-b border-border/15 px-3 py-2">{h}</th>)}
              </tr></thead>
              <tbody>
                {d.stages.map((s) => (
                  <tr key={s.stage} className={s.stage.includes("HOLDOUT") ? "bg-forecast/[0.08]" : ""}>
                    <td className="border-b border-border/10 px-3 py-2 text-text">{s.stage}</td>
                    <td className="tabular border-b border-border/10 px-3 py-2 text-text-2">{s.cutoffs}</td>
                    <td className="tabular border-b border-border/10 px-3 py-2 text-text-2">{fmt1(s.MAE)}</td>
                    <td className="tabular border-b border-border/10 px-3 py-2 font-semibold text-text">{s.ratio_vs_B0.toFixed(2)}</td>
                    <td className="tabular border-b border-border/10 px-3 py-2 text-text-2">{s.coverage80 == null ? "—" : pct(s.coverage80)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Panel>
        <Panel title="Robustness of the holdout numbers" subtitle="Same frozen forecasts, different cut-off sets">
          <div className="overflow-x-auto">
            <table className="w-full min-w-[480px] text-[13px]">
              <caption className="sr-only">Holdout metrics for alternative cut-off sets</caption>
              <thead><tr className="text-left text-[11.5px] font-semibold uppercase tracking-wide text-text-3">
                {["Cut-off set", "MAE", "Ratio", "Bias", "Coverage"].map((h) => <th key={h} scope="col" className="border-b border-border/15 px-3 py-2">{h}</th>)}
              </tr></thead>
              <tbody>
                {sets.map(([name, sub, s]) => (
                  <tr key={name}>
                    <td className="border-b border-border/10 px-3 py-2"><p className="text-text">{name}</p><p className="text-[11.5px] text-text-3">{sub}</p></td>
                    <td className="tabular border-b border-border/10 px-3 py-2 text-text-2">{fmt1(s.model.MAE)}</td>
                    <td className="tabular border-b border-border/10 px-3 py-2 font-semibold text-text">{s.model.MAE_ratio_vs_B0.toFixed(2)}</td>
                    <td className="tabular border-b border-border/10 px-3 py-2 text-text-2">{signed(s.model.bias)}</td>
                    <td className="tabular border-b border-border/10 px-3 py-2 text-text-2">{pct(s.model.coverage80)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Panel>
      </div>
      <ul className="list-disc space-y-1 pl-5 text-[12px] leading-5 text-text-3">{d.notes.map((n) => <li key={n}>{n}</li>)}</ul>
    </div>
  );
}

// ======================================================================= page
const TABS: { id: View; label: string; icon: typeof History; badge: ReactNode; blurb: string }[] = [
  { id: "retro", label: "Retrospective explorer", icon: FlaskConical, badge: <Badge tone="amber" icon={FlaskConical}>Retrospective</Badge>,
    blurb: "Development-period cut-offs (Jul 2017 – Dec 2018) with actual outcomes revealed for comparison." },
  { id: "holdout", label: "Holdout evaluation", icon: LineIcon, badge: <ProvenanceBadge kind="evaluation" />,
    blurb: "Saved Jan–Mar 2019 results for M4 – previously viewed in the M4 holdout context. Read-only." },
];

/** Internal: the plain-language holdout summary (moved off the client pages). */
function InternalReliability() {
  const run = useForecastRun();
  return (
    <Panel className="mb-6" title="Reliability summary (M4, Jan–Mar 2019 holdout – previously viewed)" actions={<ProvenanceBadge kind="evaluation" />}>
      {run.data ? <ReliabilityCard run={run.data} /> : <Loading label="Loading reliability" />}
    </Panel>
  );
}

export default function ModelPerformance() {
  const initial = (new URLSearchParams(window.location.search).get("view") as View) || "retro";
  const [view, setView] = useState<View>(TABS.some((t) => t.id === initial) ? initial : "retro");
  const base = useId();
  return (
    <div>
      <section className="panel relative mb-6 overflow-hidden p-6 animate-fade-up" aria-labelledby="studio-title">
        <div aria-hidden className="pointer-events-none absolute -right-16 -top-24 h-72 w-72 rounded-full bg-forecast opacity-[0.14] blur-3xl" />
        <div className="relative max-w-3xl">
          <div className="flex flex-wrap gap-2"><Badge tone="forecast" icon={Sigma}>M4 Poisson GLM · frozen</Badge><Badge tone="neutral">Raw records target</Badge></div>
          <h2 id="studio-title" className="mt-3 text-2xl font-semibold tracking-tight text-text">
            Model <span className="bg-gradient-to-r from-forecast-ink to-observed-ink bg-clip-text text-transparent">performance</span>
          </h2>
          <p className="mt-2 text-[14px] leading-6 text-text-2">
            How the frozen model behind the 14-day forecast works and how well it has done: a retrospective explorer for the 2017–18
            development period (actual outcomes revealed for comparison) and the saved Jan–Mar 2019 results (previously viewed in the M4 holdout context; not used to select newer models).
            The 1–14 Apr 2019 forecast itself is on the{" "}
            <a className="font-semibold text-observed-ink underline-offset-2 hover:underline" href={hrefFor("/forecast")}>14-Day Forecast</a> page.
          </p>
        </div>
      </section>

      <ModelExplainer />

      <MlComparisonPanel />

      <InternalReliability />

      <div className="mb-6 grid grid-cols-1 gap-4 lg:grid-cols-2"><IntervalMeaning /><BiasDisclosure /></div>

      <div role="tablist" aria-label="Model performance views" className="mb-5 grid grid-cols-1 gap-2 lg:grid-cols-2">
        {TABS.map((t) => {
          const active = t.id === view;
          const Icon = t.icon;
          return (
            <button key={t.id} id={`${base}-tab-${t.id}`} role="tab" type="button" aria-selected={active} aria-controls={`${base}-panel-${t.id}`}
              onClick={() => setView(t.id)}
              className={clsx("panel flex flex-col gap-2 p-4 text-left transition-all",
                active ? "ring-2 ring-inset ring-observed/45" : "opacity-80 hover:opacity-100")}>
              <span className="flex flex-wrap items-center justify-between gap-2">
                <span className="flex items-center gap-2 text-[14px] font-semibold text-text"><Icon aria-hidden className="h-4 w-4 text-text-2" />{t.label}</span>
                {t.badge}
              </span>
              <span className="text-[12.5px] leading-5 text-text-2">{t.blurb}</span>
            </button>
          );
        })}
      </div>

      <div id={`${base}-panel-${view}`} role="tabpanel" aria-labelledby={`${base}-tab-${view}`}>
        {view === "retro" && <RetroView />}
        {view === "holdout" && <HoldoutView />}
      </div>
    </div>
  );
}
