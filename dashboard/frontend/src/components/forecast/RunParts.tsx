import clsx from "clsx";
import {
  ArrowDown, ArrowRight, ArrowUp, CalendarDays, CalendarRange, CircleAlert, Download, Fingerprint, Gauge, History, Info, PartyPopper, Trophy,
} from "lucide-react";
import { fmt1, fmtDate, fmtDayMonth, fmtLong, fmtWeekday } from "../../lib/format";
import type { ForecastRun, HistoricalTier, RunDay } from "../../lib/types";
import { Badge } from "../ui/Badge";

const pct = (v: number) => `${(v * 100).toFixed(1)}%`;
export const fmtRange = (d: { lower_80: number; upper_80: number }) => `${fmt1(d.lower_80)}–${fmt1(d.upper_80)}`;

// ============================================================================ run banner
/** Always-visible statement of what this forecast is: a historical demonstration for 1-14 Apr 2019. */
export function RunBanner({ run, compact = false }: { run: ForecastRun; compact?: boolean }) {
  const m = run.manifest;
  return (
    <section aria-label="Forecast status" className="mb-6 rounded-2xl bg-forecast/[0.07] p-4 ring-1 ring-inset ring-forecast/30 animate-fade-up">
      <div className="flex flex-wrap items-start gap-3">
        <span className="grid h-9 w-9 shrink-0 place-items-center rounded-xl bg-forecast/15 text-forecast-ink ring-1 ring-inset ring-forecast/30">
          <History aria-hidden className="h-[18px] w-[18px]" />
        </span>
        <div className="min-w-0 flex-1">
          <p className="text-[14px] font-semibold text-text">
            Historical hackathon demonstration – not a live hospital forecast
          </p>
          <p className="mt-0.5 text-[12.5px] leading-5 text-text-2">
            Expected admissions for <strong className="text-text">{fmtDate(m.forecast_start)} – {fmtDate(m.forecast_end)}</strong>, produced by the
            official <span className="font-mono text-[12px]">forecast.py</span> with admissions data through {fmtDate(m.data_through)}.
            {!compact && <> {m.period_note}</>}
          </p>
          <div className="mt-2.5 flex flex-wrap gap-1.5">
            <Badge tone="forecast" icon={CalendarRange}>Forecast period 1–14 Apr 2019</Badge>
            <Badge tone="neutral">Data through {fmtDayMonth(m.data_through)} 2019</Badge>
            <Badge tone={m.sha256_matches_frozen ? "teal" : "rose"} icon={Fingerprint}>
              {m.sha256_matches_frozen ? "Official forecast.py" : "forecast.py hash mismatch"} · {m.forecast_py_sha256.slice(0, 8)}
            </Badge>
            {m.fallback_used ? <Badge tone="amber" icon={CircleAlert}>Fallback model used</Badge> : <Badge tone="teal">Main model ran normally</Badge>}
            <Badge tone="neutral">Weather &amp; AQI: not model inputs</Badge>
          </div>
        </div>
      </div>
    </section>
  );
}

// ============================================================================ reliability
export function ReliabilityCard({ run }: { run: ForecastRun }) {
  const r = run.reliability;
  const items = [
    { k: "Historical average absolute error", v: `about ${fmt1(r.mae)} admissions/day (holdout MAE)`, d: "average size of a daily miss – not a ± range or a guarantee" },
    { k: "The shaded range", v: `caught about ${Math.round(r.coverage80 * 10)} in 10 days`, d: `${pct(r.coverage80)} of ${r.n_points} test days (target at least 80%)` },
    { k: "Recent tendency", v: `under-predicted by about ${Math.round(Math.abs(r.bias))} per day`, d: "lean towards the upper bound when reading these numbers" },
  ];
  return (
    <div>
      <ul className="space-y-2.5">
        {items.map((i) => (
          <li key={i.k} className="rounded-xl bg-surface-2/60 px-3.5 py-2.5 ring-1 ring-inset ring-border/15">
            <p className="text-[12px] font-medium text-text-2">{i.k}</p>
            <p className="text-[14px] font-semibold text-text">{i.v}</p>
            <p className="text-[11.5px] text-text-3">{i.d}</p>
          </li>
        ))}
      </ul>
      <p className="mt-3 text-[11.5px] leading-4 text-text-3">
        Jan–Mar 2019, previously viewed in the M4 holdout context ({r.n_cutoffs} forecast runs); not used to select newer models. Past accuracy is not a guarantee.
      </p>
    </div>
  );
}

/** Client-facing: how to read the forecast, in plain language. No accuracy metrics or validation diagnostics. */
export function ReadingGuide() {
  const items = [
    { t: "The expected number", d: "Each day's figure is the model's central estimate of cardiac admission records for that day." },
    { t: "The shaded range", d: "The 80% prediction range: the band the model expects most days to fall inside. It is wide because daily admissions vary a lot." },
    { t: "Plan cautiously", d: "In early 2019 actual admissions ran above this model's forecasts, so the upper end of the range is the safer planning reference." },
    { t: "Historical demonstration", d: "These figures are for 1–14 April 2019, the fortnight after the hospital data end. They are not a forecast for today." },
  ];
  return (
    <ul className="grid grid-cols-1 gap-2.5 sm:grid-cols-2">
      {items.map((i) => (
        <li key={i.t} className="rounded-xl bg-surface-2/60 px-3.5 py-2.5 ring-1 ring-inset ring-border/15">
          <p className="text-[13px] font-semibold text-text">{i.t}</p>
          <p className="mt-0.5 text-[12.5px] leading-5 text-text-2">{i.d}</p>
        </li>
      ))}
    </ul>
  );
}

// ============================================================================ indicators
const TIER: Record<HistoricalTier, { icon: typeof ArrowDown; tone: "amber" | "neutral" | "observed"; short: string }> = {
  higher: { icon: ArrowUp, tone: "amber", short: "Above recent baseline" },
  typical: { icon: ArrowRight, tone: "neutral", short: "Within recent range" },
  lower: { icon: ArrowDown, tone: "observed", short: "Below recent baseline" },
};

export function TierBadge({ day }: { day: RunDay }) {
  const t = TIER[day.historical_comparison.tier];
  return <span title={day.historical_comparison.label}><Badge tone={t.tone} icon={t.icon}>{t.short}</Badge></span>;
}

/** Wrapping variant for narrow day cards (same wording as the badge). */
function TierLine({ day }: { day: RunDay }) {
  const t = TIER[day.historical_comparison.tier];
  const Icon = t.icon;
  return (
    <span title={day.historical_comparison.label} className="flex items-start gap-1 text-[11px] font-medium leading-4 text-text-2">
      <Icon aria-hidden className="mt-0.5 h-3 w-3 shrink-0" />{t.short}
    </span>
  );
}

export function BusiestBadge({ day }: { day: RunDay }) {
  if (!day.busiest_in_window) return null;
  return <Badge tone="forecast" icon={Trophy}>Busiest #{day.window_rank} in forecast</Badge>;
}

/** Both descriptive indicators, explained side by side and never merged. */
export function IndicatorExplainer({ run }: { run: ForecastRun }) {
  const c = run.summary.historical_comparison_counts;
  const hr = run.historical_comparison_rule;
  const busiest = run.days.filter((d) => d.busiest_in_window).sort((a, b) => a.window_rank - b.window_rank);
  return (
    <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
      <div className="rounded-xl bg-surface-2/60 p-4 ring-1 ring-inset ring-border/15">
        <p className="text-[11px] font-semibold uppercase tracking-wide text-text-3">1 · Historical comparison</p>
        <p className="mt-1 text-[13.5px] font-semibold text-text">How each day compares with recent same-weekday admissions</p>
        <dl className="mt-3 grid grid-cols-3 gap-2 text-center">
          {(["higher", "typical", "lower"] as const).map((k) => (
            <div key={k} className="rounded-lg bg-surface px-2 py-2 ring-1 ring-inset ring-border/15">
              <dt className="text-[11px] text-text-2">{TIER[k].short}</dt>
              <dd className="tabular text-[20px] font-semibold text-text">{c[k]}</dd>
            </div>
          ))}
        </dl>
        {hr.no_higher_days_note && <p className="mt-3 text-[12.5px] font-medium leading-5 text-text">{hr.no_higher_days_note}</p>}
        {hr.all_lower_note && <p className="mt-1 text-[12.5px] leading-5 text-text-2">{hr.all_lower_note}</p>}
        <p className="mt-2 text-[11.5px] leading-4 text-text-3">
          Baseline: observed raw records on the same weekday, {fmtDayMonth(hr.baseline_start)} – {fmtDate(hr.baseline_end)} (8 weeks). Above the
          75th percentile = above baseline; below the 25th = below. {hr.descriptive_only}
        </p>
      </div>
      <div className="rounded-xl bg-surface-2/60 p-4 ring-1 ring-inset ring-border/15">
        <p className="text-[11px] font-semibold uppercase tracking-wide text-text-3">2 · Forecast-period ranking</p>
        <p className="mt-1 text-[13.5px] font-semibold text-text">Busiest days within this 14-day forecast</p>
        <ol className="mt-3 space-y-1.5">
          {busiest.map((d) => (
            <li key={d.date} className="flex items-center justify-between gap-3 rounded-lg bg-surface px-3 py-2 ring-1 ring-inset ring-border/15 text-[13px]">
              <span className="flex items-center gap-2 font-medium text-text"><Trophy aria-hidden className="h-3.5 w-3.5 text-forecast-ink" />#{d.window_rank} · {fmtLong(d.date)}</span>
              <span className="tabular font-semibold text-text">{fmt1(d.mean)}</span>
            </li>
          ))}
        </ol>
        <p className="mt-2 text-[11.5px] leading-4 text-text-3">
          {run.window_ranking_rule.description} Average across the 14 days: {fmt1(run.summary.average_mean)} records/day.{" "}
          {run.window_ranking_rule.descriptive_only}
        </p>
      </div>
    </div>
  );
}

// ============================================================================ day cards
export function DayCard({ day }: { day: RunDay }) {
  return (
    <article className={clsx("flex flex-col rounded-xl p-3.5 ring-1 ring-inset",
      day.busiest_in_window ? "bg-forecast/[0.08] ring-forecast/35" : "bg-surface-2/60 ring-border/15")}
      aria-label={`${fmtLong(day.date)}: ${fmt1(day.mean)} expected records, 80% range ${fmtRange(day)}`}>
      <div className="flex items-baseline justify-between gap-2">
        <p className="text-[12.5px] font-semibold text-text">{fmtWeekday(day.date)} {fmtDayMonth(day.date)}</p>
        {day.busiest_in_window && <span className="text-[11px] font-semibold text-forecast-ink">#{day.window_rank}</span>}
      </div>
      <p className="tabular mt-1.5 text-[24px] font-semibold leading-none text-text">{fmt1(day.mean)}</p>
      <p className="tabular mt-1 text-[11.5px] text-text-2">range {fmtRange(day)}</p>
      <div className="mt-2.5 flex flex-1 flex-col justify-end gap-1">
        {day.holidays.map((h) => (
          <span key={h.name} className="flex items-start gap-1 rounded-md bg-teal/[0.14] px-1.5 py-0.5 text-[11px] font-semibold leading-4 text-teal-ink ring-1 ring-inset ring-teal/30">
            <PartyPopper aria-hidden className="mt-0.5 h-3 w-3 shrink-0" />{h.name}
          </span>
        ))}
        {!day.holidays.length && day.day_before_holiday && <span className="text-[11px] text-text-2">Day before a holiday</span>}
        {!day.holidays.length && day.day_after_holiday && <span className="text-[11px] text-text-2">Day after a holiday</span>}
        <TierLine day={day} />
      </div>
    </article>
  );
}

export function ForecastStrip({ run }: { run: ForecastRun }) {
  const weeks = [run.days.slice(0, 7), run.days.slice(7, 14)];
  return (
    <div className="space-y-4">
      {weeks.map((w, i) => (
        <div key={i}>
          <p className="mb-2 text-[12px] font-semibold text-text-2">Week {i + 1} · {fmtDayMonth(w[0].date)} – {fmtDayMonth(w[w.length - 1].date)} 2019</p>
          <div className="-mx-1 flex snap-x gap-2.5 overflow-x-auto px-1 pb-1 md:grid md:grid-cols-4 md:overflow-visible xl:grid-cols-7">
            {w.map((d) => <div key={d.date} className="w-[150px] shrink-0 snap-start md:w-auto"><DayCard day={d} /></div>)}
          </div>
        </div>
      ))}
    </div>
  );
}

// ============================================================================ official table + CSV
export function OfficialTable({ run }: { run: ForecastRun }) {
  const download = () => {
    const csv = ["date,mean,lower_80,upper_80", ...run.official.map((r) => `${r.date},${r.mean},${r.lower_80},${r.upper_80}`)].join("\n") + "\n";
    const url = URL.createObjectURL(new Blob([csv], { type: "text/csv" }));
    const a = document.createElement("a");
    a.href = url;
    a.download = `forecast_${run.manifest.as_of}_historical_demonstration.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };
  return (
    <div>
      <div className="overflow-x-auto rounded-xl ring-1 ring-inset ring-border/15">
        <table className="w-full min-w-[720px] text-[13px]">
          <caption className="sr-only">Official forecast.py output for 1 to 14 April 2019 with calendar and descriptive indicators</caption>
          <thead className="bg-surface-2">
            <tr className="text-left text-[11px] font-semibold uppercase tracking-wide text-text-3">
              {["Date", "Day", "Mean", "Lower 80%", "Upper 80%", "Calendar", "Historical comparison", "Forecast rank"].map((h) => (
                <th key={h} scope="col" className="px-3 py-2">{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {run.days.map((d) => (
              <tr key={d.date} className={clsx("border-t border-border/10", d.busiest_in_window && "bg-forecast/[0.05]")}>
                <td className="px-3 py-2 font-medium text-text">{fmtDate(d.date)}</td>
                <td className="px-3 py-2 text-text-2">{fmtWeekday(d.date)}</td>
                <td className="tabular px-3 py-2 font-semibold text-text">{fmt1(d.mean)}</td>
                <td className="tabular px-3 py-2 text-text-2">{fmt1(d.lower_80)}</td>
                <td className="tabular px-3 py-2 text-text-2">{fmt1(d.upper_80)}</td>
                <td className="px-3 py-2">
                  <div className="flex flex-wrap gap-1">
                    {d.holidays.map((h) => <Badge key={h.name} tone="teal">{h.name}</Badge>)}
                    {!d.holidays.length && (d.day_before_holiday || d.day_after_holiday)
                      ? <span className="text-[12px] text-text-2">{d.day_before_holiday ? "Day before a holiday" : "Day after a holiday"}</span>
                      : !d.holidays.length && <span className="text-text-3">—</span>}
                  </div>
                </td>
                <td className="px-3 py-2"><TierBadge day={d} /></td>
                <td className="tabular px-3 py-2 text-text-2">{d.window_rank} of 14{d.busiest_in_window ? " · busiest" : ""}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="mt-3 flex flex-wrap items-center justify-between gap-3">
        <p className="text-[12px] leading-5 text-text-2">
          Mean, lower and upper bounds are the official <span className="font-mono">forecast.py</span> output (shown to 1 decimal; the CSV has 3).
        </p>
        <button type="button" onClick={download}
          className="inline-flex items-center gap-1.5 rounded-lg bg-surface-2 px-3 py-1.5 text-[12.5px] font-semibold text-text ring-1 ring-inset ring-border/25 hover:bg-surface">
          <Download aria-hidden className="h-3.5 w-3.5" /> Download official 14 rows (CSV)
        </button>
      </div>
    </div>
  );
}

// ============================================================================ watch list + holidays
export function DaysToWatch({ run, limit = 4 }: { run: ForecastRun; limit?: number }) {
  const items = [...run.days_to_watch].sort((a, b) => (a.kind === b.kind ? a.date.localeCompare(b.date) : a.kind === "busiest" ? -1 : 1)).slice(0, limit);
  return (
    <ul className="space-y-2.5">
      {items.map((w) => (
        <li key={`${w.kind}-${w.date}`} className="flex gap-3 rounded-xl bg-surface-2/60 p-3 ring-1 ring-inset ring-border/15">
          <span className={clsx("grid h-8 w-8 shrink-0 place-items-center rounded-lg ring-1 ring-inset",
            w.kind === "busiest" ? "bg-forecast/15 text-forecast-ink ring-forecast/30" : "bg-teal/15 text-teal-ink ring-teal/30")}>
            {w.kind === "busiest" ? <Trophy aria-hidden className="h-4 w-4" /> : <PartyPopper aria-hidden className="h-4 w-4" />}
          </span>
          <div className="min-w-0">
            <p className="text-[12px] font-medium text-text-2">{fmtLong(w.date)}</p>
            <p className="text-[13px] font-semibold text-text">{w.title}</p>
            <p className="text-[12px] leading-5 text-text-2">{w.detail}</p>
          </div>
        </li>
      ))}
    </ul>
  );
}

export function UpcomingHolidays({ run }: { run: ForecastRun }) {
  return (
    <ul className="divide-y divide-border/10">
      {run.upcoming_holidays.map((h) => (
        <li key={`${h.date}-${h.name}`} className="flex items-center justify-between gap-3 py-2 text-[13px]">
          <span className="flex min-w-0 items-center gap-2">
            <CalendarDays aria-hidden className="h-3.5 w-3.5 shrink-0 text-teal-ink" />
            <span className="truncate font-medium text-text">{h.name}</span>
          </span>
          <span className="flex shrink-0 items-center gap-2 text-text-2">
            {fmtWeekday(h.date)} {fmtDayMonth(h.date)}
            {h.in_forecast_window ? <Badge tone="teal">In forecast</Badge> : <Badge tone="neutral">After window</Badge>}
          </span>
        </li>
      ))}
    </ul>
  );
}

export function NotModelInputs({ run }: { run: ForecastRun }) {
  return (
    <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
      <div className="rounded-xl bg-surface-2/60 p-3.5 ring-1 ring-inset ring-border/15">
        <p className="flex items-center gap-1.5 text-[12.5px] font-semibold text-text"><Gauge aria-hidden className="h-3.5 w-3.5 text-forecast-ink" />What the model uses</p>
        <ul className="mt-1.5 list-disc space-y-0.5 pl-5 text-[12.5px] text-text-2">{run.manifest.model_inputs.map((m) => <li key={m}>{m}</li>)}</ul>
      </div>
      <div className="rounded-xl bg-surface-2/60 p-3.5 ring-1 ring-inset ring-border/15">
        <p className="flex items-center gap-1.5 text-[12.5px] font-semibold text-text"><Info aria-hidden className="h-3.5 w-3.5 text-text-2" />Not used by the model</p>
        <ul className="mt-1.5 list-disc space-y-0.5 pl-5 text-[12.5px] text-text-2">{run.manifest.not_model_inputs.map((m) => <li key={m}>{m}</li>)}</ul>
        <p className="mt-1.5 text-[11.5px] leading-4 text-text-3">Live weather and AQI forecasts (Weather &amp; AQI page) cover the current period and are context only.</p>
      </div>
    </div>
  );
}
