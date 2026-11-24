import { CalendarDays, CircleAlert, Gauge, Info, Layers, Sun } from "lucide-react";
import type { ReactNode } from "react";
import { AnnualCurve, WeekdayMultipliers } from "../components/charts/StudioCharts";
import { RunBanner } from "../components/forecast/RunParts";
import { Badge } from "../components/ui/Badge";
import { Panel } from "../components/ui/Panel";
import { ErrorState, Loading } from "../components/ui/States";
import { useForecastRun, useModelExplain } from "../lib/api";
import { fmt1, fmtDayMonth, fmtWeekday } from "../lib/format";
import type { ForecastRun, ModelExplain } from "../lib/types";

/**
 * Prediction Insights (client page): plain-language explanation of the frozen M4 forecast.
 * Data: /api/model/explain (the frozen model's own fitted multipliers, reproduced from forecast.py) and /api/run
 * (official 14-day output). Every figure, rank, peak and trough below is computed from those responses.
 * No accuracy metrics, comparison tables or internal links.
 */

const change = (m: number) => `${Math.abs(Math.round((m - 1) * 100))}% ${m < 1 ? "fewer" : "more"}`;

function Unavailable({ what, onRetry }: { what: string; onRetry: () => void }) {
  return <ErrorState title={`${what} is unavailable right now`} message="The rest of this page and the forecast itself are unaffected." onRetry={onRetry} />;
}

function Section({ n, icon: Icon, title, children }: { n: number; icon: typeof Info; title: string; children: ReactNode }) {
  return (
    <Panel className="mb-6" title={<span className="flex items-center gap-2"><span className="grid h-6 w-6 place-items-center rounded-md bg-forecast/15 text-[12px] font-semibold text-forecast-ink">{n}</span>
      <Icon aria-hidden className="h-4 w-4 text-text-2" />{title}</span>}>
      {children}
    </Panel>
  );
}

const Caveat = ({ children }: { children: ReactNode }) => (
  <p className="mt-3 flex items-start gap-2 text-[12px] leading-5 text-text-3"><Info aria-hidden className="mt-0.5 h-3.5 w-3.5 shrink-0" /><span>{children}</span></p>
);

// ============================================================================ 1. weekday
function Weekday({ ex }: { ex: ModelExplain }) {
  const sorted = [...ex.weekday].sort((a, b) => b.multiplier - a.multiplier);
  const hi = sorted[0], lo = sorted[sorted.length - 1];
  return (
    <div className="grid grid-cols-1 gap-6 lg:grid-cols-5">
      <div className="lg:col-span-3"><WeekdayMultipliers data={ex} /></div>
      <div className="space-y-2.5 text-[13px] leading-6 text-text-2 lg:col-span-2">
        <p>
          The model gives each day of the week its own multiplier, measured against <strong className="text-text">Monday = 1.00</strong>.
          A value of 0.80 means about 20% fewer expected admission records than on an otherwise similar Monday.
        </p>
        <p>
          In this hospital's 2017–19 records the highest weekday multiplier is <strong className="text-text">{hi.day} (×{hi.multiplier.toFixed(2)})</strong> and
          the lowest is <strong className="text-text">{lo.day} (×{lo.multiplier.toFixed(2)}, about {change(lo.multiplier)} records)</strong>.
        </p>
        <ul className="grid grid-cols-2 gap-x-4 gap-y-1 text-[12.5px] sm:grid-cols-4 lg:grid-cols-2">
          {ex.weekday.map((w) => <li key={w.day} className="tabular flex justify-between gap-2"><span>{w.day}</span><span className="font-semibold text-text">×{w.multiplier.toFixed(2)}</span></li>)}
        </ul>
      </div>
      <div className="lg:col-span-5"><Caveat>These are associations learned from past admission records. They describe when admissions were recorded more or less often,
        not why – they are not causal explanations.</Caveat></div>
    </div>
  );
}

// ============================================================================ 2. annual seasonality
function Seasonality({ ex }: { ex: ModelExplain }) {
  const pts = ex.annual_curve;
  if (!pts.length) return <p className="text-[13px] text-text-2">No annual pattern is available for this cut-off.</p>;
  const peak = pts.reduce((a, b) => (b.multiplier > a.multiplier ? b : a));
  const low = pts.reduce((a, b) => (b.multiplier < a.multiplier ? b : a));
  const win = pts.filter((p) => p.day_of_year >= 91 && p.day_of_year <= 104);
  const winAvg = win.reduce((a, p) => a + p.multiplier, 0) / (win.length || 1);
  return (
    <div className="grid grid-cols-1 gap-6 lg:grid-cols-5">
      <div className="lg:col-span-3"><AnnualCurve data={ex} />
        <p className="mt-1 text-[11.5px] text-text-3">Shaded: the forecast window (1–14 Apr). Dashed line: the yearly average (×1.00).</p></div>
      <div className="space-y-2.5 text-[13px] leading-6 text-text-2 lg:col-span-2">
        <p>
          Admissions rise and fall over the year. The model captures this with a smooth yearly wave built from
          {" "}<strong className="text-text">{ex.fourier_harmonics} pairs of sine and cosine curves</strong> (Fourier terms): one full wave per year
          plus one twice-a-year wave. Two waves are enough for a broad seasonal shape but cannot follow sharp short-term spikes.
        </p>
        <p>
          Learned from {ex.training.days} days of records ({fmtDayMonth(ex.training.start)} {ex.training.start.slice(0, 4)} – {fmtDayMonth(ex.training.end)} {ex.training.end.slice(0, 4)}),
          the curve is highest around <strong className="text-text">{peak.label} (×{peak.multiplier.toFixed(2)})</strong> and lowest around
          {" "}<strong className="text-text">{low.label} (×{low.multiplier.toFixed(2)})</strong>. Across 1–14 April it averages
          {" "}<strong className="text-text">×{winAvg.toFixed(2)}</strong>.
        </p>
      </div>
      <div className="lg:col-span-5"><Caveat>The seasonal wave repeats the shape of past years. It does not know about anything new this year, and it describes timing,
        not what drives admissions.</Caveat></div>
    </div>
  );
}

// ============================================================================ 3. holidays
const HOLIDAY_TEXT: Record<string, string> = {
  "Holiday": "The holiday itself, from the supplied Punjab calendar.",
  "Day before a holiday": "The day immediately before a listed holiday.",
  "Day after a holiday": "The day immediately after a listed holiday.",
};

function Holidays({ ex, run }: { ex: ModelExplain; run?: ForecastRun }) {
  const inWindow = run ? run.days.filter((d) => d.holidays.length || d.day_before_holiday || d.day_after_holiday) : [];
  const both = run?.days.find((d) => d.holidays.length && (d.day_before_holiday || d.day_after_holiday));
  return (
    <div className="space-y-3">
      <p className="text-[13px] leading-6 text-text-2">
        Three separate yes/no indicators are switched on from the holiday calendar, which is known in advance. Their effects multiply with the
        weekday and seasonal effects, so a day can carry more than one{both ? ` (${fmtWeekday(both.date)} ${fmtDayMonth(both.date)} is both a holiday and the day before another)` : ""}.
      </p>
      <ul className="grid grid-cols-1 gap-2.5 sm:grid-cols-3">
        {ex.calendar.map((t) => (
          <li key={t.term} className="rounded-xl bg-surface-2/60 px-4 py-3 ring-1 ring-inset ring-border/15">
            <p className="text-[12.5px] font-semibold text-text">{t.term}</p>
            <p className="tabular mt-0.5 text-[20px] font-semibold text-text">×{t.multiplier.toFixed(2)}</p>
            <p className="text-[12px] text-text-2">about {change(t.multiplier)} records than a comparable day</p>
            <p className="mt-1.5 text-[11.5px] leading-4 text-text-3">{HOLIDAY_TEXT[t.term] ?? ""}</p>
          </li>
        ))}
      </ul>
      {run && inWindow.length > 0 && (
        <p className="text-[12.5px] leading-5 text-text-2">
          In the 1–14 April forecast these indicators apply to:{" "}
          {inWindow.map((d, i) => (
            <span key={d.date}>{i > 0 && "; "}<strong className="text-text">{fmtWeekday(d.date)} {fmtDayMonth(d.date)}</strong>{" "}
              ({[d.holidays.length ? d.holidays.map((h) => h.name).join(" & ") : null, d.day_before_holiday ? "day before a holiday" : null,
                d.day_after_holiday ? "day after a holiday" : null].filter(Boolean).join(", ")})</span>
          ))}.
        </p>
      )}
      <Caveat>These are historical patterns in this hospital's records, not proof that holidays cause admissions to change. They are modest
        compared with the weekday pattern.</Caveat>
    </div>
  );
}

// ============================================================================ 4. uncertainty
function Uncertainty({ run }: { run: ForecastRun }) {
  const days = run.days;
  const top = Math.max(...days.map((d) => d.upper_80));
  const widths = days.map((d) => ({ d, w: d.upper_80 - d.lower_80 }));
  const widest = widths.reduce((a, b) => (b.w > a.w ? b : a));
  const narrowest = widths.reduce((a, b) => (b.w < a.w ? b : a));
  return (
    <div className="grid grid-cols-1 gap-6 lg:grid-cols-5">
      <div className="space-y-2.5 text-[13px] leading-6 text-text-2 lg:col-span-2">
        <p>
          Each day has a central estimate (<strong className="text-text">mean</strong>) and an <strong className="text-text">80% prediction range</strong> from
          {" "}<span className="font-mono text-[12px]">lower_80</span> to <span className="font-mono text-[12px]">upper_80</span>.
        </p>
        <p>
          The range is built from how far the model's own earlier forecasts missed. It is designed so that, <strong className="text-text">over many
          days</strong>, roughly 8 in 10 actual daily counts fall inside their ranges. It is <strong className="text-text">not a guarantee</strong> for any
          particular day, and some days will fall outside – including runs of several days if admissions shift to a new level.
        </p>
        <p>
          Ranges are wide because daily admission counts vary a lot on their own. Days with more expected admissions tend to get wider ranges: here the widest is
          {" "}<strong className="text-text">{fmtWeekday(widest.d.date)} {fmtDayMonth(widest.d.date)}</strong> ({fmt1(widest.d.lower_80)}–{fmt1(widest.d.upper_80)})
          and the narrowest <strong className="text-text">{fmtWeekday(narrowest.d.date)} {fmtDayMonth(narrowest.d.date)}</strong> ({fmt1(narrowest.d.lower_80)}–{fmt1(narrowest.d.upper_80)}).
        </p>
      </div>
      <div className="lg:col-span-3">
        <ul className="space-y-1.5" aria-label="80% prediction range for each forecast day">
          {days.map((d) => (
            <li key={d.date} className="grid grid-cols-[72px_1fr_110px] items-center gap-3 text-[12px]">
              <span className="font-medium text-text">{fmtWeekday(d.date)} {fmtDayMonth(d.date)}</span>
              <span aria-hidden className="relative h-3 rounded-full bg-surface-2">
                <span className="absolute inset-y-0 rounded-full bg-forecast/35" style={{ left: `${(d.lower_80 / top) * 100}%`, width: `${((d.upper_80 - d.lower_80) / top) * 100}%` }} />
                <span className="absolute -top-0.5 h-4 w-[3px] rounded bg-forecast" style={{ left: `calc(${(d.mean / top) * 100}% - 1.5px)` }} />
              </span>
              <span className="tabular text-right text-text-2"><strong className="text-text">{fmt1(d.mean)}</strong> ({fmt1(d.lower_80)}–{fmt1(d.upper_80)})</span>
            </li>
          ))}
        </ul>
        <p className="mt-2 text-[11.5px] text-text-3">Bar = 80% prediction range; tick = mean. Values are the official forecast output for 1–14 Apr 2019.</p>
      </div>
    </div>
  );
}

// ============================================================================ 5. inputs & limitations
function InputsAndLimits({ run }: { run?: ForecastRun }) {
  const inputs = run?.manifest.model_inputs ?? [];
  return (
    <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
      <div className="rounded-xl bg-surface-2/60 p-4 ring-1 ring-inset ring-border/15">
        <p className="flex items-center gap-1.5 text-[13px] font-semibold text-text"><Gauge aria-hidden className="h-4 w-4 text-forecast-ink" />What the model uses</p>
        <ul className="mt-2 list-disc space-y-1 pl-5 text-[12.5px] leading-5 text-text-2">
          <li>Historical daily admission counts (all records before the cut-off) – to learn the patterns below</li>
          <li>Day of the week</li>
          <li>Annual seasonality (a smooth yearly wave)</li>
          <li>Holiday, day-before-holiday and day-after-holiday indicators from the supplied calendar</li>
        </ul>
        {inputs.length > 0 && <p className="mt-2 text-[11.5px] text-text-3">As reported by the forecast service: {inputs.join("; ")}.</p>}
        <p className="mt-2 text-[12px] leading-5 text-text-2">It does not include a separate adjustment for the most recent level of admissions or a long-term trend.</p>
      </div>
      <div className="rounded-xl bg-surface-2/60 p-4 ring-1 ring-inset ring-border/15">
        <p className="flex items-center gap-1.5 text-[13px] font-semibold text-text"><CircleAlert aria-hidden className="h-4 w-4 text-amber-ink" />Not used, and limitations</p>
        <ul className="mt-2 list-disc space-y-1 pl-5 text-[12.5px] leading-5 text-text-2">
          <li><strong className="text-text">Weather and air quality (AQI, pollutants) are not inputs</strong> to this model. Live weather and AQI shown elsewhere are context only.</li>
          <li>Environmental data was evaluated: models with weather and air-quality information were compared with this one, and none met the
            improvement rule set in advance, so this model was kept.</li>
          <li>The hospital admissions data end on <strong className="text-text">31 March 2019</strong>. This is a historical demonstration for 1–14 April 2019,
            not an operational forecast for 2026.</li>
          <li>In early 2019 actual admissions ran above this model's forecasts, so read the upper end of the range as the cautious reference.</li>
          <li>Forecasts are for total daily admission records (including repeated records), not for individual patients.</li>
        </ul>
      </div>
    </div>
  );
}

// ============================================================================ page
export default function Insights() {
  const ex = useModelExplain();
  const run = useForecastRun();
  return (
    <div>
      {run.data && <RunBanner run={run.data} compact />}
      <section className="panel relative mb-6 overflow-hidden p-6 animate-fade-up" aria-labelledby="ins-title">
        <div aria-hidden className="pointer-events-none absolute -right-16 -top-24 h-64 w-64 rounded-full bg-forecast opacity-[0.12] blur-3xl" />
        <div className="relative max-w-3xl">
          <div className="flex flex-wrap gap-2"><Badge tone="forecast" icon={Layers}>Prediction insights</Badge><Badge tone="neutral">Historical demonstration · 1–14 Apr 2019</Badge></div>
          <h2 id="ins-title" className="mt-3 text-2xl font-semibold tracking-tight text-text">Why the forecast looks the way it does</h2>
          <p className="mt-1.5 text-[14px] leading-6 text-text-2">
            The forecast multiplies a few simple patterns learned from this hospital's 2017–19 admission records: the day of the week, the time of
            year and the holiday calendar. The figures below are the model's own fitted values.
          </p>
        </div>
      </section>

      <Section n={1} icon={CalendarDays} title="Weekday patterns">
        {ex.isError ? <Unavailable what="The weekday pattern" onRetry={() => ex.refetch()} /> : !ex.data ? <Loading label="Loading weekday pattern" /> : <Weekday ex={ex.data} />}
      </Section>
      <Section n={2} icon={Sun} title="Annual seasonality">
        {ex.isError ? <Unavailable what="The seasonal pattern" onRetry={() => ex.refetch()} /> : !ex.data ? <Loading label="Loading seasonal pattern" /> : <Seasonality ex={ex.data} />}
      </Section>
      <Section n={3} icon={CalendarDays} title="Holiday effects">
        {ex.isError ? <Unavailable what="The holiday effects" onRetry={() => ex.refetch()} /> : !ex.data ? <Loading label="Loading holiday effects" /> : <Holidays ex={ex.data} run={run.data} />}
      </Section>
      <Section n={4} icon={Info} title="Forecast uncertainty">
        {run.isError ? <Unavailable what="The forecast" onRetry={() => run.refetch()} /> : !run.data ? <Loading label="Loading forecast" /> : <Uncertainty run={run.data} />}
      </Section>
      <Section n={5} icon={Gauge} title="Model inputs and limitations">
        <InputsAndLimits run={run.data} />
      </Section>
    </div>
  );
}
