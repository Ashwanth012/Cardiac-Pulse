import clsx from "clsx";
import { ArrowLeftToLine, ArrowRightToLine, CalendarDays, Info, PartyPopper } from "lucide-react";
import { RunBanner } from "../components/forecast/RunParts";
import { Badge, ProvenanceBadge } from "../components/ui/Badge";
import { Panel } from "../components/ui/Panel";
import { ErrorState, Loading } from "../components/ui/States";
import { useForecastRun, useModelExplain } from "../lib/api";
import { fmt1, fmtDate, fmtDayMonth, fmtLong, fmtWeekday } from "../lib/format";
import type { ForecastRun, RunDay } from "../lib/types";

/**
 * Calendar context for the historical 1–14 Apr 2019 admission forecast (client page).
 * Data: /api/run (official forecast.py output + official festival calendar) and /api/model/explain (the frozen
 * model's fitted holiday multipliers). No metrics, no internal links, nothing computed beyond simple counts.
 */

const TYPE: Record<string, { label: string; tone: "observed" | "teal" | "neutral" }> = {
  national: { label: "National holiday", tone: "observed" },
  state: { label: "Punjab state holiday", tone: "neutral" },
  festival: { label: "Festival", tone: "teal" },
};
const typeOf = (t?: string) => (t && TYPE[t]) || { label: t ? t : "Type not listed", tone: "neutral" as const };
const pctEffect = (m: number) => `${Math.abs(Math.round((m - 1) * 100))}% ${m < 1 ? "fewer" : "more"}`;

function DayRow({ d }: { d: RunDay }) {
  const weekend = ["Saturday", "Sunday"].includes(d.weekday);
  return (
    <li className={clsx("grid grid-cols-[88px_1fr] items-start gap-x-4 gap-y-2 rounded-xl px-4 py-3 ring-1 ring-inset sm:grid-cols-[110px_1fr_auto]",
      d.holidays.length ? "bg-teal/[0.07] ring-teal/30" : "bg-surface-2/50 ring-border/15")}
      aria-label={`${fmtLong(d.date)}: ${d.holidays.length ? d.holidays.map((h) => h.name).join(", ") : "no holiday listed"}; ${fmt1(d.mean)} expected admission records`}>
      <div>
        <p className="text-[14px] font-semibold text-text">{fmtDayMonth(d.date)}</p>
        <p className={clsx("text-[12px]", weekend ? "font-medium text-text-2" : "text-text-3")}>{d.weekday}</p>
      </div>
      <div className="min-w-0 space-y-1.5">
        {d.holidays.length ? d.holidays.map((h) => (
          <div key={h.name} className="flex flex-wrap items-center gap-2">
            <PartyPopper aria-hidden className="h-4 w-4 shrink-0 text-teal-ink" />
            <span className="text-[13.5px] font-semibold text-text">{h.name}</span>
            <Badge tone={typeOf(h.type).tone}>{typeOf(h.type).label}</Badge>
          </div>
        )) : <p className="text-[13px] text-text-3">No holiday listed</p>}
        {(d.day_before_holiday || d.day_after_holiday) && (
          <div className="flex flex-wrap gap-2 text-[12px] text-text-2">
            {d.day_before_holiday && <span className="inline-flex items-center gap-1"><ArrowRightToLine aria-hidden className="h-3.5 w-3.5" />Day before a holiday</span>}
            {d.day_after_holiday && <span className="inline-flex items-center gap-1"><ArrowLeftToLine aria-hidden className="h-3.5 w-3.5" />Day after a holiday</span>}
          </div>
        )}
      </div>
      <div className="col-span-2 flex items-baseline gap-2 sm:col-span-1 sm:block sm:text-right">
        <p className="tabular text-[20px] font-semibold leading-none text-text">{fmt1(d.mean)}</p>
        <p className="tabular text-[11.5px] text-text-2 sm:mt-1">expected · range {fmt1(d.lower_80)}–{fmt1(d.upper_80)}</p>
      </div>
    </li>
  );
}

function Agenda({ run }: { run: ForecastRun }) {
  const weeks = [run.days.slice(0, 7), run.days.slice(7, 14)];
  return (
    <div className="space-y-5">
      {weeks.map((w, i) => (
        <section key={i} aria-label={`Week ${i + 1}`}>
          <p className="mb-2 text-[12px] font-semibold uppercase tracking-wide text-text-3">
            Week {i + 1} · {fmtDayMonth(w[0].date)} – {fmtDate(w[w.length - 1].date)}
          </p>
          <ul className="space-y-2">{w.map((d) => <DayRow key={d.date} d={d} />)}</ul>
        </section>
      ))}
    </div>
  );
}

function Next30({ run }: { run: ForecastRun }) {
  const start = run.manifest.as_of;
  const end = new Date(`${start}T00:00:00Z`);
  end.setUTCDate(end.getUTCDate() + 29);
  const endIso = end.toISOString().slice(0, 10);
  const items = run.upcoming_holidays;
  return (
    <div>
      {items.length === 0 ? (
        <p className="rounded-xl bg-surface-2/60 p-4 text-[13px] text-text-2 ring-1 ring-inset ring-border/15">
          No holidays are listed in the supplied calendar between {fmtDayMonth(start)} and {fmtDate(endIso)}.
        </p>
      ) : (
        <ul className="divide-y divide-border/10 rounded-xl ring-1 ring-inset ring-border/15">
          {items.map((h) => {
            const days = Math.round((new Date(`${h.date}T00:00:00Z`).getTime() - new Date(`${start}T00:00:00Z`).getTime()) / 86_400_000);
            return (
              <li key={`${h.date}-${h.name}`} className="flex flex-wrap items-center justify-between gap-x-4 gap-y-1.5 px-4 py-2.5">
                <div className="flex min-w-0 items-center gap-3">
                  <span className="grid h-10 w-12 shrink-0 place-items-center rounded-lg bg-surface-2 text-center leading-tight ring-1 ring-inset ring-border/15">
                    <span className="text-[10.5px] font-semibold uppercase text-text-3">{fmtWeekday(h.date)}</span>
                    <span className="text-[13px] font-semibold text-text">{fmtDayMonth(h.date)}</span>
                  </span>
                  <div className="min-w-0">
                    <p className="truncate text-[13.5px] font-semibold text-text">{h.name}</p>
                    <p className="text-[12px] text-text-2">{days === 0 ? "On the cut-off day" : `${days} day${days === 1 ? "" : "s"} after the cut-off`}</p>
                  </div>
                </div>
                <div className="flex flex-wrap gap-1.5">
                  <Badge tone={typeOf(h.type).tone}>{typeOf(h.type).label}</Badge>
                  {h.in_forecast_window ? <Badge tone="forecast">In the 14-day forecast</Badge> : <Badge tone="neutral">After the forecast window</Badge>}
                </div>
              </li>
            );
          })}
        </ul>
      )}
      <p className="mt-2 text-[12px] text-text-3">
        {fmtDayMonth(start)} – {fmtDate(endIso)}, from the supplied Punjab festival calendar (covers dates up to {fmtDate(run.manifest.calendar_covered_until)}).
        Days after the forecast window have no admission forecast.
      </p>
    </div>
  );
}

function HowHolidaysAreUsed() {
  const ex = useModelExplain();
  const terms = ex.data?.calendar ?? [];
  return (
    <div className="space-y-3">
      <p className="text-[13px] leading-6 text-text-2">
        The forecasting model knows the holiday calendar in advance. For every day it checks three yes/no flags from the supplied calendar –
        <strong className="text-text"> is it a holiday</strong>, <strong className="text-text">is it the day before one</strong>, and
        <strong className="text-text"> is it the day after one</strong> – alongside the day of the week and the time of year. Each flag scales the
        expected number of admission records by an amount learned from this hospital's 2017–19 records:
      </p>
      {ex.isError ? (
        <p className="text-[12.5px] text-text-3">The model's holiday adjustments could not be loaded right now.</p>
      ) : !ex.data ? <Loading label="Loading holiday adjustments" /> : (
        <ul className="grid grid-cols-1 gap-2.5 sm:grid-cols-3">
          {terms.map((t) => (
            <li key={t.term} className="rounded-xl bg-surface-2/60 px-4 py-3 ring-1 ring-inset ring-border/15">
              <p className="text-[12.5px] font-medium text-text-2">{t.term}</p>
              <p className="tabular mt-0.5 text-[20px] font-semibold text-text">×{t.multiplier.toFixed(2)}</p>
              <p className="text-[12px] text-text-2">about {pctEffect(t.multiplier)} records than a comparable day</p>
            </li>
          ))}
        </ul>
      )}
      <p className="flex items-start gap-2 text-[12px] leading-5 text-text-3">
        <Info aria-hidden className="mt-0.5 h-3.5 w-3.5 shrink-0" />
        <span>
          These are patterns in past records, not causes, and they are modest compared with the weekday rhythm. A day can be both a holiday and
          the day before another (13–14 Apr). Weather and air quality are not part of the model.
        </span>
      </p>
    </div>
  );
}

export default function CalendarContext() {
  const run = useForecastRun();
  if (run.isError) return <ErrorState title="The forecast calendar is unavailable" message={(run.error as Error).message} onRetry={() => run.refetch()} />;
  if (!run.data) return <Loading label="Loading the forecast calendar" />;
  const r = run.data;
  const holidayDays = r.days.filter((d) => d.holidays.length);
  const adjacent = r.days.filter((d) => !d.holidays.length && (d.day_before_holiday || d.day_after_holiday));
  return (
    <div>
      <RunBanner run={r} compact />
      <section className="panel relative mb-6 overflow-hidden p-6 animate-fade-up" aria-labelledby="cal-title">
        <div aria-hidden className="pointer-events-none absolute -right-16 -top-24 h-64 w-64 rounded-full bg-teal opacity-[0.10] blur-3xl" />
        <div className="relative flex flex-wrap items-end justify-between gap-5">
          <div className="max-w-2xl">
            <div className="flex flex-wrap gap-2"><Badge tone="teal" icon={CalendarDays}>Calendar context</Badge><ProvenanceBadge kind="historical_forecast" /></div>
            <h2 id="cal-title" className="mt-3 text-2xl font-semibold tracking-tight text-text">Holidays beside the 14-day forecast</h2>
            <p className="mt-1.5 text-[14px] leading-6 text-text-2">
              Each day of the 1–14 April 2019 historical forecast with its holidays from the supplied Punjab calendar and the expected number
              of cardiac admission records.
            </p>
          </div>
          <dl className="grid grid-cols-3 gap-3 text-center">
            {[["Holiday days", holidayDays.length], ["Holidays listed", holidayDays.reduce((a, d) => a + d.holidays.length, 0)],
              ["Days next to a holiday", adjacent.length]].map(([k, v]) => (
              <div key={k as string} className="rounded-xl bg-surface-2/70 px-3 py-2.5 ring-1 ring-inset ring-border/15">
                <dd className="tabular text-[22px] font-semibold text-text">{v}</dd>
                <dt className="text-[11.5px] text-text-2">{k}</dt>
              </div>
            ))}
          </dl>
        </div>
      </section>

      <div className="grid grid-cols-1 gap-6 xl:grid-cols-5">
        <Panel className="xl:col-span-3" title="Forecast days, 1–14 Apr 2019"
          subtitle="Holiday name and type, day-before / day-after flags, and the expected admission records with their 80% range"
          actions={<ProvenanceBadge kind="historical_forecast" />}>
          <Agenda run={r} />
        </Panel>
        <div className="space-y-6 xl:col-span-2">
          <Panel title="Holidays in the next 30 days" subtitle="From the 1 Apr 2019 cut-off">
            <Next30 run={r} />
          </Panel>
          <Panel title="How holidays are used in the forecast">
            <HowHolidaysAreUsed />
          </Panel>
        </div>
      </div>
    </div>
  );
}
