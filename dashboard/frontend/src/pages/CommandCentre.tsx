import { Activity, ArrowDown, CalendarDays, ChevronRight, CloudSun, Lightbulb, ShieldAlert, Sigma, TrendingUp, Trophy } from "lucide-react";
import { HistoricalForecastChart } from "../components/charts/HistoricalForecastChart";
import { RecentTrendChart } from "../components/charts/RecentTrendChart";
import { DaysToWatch, ReadingGuide, RunBanner, UpcomingHolidays } from "../components/forecast/RunParts";
import { Badge, ProvenanceBadge } from "../components/ui/Badge";
import { Callout } from "../components/ui/Callout";
import { KpiCard } from "../components/ui/KpiCard";
import { Panel } from "../components/ui/Panel";
import { ErrorState, Loading, Skeleton } from "../components/ui/States";
import { useConditions, useForecastRun, useHistoricalForecast, useOverview } from "../lib/api";
import { useFilters } from "../lib/filters";
import { fmt1, fmtDate, fmtDayMonth, fmtLong, fmtWeekday } from "../lib/format";
import { hrefFor } from "../lib/router";
import type { ForecastRun } from "../lib/types";
import { aqiTone } from "../components/charts/EnvCharts";
import { StatusBadge, fmtRetrieved } from "./Conditions";

function Hero() {
  return (
    <section className="panel relative mb-6 overflow-hidden p-6 animate-fade-up" aria-labelledby="hero-title">
      <div aria-hidden className="pointer-events-none absolute -left-20 -top-24 h-64 w-64 rounded-full bg-observed opacity-[0.12] blur-3xl" />
      <div aria-hidden className="pointer-events-none absolute -bottom-28 right-0 h-72 w-72 rounded-full bg-forecast opacity-[0.14] blur-3xl" />
      <div className="relative flex flex-wrap items-end justify-between gap-6">
        <div className="max-w-2xl">
          <div className="flex flex-wrap gap-2">
            <ProvenanceBadge kind="historical_forecast" />
            <Badge tone="neutral">Cardiac admissions · Ludhiana · aggregate daily counts</Badge>
          </div>
          <h2 id="hero-title" className="mt-4 text-2xl font-semibold tracking-tight text-text sm:text-[28px]">
            Expected cardiac admissions,{" "}
            <span className="bg-gradient-to-r from-observed-ink via-teal-ink to-forecast-ink bg-clip-text text-transparent">next 14 days</span>
          </h2>
          <p className="mt-2 text-[14px] leading-6 text-text-2">
            The "next 14 days" here are <strong className="font-semibold text-text">1–14 April 2019</strong>: the fortnight right after the
            hackathon data end. It shows how the product works with the official forecasting model; it is not a forecast for today.
          </p>
        </div>
        <a href={hrefFor("/forecast")}
          className="inline-flex items-center gap-1.5 rounded-xl bg-forecast/15 px-4 py-2.5 text-[13px] font-semibold text-forecast-ink ring-1 ring-inset ring-forecast/35 hover:bg-forecast/20">
          Open the 14-day forecast <ChevronRight aria-hidden className="h-4 w-4" />
        </a>
      </div>
    </section>
  );
}

function Kpis({ run }: { run: ForecastRun }) {
  const s = run.summary;
  const c = s.historical_comparison_counts;
  return (
    <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">
      <KpiCard label="Expected admissions, 1–14 Apr 2019" value={Math.round(s.total_mean).toLocaleString("en-GB")} unit="records"
        detail={`About ${fmt1(s.average_mean)} per day (sum of the official daily means).`} provenance="historical_forecast" icon={Sigma} accent="forecast" />
      <KpiCard label="Busiest day in this forecast" value={fmt1(s.busiest.mean)} unit="records"
        detail={`${fmtLong(s.busiest.date)}. Ranked within these 14 days only.`} provenance="historical_forecast" icon={Trophy} accent="forecast" delay={60} />
      <KpiCard label="Days above the recent baseline" value={`${c.higher} of 14`}
        detail={c.higher ? "Forecast above the 75th percentile of recent same weekdays." : "No day exceeds the higher-demand threshold of recent same-weekday admissions."}
        provenance="historical_forecast" icon={ArrowDown} accent="dedup" delay={120} />
      <KpiCard label="Recent observed level" value={fmt1(s.recent_28_day_mean)} unit="/ day"
        detail="Mean of the 28 days before the cut-off. The forecast sits below it (known under-forecasting)." provenance="observed" icon={TrendingUp} accent="observed" delay={180} />
    </div>
  );
}

/** Live weather/AQI highlights for the CURRENT period. Kept visually and verbally apart from the 2019 admission demonstration. */
function LiveConditionsCard() {
  const q = useConditions();
  if (q.isLoading) return <Loading label="Loading live conditions" />;
  if (q.isError || !q.data)
    return <p className="rounded-xl bg-surface-2/60 p-4 text-[12.5px] text-text-2 ring-1 ring-inset ring-border/15">Live conditions are unavailable right now. The admission forecast is unaffected.</p>;
  const d = q.data;
  const w = d.weather.days.filter((x) => x.available);
  const tmax = w.map((x) => x.values.temperature_2m_max).filter((v): v is number => v != null);
  const tmin = w.map((x) => x.values.temperature_2m_min).filter((v): v is number => v != null);
  const rainy = w.filter((x) => (x.values.precipitation_sum ?? 0) >= 1).length;
  const air = d.air_quality.days.filter((x) => x.values.us_aqi_max != null);
  const worst = air.reduce<(typeof air)[number] | null>((a, b) => (!a || b.values.us_aqi_max > a.values.us_aqi_max ? b : a), null);
  const unit = d.weather.units.temperature_2m_max ?? "";
  return (
    <div className="space-y-3 text-[12.5px] leading-5 text-text-2">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <span className="font-semibold text-text">{fmtDayMonth(d.window.start)} – {fmtDate(d.window.end)} · Ludhiana</span>
        <StatusBadge status={d.weather.status === "ok" && d.air_quality.status === "ok" ? "ok" : d.weather.status === "unavailable" && d.air_quality.status === "unavailable" ? "unavailable" : "stale"} />
      </div>
      <ul className="space-y-2">
        <li className="flex justify-between gap-3 rounded-lg bg-surface-2/60 px-3 py-2 ring-1 ring-inset ring-border/15">
          <span>Temperature</span><span className="tabular font-semibold text-text">{tmax.length ? `${fmt1(Math.min(...tmin))}–${fmt1(Math.max(...tmax))} ${unit}` : "No forecast available"}</span></li>
        <li className="flex justify-between gap-3 rounded-lg bg-surface-2/60 px-3 py-2 ring-1 ring-inset ring-border/15">
          <span>Days with ≥1 {d.weather.units.precipitation_sum ?? "mm"} rain</span><span className="tabular font-semibold text-text">{w.length ? `${rainy} of ${w.length}` : "No forecast available"}</span></li>
        <li className="flex items-center justify-between gap-3 rounded-lg bg-surface-2/60 px-3 py-2 ring-1 ring-inset ring-border/15">
          <span>Worst US AQI ({d.air_quality.available_dates.count} days forecast)</span>
          {worst ? <Badge tone={aqiTone(worst.us_aqi_category)}>{worst.values.us_aqi_max} · {worst.us_aqi_category} · {fmtWeekday(worst.date)}</Badge>
            : <span className="font-semibold text-text">No forecast available</span>}</li>
      </ul>
      <p>These live forecasts are context only and never feed into the April 2019 admission forecast above. Retrieved {fmtRetrieved(d.weather.retrieved_at)}.</p>
      <a href={hrefFor("/conditions")} className="inline-flex items-center gap-1 font-semibold text-observed-ink hover:underline">Open Weather &amp; AQI <ChevronRight aria-hidden className="h-3.5 w-3.5" /></a>
    </div>
  );
}

export default function CommandCentre() {
  const { windowDays } = useFilters();
  const run = useForecastRun();
  const hist = useHistoricalForecast();
  const ov = useOverview(windowDays);
  const avgKpi = ov.data?.kpis.find((k) => k.id === "avg_daily");

  return (
    <div>
      <Hero />
      {run.isError ? (
        <div className="mb-6"><ErrorState title="The official forecast is unavailable" message={(run.error as Error).message} onRetry={() => run.refetch()} /></div>
      ) : run.data ? (
        <>
          <RunBanner run={run.data} compact />
          <Kpis run={run.data} />
        </>
      ) : (
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">{Array.from({ length: 4 }).map((_, i) => <Skeleton key={i} className="h-[196px]" />)}</div>
      )}

      <div className="mt-6 grid grid-cols-1 gap-6 xl:grid-cols-3">
        <Panel className="xl:col-span-2" title="14-day admission trend"
          subtitle="The last 14 observed days before the 1 Apr 2019 cut-off, then the official forecast with its 80% range. Busiest days ringed; holidays dashed."
          actions={<ProvenanceBadge kind="historical_forecast" />}>
          {hist.data && run.data ? (
            <HistoricalForecastChart busiest={run.data.summary.busiest_days} markHolidays height={420}
              data={{ ...hist.data, observed_before_cutoff: hist.data.observed_before_cutoff.slice(-14) }} />
          )
            : hist.isError ? <ErrorState title="Chart unavailable" message={(hist.error as Error).message} onRetry={() => hist.refetch()} />
              : <Loading label="Running the official forecast" />}
        </Panel>
        <Panel title="Days to watch" subtitle="Busiest forecast days and holidays in the window"
          actions={<a href={hrefFor("/forecast")} className="text-[12px] font-semibold text-observed-ink hover:underline">All 14 days</a>}>
          {run.data ? <DaysToWatch run={run.data} limit={5} /> : <Loading label="Loading days to watch" />}
        </Panel>
      </div>

      <div className="mt-6 grid grid-cols-1 gap-6 lg:grid-cols-2 xl:grid-cols-3">
        <Panel title="Upcoming holidays" subtitle="Official Punjab calendar, 30 days from 1 Apr 2019" actions={<CalendarDays aria-hidden className="h-4 w-4 text-teal-ink" />}>
          {run.data ? <UpcomingHolidays run={run.data} /> : <Loading label="Loading holidays" />}
        </Panel>
        <Panel title="How to read this forecast">
          <ReadingGuide />
        </Panel>
        <Panel title="Weather & air quality" subtitle="Live forecast for the current period – a different period from the admission forecast"
          actions={<Badge tone="teal" icon={CloudSun}>Live · context only</Badge>}>
          <LiveConditionsCard />
        </Panel>
      </div>

      <div className="mt-6 grid grid-cols-1 gap-6 xl:grid-cols-3">
        <Panel className="xl:col-span-2" fill title="Recent observed admissions"
          subtitle={`Raw daily records, ${windowDays === 730 ? "full two years" : `last ${windowDays} days`} to 31 Mar 2019 – before the forecast period. Change the window in the top bar.`}
          actions={<ProvenanceBadge kind="observed" />}>
          {ov.data && avgKpi ? <RecentTrendChart points={ov.data.trend.points} overallMean={avgKpi.value as number} />
            : ov.isError ? <ErrorState title="Could not load the overview" message={(ov.error as Error).message} onRetry={() => ov.refetch()} />
              : <Loading label="Loading trend" />}
        </Panel>
        <Panel title="Data & model caveats" subtitle="Read before acting on any number" actions={<ShieldAlert aria-hidden className="h-4 w-4 text-amber-ink" />}>
          {ov.data ? (
            <>
              <ul className="space-y-2.5">{ov.data.alerts.map((a) => <Callout key={a.id} alert={a} />)}</ul>
              <details className="mt-3 rounded-xl bg-surface-2/50 p-3 ring-1 ring-inset ring-border/15">
                <summary className="flex cursor-pointer items-center gap-2 text-[12.5px] font-semibold text-text">
                  <Lightbulb aria-hidden className="h-3.5 w-3.5 text-teal-ink" />Key facts from the data ({ov.data.insights.length})
                </summary>
                <ul className="mt-2 space-y-2">
                  {ov.data.insights.map((s) => (
                    <li key={s.id} className="text-[12.5px] leading-5 text-text-2">
                      <span className="font-semibold text-text">{s.title}.</span> {s.detail} <ProvenanceBadge kind={s.provenance} className="ml-1 align-middle" />
                    </li>
                  ))}
                </ul>
              </details>
            </>
          ) : <Loading label="Loading caveats" />}
        </Panel>
      </div>

      <p className="mt-6 flex items-start gap-2 text-[12px] leading-5 text-text-3">
        <Activity aria-hidden className="mt-0.5 h-3.5 w-3.5 shrink-0" />
        <span>Historical forecast = output of the frozen <span className="font-mono">forecast.py</span> for cut-off 1 Apr 2019 (a demonstration; the data end
        on 31 Mar 2019). Observed = recorded admission rows (raw, the forecast target). Forecasts describe aggregate demand, not individual patients,
        and are not capacity or staffing recommendations.</span>
      </p>
    </div>
  );
}
