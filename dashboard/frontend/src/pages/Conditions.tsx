import clsx from "clsx";
import { CircleAlert, CloudRain, CloudSun, Droplets, ExternalLink, History, MapPin, RefreshCw, Thermometer, Wind } from "lucide-react";
import type { ReactNode } from "react";
import { AqiChart, PrecipitationChart, TemperatureChart, aqiTone } from "../components/charts/EnvCharts";
import { Badge } from "../components/ui/Badge";
import { Panel } from "../components/ui/Panel";
import { ErrorState, Loading } from "../components/ui/States";
import { useConditions } from "../lib/api";
import { fmt1, fmtDate, fmtDayMonth, fmtWeekday } from "../lib/format";
import { hrefFor } from "../lib/router";
import type { Conditions as ConditionsData, ProviderStatus } from "../lib/types";

const NO_DATA = "No forecast available";
const POLLUTANTS = ["pm2_5", "pm10", "nitrogen_dioxide", "ozone", "carbon_monoxide", "sulphur_dioxide"] as const;

/** Retrieval time shown in the location's timezone (IST), e.g. "24 Nov 2026, 00:18 IST". */
export const fmtRetrieved = (iso: string | null) =>
  iso ? `${new Date(iso).toLocaleString("en-GB", { timeZone: "Asia/Kolkata", day: "numeric", month: "short", year: "numeric", hour: "2-digit", minute: "2-digit", hour12: false })} IST` : "never";
const fmtSpan = (a: string | null, b: string | null) => (a && b ? (a === b ? fmtDate(a) : `${fmtDayMonth(a)} – ${fmtDate(b)}`) : "none");

export function StatusBadge({ status }: { status: ProviderStatus }) {
  if (status === "ok") return <Badge tone="ok">Live · up to date</Badge>;
  if (status === "stale") return <Badge tone="amber" icon={History}>Stale · last good copy</Badge>;
  return <Badge tone="rose" icon={CircleAlert}>Unavailable</Badge>;
}

function Card({ icon: Icon, label, value, sub, tone = "neutral" }: { icon: typeof Thermometer; label: string; value: ReactNode; sub: ReactNode; tone?: "neutral" | "teal" | "amber" | "rose" }) {
  return (
    <div className={clsx("rounded-xl px-4 py-3 ring-1 ring-inset",
      tone === "neutral" ? "bg-surface-2/60 ring-border/15" : tone === "teal" ? "bg-teal/[0.08] ring-teal/30" : tone === "amber" ? "bg-amber/[0.08] ring-amber/30" : "bg-rose/[0.08] ring-rose/30")}>
      <p className="flex items-center gap-1.5 text-[12px] font-medium text-text-2"><Icon aria-hidden className="h-3.5 w-3.5" />{label}</p>
      <p className="tabular mt-1 text-[21px] font-semibold leading-tight text-text">{value}</p>
      <p className="mt-1 text-[11.5px] leading-4 text-text-2">{sub}</p>
    </div>
  );
}

// ============================================================================ header + separation
function Header({ d }: { d: ConditionsData }) {
  return (
    <section className="panel relative mb-6 overflow-hidden p-6 animate-fade-up" aria-labelledby="cond-title">
      <div aria-hidden className="pointer-events-none absolute -right-16 -top-24 h-64 w-64 rounded-full bg-teal opacity-[0.12] blur-3xl" />
      <div className="relative">
        <div className="flex flex-wrap gap-2">
          <Badge tone="teal" icon={CloudSun}>Live forecast · current period</Badge>
          <Badge tone="neutral" icon={MapPin}>{d.location.name}</Badge>
          <Badge tone="neutral">Context only</Badge>
        </div>
        <h2 id="cond-title" className="mt-4 text-2xl font-semibold tracking-tight text-text sm:text-[28px]">
          Weather &amp; air quality, <span className="bg-gradient-to-r from-teal-ink to-observed-ink bg-clip-text text-transparent">next 14 days</span>
        </h2>
        <p className="mt-2 max-w-3xl text-[14px] leading-6 text-text-2">
          Live forecasts for <strong className="text-text">{fmtDayMonth(d.window.start)} – {fmtDate(d.window.end)}</strong> from Open-Meteo, refreshed at most
          hourly. Weather and air-quality forecasts cover different numbers of days; each section shows exactly which dates it has.
        </p>
      </div>
      <div className="relative mt-5 grid grid-cols-1 gap-3 md:grid-cols-2">
        <div className="rounded-xl bg-teal/[0.08] p-3.5 ring-1 ring-inset ring-teal/30">
          <p className="text-[11px] font-semibold uppercase tracking-wide text-teal-ink">This page</p>
          <p className="mt-0.5 text-[14px] font-semibold text-text">Live weather &amp; air-quality forecast · {fmtDayMonth(d.window.start)} – {fmtDate(d.window.end)}</p>
        </div>
        <div className="rounded-xl bg-forecast/[0.08] p-3.5 ring-1 ring-inset ring-forecast/30">
          <p className="text-[11px] font-semibold uppercase tracking-wide text-forecast-ink">Admission forecast (other pages)</p>
          <p className="mt-0.5 text-[14px] font-semibold text-text">Historical demonstration · 1–14 Apr 2019</p>
        </div>
      </div>
      <p className="relative mt-3 flex items-start gap-2 text-[12.5px] leading-5 text-text-2">
        <CircleAlert aria-hidden className="mt-0.5 h-3.5 w-3.5 shrink-0 text-amber-ink" />
        <span>
          Weather and air quality are <strong className="text-text">not used by the admissions model</strong>. These live values do not change, and are
          unrelated to, the 1–14 Apr 2019 admission demonstration on the{" "}
          <a className="font-semibold text-observed-ink underline-offset-2 hover:underline" href={hrefFor("/forecast")}>14-Day Forecast</a> page.
        </span>
      </p>
    </section>
  );
}

function SourceStatus({ d }: { d: ConditionsData }) {
  const rows = [
    { key: "Weather", b: d.weather, horizon: "up to 16 days ahead" },
    { key: "Air quality", b: d.air_quality, horizon: "about 4–5 days ahead (CAMS global forecast)" },
  ];
  return (
    <div className="mb-6 grid grid-cols-1 gap-4 lg:grid-cols-2">
      {rows.map(({ key, b, horizon }) => (
        <div key={key} className="panel p-4" aria-label={`${key} forecast source status`}>
          <div className="flex flex-wrap items-center justify-between gap-2">
            <p className="text-[14px] font-semibold text-text">{key} forecast</p>
            <StatusBadge status={b.status} />
          </div>
          <dl className="mt-3 grid grid-cols-2 gap-x-4 gap-y-2 text-[12.5px]">
            <div><dt className="text-text-3">Forecast dates available</dt>
              <dd className="font-semibold text-text">{fmtSpan(b.available_dates.first, b.available_dates.last)} <span className="font-normal text-text-2">({b.available_dates.count} of {d.window.days} days)</span></dd></div>
            <div><dt className="text-text-3">Retrieved</dt><dd className="font-semibold text-text">{fmtRetrieved(b.retrieved_at)}</dd></div>
            <div><dt className="text-text-3">Provider horizon</dt><dd className="text-text-2">{horizon}</dd></div>
            <div><dt className="text-text-3">Grid point</dt><dd className="tabular text-text-2">
              {b.grid.latitude != null ? `${b.grid.latitude.toFixed(2)}° N, ${b.grid.longitude?.toFixed(2)}° E` : "—"}</dd></div>
          </dl>
          {b.status === "stale" && (
            <p className="mt-3 rounded-lg bg-amber/[0.08] px-3 py-2 text-[12px] leading-5 text-text-2 ring-1 ring-inset ring-amber/30">
              Refresh failed ({b.error}). Showing the last good copy, retrieved {fmtRetrieved(b.retrieved_at)} ({Math.round((b.age_minutes ?? 0) / 60)} h ago).
            </p>
          )}
          {b.status === "unavailable" && (
            <p className="mt-3 rounded-lg bg-rose/[0.08] px-3 py-2 text-[12px] leading-5 text-text-2 ring-1 ring-inset ring-rose/30">
              {key} forecast unavailable{b.error ? ` (${b.error})` : ""}. No values are shown – nothing is filled in from past observations.
            </p>
          )}
          {b.issues.length > 0 && <ul className="mt-2 list-disc pl-5 text-[11.5px] text-text-3">{b.issues.map((i) => <li key={i}>{i}</li>)}</ul>}
        </div>
      ))}
    </div>
  );
}

// ============================================================================ summaries
function Summary({ d }: { d: ConditionsData }) {
  const w = d.weather.days.filter((x) => x.available);
  const u = d.weather.units;
  const nums = (k: keyof (typeof w)[number]["values"]) => w.map((x) => x.values[k]).filter((v): v is number => v != null);
  const tmax = nums("temperature_2m_max"), tmin = nums("temperature_2m_min"), rain = nums("precipitation_sum"),
    chance = nums("precipitation_probability_max"), hum = nums("relative_humidity_2m_mean"), wind = nums("wind_speed_10m_max"),
    gust = nums("wind_gusts_10m_max");
  const rainyDays = w.filter((x) => (x.values.precipitation_sum ?? 0) >= 1).length;
  const air = d.air_quality.days.filter((x) => x.values.us_aqi_max != null);
  const worst = air.reduce<(typeof air)[number] | null>((a, b) => (!a || b.values.us_aqi_max > a.values.us_aqi_max ? b : a), null);
  const today = d.air_quality.days.find((x) => x.date === d.today);
  return (
    <div className="mb-6 grid grid-cols-2 gap-3 lg:grid-cols-3 2xl:grid-cols-6">
      <Card icon={Thermometer} label="Temperature range" value={tmax.length ? `${fmt1(Math.min(...tmin))}–${fmt1(Math.max(...tmax))} ${u.temperature_2m_max ?? ""}` : NO_DATA}
        sub={tmax.length ? `daily lows to highs over ${w.length} forecast days` : "weather forecast unavailable"} />
      <Card icon={CloudRain} label="Precipitation" value={rain.length ? `${fmt1(rain.reduce((a, b) => a + b, 0))} ${u.precipitation_sum ?? ""}` : NO_DATA}
        sub={rain.length ? `${rainyDays} day${rainyDays === 1 ? "" : "s"} with ≥1 ${u.precipitation_sum}; highest chance ${chance.length ? Math.max(...chance) : "–"}${u.precipitation_probability_max ?? ""}` : "weather forecast unavailable"} />
      <Card icon={Droplets} label="Relative humidity" value={hum.length ? `${Math.min(...hum)}–${Math.max(...hum)}${u.relative_humidity_2m_mean ?? ""}` : NO_DATA}
        sub="range of daily means" />
      <Card icon={Wind} label="Wind" value={wind.length ? `${fmt1(Math.max(...wind))} ${u.wind_speed_10m_max ?? ""}` : NO_DATA}
        sub={gust.length ? `highest daily max; gusts up to ${fmt1(Math.max(...gust))} ${u.wind_gusts_10m_max ?? ""}` : "highest daily max"} />
      <Card icon={CloudSun} label={`US AQI today (${fmtDayMonth(d.today)})`} tone={today?.us_aqi_category ? aqiTone(today.us_aqi_category) : "neutral"}
        value={today?.values.us_aqi_max != null ? today.values.us_aqi_max : NO_DATA} sub={today?.us_aqi_category ?? "air-quality forecast unavailable for today"} />
      <Card icon={CircleAlert} label="Worst forecast US AQI" tone={worst ? aqiTone(worst.us_aqi_category) : "neutral"}
        value={worst ? worst.values.us_aqi_max : NO_DATA} sub={worst ? `${fmtWeekday(worst.date)} ${fmtDayMonth(worst.date)} · ${worst.us_aqi_category}` : "no air-quality forecast available"} />
    </div>
  );
}

// ============================================================================ pollutants
function PollutantTable({ d }: { d: ConditionsData }) {
  const a = d.air_quality;
  const present = POLLUTANTS.filter((p) => a.units[p]);
  if (!present.length) return <p className="text-[13px] text-text-2">{NO_DATA}: the air-quality provider returned no pollutant data.</p>;
  return (
    <div className="overflow-x-auto rounded-xl ring-1 ring-inset ring-border/15">
      <table className="w-full min-w-[720px] text-[12.5px]">
        <caption className="sr-only">Daily mean pollutant concentrations forecast, with daily maximum in brackets</caption>
        <thead className="bg-surface-2"><tr className="text-left text-[11px] font-semibold uppercase tracking-wide text-text-3">
          <th scope="col" className="px-3 py-2">Date</th>
          <th scope="col" className="px-3 py-2">US AQI (max)</th>
          {present.map((p) => <th key={p} scope="col" className="px-3 py-2 text-right normal-case">{a.labels[p]} <span className="font-normal">({a.units[p]})</span></th>)}
        </tr></thead>
        <tbody>
          {a.days.map((day) => (
            <tr key={day.date} className={clsx("border-t border-border/10", !day.available && "text-text-3")}>
              <td className="whitespace-nowrap px-3 py-2 font-medium text-text">{fmtWeekday(day.date)} {fmtDayMonth(day.date)}</td>
              {day.available ? (
                <>
                  <td className="px-3 py-2"><Badge tone={aqiTone(day.us_aqi_category)}>{day.values.us_aqi_max} · {day.us_aqi_category}</Badge></td>
                  {present.map((p) => (
                    <td key={p} className="tabular px-3 py-2 text-right text-text">
                      {day.values[`${p}_mean`] != null ? fmt1(day.values[`${p}_mean`]) : "—"}
                      {day.values[`${p}_max`] != null && <span className="text-text-3"> ({fmt1(day.values[`${p}_max`])})</span>}
                    </td>
                  ))}
                </>
              ) : (
                <td colSpan={present.length + 1} className="px-3 py-2 italic text-text-2">
                  {NO_DATA}{day.status === "partial" ? ` – only ${day.hours_with_data} of 24 hourly values, not summarised` : " – beyond the air-quality forecast horizon"}
                </td>
              )}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

// ============================================================================ daily breakdown
function DailyTable({ d }: { d: ConditionsData }) {
  const u = d.weather.units;
  const air = new Map(d.air_quality.days.map((x) => [x.date, x]));
  const f = (v: number | undefined, unit?: string) => (v == null ? "—" : `${fmt1(v)}${unit ? ` ${unit}` : ""}`);
  return (
    <div className="overflow-x-auto rounded-xl ring-1 ring-inset ring-border/15">
      <table className="w-full min-w-[860px] text-[12.5px]">
        <caption className="sr-only">Daily live forecast for Ludhiana: weather and US AQI</caption>
        <thead className="bg-surface-2"><tr className="text-left text-[11px] font-semibold uppercase tracking-wide text-text-3">
          {["Date", `Max / min (${u.temperature_2m_max ?? "°C"})`, `Humidity (${u.relative_humidity_2m_mean ?? "%"})`, `Precipitation (${u.precipitation_sum ?? "mm"})`,
            `Chance (${u.precipitation_probability_max ?? "%"})`, `Wind / gusts (${u.wind_speed_10m_max ?? "km/h"})`, "US AQI (daily max)"].map((h) =>
            <th key={h} scope="col" className="px-3 py-2 normal-case">{h}</th>)}
        </tr></thead>
        <tbody>
          {d.weather.days.map((w) => {
            const a = air.get(w.date);
            const v = w.values;
            return (
              <tr key={w.date} className="border-t border-border/10">
                <td className="whitespace-nowrap px-3 py-2 font-medium text-text">{fmtWeekday(w.date)} {fmtDayMonth(w.date)}{w.date === d.today && <span className="ml-1.5 text-[11px] text-teal-ink">today</span>}</td>
                {w.available ? (
                  <>
                    <td className="tabular px-3 py-2 text-text">{f(v.temperature_2m_max)} / {f(v.temperature_2m_min)}</td>
                    <td className="tabular px-3 py-2 text-text-2">{v.relative_humidity_2m_mean ?? "—"}</td>
                    <td className="tabular px-3 py-2 text-text-2">{f(v.precipitation_sum)}</td>
                    <td className="tabular px-3 py-2 text-text-2">{v.precipitation_probability_max ?? "—"}</td>
                    <td className="tabular px-3 py-2 text-text-2">{f(v.wind_speed_10m_max)} / {f(v.wind_gusts_10m_max)}</td>
                  </>
                ) : <td colSpan={5} className="px-3 py-2 italic text-text-2">{NO_DATA} (weather)</td>}
                <td className="px-3 py-2">
                  {a?.available ? <Badge tone={aqiTone(a.us_aqi_category)}>{a.values.us_aqi_max} · {a.us_aqi_category}</Badge>
                    : <span className="italic text-text-3">{a?.status === "partial" ? `Partial data (${a.hours_with_data}/24 h)` : NO_DATA}</span>}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

// ============================================================================ attribution
function Attribution({ d }: { d: ConditionsData }) {
  return (
    <footer className="mt-6 rounded-xl bg-surface-2/50 p-4 text-[12px] leading-5 text-text-2 ring-1 ring-inset ring-border/15">
      <p className="font-semibold text-text">Sources &amp; attribution</p>
      <ul className="mt-1 space-y-1">
        {[d.weather.attribution, d.air_quality.attribution].map((a) => (
          <li key={a.text}>
            {a.text} Licence: {a.licence}.{" "}
            <a className="inline-flex items-center gap-0.5 font-semibold text-observed-ink hover:underline" href={a.url} target="_blank" rel="noreferrer noopener">
              Open-Meteo<ExternalLink aria-hidden className="h-3 w-3" /></a>
          </li>
        ))}
      </ul>
      <p className="mt-2 text-text-3">
        Air quality uses the CAMS (Copernicus) global model at roughly 45 km resolution, so values describe the wider Ludhiana area, not a single street.
        {" "}{d.aqi_scale.note} Daily US AQI = maximum of the hourly values; pollutants = daily mean (maximum in brackets) of hourly values; a day needs at
        least {d.air_quality.aggregation?.min_hours ?? 18} of 24 hourly values to be summarised. The server keeps each response for {d.cache_policy.ttl_minutes} minutes;
        if a refresh fails, the last good copy is shown as stale for up to {d.cache_policy.stale_max_hours} hours.
      </p>
    </footer>
  );
}

// ============================================================================ page
export default function Conditions() {
  const q = useConditions();
  if (q.isLoading) return <Loading label="Loading live weather and air-quality forecasts" />;
  if (q.isError || !q.data)
    return (
      <div className="space-y-4">
        <ErrorState title="Live conditions are unavailable" message={(q.error as Error)?.message ?? "Unknown error"} onRetry={() => q.refetch()} />
        <p className="text-[12.5px] text-text-2">The rest of the dashboard, including the admission forecast, is unaffected.</p>
      </div>
    );
  const d = q.data;
  return (
    <div>
      <Header d={d} />
      <SourceStatus d={d} />
      <Summary d={d} />

      <div className="mb-6 grid grid-cols-1 gap-6 xl:grid-cols-2">
        <Panel title="Temperature" subtitle={`Daily maximum and minimum, ${fmtDayMonth(d.window.start)} – ${fmtDate(d.window.end)}`}
          actions={<StatusBadge status={d.weather.status} />}>
          <TemperatureChart data={d} />
        </Panel>
        <Panel title="Precipitation" subtitle="Daily total, with the day's highest chance of precipitation above each bar"
          actions={<StatusBadge status={d.weather.status} />}>
          <PrecipitationChart data={d} />
        </Panel>
      </div>

      <Panel className="mb-6" title="Air quality – US AQI"
        subtitle={d.air_quality.available_dates.count
          ? `Daily maximum US AQI. Forecast available for ${fmtSpan(d.air_quality.available_dates.first, d.air_quality.available_dates.last)} (${d.air_quality.available_dates.count} of ${d.window.days} days); later dates have no forecast.`
          : "No air-quality forecast is available right now; no values are shown."}
        actions={<StatusBadge status={d.air_quality.status} />}>
        <AqiChart data={d} />
        <p className="mt-2 text-[12px] leading-5 text-text-2">US AQI categories (EPA): Good 0–50 · Moderate 51–100 · Unhealthy for sensitive groups 101–150 · Unhealthy 151–200 ·
          Very unhealthy 201–300 · Hazardous 301+. {d.aqi_scale.note}</p>
      </Panel>

      <Panel className="mb-6" title="Pollutants" subtitle="Daily mean concentration (daily maximum in brackets), units as supplied by the provider"
        actions={<StatusBadge status={d.air_quality.status} />}>
        <PollutantTable d={d} />
      </Panel>

      <Panel title="Daily breakdown" subtitle={`All ${d.window.days} days of the current window. Retrieved: weather ${fmtRetrieved(d.weather.retrieved_at)}, air quality ${fmtRetrieved(d.air_quality.retrieved_at)}.`}
        actions={<button type="button" onClick={() => q.refetch()} className="inline-flex items-center gap-1.5 rounded-lg bg-surface-2 px-2.5 py-1 text-[12px] font-semibold text-text ring-1 ring-inset ring-border/25 hover:bg-surface">
          <RefreshCw aria-hidden className={clsx("h-3.5 w-3.5", q.isFetching && "animate-spin")} /> Check for updates</button>}>
        <DailyTable d={d} />
      </Panel>

      <Attribution d={d} />
    </div>
  );
}
