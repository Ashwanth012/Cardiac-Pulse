import { Area, Bar, BarChart, CartesianGrid, Cell, ComposedChart, LabelList, Line, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { fmt1, fmtDayMonth, fmtLong, fmtWeekday } from "../../lib/format";
import { useChartColors, usePrefersReducedMotion } from "../../lib/theme";
import type { AirDay, Conditions, WeatherDay } from "../../lib/types";
import { LegendItem, TooltipCard, TooltipRow } from "./ChartParts";

/** Live environmental forecast charts (context only). Days without provider data are gaps, never filled. */

export type AqiTone = "teal" | "amber" | "rose";
export const aqiTone = (category?: string | null): AqiTone =>
  !category ? "amber" : category === "Good" ? "teal" : category === "Moderate" || category.startsWith("Unhealthy for") ? "amber" : "rose";

const tick = (iso: string) => `${fmtWeekday(iso).slice(0, 2)} ${fmtDayMonth(iso)}`;
const NoData = "No forecast available";

// ============================================================================ temperature
export function TemperatureChart({ data }: { data: Conditions }) {
  const c = useChartColors();
  const reduced = usePrefersReducedMotion();
  const u = data.weather.units.temperature_2m_max ?? data.weather.units.temperature_2m_min ?? "";
  const rows = data.weather.days.map((d: WeatherDay) => ({
    date: d.date, max: d.values.temperature_2m_max ?? null, min: d.values.temperature_2m_min ?? null,
    band: d.values.temperature_2m_max != null && d.values.temperature_2m_min != null
      ? [d.values.temperature_2m_min, d.values.temperature_2m_max] as [number, number] : null,
    available: d.available,
  }));
  const vals = rows.flatMap((r) => [r.max, r.min]).filter((v): v is number => v != null);
  if (!vals.length) return <p className="py-10 text-center text-[13px] text-text-2">{NoData} for temperature.</p>;
  const lo = Math.floor(Math.min(...vals) / 5) * 5 - 5, hi = Math.ceil(Math.max(...vals) / 5) * 5 + 5;
  return (
    <div>
      <div className="mb-3 flex flex-wrap gap-x-5 gap-y-2">
        <LegendItem color={c.rose} label={`Daily maximum (${u})`} />
        <LegendItem color={c.teal} label={`Daily minimum (${u})`} kind="dashed" />
      </div>
      <div className="h-[250px] w-full" role="img"
        aria-label={`Forecast daily temperatures for Ludhiana, ${fmtLong(rows[0].date)} to ${fmtLong(rows[rows.length - 1].date)}: minimum ${fmt1(Math.min(...vals))}${u}, maximum ${fmt1(Math.max(...vals))}${u}.`}>
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={rows} margin={{ top: 8, right: 8, bottom: 0, left: -14 }}>
            <CartesianGrid vertical={false} stroke={c.border} strokeOpacity={0.14} />
            <XAxis dataKey="date" tickFormatter={tick} interval="preserveStartEnd" minTickGap={10} tick={{ fill: c["text-2"], fontSize: 11 }}
              axisLine={{ stroke: c.border, strokeOpacity: 0.3 }} tickLine={false} />
            <YAxis domain={[lo, hi]} ticks={Array.from({ length: (hi - lo) / 5 + 1 }, (_, i) => lo + i * 5)} tickFormatter={(v) => `${v}°`} tick={{ fill: c["text-2"], fontSize: 11.5 }} axisLine={false} tickLine={false} width={44} />
            <Tooltip isAnimationActive={false} cursor={{ stroke: c.border, strokeOpacity: 0.4 }}
              content={({ active, payload }) => {
                if (!active || !payload?.length) return null;
                const r = payload[0].payload as (typeof rows)[number];
                return (
                  <TooltipCard title={fmtLong(r.date)}>
                    {r.available ? (
                      <>
                        {r.max != null && <TooltipRow swatch={c.rose} label="Maximum" value={`${fmt1(r.max)} ${u}`} />}
                        {r.min != null && <TooltipRow swatch={c.teal} dashed label="Minimum" value={`${fmt1(r.min)} ${u}`} />}
                        <p className="pt-1 text-[11px] text-text-3">Live forecast · context only</p>
                      </>
                    ) : <p className="text-[12px] text-text-2">{NoData}</p>}
                  </TooltipCard>
                );
              }} />
            <Area dataKey="band" stroke="none" fill={c.rose} fillOpacity={0.1} isAnimationActive={!reduced} connectNulls={false} />
            <Line dataKey="max" stroke={c.rose} strokeWidth={2.25} dot={{ r: 2.5, fill: c.rose, strokeWidth: 0 }} isAnimationActive={!reduced} connectNulls={false} />
            <Line dataKey="min" stroke={c.teal} strokeWidth={2.25} strokeDasharray="6 4" dot={{ r: 2.5, fill: c.teal, strokeWidth: 0 }} isAnimationActive={!reduced} connectNulls={false} />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

// ============================================================================ precipitation
export function PrecipitationChart({ data }: { data: Conditions }) {
  const c = useChartColors();
  const reduced = usePrefersReducedMotion();
  const u = data.weather.units.precipitation_sum ?? "mm";
  const pu = data.weather.units.precipitation_probability_max ?? "%";
  const rows = data.weather.days.map((d) => ({ date: d.date, mm: d.values.precipitation_sum ?? null,
    chance: d.values.precipitation_probability_max ?? null, available: d.available }));
  const vals = rows.map((r) => r.mm).filter((v): v is number => v != null);
  if (!vals.length) return <p className="py-10 text-center text-[13px] text-text-2">{NoData} for precipitation.</p>;
  const total = vals.reduce((a, b) => a + b, 0);
  return (
    <div>
      {total === 0 && (
        <p className="mb-2 rounded-lg bg-surface-2/60 px-3 py-1.5 text-[12px] text-text-2 ring-1 ring-inset ring-border/15">
          No measurable precipitation is forecast on any of the {vals.length} days (0.0 {u} each day).
        </p>
      )}
      <div className="mb-3 flex flex-wrap gap-x-5 gap-y-2">
        <LegendItem color={c.observed} label={`Daily precipitation (${u})`} kind="bar" />
        <span className="text-[12px] text-text-2">Labels: chance of precipitation ({pu})</span>
      </div>
      <div className="h-[250px] w-full" role="img" aria-label={`Forecast daily precipitation: ${fmt1(total)} ${u} in total over ${vals.length} days with a forecast.`}>
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={rows} margin={{ top: 18, right: 8, bottom: 0, left: -14 }} barCategoryGap="25%">
            <CartesianGrid vertical={false} stroke={c.border} strokeOpacity={0.14} />
            <XAxis dataKey="date" tickFormatter={tick} interval="preserveStartEnd" minTickGap={10} tick={{ fill: c["text-2"], fontSize: 11 }}
              axisLine={{ stroke: c.border, strokeOpacity: 0.3 }} tickLine={false} />
            <YAxis domain={[0, Math.max(5, Math.ceil(Math.max(...vals) / 5) * 5)]} tickFormatter={(v) => `${v}`} tick={{ fill: c["text-2"], fontSize: 11.5 }}
              axisLine={false} tickLine={false} width={44} label={{ value: u, angle: -90, position: "insideLeft", offset: 20, fill: c["text-2"], fontSize: 11 }} />
            <Tooltip isAnimationActive={false} cursor={{ fill: c.border, fillOpacity: 0.08 }}
              content={({ active, payload }) => {
                if (!active || !payload?.length) return null;
                const r = payload[0].payload as (typeof rows)[number];
                return (
                  <TooltipCard title={fmtLong(r.date)}>
                    {r.available ? (
                      <>
                        {r.mm != null && <TooltipRow label="Precipitation" value={`${fmt1(r.mm)} ${u}`} />}
                        {r.chance != null && <TooltipRow label="Chance (max)" value={`${Math.round(r.chance)}${pu}`} />}
                      </>
                    ) : <p className="text-[12px] text-text-2">{NoData}</p>}
                  </TooltipCard>
                );
              }} />
            <Bar dataKey="mm" fill={c.observed} fillOpacity={0.75} radius={[4, 4, 0, 0]} minPointSize={2} isAnimationActive={!reduced}>
              <LabelList dataKey="chance" position="top" fontSize={10.5} fill={c["text-2"]} formatter={(v: number | null) => (v == null ? "" : `${Math.round(v)}%`)} />
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

// ============================================================================ US AQI
export function AqiChart({ data }: { data: Conditions }) {
  const c = useChartColors();
  const reduced = usePrefersReducedMotion();
  const tone = (cat?: string) => c[aqiTone(cat)];
  const vals = data.air_quality.days.map((d) => d.values.us_aqi_max).filter((v): v is number => v != null);
  const top = Math.max(250, ...vals.map((v) => Math.ceil((v + 30) / 50) * 50));
  // days without an air-quality forecast get an outlined, empty placeholder (not a value) so the gap is visible
  const rows = data.air_quality.days.map((d: AirDay) => ({ date: d.date, aqi: d.values.us_aqi_max ?? null, cat: d.us_aqi_category,
    status: d.status, hours: d.hours_with_data ?? 0, placeholder: d.values.us_aqi_max == null ? Math.round(top * 0.14) : null }));
  return (
    <div>
      <div className="mb-3 flex flex-wrap gap-x-5 gap-y-2">
        <LegendItem color={c.teal} label="Good (0–50)" kind="bar" />
        <LegendItem color={c.amber} label="Moderate / Unhealthy for sensitive groups (51–150)" kind="bar" />
        <LegendItem color={c.rose} label="Unhealthy or worse (151+)" kind="bar" />
        <span className="inline-flex items-center gap-2 text-[12px] font-medium text-text-2">
          <span aria-hidden className="h-3 w-3 rounded-sm border border-dashed" style={{ borderColor: c["text-3"] }} />{NoData}
        </span>
      </div>
      <div className="h-[260px] w-full" role="img"
        aria-label={`Daily maximum US AQI forecast. ${vals.length} of ${rows.length} days have a forecast` +
          (vals.length ? `, ranging ${Math.min(...vals)} to ${Math.max(...vals)}.` : ".")}>
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={rows} margin={{ top: 18, right: 8, bottom: 0, left: -14 }} barCategoryGap="25%">
            <CartesianGrid vertical={false} stroke={c.border} strokeOpacity={0.14} />
            <XAxis dataKey="date" tickFormatter={tick} interval="preserveStartEnd" minTickGap={10} tick={{ fill: c["text-2"], fontSize: 11 }}
              axisLine={{ stroke: c.border, strokeOpacity: 0.3 }} tickLine={false} />
            <YAxis domain={[0, top]} ticks={[0, 50, 100, 150, 200, 300].filter((t) => t <= top)} tick={{ fill: c["text-2"], fontSize: 11.5 }}
              axisLine={false} tickLine={false} width={44} />
            {[50, 100, 150, 200].map((t) => <ReferenceLine key={t} y={t} stroke={c.border} strokeOpacity={0.35} strokeDasharray="3 4" />)}
            <Tooltip isAnimationActive={false} cursor={{ fill: c.border, fillOpacity: 0.08 }}
              content={({ active, payload }) => {
                if (!active || !payload?.length) return null;
                const r = payload[0].payload as (typeof rows)[number];
                return (
                  <TooltipCard title={fmtLong(r.date)}>
                    {r.aqi != null ? (
                      <>
                        <TooltipRow label="US AQI (daily max)" value={r.aqi} />
                        <p className="text-[12px] font-semibold text-text">{r.cat}</p>
                      </>
                    ) : (
                      <p className="text-[12px] text-text-2">
                        {NoData}{r.status === "partial" ? ` – only ${r.hours} of 24 hourly values, not summarised` : " – beyond the provider's air-quality horizon"}
                      </p>
                    )}
                  </TooltipCard>
                );
              }} />
            <Bar dataKey="aqi" stackId="a" radius={[4, 4, 0, 0]} isAnimationActive={!reduced}>
              {rows.map((r) => <Cell key={r.date} fill={tone(r.cat)} fillOpacity={0.85} />)}
              <LabelList dataKey="aqi" position="top" fontSize={11} fill={c.text} />
            </Bar>
            <Bar dataKey="placeholder" stackId="a" fill="transparent" isAnimationActive={false} legendType="none">
              {rows.map((r) => <Cell key={r.date} fill="none" stroke={r.placeholder ? c["text-3"] : "none"} strokeDasharray="3 3" />)}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
