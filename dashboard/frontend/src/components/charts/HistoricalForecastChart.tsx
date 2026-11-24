import { Area, CartesianGrid, ComposedChart, Line, ReferenceArea, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { fmt1, fmtDayMonth, fmtLong } from "../../lib/format";
import { useChartColors, usePrefersReducedMotion } from "../../lib/theme";
import type { HistoricalForecast } from "../../lib/types";
import { LegendItem, TooltipCard, TooltipRow } from "./ChartParts";

interface Row {
  date: string;
  observed?: number;
  incomplete?: boolean;
  mean?: number;
  band?: [number, number];
  lower?: number;
  upper?: number;
  holidays?: string[];
}

/**
 * Observed records BEFORE the cut-off and the historical forecast AFTER it.
 * No actual outcomes exist (or are requested) for the forecast dates.
 */
export function HistoricalForecastChart({ data, busiest = [], markHolidays = false, height = 320 }: {
  data: HistoricalForecast;
  /** dates ranked busiest within this forecast window (descriptive; highlighted with a ring) */
  busiest?: string[];
  /** draw a dashed marker at each holiday in the forecast window */
  markHolidays?: boolean;
  height?: number;
}) {
  const c = useChartColors();
  const reduced = usePrefersReducedMotion();
  const rows: Row[] = [
    ...data.observed_before_cutoff.map((o) => ({ date: o.date, observed: o.admissions, incomplete: o.potentially_incomplete })),
    ...data.forecast.map((f) => ({ date: f.date, mean: f.mean, band: [f.lower_80, f.upper_80] as [number, number],
      lower: f.lower_80, upper: f.upper_80, holidays: f.holidays })),
  ];
  const lastObs = data.observed_before_cutoff[data.observed_before_cutoff.length - 1];
  const end = data.forecast[data.forecast.length - 1].date;
  const ymax = Math.ceil((Math.max(...rows.map((r) => r.upper ?? r.observed ?? 0)) + 4) / 10) * 10;
  const avg = data.forecast.reduce((s, f) => s + f.mean, 0) / data.forecast.length;
  const summary = `Observed daily records up to ${fmtLong(lastObs.date)}, then a historical forecast for ${fmtLong(data.forecast[0].date)} ` +
    `to ${fmtLong(end)} averaging ${fmt1(avg)} per day with an 80% prediction interval.`;

  const dot = (props: { cx?: number; cy?: number; payload?: Row; index?: number }) => {
    const { cx, cy, payload } = props;
    if (cx == null || cy == null || !payload?.incomplete) return <g key={`d-${props.index}`} />;
    return <circle key={`d-${props.index}`} cx={cx} cy={cy} r={5.5} fill={c.surface} stroke={c.amber} strokeWidth={2.5} />;
  };

  return (
    <div>
      <div className="mb-3 flex flex-wrap gap-x-5 gap-y-2">
        <LegendItem color={c.observed} label="Observed before cut-off (raw)" />
        <LegendItem color={c.forecast} label="Forecast mean (M4)" kind="dashed" />
        <LegendItem color={c.forecast} label="80% prediction interval" kind="band" />
        <LegendItem color={c.amber} label="31 Mar 2019 potentially incomplete" kind="ring" />
        {busiest.length > 0 && <LegendItem color={c.text} label="Busiest days in this forecast" kind="ring" />}
        {markHolidays && <LegendItem color={c.text} label="Holiday (official calendar)" kind="dashed" />}
      </div>
      <div className="w-full" style={{ height }} role="img" aria-label={summary}>
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={rows} margin={{ top: 22, right: 12, bottom: 0, left: -12 }}>
            <CartesianGrid vertical={false} stroke={c.border} strokeOpacity={0.14} />
            <ReferenceArea x1={data.cutoff} x2={end} fill={c.forecast} fillOpacity={0.05} ifOverflow="extendDomain" />
            <XAxis dataKey="date" tickFormatter={fmtDayMonth} minTickGap={26} tick={{ fill: c["text-2"], fontSize: 11.5 }}
              axisLine={{ stroke: c.border, strokeOpacity: 0.3 }} tickLine={false} />
            <YAxis domain={[0, ymax]} tick={{ fill: c["text-2"], fontSize: 11.5 }} axisLine={false} tickLine={false} width={44} allowDecimals={false} />
            {markHolidays && data.forecast.filter((f) => f.holidays.length).map((f) => (
              <ReferenceLine key={`hol-${f.date}`} x={f.date} stroke={c.text} strokeDasharray="2 4" strokeWidth={1.5} />
            ))}
            <ReferenceLine x={data.cutoff} stroke={c["text-3"]} strokeDasharray="3 4"
              label={{ value: "Cut-off 1 Apr 2019", position: "insideTopLeft", fill: c["text-2"], fontSize: 11.5, offset: 8 }} />
            <Tooltip isAnimationActive={false} cursor={{ stroke: c.border, strokeOpacity: 0.4 }}
              content={({ active, payload }) => {
                if (!active || !payload?.length) return null;
                const r = payload[0].payload as Row;
                return (
                  <TooltipCard title={fmtLong(r.date)}>
                    {r.observed != null && <TooltipRow swatch={c.observed} label="Observed (raw)" value={r.observed} />}
                    {r.incomplete && <p className="text-[11.5px] font-semibold text-amber-ink">Potentially incomplete day</p>}
                    {r.mean != null && (
                      <>
                        <TooltipRow swatch={c.forecast} dashed label="Forecast mean" value={fmt1(r.mean)} />
                        <TooltipRow label="80% interval" value={`${fmt1(r.lower!)} – ${fmt1(r.upper!)}`} />
                        {!!r.holidays?.length && <p className="pt-0.5 text-[11.5px] text-text-2">Holiday: {r.holidays.join(", ")}</p>}
                        {busiest.includes(r.date) && <p className="pt-0.5 text-[11.5px] font-semibold text-text">Busiest #{busiest.indexOf(r.date) + 1} in this forecast</p>}
                        <p className="pt-1 text-[11px] text-text-3">Historical forecast · no actuals shown</p>
                      </>
                    )}
                  </TooltipCard>
                );
              }} />
            <Area dataKey="band" stroke="none" fill={c.forecast} fillOpacity={0.2} isAnimationActive={!reduced} animationDuration={900} connectNulls={false} />
            <Line dataKey="observed" type="monotone" stroke={c.observed} strokeWidth={2.25} dot={dot}
              activeDot={{ r: 4, stroke: c.surface, strokeWidth: 2 }} isAnimationActive={!reduced} animationDuration={800} connectNulls={false} />
            <Line dataKey="mean" type="monotone" stroke={c.forecast} strokeWidth={2.5} strokeDasharray="6 4"
              dot={(props: { cx?: number; cy?: number; payload?: Row; index?: number }) => {
                const { cx, cy, payload, index } = props;
                if (cx == null || cy == null || payload?.mean == null) return <g key={`m-${index}`} />;
                const top = busiest.includes(payload.date);
                return (
                  <g key={`m-${index}`}>
                    {top && <circle cx={cx} cy={cy} r={7.5} fill="none" stroke={c.text} strokeWidth={2} />}
                    <circle cx={cx} cy={cy} r={3} fill={c.forecast} stroke={c.surface} strokeWidth={1.5} />
                  </g>
                );
              }} activeDot={{ r: 5, stroke: c.surface, strokeWidth: 2 }}
              isAnimationActive={!reduced} animationDuration={1100} connectNulls={false} />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
