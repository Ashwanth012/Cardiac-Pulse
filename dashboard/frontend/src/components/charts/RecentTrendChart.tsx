import { Bar, CartesianGrid, Cell, ComposedChart, Line, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { fmt1, fmtDayMonth, fmtLong } from "../../lib/format";
import { useChartColors, usePrefersReducedMotion } from "../../lib/theme";
import type { TrendPoint } from "../../lib/types";
import { LegendItem, TooltipCard, TooltipRow } from "./ChartParts";

interface Props {
  points: TrendPoint[];
  overallMean: number;
}

/** Observed raw daily records (bars) with the trailing 28-day mean (line). Single y-axis. */
export function RecentTrendChart({ points, overallMean }: Props) {
  const c = useChartColors();
  const reduced = usePrefersReducedMotion();
  const last = points[points.length - 1];
  const max = Math.max(...points.map((p) => p.admissions));
  const summary = `Daily raw admission records from ${fmtLong(points[0].date)} to ${fmtLong(last.date)}. ` +
    `Range ${Math.min(...points.map((p) => p.admissions))} to ${max} per day; latest 28-day mean ${last.trailing_28d_mean ? fmt1(last.trailing_28d_mean) : "n/a"}.`;
  const vals = points.map((p) => p.admissions);
  const mean = vals.reduce((a, b) => a + b, 0) / vals.length;
  const hi = points.reduce((a, b) => (b.admissions > a.admissions ? b : a));
  const lo = points.reduce((a, b) => (b.admissions < a.admissions ? b : a));
  const flagged = points.filter((p) => p.low_count_flag || p.potentially_incomplete).length;
  const stats = [
    { label: "Window mean", value: fmt1(mean), sub: "records / day" },
    { label: "Highest day", value: String(hi.admissions), sub: fmtLong(hi.date) },
    { label: "Lowest day", value: String(lo.admissions), sub: `${fmtLong(lo.date)}${lo.potentially_incomplete ? " · potentially incomplete" : ""}` },
    { label: "Flagged days", value: String(flagged), sub: "low-count or potentially incomplete" },
  ];
  return (
    <div className="flex h-full flex-1 flex-col">
      <div className="mb-3 flex flex-wrap gap-x-5 gap-y-2">
        <LegendItem color={c.observed} label="Daily records (raw)" kind="bar" />
        <LegendItem color={c.observed} label="Trailing 28-day mean" />
        <LegendItem color={c["text-3"]} label={`Two-year average ${fmt1(overallMean)}`} kind="dashed" />
        <LegendItem color={c.amber} label="Potentially incomplete / low-count day" kind="ring" />
      </div>
      <div className="min-h-[300px] w-full flex-1" role="img" aria-label={summary}>
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={points} margin={{ top: 8, right: 8, bottom: 0, left: -12 }} barCategoryGap={points.length > 200 ? 0 : 1}>
            <CartesianGrid vertical={false} stroke={c.border} strokeOpacity={0.14} />
            <XAxis dataKey="date" tickFormatter={fmtDayMonth} minTickGap={28} tick={{ fill: c["text-2"], fontSize: 11.5 }}
              axisLine={{ stroke: c.border, strokeOpacity: 0.3 }} tickLine={false} />
            <YAxis domain={[0, Math.ceil((max + 4) / 10) * 10]} tick={{ fill: c["text-2"], fontSize: 11.5 }} axisLine={false} tickLine={false}
              allowDecimals={false} width={44} />
            <ReferenceLine y={overallMean} stroke={c["text-3"]} strokeDasharray="5 5" strokeWidth={1.5} />
            <Tooltip cursor={{ fill: c.border, fillOpacity: 0.08 }} isAnimationActive={false}
              content={({ active, payload }) => {
                if (!active || !payload?.length) return null;
                const p = payload[0].payload as TrendPoint;
                return (
                  <TooltipCard title={fmtLong(p.date)}>
                    <TooltipRow swatch={c.observed} label="Raw records" value={p.admissions} />
                    {p.trailing_28d_mean != null && <TooltipRow swatch={c.observed} label="28-day mean" value={fmt1(p.trailing_28d_mean)} />}
                    {p.potentially_incomplete && <p className="pt-1 text-[11.5px] font-semibold text-amber-ink">Potentially incomplete (last day in file)</p>}
                    {p.low_count_flag && !p.potentially_incomplete && <p className="pt-1 text-[11.5px] font-semibold text-amber-ink">Flagged low-count day</p>}
                  </TooltipCard>
                );
              }} />
            <Bar dataKey="admissions" isAnimationActive={!reduced} animationDuration={700} radius={[3, 3, 0, 0]}>
              {points.map((p) => {
                const flagged = p.potentially_incomplete || p.low_count_flag;
                return (
                  <Cell key={p.date} fill={flagged ? c.amber : c.observed} fillOpacity={flagged ? 0.85 : 0.4}
                    stroke={flagged ? c.amber : "none"} strokeWidth={flagged ? 1.5 : 0} />
                );
              })}
            </Bar>
            <Line dataKey="trailing_28d_mean" type="monotone" stroke={c.observed} strokeWidth={2.5} dot={false}
              activeDot={{ r: 4, strokeWidth: 2, stroke: c.surface }} isAnimationActive={!reduced} animationDuration={900} connectNulls />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
      <dl className="mt-4 grid grid-cols-2 gap-3 md:grid-cols-4">
        {stats.map((s) => (
          <div key={s.label} className="rounded-xl bg-surface-2/60 px-3.5 py-3 ring-1 ring-inset ring-border/15">
            <dt className="text-[11.5px] font-medium text-text-2">{s.label}</dt>
            <dd className="tabular mt-0.5 text-[20px] font-semibold leading-tight text-text">{s.value}</dd>
            <dd className="mt-0.5 text-[11.5px] leading-4 text-text-3">{s.sub}</dd>
          </div>
        ))}
      </dl>
    </div>
  );
}
