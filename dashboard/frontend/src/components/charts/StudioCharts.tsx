import {
  Area, Bar, BarChart, CartesianGrid, Cell, ComposedChart, Line, LineChart, ReferenceArea, ReferenceLine,
  ResponsiveContainer, Scatter, Tooltip, XAxis, YAxis,
} from "recharts";
import { fmt1, fmtDayMonth, fmtLong } from "../../lib/format";
import { useChartColors, usePrefersReducedMotion } from "../../lib/theme";
import type { HoldoutEvaluation, ModelExplain, RetroForecast } from "../../lib/types";
import { LegendItem, TooltipCard, TooltipRow } from "./ChartParts";

const axisTick = (c: ReturnType<typeof useChartColors>) => ({ fill: c["text-2"], fontSize: 11.5 });
const pct = (v: number) => `${Math.round(v * 100)}%`;
const mult = (v: number) => `×${v.toFixed(2)}`;

// ============================================================ model explanation
export function WeekdayMultipliers({ data }: { data: ModelExplain }) {
  const c = useChartColors();
  const reduced = usePrefersReducedMotion();
  const rows = data.weekday.map((w) => ({ ...w, short: w.day.slice(0, 3) }));
  return (
    <div className="h-[210px]" role="img"
      aria-label={`Weekday multipliers relative to Monday: ${rows.map((r) => `${r.day} ${r.multiplier.toFixed(2)}`).join(", ")}`}>
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={rows} margin={{ top: 18, right: 6, bottom: 0, left: -4 }}>
          <CartesianGrid vertical={false} stroke={c.border} strokeOpacity={0.14} />
          <XAxis dataKey="short" tick={axisTick(c)} axisLine={{ stroke: c.border, strokeOpacity: 0.3 }} tickLine={false} />
          <YAxis domain={[0, 1.2]} ticks={[0, 0.5, 1]} tick={axisTick(c)} axisLine={false} tickLine={false} width={40} />
          <ReferenceLine y={1} stroke={c["text-3"]} strokeDasharray="4 4" />
          <Tooltip cursor={{ fill: c.border, fillOpacity: 0.08 }} isAnimationActive={false}
            content={({ active, payload }) => active && payload?.length ? (
              <TooltipCard title={(payload[0].payload as { day: string }).day}>
                <TooltipRow label="Multiplier vs Monday" value={mult(payload[0].value as number)} />
              </TooltipCard>) : null} />
          <Bar dataKey="multiplier" radius={[5, 5, 0, 0]} isAnimationActive={!reduced} fill={c.forecast} fillOpacity={0.75}
            label={{ position: "top", fill: c["text-2"], fontSize: 11, formatter: (v: number) => v.toFixed(2) }} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

export function AnnualCurve({ data }: { data: ModelExplain }) {
  const c = useChartColors();
  const reduced = usePrefersReducedMotion();
  const pts = data.annual_curve;
  if (!pts.length) return <p className="text-[12.5px] text-text-2">No annual cycle: less than a year of history before this cut-off.</p>;
  const peak = pts.reduce((a, b) => (b.multiplier > a.multiplier ? b : a));
  const low = pts.reduce((a, b) => (b.multiplier < a.multiplier ? b : a));
  return (
    <div className="h-[210px]" role="img"
      aria-label={`Annual seasonal multiplier: highest ${mult(peak.multiplier)} around ${peak.label}, lowest ${mult(low.multiplier)} around ${low.label}.`}>
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={pts} margin={{ top: 10, right: 8, bottom: 0, left: -4 }}>
          <CartesianGrid vertical={false} stroke={c.border} strokeOpacity={0.14} />
          <ReferenceArea x1={91} x2={104} fill={c.forecast} fillOpacity={0.1} />
          <XAxis dataKey="day_of_year" type="number" domain={[1, 365]} ticks={[1, 91, 182, 274, 365]}
            tickFormatter={(d: number) => ["Jan", "Apr", "Jul", "Oct", "Dec"][[1, 91, 182, 274, 365].indexOf(d)]}
            tick={axisTick(c)} axisLine={{ stroke: c.border, strokeOpacity: 0.3 }} tickLine={false} />
          <YAxis domain={[0.8, 1.3]} ticks={[0.8, 1, 1.2]} tick={axisTick(c)} axisLine={false} tickLine={false} width={40} />
          <ReferenceLine y={1} stroke={c["text-3"]} strokeDasharray="4 4" />
          <Tooltip isAnimationActive={false} content={({ active, payload }) => active && payload?.length ? (
            <TooltipCard title={(payload[0].payload as { label: string }).label}>
              <TooltipRow swatch={c.forecast} label="Seasonal multiplier" value={mult(payload[0].value as number)} />
            </TooltipCard>) : null} />
          <Line dataKey="multiplier" stroke={c.forecast} strokeWidth={2.5} dot={false} isAnimationActive={!reduced} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

// ============================================================== retrospective
interface RetroRow {
  date: string;
  observed?: number;
  mean?: number;
  band?: [number, number];
  lower?: number;
  upper?: number;
  actual?: number;
  dedup?: number;
  baseline?: number;
  holidays?: string[];
}

export function RetroForecastChart({ data, showDedup, showBaseline }: { data: RetroForecast; showDedup: boolean; showBaseline: boolean }) {
  const c = useChartColors();
  const reduced = usePrefersReducedMotion();
  const act = new Map(data.actual_after_cutoff.points.map((p) => [p.date, p.value]));
  const ded = new Map(data.actual_after_cutoff_dedup.points.map((p) => [p.date, p.value]));
  const base = new Map(data.baseline.points.map((p) => [p.date, p.value]));
  const rows: RetroRow[] = [
    ...data.observed_before_cutoff.map((o) => ({ date: o.date, observed: o.admissions })),
    ...data.forecast.map((f) => ({ date: f.date, mean: f.mean, band: [f.lower_80, f.upper_80] as [number, number],
      lower: f.lower_80, upper: f.upper_80, actual: act.get(f.date), dedup: ded.get(f.date), baseline: base.get(f.date),
      holidays: f.holidays })),
  ];
  const end = data.forecast[data.forecast.length - 1].date;
  const ymax = Math.ceil((Math.max(...rows.map((r) => Math.max(r.upper ?? 0, r.observed ?? 0, r.actual ?? 0))) + 4) / 10) * 10;
  return (
    <div>
      <div className="mb-3 flex flex-wrap gap-x-5 gap-y-2">
        <LegendItem color={c.observed} label="Observed before cut-off (raw)" />
        <LegendItem color={c.forecast} label="Forecast mean (M4)" kind="dashed" />
        <LegendItem color={c.forecast} label="80% interval" kind="band" />
        <LegendItem color={c.observed} label="Actual after cut-off (raw, retrospective)" kind="ring" />
        {showDedup && <LegendItem color={c.dedup} label="Actual de-duplicated (sensitivity)" kind="ring" />}
        {showBaseline && <LegendItem color={c["text-3"]} label="Seasonal-naive baseline" kind="dashed" />}
      </div>
      <div className="h-[330px]" role="img"
        aria-label={`Retrospective forecast from cut-off ${fmtLong(data.cutoff)}: forecast mean, 80% interval and the actual raw records for the 14 days.`}>
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={rows} margin={{ top: 22, right: 12, bottom: 0, left: -12 }}>
            <CartesianGrid vertical={false} stroke={c.border} strokeOpacity={0.14} />
            <ReferenceArea x1={data.cutoff} x2={end} fill={c.forecast} fillOpacity={0.05} />
            <XAxis dataKey="date" tickFormatter={fmtDayMonth} minTickGap={26} tick={axisTick(c)} axisLine={{ stroke: c.border, strokeOpacity: 0.3 }} tickLine={false} />
            <YAxis domain={[0, ymax]} tick={axisTick(c)} axisLine={false} tickLine={false} width={44} allowDecimals={false} />
            <ReferenceLine x={data.cutoff} stroke={c["text-3"]} strokeDasharray="3 4"
              label={{ value: `Cut-off ${fmtDayMonth(data.cutoff)}`, position: "insideTopLeft", fill: c["text-2"], fontSize: 11.5, offset: 8 }} />
            <Tooltip isAnimationActive={false} cursor={{ stroke: c.border, strokeOpacity: 0.4 }}
              content={({ active, payload }) => {
                if (!active || !payload?.length) return null;
                const r = payload[0].payload as RetroRow;
                return (
                  <TooltipCard title={fmtLong(r.date)}>
                    {r.observed != null && <TooltipRow swatch={c.observed} label="Observed (raw)" value={r.observed} />}
                    {r.mean != null && <>
                      <TooltipRow swatch={c.forecast} dashed label="Forecast mean" value={fmt1(r.mean)} />
                      <TooltipRow label="80% interval" value={`${fmt1(r.lower!)} – ${fmt1(r.upper!)}`} />
                      <TooltipRow swatch={c.observed} label="Actual (raw)" value={r.actual ?? "–"} />
                      {showDedup && <TooltipRow swatch={c.dedup} label="Actual de-duplicated" value={r.dedup ?? "–"} />}
                      {showBaseline && <TooltipRow swatch={c["text-3"]} dashed label="Seasonal naive" value={r.baseline ?? "–"} />}
                      {r.actual != null && <p className="pt-0.5 text-[11.5px] text-text-2">
                        {r.actual >= r.lower! && r.actual <= r.upper! ? "Inside the 80% interval" : "Outside the 80% interval"}</p>}
                      {!!r.holidays?.length && <p className="text-[11.5px] text-text-2">Holiday: {r.holidays.join(", ")}</p>}
                    </>}
                  </TooltipCard>
                );
              }} />
            <Area dataKey="band" stroke="none" fill={c.forecast} fillOpacity={0.2} isAnimationActive={!reduced} />
            <Line dataKey="observed" type="linear" stroke={c.observed} strokeWidth={2} dot={false} isAnimationActive={!reduced} connectNulls={false} />
            {showBaseline && <Line dataKey="baseline" type="linear" stroke={c["text-3"]} strokeWidth={1.75} strokeDasharray="2 4" dot={false} isAnimationActive={false} />}
            <Line dataKey="mean" type="linear" stroke={c.forecast} strokeWidth={2.5} strokeDasharray="6 4"
              dot={{ r: 2.5, fill: c.forecast, stroke: c.surface, strokeWidth: 1 }} isAnimationActive={!reduced} connectNulls={false} />
            <Scatter dataKey="actual" fill={c.surface} stroke={c.observed} strokeWidth={2.25} shape="circle" isAnimationActive={false} />
            {showDedup && <Scatter dataKey="dedup" fill={c.surface} stroke={c.dedup} strokeWidth={2.25} shape="diamond" isAnimationActive={false} />}
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

// ================================================================== holdout
export function HorizonErrorChart({ data }: { data: HoldoutEvaluation }) {
  const c = useChartColors();
  const reduced = usePrefersReducedMotion();
  return (
    <div>
      <div className="mb-3 flex flex-wrap gap-x-5 gap-y-2">
        <LegendItem color={c.forecast} label="M4 (frozen forecast.py)" />
        <LegendItem color={c["text-3"]} label="Seasonal-naive baseline" kind="dashed" />
      </div>
      <div className="h-[240px]" role="img" aria-label="Holdout mean absolute error by forecast day, model versus baseline.">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data.by_horizon} margin={{ top: 8, right: 8, bottom: 14, left: -16 }}>
            <CartesianGrid vertical={false} stroke={c.border} strokeOpacity={0.14} />
            <XAxis dataKey="h" tick={axisTick(c)} axisLine={{ stroke: c.border, strokeOpacity: 0.3 }} tickLine={false}
              label={{ value: "Forecast day (1 = cut-off day)", position: "insideBottom", offset: -8, fill: c["text-2"], fontSize: 11.5 }} />
            <YAxis domain={[0, 10]} tick={axisTick(c)} axisLine={false} tickLine={false} width={40} />
            <Tooltip isAnimationActive={false} content={({ active, payload }) => active && payload?.length ? (
              <TooltipCard title={`Forecast day ${(payload[0].payload as { h: number }).h}`}>
                <TooltipRow swatch={c.forecast} label="M4 MAE" value={fmt1((payload[0].payload as { MAE: number }).MAE)} />
                <TooltipRow swatch={c["text-3"]} dashed label="Baseline MAE" value={fmt1((payload[0].payload as { MAE_B0: number }).MAE_B0)} />
              </TooltipCard>) : null} />
            <Line dataKey="MAE_B0" stroke={c["text-3"]} strokeWidth={2} strokeDasharray="5 4" dot={{ r: 2.5 }} isAnimationActive={!reduced} />
            <Line dataKey="MAE" stroke={c.forecast} strokeWidth={2.5} dot={{ r: 3, fill: c.forecast }} isAnimationActive={!reduced} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

export function HorizonCoverageChart({ data }: { data: HoldoutEvaluation }) {
  const c = useChartColors();
  const reduced = usePrefersReducedMotion();
  return (
    <div className="h-[240px]" role="img" aria-label="Holdout 80% interval coverage by forecast day, with the 80% target.">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data.by_horizon} margin={{ top: 22, right: 8, bottom: 14, left: 4 }}>
          <CartesianGrid vertical={false} stroke={c.border} strokeOpacity={0.14} />
          <XAxis dataKey="h" tick={axisTick(c)} axisLine={{ stroke: c.border, strokeOpacity: 0.3 }} tickLine={false}
            label={{ value: "Forecast day", position: "insideBottom", offset: -8, fill: c["text-2"], fontSize: 11.5 }} />
          <YAxis domain={[0, 1]} ticks={[0, 0.4, 0.8, 1]} tickFormatter={pct} tick={axisTick(c)} axisLine={false} tickLine={false} width={46} />
          <ReferenceLine y={0.8} stroke={c.amber} strokeDasharray="4 4" label={{ value: "80% target", position: "top", fill: c["text-2"], fontSize: 11, offset: 6 }} />
          <Tooltip cursor={{ fill: c.border, fillOpacity: 0.08 }} isAnimationActive={false} content={({ active, payload }) => active && payload?.length ? (
            <TooltipCard title={`Forecast day ${(payload[0].payload as { h: number }).h}`}>
              <TooltipRow label="Coverage (19 cut-offs)" value={pct(payload[0].value as number)} />
            </TooltipCard>) : null} />
          <Bar dataKey="coverage80" fill={c.forecast} fillOpacity={0.7} radius={[4, 4, 0, 0]} isAnimationActive={!reduced} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

export function PerCutoffRatioChart({ data }: { data: HoldoutEvaluation }) {
  const c = useChartColors();
  const reduced = usePrefersReducedMotion();
  const rows = data.per_cutoff.map((r) => ({ ...r, label: fmtDayMonth(r.cutoff) }));
  return (
    <div className="h-[240px]" role="img"
      aria-label={`MAE ratio versus baseline for each of ${rows.length} holdout cut-offs; below 1 means the model beat the baseline.`}>
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={rows} margin={{ top: 16, right: 8, bottom: 0, left: -10 }}>
          <CartesianGrid vertical={false} stroke={c.border} strokeOpacity={0.14} />
          <XAxis dataKey="label" interval={2} tick={axisTick(c)} axisLine={{ stroke: c.border, strokeOpacity: 0.3 }} tickLine={false} />
          <YAxis domain={[0, 1.6]} ticks={[0, 0.5, 1, 1.5]} tick={axisTick(c)} axisLine={false} tickLine={false} width={40} />
          <ReferenceLine y={1} stroke={c["text-3"]} strokeDasharray="4 4" label={{ value: "baseline = 1", position: "insideTopLeft", fill: c["text-2"], fontSize: 11, offset: 4 }} />
          <Tooltip cursor={{ fill: c.border, fillOpacity: 0.08 }} isAnimationActive={false} content={({ active, payload }) => {
            if (!active || !payload?.length) return null;
            const r = payload[0].payload as (typeof rows)[number];
            return (
              <TooltipCard title={`Cut-off ${fmtLong(r.cutoff)}`}>
                <TooltipRow label="MAE ratio vs baseline" value={r.ratio.toFixed(2)} />
                <TooltipRow swatch={c.forecast} label="M4 MAE" value={fmt1(r.MAE)} />
                <TooltipRow swatch={c["text-3"]} dashed label="Baseline MAE" value={fmt1(r.MAE_B0)} />
                <TooltipRow label="80% coverage" value={pct(r.coverage80)} />
              </TooltipCard>
            );
          }} />
          <Bar dataKey="ratio" radius={[4, 4, 0, 0]} isAnimationActive={!reduced}>
            {rows.map((r) => <Cell key={r.cutoff} fill={r.ratio > 1 ? c.amber : c.forecast} fillOpacity={0.75} />)}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

export function DiagnosisChart({ data }: { data: HoldoutEvaluation }) {
  const c = useChartColors();
  const reduced = usePrefersReducedMotion();
  const rows = data.diagnosis_by_month.map((d) => ({ ...d,
    label: new Date(`${d.month}-01T00:00:00`).toLocaleDateString("en-GB", { month: "short", year: "numeric" }) }));
  return (
    <div>
      <div className="mb-3 flex flex-wrap gap-x-5 gap-y-2">
        <LegendItem color={c.forecast} label="M4 forecast (mean)" kind="bar" />
        <LegendItem color={c.observed} label="Actual raw records (scored target)" kind="bar" />
        <LegendItem color={c.dedup} label="Actual de-duplicated (sensitivity)" kind="bar" />
      </div>
      <div className="h-[250px]" role="img" aria-label="Average daily forecast versus actual raw and de-duplicated records by month of the holdout.">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={rows} margin={{ top: 16, right: 8, bottom: 0, left: -16 }} barGap={3} barCategoryGap="22%">
            <CartesianGrid vertical={false} stroke={c.border} strokeOpacity={0.14} />
            <XAxis dataKey="label" tick={axisTick(c)} axisLine={{ stroke: c.border, strokeOpacity: 0.3 }} tickLine={false} />
            <YAxis domain={[0, 32]} tick={axisTick(c)} axisLine={false} tickLine={false} width={40} />
            <Tooltip cursor={{ fill: c.border, fillOpacity: 0.08 }} isAnimationActive={false} content={({ active, payload }) => {
              if (!active || !payload?.length) return null;
              const r = payload[0].payload as (typeof rows)[number];
              return (
                <TooltipCard title={r.label}>
                  <TooltipRow swatch={c.forecast} label="Forecast" value={fmt1(r.forecast)} />
                  <TooltipRow swatch={c.observed} label="Actual raw" value={fmt1(r.actual_raw)} />
                  <TooltipRow swatch={c.dedup} label="Actual de-duplicated" value={fmt1(r.actual_dedup)} />
                  <TooltipRow label="Repeat rows" value={`${fmt1(r.repeat_rows_pct)}%`} />
                </TooltipCard>
              );
            }} />
            {(["forecast", "actual_raw", "actual_dedup"] as const).map((k, i) => (
              <Bar key={k} dataKey={k} fill={[c.forecast, c.observed, c.dedup][i]} fillOpacity={0.8} radius={[4, 4, 0, 0]} isAnimationActive={!reduced}
                label={{ position: "top", fill: c["text-2"], fontSize: 10.5, formatter: (v: number) => v.toFixed(1) }} />
            ))}
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
