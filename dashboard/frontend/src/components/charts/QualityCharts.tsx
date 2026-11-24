import { useMemo, useState } from "react";
import { Bar, BarChart, CartesianGrid, Cell, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { fmt1, fmtDayMonth, fmtInt, fmtLong } from "../../lib/format";
import { useChartColors, usePrefersReducedMotion } from "../../lib/theme";
import type { CaseMix, QualityDuplicates, RepeatRates } from "../../lib/types";
import { LegendItem, TooltipCard, TooltipRow } from "./ChartParts";

export type RepeatDef = "exact" | "mrd";
export const DEF_LABEL: Record<RepeatDef, string> = { exact: "Exact repeats removed", mrd: "One record per MRD + admission date" };
export const DEF_SHORT: Record<RepeatDef, string> = { exact: "exact repeats", mrd: "MRD + admission-date repeats" };

const pct = (v: number) => `${(v * 100).toFixed(1)}%`;
const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
export const fmtMonth = (m: string, long = false) => {
  const [y, mo] = m.split("-");
  return long ? `${MONTHS[+mo - 1]} ${y}` : `${MONTHS[+mo - 1]} ${y.slice(2)}`;
};
export const rateOf = (r: RepeatRates, def: RepeatDef) => (def === "exact" ? r.exact_rate : r.mrd_rate);
const extraOf = (r: RepeatRates, def: RepeatDef) => (def === "exact" ? r.exact_extra : r.mrd_extra);

// ======================================================================= raw vs de-duplicated
type Daily = QualityDuplicates["daily"][number];
interface LinePoint { date: string; end: string; raw: number; dedup: number; days: number; flagged: boolean }

function aggregate(daily: Daily[], def: RepeatDef, weekly: boolean): LinePoint[] {
  const pick = (d: Daily) => (def === "exact" ? d.dedup_exact : d.dedup_mrd);
  if (!weekly)
    return daily.map((d) => ({ date: d.date, end: d.date, raw: d.raw, dedup: pick(d), days: 1, flagged: d.potentially_incomplete || d.low_count_flag }));
  const out: LinePoint[] = [];
  let cur: Daily[] = [];
  const flush = () => {
    if (!cur.length) return;
    const n = cur.length;
    out.push({ date: cur[0].date, end: cur[n - 1].date, raw: cur.reduce((a, d) => a + d.raw, 0) / n,
      dedup: cur.reduce((a, d) => a + pick(d), 0) / n, days: n, flagged: cur.some((d) => d.potentially_incomplete) });
    cur = [];
  };
  for (const d of daily) {
    if (new Date(`${d.date}T00:00:00Z`).getUTCDay() === 1) flush();   // weeks start on Monday
    cur.push(d);
  }
  flush();
  return out;
}

/** Raw (scored target, solid) and one de-duplicated sensitivity series (dashed). Single y-axis; series never merged. */
export function RawVsDedupChart({ daily, def, weekly }: { daily: Daily[]; def: RepeatDef; weekly: boolean }) {
  const c = useChartColors();
  const reduced = usePrefersReducedMotion();
  const pts = useMemo(() => aggregate(daily, def, weekly), [daily, def, weekly]);
  const max = Math.max(...pts.map((p) => p.raw));
  const totRaw = daily.reduce((a, d) => a + d.raw, 0);
  const totDedup = daily.reduce((a, d) => a + (def === "exact" ? d.dedup_exact : d.dedup_mrd), 0);
  const summary = `${weekly ? "Weekly mean" : "Daily"} raw records and ${DEF_LABEL[def].toLowerCase()} from ${fmtLong(daily[0].date)} to ` +
    `${fmtLong(daily[daily.length - 1].date)}: ${fmtInt(totRaw)} raw vs ${fmtInt(totDedup)} de-duplicated records.`;
  return (
    <div className="flex flex-1 flex-col">
      <div className="mb-3 flex flex-wrap gap-x-5 gap-y-2">
        <LegendItem color={c.observed} label={`Raw records (scored target)${weekly ? ", weekly mean" : ""}`} />
        <LegendItem color={c.dedup} label={`${DEF_LABEL[def]} (sensitivity)`} kind="dashed" />
      </div>
      <div className="h-[300px] w-full" role="img" aria-label={summary}>
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={pts} margin={{ top: 8, right: 8, bottom: 0, left: -12 }}>
            <CartesianGrid vertical={false} stroke={c.border} strokeOpacity={0.14} />
            <XAxis dataKey="date" tickFormatter={fmtDayMonth} minTickGap={32} tick={{ fill: c["text-2"], fontSize: 11.5 }}
              axisLine={{ stroke: c.border, strokeOpacity: 0.3 }} tickLine={false} />
            <YAxis domain={[0, Math.ceil((max + 4) / 10) * 10]} tick={{ fill: c["text-2"], fontSize: 11.5 }} axisLine={false} tickLine={false} width={44} />
            <Tooltip isAnimationActive={false} cursor={{ stroke: c.border, strokeOpacity: 0.4 }}
              content={({ active, payload }) => {
                if (!active || !payload?.length) return null;
                const p = payload[0].payload as LinePoint;
                const f = weekly ? fmt1 : fmtInt;
                return (
                  <TooltipCard title={weekly ? `Week of ${fmtLong(p.date)}${p.days < 7 ? ` (${p.days} days)` : ""}` : fmtLong(p.date)}>
                    <TooltipRow swatch={c.observed} label={weekly ? "Raw, mean/day" : "Raw records"} value={f(p.raw)} />
                    <TooltipRow swatch={c.dedup} dashed label={weekly ? "De-duplicated, mean/day" : "De-duplicated"} value={f(p.dedup)} />
                    <TooltipRow label="Repeats" value={f(p.raw - p.dedup)} />
                    {p.flagged && <p className="pt-1 text-[11.5px] font-semibold text-amber-ink">
                      {weekly ? "Includes 31 Mar 2019 (potentially incomplete)" : p.date === "2019-03-31" ? "Potentially incomplete (last day in file)" : "Flagged low-count day"}
                    </p>}
                  </TooltipCard>
                );
              }} />
            <Line dataKey="raw" type="monotone" stroke={c.observed} strokeWidth={2} dot={false} isAnimationActive={!reduced} animationDuration={700}
              activeDot={{ r: 4, strokeWidth: 2, stroke: c.surface }} />
            <Line dataKey="dedup" type="monotone" stroke={c.dedup} strokeWidth={2} strokeDasharray="6 4" dot={false} isAnimationActive={!reduced}
              animationDuration={700} activeDot={{ r: 4, strokeWidth: 2, stroke: c.surface }} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

// ======================================================================= repeat rate bars
export function RepeatRateBars({ rows, def, labelKey, height = 260, tickFormatter, overall }: {
  rows: (RepeatRates & { month?: string; weekday?: string })[]; def: RepeatDef; labelKey: "month" | "weekday"; height?: number;
  tickFormatter: (v: string) => string; overall: number;
}) {
  const c = useChartColors();
  const reduced = usePrefersReducedMotion();
  const data = rows.map((r) => ({ key: String(r[labelKey]), rate: rateOf(r, def) * 100, extra: extraOf(r, def), rows: r.rows }));
  const max = Math.max(...data.map((d) => d.rate));
  const top = data.reduce((a, b) => (b.rate > a.rate ? b : a));
  const summary = `Share of raw records that are ${DEF_SHORT[def]}, by ${labelKey}. Highest ${tickFormatter(top.key)} at ${fmt1(top.rate)}%; ` +
    `overall ${pct(overall)}.`;
  return (
    <div style={{ height }} className="w-full" role="img" aria-label={summary}>
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} margin={{ top: 16, right: 8, bottom: 0, left: -12 }} barCategoryGap="22%">
          <CartesianGrid vertical={false} stroke={c.border} strokeOpacity={0.14} />
          <XAxis dataKey="key" tickFormatter={tickFormatter} interval={labelKey === "month" ? "preserveStartEnd" : 0} minTickGap={6}
            tick={{ fill: c["text-2"], fontSize: 11.5 }} axisLine={{ stroke: c.border, strokeOpacity: 0.3 }} tickLine={false} />
          <YAxis domain={[0, Math.max(5, Math.ceil(max / 5) * 5)]} tickFormatter={(v) => `${v}%`} tick={{ fill: c["text-2"], fontSize: 11.5 }}
            axisLine={false} tickLine={false} width={48} />
          <ReferenceLine y={overall * 100} stroke={c["text-3"]} strokeDasharray="5 5" strokeWidth={1.5}
            label={{ value: `overall ${pct(overall)}`, position: "insideTopRight", fill: c["text-2"], fontSize: 11 }} />
          <Tooltip isAnimationActive={false} cursor={{ fill: c.border, fillOpacity: 0.08 }}
            content={({ active, payload }) => {
              if (!active || !payload?.length) return null;
              const p = payload[0].payload as (typeof data)[number];
              return (
                <TooltipCard title={labelKey === "month" ? fmtMonth(p.key, true) : p.key}>
                  <TooltipRow label="Repeat share" value={`${fmt1(p.rate)}%`} />
                  <TooltipRow label="Repeat records" value={fmtInt(p.extra)} />
                  <TooltipRow label="Raw records" value={fmtInt(p.rows)} />
                </TooltipCard>
              );
            }} />
          <Bar dataKey="rate" fill={c.dedup} fillOpacity={0.8} radius={[4, 4, 0, 0]} isAnimationActive={!reduced} animationDuration={600} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

// ======================================================================= calendar heatmap
const BINS = [
  { min: 0, max: 0, label: "0", alpha: 0 },
  { min: 1, max: 1, label: "1", alpha: 0.28 },
  { min: 2, max: 3, label: "2–3", alpha: 0.5 },
  { min: 4, max: 6, label: "4–6", alpha: 0.75 },
  { min: 7, max: Infinity, label: "7+", alpha: 1 },
];
const binOf = (v: number) => BINS.find((b) => v >= b.min && v <= b.max) ?? BINS[BINS.length - 1];
const WD = ["Mon", "", "Wed", "", "Fri", "", "Sun"];

/** Extra (repeat) records per day, one strip per data year (Apr-Mar). Sequential single hue; counts in the tooltip. */
export function RepeatHeatmap({ daily, def }: { daily: Daily[]; def: RepeatDef }) {
  const c = useChartColors();
  const [hover, setHover] = useState<Daily | null>(null);
  const CELL = 11, GAP = 2, STEP = CELL + GAP, LEFT = 30, TOP = 16;
  const years = useMemo(() => {
    const ys: { label: string; days: Daily[] }[] = [];
    for (const d of daily) {
      const y = +d.date.slice(0, 4), m = +d.date.slice(5, 7);
      const fy = m >= 4 ? y : y - 1;
      const label = `Apr ${fy} – Mar ${fy + 1}`;
      if (!ys.length || ys[ys.length - 1].label !== label) ys.push({ label, days: [] });
      ys[ys.length - 1].days.push(d);
    }
    return ys;
  }, [daily]);
  const extra = (d: Daily) => (def === "exact" ? d.extra_exact : d.extra_mrd);
  const daysWith = daily.filter((d) => extra(d) > 0).length;
  return (
    <div>
      <div className="mb-3 flex flex-wrap items-center gap-x-4 gap-y-2 text-[12px] text-text-2">
        <span className="font-medium">Repeat records per day:</span>
        {BINS.map((b) => (
          <span key={b.label} className="inline-flex items-center gap-1.5">
            <span aria-hidden className="h-3 w-3 rounded-[3px] ring-1 ring-inset ring-border/25"
              style={{ background: b.alpha ? c.dedup : c["surface-2"], opacity: b.alpha ? Math.max(b.alpha, 0.28) : 1 }} />
            {b.label}
          </span>
        ))}
        <span className="inline-flex items-center gap-1.5"><span aria-hidden className="h-3 w-3 rounded-[3px] border-2" style={{ borderColor: c.amber }} />potentially incomplete day</span>
      </div>
      <div className="overflow-x-auto pb-1" role="img"
        aria-label={`Calendar of ${DEF_SHORT[def]} per day, April 2017 to March 2019. ${daysWith} of ${daily.length} days have at least one repeat record.`}>
        {years.map((y) => {
          const first = new Date(`${y.days[0].date}T00:00:00Z`);
          const offset = (first.getUTCDay() + 6) % 7;            // Monday = 0
          const weeks = Math.ceil((offset + y.days.length) / 7);
          const monthTicks: { x: number; label: string }[] = [];
          y.days.forEach((d, i) => {
            if (d.date.endsWith("-01")) monthTicks.push({ x: LEFT + Math.floor((offset + i) / 7) * STEP, label: MONTHS[+d.date.slice(5, 7) - 1] });
          });
          return (
            <div key={y.label} className="mb-2">
              <p className="mb-1 text-[11.5px] font-semibold text-text-2">{y.label}</p>
              <svg width={LEFT + weeks * STEP} height={TOP + 7 * STEP} className="block" onMouseLeave={() => setHover(null)}>
                {monthTicks.map((t) => <text key={t.label + t.x} x={t.x} y={10} fontSize={10} fill={c["text-2"]}>{t.label}</text>)}
                {WD.map((w, i) => w && <text key={w} x={0} y={TOP + i * STEP + CELL - 2} fontSize={10} fill={c["text-2"]}>{w}</text>)}
                {y.days.map((d, i) => {
                  const k = offset + i, b = binOf(extra(d));
                  return (
                    <rect key={d.date} x={LEFT + Math.floor(k / 7) * STEP} y={TOP + (k % 7) * STEP} width={CELL} height={CELL} rx={2.5}
                      fill={b.alpha ? c.dedup : c["surface-2"]} fillOpacity={b.alpha || 1}
                      stroke={d.potentially_incomplete ? c.amber : hover?.date === d.date ? c.text : "none"} strokeWidth={d.potentially_incomplete || hover?.date === d.date ? 1.5 : 0}
                      onMouseEnter={() => setHover(d)} />
                  );
                })}
              </svg>
            </div>
          );
        })}
      </div>
      <p className="mt-1 min-h-[20px] text-[12.5px] text-text-2" aria-live="polite">
        {hover ? (
          <><strong className="text-text">{fmtLong(hover.date)}</strong>: {hover.raw} raw records, {extra(hover)} {DEF_SHORT[def]}
            {" "}({hover.raw ? fmt1((100 * extra(hover)) / hover.raw) : "0"}%){hover.potentially_incomplete ? " · potentially incomplete" : ""}</>
        ) : (
          <>Hover a day for its counts. {daysWith} of {daily.length} days have at least one repeat record.</>
        )}
      </p>
    </div>
  );
}

// ======================================================================= Emergency / OPD share
export function OpdShareChart({ data }: { data: CaseMix }) {
  const c = useChartColors();
  const reduced = usePrefersReducedMotion();
  const rows = data.months.map((m) => ({ ...m, share: m.opd_share * 100 }));
  const [lo, hi] = data.other_development_months_opd_share_range;
  return (
    <div className="flex flex-1 flex-col">
      <div className="mb-3 flex flex-wrap gap-x-5 gap-y-2">
        <LegendItem color={c.observed} label="OPD-coded share of records" kind="bar" />
        <LegendItem color={c.amber} label="Coding anomaly (Nov–Dec 2017)" kind="bar" />
      </div>
      <div className="h-[240px] w-full" role="img"
        aria-label={`Monthly share of records coded OPD. Nov and Dec 2017 reach ${rows.filter((r) => r.anomaly).map((r) => `${fmt1(r.share)}%`).join(" and ")}, versus ${pct(lo)} to ${pct(hi)} in other 2017-18 months.`}>
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={rows} margin={{ top: 8, right: 8, bottom: 0, left: -12 }} barCategoryGap="22%">
            <CartesianGrid vertical={false} stroke={c.border} strokeOpacity={0.14} />
            <XAxis dataKey="month" tickFormatter={(m) => fmtMonth(m)} interval="preserveStartEnd" minTickGap={8}
              tick={{ fill: c["text-2"], fontSize: 11.5 }} axisLine={{ stroke: c.border, strokeOpacity: 0.3 }} tickLine={false} />
            <YAxis domain={[0, 60]} ticks={[0, 20, 40, 60]} tickFormatter={(v) => `${v}%`} tick={{ fill: c["text-2"], fontSize: 11.5 }}
              axisLine={false} tickLine={false} width={48} />
            <ReferenceLine y={data.overall.opd_share * 100} stroke={c["text-3"]} strokeDasharray="5 5" strokeWidth={1.5}
              label={{ value: `overall ${pct(data.overall.opd_share)}`, position: "insideTopRight", fill: c["text-2"], fontSize: 11 }} />
            <Tooltip isAnimationActive={false} cursor={{ fill: c.border, fillOpacity: 0.08 }}
              content={({ active, payload }) => {
                if (!active || !payload?.length) return null;
                const p = payload[0].payload as (typeof rows)[number];
                return (
                  <TooltipCard title={fmtMonth(p.month, true)}>
                    <TooltipRow label="OPD share" value={`${fmt1(p.share)}%`} />
                    <TooltipRow label="Emergency records" value={fmtInt(p.emergency)} />
                    <TooltipRow label="OPD records" value={fmtInt(p.opd)} />
                    {p.anomaly && <p className="pt-1 text-[11.5px] font-semibold text-amber-ink">Coding anomaly month</p>}
                  </TooltipCard>
                );
              }} />
            <Bar dataKey="share" radius={[4, 4, 0, 0]} isAnimationActive={!reduced} animationDuration={600}>
              {rows.map((r) => <Cell key={r.month} fill={r.anomaly ? c.amber : c.observed} fillOpacity={r.anomaly ? 0.9 : 0.6} />)}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
