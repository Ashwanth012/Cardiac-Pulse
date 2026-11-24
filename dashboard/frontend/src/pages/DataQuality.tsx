import clsx from "clsx";
import { CalendarX2, CircleAlert, Copy, Database, FileSearch, Info, Layers, ListChecks, Wind } from "lucide-react";
import { useMemo, useState, type ReactNode } from "react";
import {
  DEF_LABEL, DEF_SHORT, OpdShareChart, RawVsDedupChart, RepeatHeatmap, RepeatRateBars, fmtMonth, rateOf, type RepeatDef,
} from "../components/charts/QualityCharts";
import { Badge, ProvenanceBadge } from "../components/ui/Badge";
import { KpiCard } from "../components/ui/KpiCard";
import { Panel } from "../components/ui/Panel";
import { Segmented } from "../components/ui/Segmented";
import { ErrorState, Loading } from "../components/ui/States";
import { useCaseMix, usePollutionLog, useQualityDuplicates, useQualityMissing, useQualitySummary } from "../lib/api";
import { WINDOWS, useFilters } from "../lib/filters";
import { fmtDate, fmtInt, fmtWeekday } from "../lib/format";
import { hrefFor } from "../lib/router";
import type { LogAction, PollutionLog, QualityMissing, QualitySummary } from "../lib/types";

const pct = (v: number) => `${(v * 100).toFixed(1)}%`;
const ACTION_LABEL: Record<LogAction, string> = { corrected: "Corrected", set_missing: "Set missing", imputed: "Imputed", flagged: "Flagged" };
const ACTION_TONE: Record<LogAction, "teal" | "amber" | "observed" | "neutral"> = {
  corrected: "teal", set_missing: "amber", imputed: "observed", flagged: "neutral",
};

function Note({ icon: Icon = Info, tone = "info", title, children }: { icon?: typeof Info; tone?: "info" | "warn"; title: string; children: ReactNode }) {
  return (
    <div className={clsx("flex gap-3 rounded-xl p-3.5 ring-1 ring-inset", tone === "warn" ? "bg-amber/[0.07] ring-amber/30" : "bg-observed/[0.06] ring-observed/25")}>
      <Icon aria-hidden className={clsx("mt-0.5 h-4 w-4 shrink-0", tone === "warn" ? "text-amber-ink" : "text-observed-ink")} />
      <div className="text-[12.5px] leading-5 text-text-2">
        <p className="font-semibold text-text">{title}</p>
        <div className="mt-0.5">{children}</div>
      </div>
    </div>
  );
}

function Failed({ what, error, retry }: { what: string; error: Error | null; retry: () => void }) {
  return <ErrorState title={`Could not load ${what}`} message={error?.message ?? "Unknown error"} onRetry={retry} />;
}

const th = "px-3 py-2 text-left text-[11px] font-semibold uppercase tracking-wide text-text-3";
const td = "px-3 py-2 align-top";

// ====================================================================== header + KPIs
function Header() {
  return (
    <section className="panel relative mb-6 overflow-hidden p-6 animate-fade-up" aria-labelledby="dq-title">
      <div aria-hidden className="pointer-events-none absolute -right-16 -top-24 h-64 w-64 rounded-full bg-dedup opacity-[0.10] blur-3xl" />
      <div className="relative max-w-3xl">
        <div className="flex flex-wrap gap-2">
          <ProvenanceBadge kind="observed" />
          <ProvenanceBadge kind="sensitivity" />
          <Badge tone="neutral">Apr 2017 – Mar 2019 · aggregate daily counts</Badge>
        </div>
        <h2 id="dq-title" className="mt-4 text-2xl font-semibold tracking-tight text-text sm:text-[28px]">
          Data quality <span className="bg-gradient-to-r from-observed-ink to-dedup-ink bg-clip-text text-transparent">&amp; monitoring</span>
        </h2>
        <p className="mt-2 text-[14px] leading-6 text-text-2">
          How trustworthy are the counts behind every chart? The forecasting target is the <strong className="text-text">raw</strong> number
          of admission records per day, exactly as the harness scores it. <strong className="text-text">Nothing has been removed</strong>:
          repeated records are measured here and shown as separate, de-duplicated sensitivity series. Only aggregate counts are shown – no
          patient-level information.
        </p>
      </div>
    </section>
  );
}

function Kpis({ s }: { s: QualitySummary }) {
  const r = s.records;
  const pi = s.days.potentially_incomplete;
  return (
    <div className="mb-6 grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-3">
      <KpiCard label="Raw admission records" value={fmtInt(r.raw)} unit="records" icon={Database} accent="observed" provenance="observed"
        detail={`${s.period.days} days, ${s.days.zero_count} days without records. This is the scored target; nothing removed.`} />
      <KpiCard label="Exact repeat records" value={fmtInt(r.exact_extra)} unit={pct(r.exact_extra_rate)} icon={Copy} accent="dedup" provenance="sensitivity"
        detail={`${fmtInt(r.dedup_exact)} records if exact repeats were removed (sensitivity view only).`} delay={60} />
      <KpiCard label="MRD + admission-date repeats" value={fmtInt(r.mrd_extra)} unit={pct(r.mrd_extra_rate)} icon={Layers} accent="dedup" provenance="sensitivity"
        detail={`${fmtInt(r.dedup_mrd)} records with one per MRD + date; ${fmtInt(r.near_extra)} of the repeats differ in some field.`} delay={120} />
      <KpiCard label="Flagged low-count days" value={s.days.low_count} unit="days" icon={CalendarX2} accent="teal" provenance="observed"
        detail={`Includes ${fmtDate(pi.date)} (${pi.admissions} records vs a 28-day median of ${pi.median_previous_28_days}), potentially incomplete.`} delay={180} />
      <KpiCard label="Air-quality values imputed" value={s.environment_cells.imputed} unit="cells" icon={Wind} accent="teal" provenance="observed"
        detail={`${s.environment_cells.set_missing} set missing, ${s.environment_cells.corrected} corrected, ${s.environment_cells.flagged} flagged and kept.`} delay={240} />
      <KpiCard label="Cleaning-log entries" value={s.cleaning.total} unit="entries" icon={ListChecks} accent="observed" provenance="observed"
        detail="Every check from the Step 2 build. Admission-file entries are flags only; no record changed." delay={300} />
    </div>
  );
}

// ====================================================================== repeats section
function RepeatsSection({ def, setDef }: { def: RepeatDef; setDef: (d: RepeatDef) => void }) {
  const q = useQualityDuplicates();
  const { windowDays } = useFilters();
  const [weekly, setWeekly] = useState(false);
  const windowed = useMemo(() => q.data?.daily.slice(-windowDays) ?? [], [q.data, windowDays]);
  const winLabel = WINDOWS.find((w) => w.days === windowDays)?.label ?? `${windowDays} days`;
  if (q.isLoading) return <Panel title="Repeated records"><Loading label="Loading repeat analysis" /></Panel>;
  if (q.isError || !q.data) return <Failed what="the repeat analysis" error={q.error} retry={() => q.refetch()} />;
  const d = q.data;
  const all = d.periods.reduce((a, p) => ({ rows: a.rows + p.rows, exact: a.exact + p.exact_extra, mrd: a.mrd + p.mrd_extra }), { rows: 0, exact: 0, mrd: 0 });
  const overall = (def === "exact" ? all.exact : all.mrd) / all.rows;
  return (
    <section aria-labelledby="repeats-h" className="mb-6">
      <div className="mb-4 flex flex-wrap items-end justify-between gap-3">
        <div className="max-w-3xl">
          <h2 id="repeats-h" className="text-[17px] font-semibold tracking-tight text-text">Repeated records</h2>
          <p className="mt-1 text-[13px] leading-5 text-text-2">{d.definitions[def === "exact" ? "exact" : "mrd_date"]} {d.note}</p>
        </div>
        <div className="flex flex-col items-start gap-1">
          <span className="text-[11.5px] font-medium text-text-2">Repeat definition</span>
          <Segmented label="Repeat definition" value={def} onChange={setDef}
            options={[{ value: "exact", label: "Exact" }, { value: "mrd", label: "MRD + admission date" }]} />
        </div>
      </div>

      <div className="grid grid-cols-1 gap-6 xl:grid-cols-3">
        <Panel className="xl:col-span-2" fill title="Raw vs de-duplicated records"
          subtitle={`Last ${winLabel === "All" ? "two years" : winLabel} to 31 Mar 2019 (trend window in the top bar). Raw is solid; the de-duplicated sensitivity series is dashed.`}
          actions={<>
            <Segmented label="Resolution" value={weekly ? "w" : "d"} onChange={(v) => setWeekly(v === "w")}
              options={[{ value: "d", label: "Daily" }, { value: "w", label: "Weekly mean" }]} />
            <ProvenanceBadge kind="sensitivity" />
          </>}>
          <RawVsDedupChart daily={windowed} def={def} weekly={weekly} />
        </Panel>

        <Panel title="Repeat share by period" subtitle={`Share of raw records that are ${DEF_SHORT[def]}`} actions={<ProvenanceBadge kind="sensitivity" />}>
          <ul className="space-y-2.5">
            {d.periods.map((p) => {
              const r = rateOf(p, def);
              return (
                <li key={p.label}>
                  <div className="flex items-baseline justify-between text-[12.5px]">
                    <span className="font-medium text-text-2">{p.label.replace("-", "–")}</span>
                    <span className="tabular font-semibold text-text">{pct(r)} <span className="font-normal text-text-3">of {fmtInt(p.rows)}</span></span>
                  </div>
                  <div className="mt-1 h-2 rounded-full bg-surface-2" aria-hidden>
                    <div className="h-2 rounded-full bg-dedup/80" style={{ width: `${Math.min(100, r * 500)}%` }} />
                  </div>
                </li>
              );
            })}
          </ul>
          <p className="mt-2 text-[11.5px] text-text-3">Bar scale: full width = 20%.</p>
          <h3 className="mt-5 text-[13px] font-semibold text-text">By weekday (two years)</h3>
          <RepeatRateBars rows={d.weekday} def={def} labelKey="weekday" height={170} tickFormatter={(w) => w.slice(0, 3)} overall={overall} />
        </Panel>
      </div>

      <div className="mt-6 grid grid-cols-1 gap-6">
        <Panel title="Repeat share by month" subtitle="The repeat share is higher in 2018–19 than in 2017 and varies month to month. Repeats are part of the raw target the harness scores, so they add non-clinical level shifts."
          actions={<ProvenanceBadge kind="sensitivity" />}>
          <RepeatRateBars rows={d.monthly} def={def} labelKey="month" tickFormatter={(m) => fmtMonth(m)} overall={overall} />
        </Panel>
        <Panel title="Repeat records per day" subtitle={`Calendar view of ${DEF_SHORT[def]}, April 2017 – March 2019`} actions={<ProvenanceBadge kind="sensitivity" />}>
          <RepeatHeatmap daily={d.daily} def={def} />
        </Panel>
      </div>
    </section>
  );
}

// ====================================================================== flagged days + case mix
function FlaggedDays({ s }: { s: QualitySummary }) {
  return (
    <Panel title="Flagged low-count days" subtitle="Days below 40% of the 29-day rolling median. Kept in the data; shown for context."
      actions={<ProvenanceBadge kind="observed" />}>
      <div className="max-h-[360px] overflow-auto rounded-xl ring-1 ring-inset ring-border/15">
        <table className="w-full text-[12.5px]">
          <caption className="sr-only">Low-count days with their raw record counts and calendar context</caption>
          <thead className="sticky top-0 bg-surface-2"><tr><th className={th}>Date</th><th className={th}>Day</th><th className={clsx(th, "text-right")}>Records</th><th className={th}>Context</th></tr></thead>
          <tbody>
            {s.days.low_count_days.map((d) => (
              <tr key={d.date} className="border-t border-border/10">
                <td className={clsx(td, "font-medium text-text")}>{fmtDate(d.date)}</td>
                <td className={clsx(td, "text-text-2")}>{fmtWeekday(d.date)}</td>
                <td className={clsx(td, "tabular text-right font-semibold text-text")}>{d.admissions}</td>
                <td className={td}>
                  <div className="flex flex-wrap gap-1">
                    {d.potentially_incomplete && <Badge tone="amber">Potentially incomplete</Badge>}
                    {d.holidays.map((h) => <Badge key={h} tone="teal">{h}</Badge>)}
                    {!d.potentially_incomplete && !d.holidays.length && <span className="text-text-3">—</span>}
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="mt-3 text-[12px] leading-5 text-text-2">
        {s.days.zero_count} days have no records. 31 Mar 2019 is the last day in the file and is treated as potentially incomplete, never as a
        confirmed drop in admissions.
      </p>
    </Panel>
  );
}

function CaseMixPanel() {
  const q = useCaseMix();
  return (
    <Panel fill title="Emergency vs OPD coding" subtitle="Monthly share of records coded OPD (planned) rather than emergency"
      actions={<ProvenanceBadge kind="observed" />}>
      {q.isLoading ? <Loading label="Loading case mix" /> : q.isError || !q.data ? <Failed what="the case mix" error={q.error} retry={() => q.refetch()} /> : (
        <>
          <OpdShareChart data={q.data} />
          <div className="mt-3">
            <Note tone="warn" icon={CircleAlert} title="Unreliable around Nov–Dec 2017">
              OPD-coded share reached {q.data.months.filter((m) => m.anomaly).map((m) => pct(m.opd_share)).join(" and ")} in Nov–Dec 2017, against{" "}
              {pct(q.data.other_development_months_opd_share_range[0])}–{pct(q.data.other_development_months_opd_share_range[1])} in other
              2017–18 months, while emergency counts stayed flat. The split is context only and is not forecast.
            </Note>
          </div>
        </>
      )}
    </Panel>
  );
}

// ====================================================================== completeness
function MonthStrip({ months }: { months: QualityMissing["groups"][number]["variables"][number]["months"] }) {
  return (
    <div className="flex gap-[2px]" aria-hidden>
      {months.map((m) => (
        <span key={m.month} title={`${fmtMonth(m.month, true)}: ${m.missing} missing, ${m.imputed} imputed of ${m.days} days`}
          className={clsx("h-3.5 w-2.5 rounded-[2px]", m.missing ? "bg-amber" : m.imputed ? "bg-observed" : "bg-surface-2 ring-1 ring-inset ring-border/15")} />
      ))}
    </div>
  );
}

function Completeness() {
  const q = useQualityMissing();
  if (q.isLoading) return <Panel title="Completeness of environment data"><Loading label="Loading completeness" /></Panel>;
  if (q.isError || !q.data) return <Failed what="completeness data" error={q.error} retry={() => q.refetch()} />;
  const d = q.data;
  return (
    <Panel title="Completeness of environment data"
      subtitle={`Days missing or imputed per variable after cleaning, ${fmtDate(d.period.start)} – ${fmtDate(d.period.end)} (${d.period.days} days). Admission counts: ${d.admissions.missing_days} missing days.`}
      actions={<ProvenanceBadge kind="observed" />}>
      <div className="mb-3 flex flex-wrap gap-x-5 gap-y-1 text-[12px] text-text-2">
        <span className="inline-flex items-center gap-1.5"><span aria-hidden className="h-3 w-2.5 rounded-[2px] bg-surface-2 ring-1 ring-inset ring-border/15" />Complete month</span>
        <span className="inline-flex items-center gap-1.5"><span aria-hidden className="h-3 w-2.5 rounded-[2px] bg-observed" />Month with imputed days</span>
        <span className="inline-flex items-center gap-1.5"><span aria-hidden className="h-3 w-2.5 rounded-[2px] bg-amber" />Month with missing days</span>
      </div>
      <div className="overflow-x-auto rounded-xl ring-1 ring-inset ring-border/15">
        <table className="w-full min-w-[640px] text-[12.5px]">
          <caption className="sr-only">Missing and imputed days per environment variable</caption>
          <thead className="bg-surface-2"><tr>
            <th className={th}>Variable</th><th className={clsx(th, "text-right")}>Missing</th><th className={clsx(th, "text-right")}>Imputed</th>
            <th className={th}>Apr 2017 → Mar 2019 (one cell per month)</th>
          </tr></thead>
          {d.groups.map((g) => (
            <tbody key={g.group}>
              <tr className="border-t border-border/15 bg-surface-2/40"><th colSpan={4} scope="colgroup" className="px-3 py-1.5 text-left text-[11.5px] font-semibold text-text-2">{g.label}</th></tr>
              {g.variables.map((v) => (
                <tr key={v.variable} className="border-t border-border/10">
                  <td className={clsx(td, "text-text")}>{v.label} <span className="text-text-3">({v.unit})</span></td>
                  <td className={clsx(td, "tabular text-right", v.missing ? "font-semibold text-amber-ink" : "text-text-3")}>{v.missing}</td>
                  <td className={clsx(td, "tabular text-right", v.imputed ? "font-semibold text-observed-ink" : "text-text-3")}>{v.imputed}</td>
                  <td className={td}><MonthStrip months={v.months} /></td>
                </tr>
              ))}
            </tbody>
          ))}
        </table>
      </div>
      <ul className="mt-3 list-disc space-y-1 pl-5 text-[12px] leading-5 text-text-2">{d.notes.map((n) => <li key={n}>{n}</li>)}</ul>
    </Panel>
  );
}

// ====================================================================== cleaning
function CleaningSummary({ s }: { s: QualitySummary }) {
  return (
    <Panel title="Cleaning summary by rule" subtitle={`${s.cleaning.total} entries from the Step 2 build, grouped by source and rule`} actions={<ProvenanceBadge kind="observed" />}>
      {(["pollution", "admissions"] as const).map((src) => (
        <div key={src} className="mb-4 last:mb-0">
          <h3 className="mb-2 text-[13px] font-semibold text-text">{src === "pollution" ? "Pollution-station file" : "Admission file (counts only)"}</h3>
          <div className="overflow-x-auto rounded-xl ring-1 ring-inset ring-border/15">
            <table className="w-full min-w-[520px] text-[12.5px]">
              <caption className="sr-only">{src} cleaning rules with entry counts by action</caption>
              <thead className="bg-surface-2"><tr><th className={th}>Rule</th><th className={th}>Check</th><th className={th}>Actions</th><th className={clsx(th, "text-right")}>Entries</th></tr></thead>
              <tbody>
                {s.cleaning.by_rule.filter((r) => r.source === src).map((r) => (
                  <tr key={r.rule} className="border-t border-border/10">
                    <td className={clsx(td, "tabular font-semibold text-text")}>{r.rule}</td>
                    <td className={clsx(td, "text-text-2")}>{r.description}{r.fields?.length ? <span className="block text-[11.5px] text-text-3">{r.fields.join(", ")}</span> : null}</td>
                    <td className={td}><div className="flex flex-wrap gap-1">{(Object.entries(r.actions) as [LogAction, number][]).map(([a, n]) =>
                      <Badge key={a} tone={ACTION_TONE[a]}>{ACTION_LABEL[a]} {n}</Badge>)}</div></td>
                    <td className={clsx(td, "tabular text-right font-semibold text-text")}>{r.entries}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      ))}
      <p className="mt-3 text-[12px] leading-5 text-text-2">{s.cleaning.admissions_note}</p>
    </Panel>
  );
}

const FILTERS: { value: "all" | LogAction; label: string }[] = [
  { value: "all", label: "All" }, { value: "corrected", label: "Corrected" }, { value: "set_missing", label: "Set missing" },
  { value: "imputed", label: "Imputed" }, { value: "flagged", label: "Flagged" },
];

function PollutionLogPanel() {
  const q = usePollutionLog();
  const [f, setF] = useState<"all" | LogAction>("all");
  const rows = useMemo(() => (q.data?.rows ?? []).filter((r) => f === "all" || r.action === f), [q.data, f]);
  return (
    <Panel title="Pollution cleaning log" subtitle={q.data ? `${q.data.entries} entries from the ${q.data.source}. Station readings only – no patient data.` : undefined}
      actions={<ProvenanceBadge kind="observed" />}>
      {q.isLoading ? <Loading label="Loading pollution log" /> : q.isError || !q.data ? <Failed what="the pollution log" error={q.error} retry={() => q.refetch()} /> : (
        <>
          <div className="mb-3 overflow-x-auto"><Segmented label="Filter by action" value={f} onChange={setF} options={FILTERS} /></div>
          <LogTable rows={rows} actions={q.data.actions} />
        </>
      )}
    </Panel>
  );
}

function LogTable({ rows, actions }: { rows: PollutionLog["rows"]; actions: PollutionLog["actions"] }) {
  return (
    <div className="max-h-[420px] overflow-auto rounded-xl ring-1 ring-inset ring-border/15">
      <table className="w-full min-w-[640px] text-[12.5px]">
        <caption className="sr-only">Pollution-station cleaning log entries</caption>
        <thead className="sticky top-0 bg-surface-2"><tr>
          <th className={th}>Date</th><th className={th}>Field</th><th className={clsx(th, "text-right")}>Original</th><th className={clsx(th, "text-right")}>Cleaned</th>
          <th className={th}>Action</th><th className={th}>Reason</th>
        </tr></thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={`${r.date}-${r.field}-${r.rule}-${i}`} className="border-t border-border/10">
              <td className={clsx(td, "whitespace-nowrap text-text")}>{fmtDate(r.date)}</td>
              <td className={clsx(td, "whitespace-nowrap text-text-2")}>{r.field}</td>
              <td className={clsx(td, "tabular text-right text-text-2")}>{r.original_value ?? <span className="text-text-3">blank</span>}</td>
              <td className={clsx(td, "tabular text-right text-text")}>{r.cleaned_value ?? <span className="text-text-3">{r.action === "flagged" ? "kept" : "missing"}</span>}</td>
              <td className={td}><span title={actions[r.action]}><Badge tone={ACTION_TONE[r.action]}>{ACTION_LABEL[r.action]}</Badge></span></td>
              <td className={clsx(td, "text-text-2")}><span className="tabular font-semibold text-text-3">{r.rule}</span> {r.reason}</td>
            </tr>
          ))}
          {!rows.length && <tr><td colSpan={6} className="px-3 py-6 text-center text-text-3">No entries for this action.</td></tr>}
        </tbody>
      </table>
    </div>
  );
}

// ====================================================================== limitations
function Limitations() {
  const items: [string, ReactNode][] = [
    ["Raw target, repeats included", <>The harness scores raw records, so the forecast and every headline count include repeated records. De-duplicated
      series are sensitivity views and are never mixed into the target.</>],
    ["Repeats are approximate", <>Exact and MRD + admission-date repeats are identified by rule, not confirmed against source systems; some may be
      genuine same-day events. No record was removed.</>],
    ["Last day potentially incomplete", <>31 Mar 2019 is the final day in the file and has unusually few records; it is labelled potentially incomplete
      rather than treated as a real drop.</>],
    ["Emergency/OPD split", <>The coding of admission type shifts in Nov–Dec 2017, so the split is shown for context only.</>],
    ["Environment data", <>Air-quality values are station readings with gaps filled only across 3 days or fewer; the AQI is a sub-index scale. Station
      weather is a cross-check; Open-Meteo is the primary weather source. Environment data are not used by the forecast model.</>],
  ];
  return (
    <Panel title="Known limitations" subtitle="Read every page of this dashboard with these in mind" actions={<Badge tone="amber" icon={CircleAlert}>Caveats</Badge>}>
      <dl className="grid grid-cols-1 gap-3 md:grid-cols-2 xl:grid-cols-3">
        {items.map(([t, body]) => (
          <div key={t} className="rounded-xl bg-surface-2/60 p-3.5 ring-1 ring-inset ring-border/15">
            <dt className="text-[13px] font-semibold text-text">{t}</dt>
            <dd className="mt-1 text-[12.5px] leading-5 text-text-2">{body}</dd>
          </div>
        ))}
        <div className="rounded-xl bg-surface-2/60 p-3.5 ring-1 ring-inset ring-border/15">
          <dt className="text-[13px] font-semibold text-text">Where this matters</dt>
          <dd className="mt-1 text-[12.5px] leading-5 text-text-2">
            The rise in repeats in 2019 is part of why the frozen model under-forecast the holdout. See the{" "}
            <a className="font-semibold text-observed-ink underline-offset-2 hover:underline" href={`?view=holdout${hrefFor("/performance")}`}>Model performance</a> page.
          </dd>
        </div>
      </dl>
    </Panel>
  );
}

// ====================================================================== page
export default function DataQuality() {
  const s = useQualitySummary();
  const [def, setDef] = useState<RepeatDef>("mrd");
  return (
    <div>
      <Header />
      {s.isLoading ? <Panel className="mb-6"><Loading label="Loading data-quality summary" /></Panel>
        : s.isError || !s.data ? <div className="mb-6"><Failed what="the data-quality summary" error={s.error} retry={() => s.refetch()} /></div>
          : <Kpis s={s.data} />}
      <RepeatsSection def={def} setDef={setDef} />
      <div className="mb-6 grid grid-cols-1 gap-6 xl:grid-cols-2">
        {s.data ? <FlaggedDays s={s.data} /> : <Panel title="Flagged low-count days"><Loading label="Loading flagged days" /></Panel>}
        <CaseMixPanel />
      </div>
      <div className="mb-6"><Completeness /></div>
      <div className="mb-6 grid grid-cols-1 gap-6 2xl:grid-cols-2">
        {s.data ? <CleaningSummary s={s.data} /> : <Panel title="Cleaning summary by rule"><Loading label="Loading cleaning summary" /></Panel>}
        <PollutionLogPanel />
      </div>
      <Limitations />
      <p className="mt-6 flex items-center gap-2 text-[12px] text-text-3">
        <FileSearch aria-hidden className="h-3.5 w-3.5" />
        Sources: saved Step 2 outputs (daily admissions, repeat counts, cleaned environment data, cleaning log). {DEF_LABEL[def]} is the active repeat view.
      </p>
    </div>
  );
}
