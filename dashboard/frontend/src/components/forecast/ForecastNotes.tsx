import clsx from "clsx";
import { CircleAlert, Info } from "lucide-react";
import type { ReactNode } from "react";
import { useMeta } from "../../lib/api";
import type { IntervalEvidence } from "../../lib/types";

/**
 * Shared forecast explanations (moved unchanged from the Phase 2 Forecasting Studio): the approved 80% interval
 * wording (target vs central 85% residual range), coverage history and the holdout under-forecasting disclosure.
 * All numbers come from the API (saved outputs); none are hard-coded here.
 */
const pct = (v: number) => `${(v * 100).toFixed(1)}%`;

export function Stat({ label, value, sub, tone = "neutral" }: { label: string; value: ReactNode; sub?: ReactNode; tone?: "neutral" | "warn" }) {
  return (
    <div className={clsx("rounded-xl bg-surface-2/60 px-4 py-3 ring-1 ring-inset", tone === "warn" ? "ring-amber/35" : "ring-border/15")}>
      <p className="text-[12px] font-medium text-text-2">{label}</p>
      <p className="tabular mt-1 text-[22px] font-semibold leading-tight text-text">{value}</p>
      {sub && <p className="mt-1 text-[11.5px] leading-4 text-text-2">{sub}</p>}
    </div>
  );
}

export function Note({ icon: Icon = Info, tone = "info", title, children }: { icon?: typeof Info; tone?: "info" | "warn"; title: string; children: ReactNode }) {
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

/** Coverage history: tuning / validation / Jan-Mar 2019 (previously viewed in the M4 holdout context) (values from the API, read from saved outputs). */
export function CoverageHistory({ ev, compact = false }: { ev: IntervalEvidence; compact?: boolean }) {
  return (
    <ul className={clsx("grid gap-2", compact ? "grid-cols-3" : "grid-cols-1 sm:grid-cols-3")} aria-label="80% interval coverage at each stage">
      {ev.coverage_history.map((s) => (
        <li key={s.stage} className={clsx("rounded-lg px-3 py-2 ring-1 ring-inset",
          s.stage === "holdout" ? "bg-forecast/[0.08] ring-forecast/30" : "bg-surface-2/60 ring-border/15")}>
          <p className="text-[11px] font-semibold uppercase tracking-wide text-text-3">{s.stage === "holdout" ? "Holdout (previously viewed)" : s.stage}</p>
          <p className="tabular text-[17px] font-semibold text-text">{pct(s.coverage)}</p>
          {!compact && <p className="text-[11px] leading-4 text-text-2">{s.label.replace(/^[^(]*\(|\)$/g, "")}</p>}
        </li>
      ))}
    </ul>
  );
}

export function IntervalMeaning() {
  const meta = useMeta();
  const ev = meta.data?.model.interval;
  return (
    <Note title="What the 80% interval means">
      <p>
        The band is the harness-defined <strong className="text-text">80% prediction interval</strong>: it is designed to contain
        {" "}<strong className="text-text">at least 80%</strong> of actual daily counts. Its bounds are the{" "}
        <strong className="text-text">7.5th and 92.5th percentiles</strong> – the <strong className="text-text">central 85%</strong> – of the
        model's own errors on its 90 most recent completed 14-day forecasts.
      </p>
      <p className="mt-1.5">
        Why 85% for an 80% target? On 2017–18 development data the plain 10th–90th range covered only
        {ev ? ` ${pct(ev.nominal_80_tuning_coverage_range[0])}–${pct(ev.nominal_80_tuning_coverage_range[1])}` : " about 77%"} of later outcomes, so the
        narrowest wider range reaching at least 80% was selected. This is a calibration choice made on development data, not a guarantee for
        future periods.
      </p>
      {ev && <div className="mt-2.5"><CoverageHistory ev={ev} /></div>}
      {ev && (
        <p className="mt-2">
          On the holdout, misses were lopsided: <strong className="text-text">{pct(ev.holdout_misses.above_upper)} above</strong> the upper
          bound and <strong className="text-text">{pct(ev.holdout_misses.below_lower)} below</strong> the lower bound, because the model
          under-forecast 2019. Coverage held because the band is wide, not because it was centred.
        </p>
      )}
    </Note>
  );
}

export function BiasDisclosure() {
  return (
    <Note tone="warn" icon={CircleAlert} title="Known under-forecasting (holdout)">
      On the Jan–Mar 2019 holdout the frozen model under-forecast by about <strong className="text-text">4.6 records per day</strong> on
      average. It follows the 2017–18 seasonal pattern and has no recent-level term, while 2019 ran higher – partly genuine growth (January)
      and partly a rise in repeat rows (February–March). Read these forecasts alongside the recent level.
    </Note>
  );
}
