import clsx from "clsx";
import { CircleAlert, Info, OctagonAlert, type LucideIcon } from "lucide-react";
import type { Alert } from "../../lib/types";

const SEV: Record<Alert["severity"], { icon: LucideIcon; label: string; ring: string; icoBg: string; text: string }> = {
  info: { icon: Info, label: "Info", ring: "ring-observed/25", icoBg: "bg-observed/[0.14] text-observed-ink", text: "text-observed-ink" },
  warning: { icon: CircleAlert, label: "Warning", ring: "ring-amber/30", icoBg: "bg-amber/[0.14] text-amber-ink", text: "text-amber-ink" },
  critical: { icon: OctagonAlert, label: "Critical", ring: "ring-rose/30", icoBg: "bg-rose/[0.14] text-rose-ink", text: "text-rose-ink" },
};

/** Severity is always conveyed by icon + text label, never by colour alone. */
export function Callout({ alert }: { alert: Alert }) {
  const s = SEV[alert.severity];
  const Icon = s.icon;
  return (
    <li className={clsx("flex gap-3 rounded-xl bg-surface-2/60 p-3 ring-1 ring-inset", s.ring)}>
      <span className={clsx("mt-0.5 grid h-8 w-8 shrink-0 place-items-center rounded-lg", s.icoBg)}>
        <Icon aria-hidden className="h-4 w-4" strokeWidth={2.25} />
      </span>
      <div className="min-w-0">
        <p className="text-[13px] font-semibold text-text">
          <span className={clsx("mr-1.5 text-[11px] font-bold uppercase tracking-wide", s.text)}>{s.label}</span>
          {alert.title}
        </p>
        <p className="mt-1 text-[12.5px] leading-5 text-text-2">{alert.detail}</p>
      </div>
    </li>
  );
}
