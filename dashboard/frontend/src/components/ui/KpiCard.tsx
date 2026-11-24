import clsx from "clsx";
import type { LucideIcon } from "lucide-react";
import type { ReactNode } from "react";
import type { Provenance } from "../../lib/types";
import { ProvenanceBadge } from "./Badge";

export type Accent = "observed" | "forecast" | "dedup" | "teal";

// gradient accents (decorative only; all text sits on the panel surface and passes contrast checks)
const ACCENT: Record<Accent, { bar: string; glow: string; chip: string }> = {
  observed: { bar: "from-observed via-teal to-observed/40", glow: "bg-observed", chip: "from-observed/25 to-teal/10 text-observed-ink ring-observed/30" },
  teal: { bar: "from-teal via-observed to-teal/40", glow: "bg-teal", chip: "from-teal/25 to-observed/10 text-teal-ink ring-teal/30" },
  forecast: { bar: "from-forecast via-observed/70 to-forecast/40", glow: "bg-forecast", chip: "from-forecast/25 to-observed/10 text-forecast-ink ring-forecast/30" },
  dedup: { bar: "from-dedup via-forecast/70 to-dedup/40", glow: "bg-dedup", chip: "from-dedup/25 to-forecast/10 text-dedup-ink ring-dedup/30" },
};

interface KpiCardProps {
  label: string;
  value: ReactNode;
  unit?: string;
  detail: string;
  provenance: Provenance;
  icon: LucideIcon;
  accent: Accent;
  footer?: ReactNode;
  delay?: number;
}

export function KpiCard({ label, value, unit, detail, provenance, icon: Icon, accent, footer, delay = 0 }: KpiCardProps) {
  const a = ACCENT[accent];
  return (
    <article className="panel group relative overflow-hidden p-5 transition-transform duration-300 hover:-translate-y-0.5 animate-fade-up"
      style={{ animationDelay: `${delay}ms` }} aria-label={`${label}: ${typeof value === "string" || typeof value === "number" ? value : ""} ${unit ?? ""}`}>
      <div aria-hidden className={clsx("absolute inset-x-0 top-0 h-[3px] bg-gradient-to-r", a.bar)} />
      <div aria-hidden className={clsx("pointer-events-none absolute -right-10 -top-12 h-36 w-36 rounded-full opacity-[0.16] blur-2xl transition-opacity duration-300 group-hover:opacity-25", a.glow)} />
      <div className="relative flex items-start justify-between gap-3">
        <span className={clsx("grid h-10 w-10 place-items-center rounded-xl bg-gradient-to-br ring-1 ring-inset", a.chip)}>
          <Icon aria-hidden className="h-5 w-5" strokeWidth={2} />
        </span>
        <ProvenanceBadge kind={provenance} />
      </div>
      <p className="relative mt-4 text-[13px] font-medium text-text-2">{label}</p>
      <p className="relative mt-1 flex items-baseline gap-1.5">
        <span className="tabular text-[30px] font-semibold leading-none tracking-tight text-text">{value}</span>
        {unit && <span className="text-[13px] font-medium text-text-2">{unit}</span>}
      </p>
      <p className="relative mt-2 text-[12.5px] leading-5 text-text-2">{detail}</p>
      {footer && <div className="relative mt-3">{footer}</div>}
    </article>
  );
}
