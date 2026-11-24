import clsx from "clsx";
import { Eye, FlaskConical, History, Layers, LineChart, TriangleAlert, type LucideIcon } from "lucide-react";
import type { Provenance } from "../../lib/types";

type Tone = "observed" | "forecast" | "dedup" | "teal" | "amber" | "rose" | "ok" | "neutral";

const TONE: Record<Tone, string> = {
  observed: "text-observed-ink bg-observed/[0.14] ring-observed/30",
  forecast: "text-forecast-ink bg-forecast/[0.14] ring-forecast/30",
  dedup: "text-dedup-ink bg-dedup/[0.14] ring-dedup/30",
  teal: "text-teal-ink bg-teal/[0.14] ring-teal/30",
  amber: "text-amber-ink bg-amber/[0.14] ring-amber/30",
  rose: "text-rose-ink bg-rose/[0.14] ring-rose/30",
  ok: "text-ok-ink bg-ok/[0.14] ring-ok/30",
  neutral: "text-text-2 bg-surface-2 ring-border/25",
};

export function Badge({ tone = "neutral", icon: Icon, children, className }: {
  tone?: Tone; icon?: LucideIcon; children: React.ReactNode; className?: string;
}) {
  return (
    <span className={clsx("inline-flex items-center gap-1 whitespace-nowrap rounded-full px-2 py-0.5 text-[11px] font-semibold uppercase tracking-wide ring-1 ring-inset", TONE[tone], className)}>
      {Icon && <Icon aria-hidden className="h-3 w-3" strokeWidth={2.5} />}
      {children}
    </span>
  );
}

const PROV: Record<Provenance, { tone: Tone; label: string; icon: LucideIcon }> = {
  observed: { tone: "observed", label: "Observed", icon: Eye },
  historical_forecast: { tone: "forecast", label: "Historical forecast", icon: History },
  sensitivity: { tone: "dedup", label: "Sensitivity", icon: Layers },
  exploratory: { tone: "amber", label: "Exploratory", icon: FlaskConical },
  evaluation: { tone: "teal", label: "Evaluation", icon: LineChart },
};

export function ProvenanceBadge({ kind, className }: { kind: Provenance; className?: string }) {
  const p = PROV[kind];
  return <Badge tone={p.tone} icon={p.icon} className={className}>{p.label}</Badge>;
}

export function WarningBadge({ children }: { children: React.ReactNode }) {
  return <Badge tone="amber" icon={TriangleAlert}>{children}</Badge>;
}
