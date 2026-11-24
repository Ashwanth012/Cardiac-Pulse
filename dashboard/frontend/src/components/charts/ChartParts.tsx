import clsx from "clsx";
import type { ReactNode } from "react";

/** Glass tooltip shell shared by all charts. Text uses text tokens, never series colours. */
export function TooltipCard({ title, children }: { title: ReactNode; children: ReactNode }) {
  return (
    <div className="min-w-[200px] rounded-xl border border-border/20 bg-surface/95 px-3 py-2.5 text-[12.5px] shadow-panel backdrop-blur-md">
      <p className="mb-1.5 font-semibold text-text">{title}</p>
      <div className="space-y-1">{children}</div>
    </div>
  );
}

export function TooltipRow({ swatch, label, value, dashed }: { swatch?: string; label: ReactNode; value: ReactNode; dashed?: boolean }) {
  return (
    <div className="flex items-center justify-between gap-4">
      <span className="flex items-center gap-2 text-text-2">
        {swatch && <span aria-hidden className={clsx("inline-block h-0 w-4 border-t-2", dashed && "border-dashed")} style={{ borderColor: swatch }} />}
        {label}
      </span>
      <span className="tabular font-semibold text-text">{value}</span>
    </div>
  );
}

export type LegendKind = "line" | "dashed" | "band" | "bar" | "ring";

export function LegendItem({ color, label, kind = "line" }: { color: string; label: string; kind?: LegendKind }) {
  return (
    <span className="inline-flex items-center gap-2 text-[12px] font-medium text-text-2">
      {kind === "band" && <span aria-hidden className="h-3 w-5 rounded-sm" style={{ background: color, opacity: 0.3 }} />}
      {kind === "bar" && <span aria-hidden className="h-3 w-2 rounded-sm" style={{ background: color, opacity: 0.55 }} />}
      {kind === "ring" && <span aria-hidden className="h-2.5 w-2.5 rounded-full border-2" style={{ borderColor: color }} />}
      {(kind === "line" || kind === "dashed") && (
        <span aria-hidden className={clsx("inline-block h-0 w-5 border-t-[2.5px]", kind === "dashed" && "border-dashed")} style={{ borderColor: color }} />
      )}
      {label}
    </span>
  );
}
