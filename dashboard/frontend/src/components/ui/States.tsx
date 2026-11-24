import clsx from "clsx";
import { RefreshCw, ServerCrash } from "lucide-react";

export function Skeleton({ className }: { className?: string }) {
  return <div aria-hidden className={clsx("skeleton", className)} />;
}

export function Loading({ label, className }: { label: string; className?: string }) {
  return (
    <div role="status" aria-live="polite" className={clsx("flex flex-col gap-3", className)}>
      <span className="sr-only">{label}</span>
      <Skeleton className="h-4 w-1/3" />
      <Skeleton className="h-40 w-full" />
    </div>
  );
}

export function ErrorState({ title, message, onRetry }: { title: string; message: string; onRetry?: () => void }) {
  return (
    <div role="alert" className="flex flex-col items-start gap-3 rounded-xl bg-rose/[0.08] p-4 ring-1 ring-inset ring-rose/30">
      <div className="flex items-center gap-2 text-rose-ink">
        <ServerCrash aria-hidden className="h-4 w-4" />
        <p className="text-[13px] font-semibold">{title}</p>
      </div>
      <p className="text-[12.5px] text-text-2">{message}</p>
      {onRetry && (
        <button type="button" onClick={onRetry}
          className="inline-flex items-center gap-1.5 rounded-lg bg-surface-2 px-3 py-1.5 text-[12.5px] font-medium text-text ring-1 ring-inset ring-border/25 hover:bg-surface">
          <RefreshCw aria-hidden className="h-3.5 w-3.5" /> Retry
        </button>
      )}
    </div>
  );
}
