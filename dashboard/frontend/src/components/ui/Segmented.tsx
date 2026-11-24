import clsx from "clsx";

interface Option<T> {
  value: T;
  label: string;
}

export function Segmented<T extends string | number>({ options, value, onChange, label }: {
  options: Option<T>[]; value: T; onChange: (v: T) => void; label: string;
}) {
  return (
    <div role="radiogroup" aria-label={label} className="inline-flex rounded-xl bg-surface-2/70 p-1 ring-1 ring-inset ring-border/20">
      {options.map((o) => {
        const active = o.value === value;
        return (
          <button key={String(o.value)} type="button" role="radio" aria-checked={active} onClick={() => onChange(o.value)}
            className={clsx("tabular rounded-lg px-2.5 py-1 text-[12px] font-semibold transition-colors",
              active ? "bg-surface text-text shadow-sm ring-1 ring-inset ring-observed/40" : "text-text-2 hover:text-text")}>
            {o.label}
          </button>
        );
      })}
    </div>
  );
}
