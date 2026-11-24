import clsx from "clsx";
import type { ReactNode } from "react";

interface PanelProps {
  title?: ReactNode;
  subtitle?: ReactNode;
  actions?: ReactNode;
  children: ReactNode;
  className?: string;
  bodyClassName?: string;
  as?: "section" | "div" | "aside";
  labelledBy?: string;
  /** stretch the body to the panel height (for charts sharing a row with a taller panel) */
  fill?: boolean;
}

export function Panel({ title, subtitle, actions, children, className, bodyClassName, as = "section", labelledBy, fill }: PanelProps) {
  const Tag = as;
  const id = labelledBy ?? (typeof title === "string" ? `p-${title.toLowerCase().replace(/[^a-z0-9]+/g, "-")}` : undefined);
  return (
    <Tag className={clsx("panel animate-fade-up", fill && "flex flex-col", className)} aria-labelledby={title ? id : undefined}>
      {(title || actions) && (
        <header className="flex flex-wrap items-start justify-between gap-3 px-5 pt-5">
          <div className="min-w-0">
            {title && <h2 id={id} className="text-[15px] font-semibold tracking-tight text-text">{title}</h2>}
            {subtitle && <p className="mt-1 text-[13px] leading-5 text-text-2">{subtitle}</p>}
          </div>
          {actions && <div className="flex flex-wrap items-center gap-2">{actions}</div>}
        </header>
      )}
      <div className={clsx("px-5 pb-5", title ? "pt-4" : "pt-5", fill && "flex flex-1 flex-col", bodyClassName)}>{children}</div>
    </Tag>
  );
}
