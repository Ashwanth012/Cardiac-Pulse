import clsx from "clsx";
import { ChevronsLeft, ChevronsRight, HeartPulse, X } from "lucide-react";
import { ROUTES, hrefFor, type RouteDef } from "../../lib/router";

interface SidebarProps {
  route: RouteDef;
  collapsed: boolean;
  onToggleCollapsed: () => void;
  mobileOpen: boolean;
  onCloseMobile: () => void;
}

export function Sidebar({ route, collapsed, onToggleCollapsed, mobileOpen, onCloseMobile }: SidebarProps) {
  return (
    <>
      {mobileOpen && <div className="fixed inset-0 z-30 bg-bg/70 backdrop-blur-sm lg:hidden" onClick={onCloseMobile} aria-hidden />}
      <aside aria-label="Primary"
        className={clsx(
          "fixed inset-y-0 left-0 z-40 flex flex-col border-r border-border/15 bg-surface/90 backdrop-blur-xl transition-[width,transform] duration-300",
          collapsed ? "lg:w-[76px]" : "lg:w-[264px]",
          "w-[264px]", mobileOpen ? "translate-x-0" : "-translate-x-full lg:translate-x-0",
        )}>
        <div className="flex h-16 items-center gap-3 px-4">
          <span className="grid h-10 w-10 shrink-0 place-items-center rounded-xl bg-gradient-to-br from-observed via-teal to-forecast shadow-[0_0_24px_-4px_rgb(var(--c-observed)/0.6)]">
            <HeartPulse aria-hidden className="h-5 w-5 text-[rgb(6_10_24)]" strokeWidth={2.5} />
          </span>
          <div className={clsx("min-w-0 transition-opacity", collapsed && "lg:pointer-events-none lg:opacity-0")}>
            <p className="truncate text-[15px] font-bold tracking-tight text-text">CardioPulse</p>
            <p className="truncate text-[11px] font-medium uppercase tracking-[0.12em] text-text-3">PS-06 Intelligence</p>
          </div>
          <button type="button" onClick={onCloseMobile} className="ml-auto rounded-lg p-1.5 text-text-2 hover:bg-surface-2 lg:hidden" aria-label="Close navigation">
            <X className="h-5 w-5" />
          </button>
        </div>

        <nav className="mt-2 flex-1 space-y-1 overflow-y-auto px-3" aria-label="Sections">
          {/* Internal (analyst) pages are not linked from client pages: the group appears only on an internal page
              or when the app is opened with ?internal (e.g. http://127.0.0.1:4173/?internal#/performance). */}
          {(route.group === "secondary" || new URLSearchParams(window.location.search).has("internal")
            ? (["primary", "secondary"] as const) : (["primary"] as const)).map((g) => (
          <div key={g} className={g === "secondary" ? "pt-3" : undefined}>
          <p className={clsx("px-3 pb-1 pt-2 text-[10.5px] font-semibold uppercase tracking-[0.14em] text-text-3", collapsed && "lg:sr-only")}>
            {g === "primary" ? "Forecasting" : "Internal · data & model"}
          </p>
          {ROUTES.filter((r) => r.group === g).map((r) => {
            const active = r.path === route.path;
            const Icon = r.icon;
            return (
              <a key={r.path} href={hrefFor(r.path)} onClick={onCloseMobile} aria-current={active ? "page" : undefined}
                title={collapsed ? r.short : undefined}
                className={clsx(
                  "group relative flex items-center gap-3 rounded-xl px-3 py-2.5 text-[13.5px] font-medium transition-colors",
                  active ? "bg-gradient-to-r from-observed/[0.16] to-forecast/[0.08] text-text ring-1 ring-inset ring-observed/25"
                    : "text-text-2 hover:bg-surface-2 hover:text-text",
                )}>
                {active && <span aria-hidden className="absolute left-0 top-2 bottom-2 w-[3px] rounded-r bg-gradient-to-b from-observed to-forecast" />}
                <Icon aria-hidden className={clsx("h-[18px] w-[18px] shrink-0", active ? "text-observed-ink" : "text-text-3 group-hover:text-text-2")} />
                <span className={clsx("truncate", collapsed && "lg:sr-only")}>{r.short}</span>
                {!r.ready && (
                  <span className={clsx("ml-auto rounded-md bg-surface-2 px-1.5 py-0.5 text-[10px] font-semibold text-text-3 ring-1 ring-inset ring-border/20", collapsed && "lg:hidden")}>
                    Stage {r.stage}
                  </span>
                )}
              </a>
            );
          })}
          </div>
          ))}
        </nav>

        <div className="border-t border-border/15 p-3">
          <div className={clsx("rounded-xl bg-surface-2/70 p-3 text-[11.5px] leading-4 text-text-2 ring-1 ring-inset ring-border/15", collapsed && "lg:hidden")}>
            <p className="font-semibold text-text">Aggregate data only</p>
            <p className="mt-1">Admission forecast: historical demonstration for 1–14 Apr 2019 (data end 31 Mar 2019). No patient-level information is shown.</p>
          </div>
          <button type="button" onClick={onToggleCollapsed}
            className="mt-2 hidden w-full items-center justify-center gap-2 rounded-xl px-3 py-2 text-[12.5px] font-medium text-text-2 hover:bg-surface-2 hover:text-text lg:flex"
            aria-label={collapsed ? "Expand sidebar" : "Collapse sidebar"} aria-expanded={!collapsed}>
            {collapsed ? <ChevronsRight className="h-4 w-4" /> : <><ChevronsLeft className="h-4 w-4" /> Collapse</>}
          </button>
        </div>
      </aside>
    </>
  );
}
