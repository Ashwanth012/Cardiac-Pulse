import { CalendarClock, Menu, Moon, Sun } from "lucide-react";
import { WINDOWS, useFilters } from "../../lib/filters";
import type { RouteDef } from "../../lib/router";
import { useTheme } from "../../lib/theme";
import { Segmented } from "../ui/Segmented";

export function TopBar({ route, onOpenMobile }: { route: RouteDef; onOpenMobile: () => void }) {
  const { theme, toggle } = useTheme();
  const { windowDays, setWindowDays } = useFilters();
  return (
    <header className="sticky top-0 z-20 border-b border-border/15 bg-bg/75 backdrop-blur-xl">
      <div className="mx-auto flex max-w-[1600px] flex-wrap items-center gap-x-4 gap-y-3 px-4 py-3 sm:px-6 lg:px-8">
        <button type="button" onClick={onOpenMobile} className="rounded-lg p-2 text-text-2 hover:bg-surface-2 lg:hidden" aria-label="Open navigation">
          <Menu className="h-5 w-5" />
        </button>
        <div className="min-w-[240px] flex-1">
          <p className="truncate text-[11px] font-semibold uppercase tracking-[0.14em] text-text-3">PS-06 · Cardiac admissions analytics · Ludhiana</p>
          <h1 className="truncate text-lg font-semibold tracking-tight text-text sm:text-xl">{route.label}</h1>
        </div>
        <div className="flex flex-wrap items-center gap-2 sm:gap-3">
          {(route.path === "/" || route.path === "/quality") && (
            <div className="flex items-center gap-2">
              <span className="hidden text-[12px] font-medium text-text-2 md:inline">Trend window</span>
              <Segmented label="Trend window ending 31 Mar 2019" value={windowDays} onChange={setWindowDays}
                options={WINDOWS.map((w) => ({ value: w.days, label: w.label }))} />
            </div>
          )}
          <span className="inline-flex items-center gap-1.5 rounded-xl bg-surface-2/70 px-3 py-1.5 text-[12px] font-medium text-text-2 ring-1 ring-inset ring-border/20">
            <CalendarClock aria-hidden className="h-3.5 w-3.5 text-observed-ink" />
            Admissions data through <span className="font-semibold text-text">31 Mar 2019</span>
          </span>
          <button type="button" onClick={toggle} aria-label={`Switch to ${theme === "dark" ? "light" : "dark"} theme`}
            className="grid h-9 w-9 place-items-center rounded-xl bg-surface-2/70 text-text-2 ring-1 ring-inset ring-border/20 transition-colors hover:text-text">
            {theme === "dark" ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
          </button>
        </div>
      </div>
    </header>
  );
}
