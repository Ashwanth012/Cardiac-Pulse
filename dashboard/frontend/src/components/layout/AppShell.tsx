import clsx from "clsx";
import { useEffect, useState, type ReactNode } from "react";
import type { RouteDef } from "../../lib/router";
import { Sidebar } from "./Sidebar";
import { TopBar } from "./TopBar";

const KEY = "ps06-sidebar-collapsed";

export function AppShell({ route, children }: { route: RouteDef; children: ReactNode }) {
  const [collapsed, setCollapsed] = useState(() => {
    try {
      return localStorage.getItem(KEY) === "1";
    } catch {
      return false;
    }
  });
  const [mobileOpen, setMobileOpen] = useState(false);
  useEffect(() => {
    try {
      localStorage.setItem(KEY, collapsed ? "1" : "0");
    } catch {
      /* ignore */
    }
  }, [collapsed]);
  useEffect(() => setMobileOpen(false), [route.path]);

  return (
    <div className="min-h-screen">
      <button type="button" onClick={() => document.getElementById("main")?.focus()}
        className="sr-only focus:not-sr-only focus:fixed focus:left-3 focus:top-3 focus:z-50 focus:rounded-lg focus:bg-surface focus:px-3 focus:py-2">
        Skip to content
      </button>
      <Sidebar route={route} collapsed={collapsed} onToggleCollapsed={() => setCollapsed((c) => !c)}
        mobileOpen={mobileOpen} onCloseMobile={() => setMobileOpen(false)} />
      <div className={clsx("transition-[padding] duration-300", collapsed ? "lg:pl-[76px]" : "lg:pl-[264px]")}>
        <TopBar route={route} onOpenMobile={() => setMobileOpen(true)} />
        <main id="main" tabIndex={-1} className="mx-auto max-w-[1600px] px-4 pb-16 pt-6 outline-none sm:px-6 lg:px-8">{children}</main>
      </div>
    </div>
  );
}
