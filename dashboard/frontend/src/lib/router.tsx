import { useEffect, useState } from "react";
import { Activity, CalendarDays, CloudSun, Gauge, LayoutDashboard, Lightbulb, ShieldCheck, type LucideIcon } from "lucide-react";

/** Minimal hash router: static internal routes only, no external navigation (replaces react-router). */
export interface RouteDef {
  path: string;
  label: string;
  short: string;
  icon: LucideIcon;
  /** primary = main product navigation; secondary = "About the data & model" */
  group: "primary" | "secondary";
  /** redesign stage that delivers the page (0 = carried over from earlier phases) */
  stage: 0 | 1 | 2 | 3;
  ready: boolean;
  description: string;
}

export const ROUTES: RouteDef[] = [
  { path: "/", label: "Command Centre", short: "Command Centre", icon: LayoutDashboard, group: "primary", stage: 1, ready: true,
    description: "Expected admissions for the 14-day forecast, days to watch, holidays and reliability" },
  { path: "/forecast", label: "14-Day Admission Forecast", short: "14-Day Forecast", icon: Activity, group: "primary", stage: 1, ready: true,
    description: "Daily expected admissions with 80% prediction bounds, calendar and demand indicators" },
  { path: "/conditions", label: "Weather & Air Quality", short: "Weather & AQI", icon: CloudSun, group: "primary", stage: 2, ready: true,
    description: "Live upcoming 14-day weather and US AQI forecasts for Ludhiana (Open-Meteo) - a different, current period, shown as context only" },
  { path: "/calendar", label: "Calendar Context", short: "Calendar", icon: CalendarDays, group: "primary", stage: 3, ready: true,
    description: "Holidays, festivals and weekdays beside the 14-day admission forecast" },
  { path: "/insights", label: "Prediction Insights", short: "Prediction Insights", icon: Lightbulb, group: "primary", stage: 3, ready: true,
    description: "Why the forecast has its shape, in plain language - model inputs kept separate from context" },
  { path: "/performance", label: "Model Performance", short: "Model performance", icon: Gauge, group: "secondary", stage: 0, ready: true,
    description: "How the frozen model works, a retrospective explorer and the saved 2019 holdout evaluation" },
  { path: "/quality", label: "Data Quality & Monitoring", short: "Data quality", icon: ShieldCheck, group: "secondary", stage: 0, ready: true,
    description: "Raw vs de-duplicated records, completeness and the cleaning summary" },
];

/** Old links keep working: retired research routes point at their replacements. */
export const ALIASES: Record<string, string> = { "/environment": "/conditions", "/workload": "/forecast", "/studio": "/performance" };

const current = () => {
  const raw = window.location.hash.replace(/^#/, "") || "/";
  const view = new URLSearchParams(window.location.search).get("view");
  // Phase 2 links (?view=retro|holdout#/forecast) open the Model Performance page they used to show.
  const p = raw === "/forecast" && (view === "retro" || view === "holdout") ? "/performance" : ALIASES[raw] ?? raw;
  return ROUTES.some((r) => r.path === p) ? p : "/";
};

export function useRoute(): RouteDef {
  const [path, setPath] = useState(current);
  useEffect(() => {
    const on = () => {
      setPath(current());
      window.scrollTo({ top: 0 });
    };
    window.addEventListener("hashchange", on);
    return () => window.removeEventListener("hashchange", on);
  }, []);
  return ROUTES.find((r) => r.path === path) ?? ROUTES[0];
}

export const hrefFor = (path: string) => `#${path}`;
