import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from "react";

export type Theme = "dark" | "light";
const KEY = "ps06-theme";

interface ThemeCtx {
  theme: Theme;
  toggle: () => void;
}
const Ctx = createContext<ThemeCtx>({ theme: "dark", toggle: () => {} });

export function ThemeProvider({ children }: { children: ReactNode }) {
  const [theme, setTheme] = useState<Theme>(() =>
    document.documentElement.dataset.theme === "light" ? "light" : "dark",
  );
  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    try {
      localStorage.setItem(KEY, theme);
    } catch {
      /* storage unavailable: theme still applies for this session */
    }
  }, [theme]);
  const toggle = useCallback(() => setTheme((t) => (t === "dark" ? "light" : "dark")), []);
  return <Ctx.Provider value={useMemo(() => ({ theme, toggle }), [theme, toggle])}>{children}</Ctx.Provider>;
}

export const useTheme = () => useContext(Ctx);

const TOKENS = ["observed", "forecast", "dedup", "teal", "amber", "rose", "ok", "text", "text-2", "text-3", "border", "surface", "surface-2"] as const;
export type ChartColors = Record<(typeof TOKENS)[number], string>;

/** Resolved token colours for SVG chart attributes (CSS variables do not resolve inside SVG attributes). */
export function useChartColors(): ChartColors {
  const { theme } = useTheme();
  return useMemo(() => {
    const cs = getComputedStyle(document.documentElement);
    return Object.fromEntries(
      TOKENS.map((t) => [t, `rgb(${cs.getPropertyValue(`--c-${t}`).trim().split(/\s+/).join(",")})`]),
    ) as ChartColors;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [theme]);
}

export function usePrefersReducedMotion() {
  const [reduced, setReduced] = useState(() => window.matchMedia?.("(prefers-reduced-motion: reduce)").matches ?? false);
  useEffect(() => {
    const mq = window.matchMedia?.("(prefers-reduced-motion: reduce)");
    if (!mq) return;
    const on = () => setReduced(mq.matches);
    mq.addEventListener("change", on);
    return () => mq.removeEventListener("change", on);
  }, []);
  return reduced;
}
