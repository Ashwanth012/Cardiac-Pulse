import { createContext, useContext, useMemo, useState, type ReactNode } from "react";

export const WINDOWS = [
  { days: 30, label: "30D" },
  { days: 90, label: "90D" },
  { days: 180, label: "6M" },
  { days: 365, label: "1Y" },
  { days: 730, label: "All" },
] as const;

interface FilterCtx {
  windowDays: number;
  setWindowDays: (d: number) => void;
}
const Ctx = createContext<FilterCtx>({ windowDays: 90, setWindowDays: () => {} });

export function FilterProvider({ children }: { children: ReactNode }) {
  const [windowDays, setWindowDays] = useState(90);
  return <Ctx.Provider value={useMemo(() => ({ windowDays, setWindowDays }), [windowDays])}>{children}</Ctx.Provider>;
}
export const useFilters = () => useContext(Ctx);
