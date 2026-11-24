const int = new Intl.NumberFormat("en-GB", { maximumFractionDigits: 0 });
const one = new Intl.NumberFormat("en-GB", { minimumFractionDigits: 1, maximumFractionDigits: 1 });

export const fmtInt = (v: number) => int.format(v);
export const fmt1 = (v: number) => one.format(v);
export const fmtPct = (v: number) => `${v > 0 ? "+" : ""}${one.format(v)}%`;

const parse = (iso: string) => new Date(`${iso}T00:00:00`);
export const fmtDate = (iso: string) =>
  parse(iso).toLocaleDateString("en-GB", { day: "numeric", month: "short", year: "numeric" });
export const fmtDayMonth = (iso: string) => parse(iso).toLocaleDateString("en-GB", { day: "numeric", month: "short" });
export const fmtWeekday = (iso: string) => parse(iso).toLocaleDateString("en-GB", { weekday: "short" });
export const fmtLong = (iso: string) =>
  parse(iso).toLocaleDateString("en-GB", { weekday: "long", day: "numeric", month: "long", year: "numeric" });
