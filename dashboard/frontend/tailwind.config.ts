import type { Config } from "tailwindcss";

const v = (name: string) => `rgb(var(--c-${name}) / <alpha-value>)`;
const names = ["bg", "bg-2", "surface", "surface-2", "border", "text", "text-2", "text-3",
  "observed", "forecast", "dedup", "teal", "amber", "rose", "ok",
  "observed-ink", "forecast-ink", "dedup-ink", "teal-ink", "amber-ink", "rose-ink", "ok-ink"];

export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  darkMode: ["selector", '[data-theme="dark"]'],
  theme: {
    extend: {
      colors: Object.fromEntries(names.map((n) => [n, v(n)])),
      fontFamily: {
        sans: ["Inter", "ui-sans-serif", "system-ui", "Segoe UI", "Roboto", "Arial", "sans-serif"],
        mono: ["JetBrains Mono", "ui-monospace", "SFMono-Regular", "Consolas", "monospace"],
      },
      boxShadow: { panel: "var(--shadow)" },
      borderRadius: { xl2: "1.125rem" },
      keyframes: {
        "fade-up": { "0%": { opacity: "0", transform: "translateY(6px)" }, "100%": { opacity: "1", transform: "none" } },
        shimmer: { "0%": { backgroundPosition: "-400px 0" }, "100%": { backgroundPosition: "400px 0" } },
      },
      animation: { "fade-up": "fade-up 420ms cubic-bezier(.2,.7,.2,1) both", shimmer: "shimmer 1.4s linear infinite" },
    },
  },
  plugins: [],
} satisfies Config;
