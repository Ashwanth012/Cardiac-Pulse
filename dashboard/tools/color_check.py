"""Accessibility and colour-vision check for the dashboard design tokens (no dependencies).

Reads dashboard/frontend/src/theme/tokens.css (dark and light blocks) and checks:
  1. WCAG 2.1 contrast ratio for every text token on every background it is used on
     (normal text >= 4.5:1), including tinted badge backgrounds.
  2. Non-text contrast (WCAG 1.4.11) for chart marks and UI accents on panel and page
     backgrounds (>= 3:1).
  3. Colour-vision separation for chart series that appear together: OKLab distance (x100)
     between each pair under normal vision and simulated protanopia, deuteranopia and
     tritanopia (Machado, Oliveira & Fernandes 2009, severity 1.0). Normal-vision floor 15;
     CVD target 8 (6-8 only allowed with secondary encoding such as dashes, which we use).

Usage:  python dashboard/tools/color_check.py      (exit code 1 on any failure)
"""
from __future__ import annotations

import itertools
import re
import sys
from pathlib import Path

TOKENS = Path(__file__).resolve().parents[1] / "frontend" / "src" / "theme" / "tokens.css"

TEXT_TOKENS = ["text", "text-2", "text-3"]
TEXT_BACKGROUNDS = ["bg", "bg-2", "surface", "surface-2"]
INK_TOKENS = ["observed-ink", "forecast-ink", "dedup-ink", "teal-ink", "amber-ink", "rose-ink", "ok-ink"]
MARK_TOKENS = ["observed", "forecast", "dedup", "teal", "amber", "rose", "ok"]
MARK_BACKGROUNDS = ["surface", "bg"]
BADGE_TINT_ALPHA = 0.14          # badge background = mark colour at this alpha over the panel surface
PANEL_ALPHA = {"dark": 0.72, "light": 0.9}
# chart series that are drawn together in one chart (pairs are checked)
SERIES_SETS = {
    "forecast chart": ["observed", "forecast"],
    "raw vs de-duplicated": ["observed", "dedup"],
    "three-series comparison": ["observed", "forecast", "dedup"],
    "alerts": ["amber", "rose", "observed"],
    "interval miss breakdown": ["observed", "forecast", "amber"],
    "completeness strip": ["observed", "amber"],
    "OPD share anomaly": ["observed", "amber"],
    "14-day forecast with holiday markers": ["observed", "forecast", "amber", "text"],
    "live temperature (max / min)": ["rose", "teal"],
    "US AQI categories (Good = teal)": ["teal", "amber", "rose"],
}
# sequential ramps: (hue token, alphas over the panel surface; 0 = empty cell drawn in surface-2).
# Adjacent steps must stay distinguishable (OKLab x100 >= RAMP_FLOOR under every vision type); exact
# counts are always available in the hover readout, so this is a secondary check, not WCAG 1.4.11.
RAMPS = {"repeat-records heatmap": ("dedup", [0, 0.28, 0.5, 0.75, 1.0])}
RAMP_FLOOR = 5.0
CVD_FLOOR, CVD_TARGET, NORMAL_FLOOR = 6.0, 8.0, 15.0

MACHADO = {  # severity 1.0, applied in linear RGB
    "protanopia": [[0.152286, 1.052583, -0.204868], [0.114503, 0.786281, 0.099216], [-0.003882, -0.048116, 1.051998]],
    "deuteranopia": [[0.367322, 0.860646, -0.227968], [0.280085, 0.672501, 0.047413], [-0.011820, 0.042940, 0.968881]],
    "tritanopia": [[1.255528, -0.076749, -0.178779], [-0.078411, 0.930809, 0.147602], [0.004733, 0.691367, 0.303900]],
}


def parse_tokens(path=TOKENS) -> dict:
    css = path.read_text(encoding="utf-8")
    themes = {}
    for name in ("dark", "light"):
        block = re.search(r"/\* theme: " + name + r" \*/.*?\{(.*?)\}", css, re.S).group(1)
        themes[name] = {k: tuple(int(x) for x in v.split())
                        for k, v in re.findall(r"--c-([\w-]+):\s*([\d ]+);", block)}
    return themes


def lin(c):
    c = c / 255
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def unlin(c):
    c = min(max(c, 0.0), 1.0)
    return 255 * (12.92 * c if c <= 0.0031308 else 1.055 * c ** (1 / 2.4) - 0.055)


def luminance(rgb):
    r, g, b = (lin(x) for x in rgb)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast(a, b):
    la, lb = sorted((luminance(a), luminance(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


def blend(fg, bg, alpha):
    return tuple(round(alpha * f + (1 - alpha) * b) for f, b in zip(fg, bg))


def simulate(rgb, kind):
    if kind == "normal":
        return rgb
    m = MACHADO[kind]
    l = [lin(x) for x in rgb]
    return tuple(unlin(sum(m[i][j] * l[j] for j in range(3))) for i in range(3))


def oklab(rgb):
    r, g, b = (lin(x) for x in rgb)
    l = (0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b) ** (1 / 3)
    m = (0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b) ** (1 / 3)
    s = (0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b) ** (1 / 3)
    return (0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s,
            1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s,
            0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s)


def delta(a, b):
    return 100 * sum((x - y) ** 2 for x, y in zip(oklab(a), oklab(b))) ** 0.5


def run(path=TOKENS) -> tuple[list, list]:
    themes = parse_tokens(path)
    rows, failures = [], []

    def check(theme, kind, item, against, value, floor):
        ok = value >= floor
        rows.append((theme, kind, item, against, round(value, 2), floor, "PASS" if ok else "FAIL"))
        if not ok:
            failures.append(rows[-1])

    for theme, t in themes.items():
        page = t["bg"]
        panel = blend(t["surface"], page, PANEL_ALPHA[theme])          # translucent panel over the page
        bgs = {"bg": page, "bg-2": t["bg-2"], "surface": panel, "surface-2": blend(t["surface-2"], page, PANEL_ALPHA[theme])}
        for tok in TEXT_TOKENS:
            for bg in TEXT_BACKGROUNDS:
                check(theme, "text 4.5:1", tok, bg, contrast(t[tok], bgs[bg]), 4.5)
        for tok in INK_TOKENS:
            mark = tok.replace("-ink", "")
            check(theme, "badge text 4.5:1", tok, "surface", contrast(t[tok], bgs["surface"]), 4.5)
            tint = blend(t[mark], bgs["surface"], BADGE_TINT_ALPHA)
            check(theme, "badge text 4.5:1", tok, f"{mark} tint {BADGE_TINT_ALPHA:.0%}", contrast(t[tok], tint), 4.5)
        for tok in MARK_TOKENS:
            for bg in MARK_BACKGROUNDS:
                check(theme, "graphic 3:1", tok, bg, contrast(t[tok], bgs[bg]), 3.0)
        for set_name, toks in SERIES_SETS.items():
            for a, b in itertools.combinations(toks, 2):
                for vision in ("normal", "protanopia", "deuteranopia", "tritanopia"):
                    d = delta(simulate(t[a], vision), simulate(t[b], vision))
                    floor = NORMAL_FLOOR if vision == "normal" else CVD_FLOOR
                    check(theme, f"separation {vision}", f"{a} vs {b}", set_name, d, floor)
        for name, (tok, alphas) in RAMPS.items():
            steps = [t["surface-2"] if a == 0 else blend(t[tok], panel, a) for a in alphas]
            for i in range(len(steps) - 1):
                for vision in ("normal", "protanopia", "deuteranopia", "tritanopia"):
                    d = delta(simulate(steps[i], vision), simulate(steps[i + 1], vision))
                    check(theme, f"ramp step {vision}", f"{tok} {alphas[i]}->{alphas[i + 1]}", name, d, RAMP_FLOOR)
    return rows, failures


def main():
    rows, failures = run()
    w = max(len(r[2]) for r in rows)
    for r in rows:
        print(f"{r[0]:5s}  {r[1]:26s} {r[2]:<{w}s}  on/in {r[3]:24s} {r[4]:7.2f}  (min {r[5]:>4})  {r[6]}")
    print(f"\n{len(rows)} checks, {len(failures)} failures")
    weak = [r for r in rows if r[1].startswith("separation") and "normal" not in r[1] and r[4] < CVD_TARGET]
    if weak:
        print(f"{len(weak)} CVD pairs between floor {CVD_FLOOR} and target {CVD_TARGET} (secondary encoding required):")
        for r in weak:
            print("   ", r[0], r[1], r[2], r[4])
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
