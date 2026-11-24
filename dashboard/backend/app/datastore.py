"""Read-only access to aggregate project outputs, with column whitelists and a PII guard.

Patient-level sources (the raw admission and mortality files, outputs/duplicate_rows.csv and the
admission rows of outputs/cleaning_log.csv) are never loaded here. Every API response passes
`assert_no_pii` before it is returned.
"""
from __future__ import annotations

import json
import re
from functools import cached_property

import numpy as np
import pandas as pd

from . import config as C

DAILY_COLUMNS = ["date", "admissions", "admissions_emergency", "admissions_opd", "extra_exact_copies",
                 "extra_near_copies", "admissions_dedup_exact", "admissions_dedup_mrd_doa", "low_count_flag",
                 "zero_count_day"]
DUP_MONTH_COLUMNS = ["month", "rows", "exact_extra", "near_extra", "mrd_doa_extra", "exact_extra_pct", "mrd_doa_extra_pct"]
FESTIVAL_COLUMNS = ["date", "name", "type"]
HOLDOUT_METRIC_COLUMNS = ["MAE", "RMSE", "WAPE", "bias", "MAE_ratio_vs_B0", "coverage80", "width80", "n_cutoffs", "n_points"]

SERIES = {
    "raw": ("admissions", "Raw admission records (scored target)"),
    "dedup_exact": ("admissions_dedup_exact", "Exact duplicates removed (sensitivity)"),
    "dedup_mrd": ("admissions_dedup_mrd_doa", "One record per MRD + admission date (sensitivity)"),
}

FORBIDDEN_KEYS = {"sno", "s.no", "mrd", "mrd no.", "row_ref", "raw_value", "new_value", "age", "gender",
                  "rural", "outcome", "d.o.d", "dod"}
FORBIDDEN_TEXT = re.compile(r"\b(?:S\.?NO|MRD(?:\s*No\.?)?)\s*[:#]?\s*\d{2,}", re.IGNORECASE)


class PIIError(RuntimeError):
    pass


def assert_no_pii(obj, path="$"):
    """Raise PIIError if a response contains a patient-identifier key or an identifier-like value."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if str(k).strip().lower() in FORBIDDEN_KEYS:
                raise PIIError(f"forbidden key {k!r} at {path}")
            assert_no_pii(v, f"{path}.{k}")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            assert_no_pii(v, f"{path}[{i}]")
    elif isinstance(obj, str) and FORBIDDEN_TEXT.search(obj):
        raise PIIError(f"identifier-like text at {path}")
    return obj


def _read(path, columns, **kw):
    df = pd.read_csv(path, **kw)
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise RuntimeError(f"{path.name}: expected columns missing: {missing}")
    return df[columns]


class DataStore:
    """Loads whitelisted aggregate tables once; all methods are pure reads."""

    def __init__(self, root=None):
        self._root = root or C.ROOT
        out = self._root / "outputs"
        d = _read(out / "daily_admissions.csv", DAILY_COLUMNS, parse_dates=["date"])
        self.daily = d.set_index("date").sort_index()
        expected = pd.date_range(C.DATA_START, C.DATA_END, freq="D")
        if not self.daily.index.equals(expected):
            raise RuntimeError("daily_admissions.csv does not cover the expected calendar exactly once")
        self.dup_month = _read(out / "duplicates_by_month.csv", DUP_MONTH_COLUMNS).set_index("month")
        f = _read((root or C.ROOT) / "festival_calendar_punjab.csv", FESTIVAL_COLUMNS)
        f["date"] = pd.to_datetime(f["date"], format="%Y-%m-%d")
        self.festivals = f
        h = pd.read_csv(out / "holdout" / "tables" / "metrics_primary_grid.csv", index_col=0)
        self.holdout_m4 = h.loc["M4 (forecast.py)", HOLDOUT_METRIC_COLUMNS].astype(float)
        self.holdout_b0 = h.loc["Seasonal naive (harness baseline)", HOLDOUT_METRIC_COLUMNS].astype(float)
        iv = pd.read_csv(out / "backtest" / "tables" / "interval_settings_chosen.csv").set_index("model")
        self.interval_m4 = iv.loc["M4_glm_Nall_K2_T0_L0", ["q_lo", "q_hi", "pool"]]
        val = pd.read_csv(out / "backtest" / "tables" / "validation_family_winners.csv", index_col=0)
        self.validation_m4 = val.loc["M4", ["MAE", "ratio_vs_B0", "coverage80", "width80", "n_cutoffs"]]

    # ------------------------------------------------------------------ helpers
    def holidays_between(self, start, end) -> dict:
        f = self.festivals[(self.festivals.date >= pd.Timestamp(start)) & (self.festivals.date <= pd.Timestamp(end))]
        return {d.strftime("%Y-%m-%d"): sorted(g["name"].tolist()) for d, g in f.groupby("date")}

    @cached_property
    def trailing_28(self) -> pd.Series:
        return self.daily["admissions"].rolling(28, min_periods=28).mean()

    def incomplete_day_context(self) -> dict:
        day = pd.Timestamp(C.POTENTIALLY_INCOMPLETE_DAY)
        prev = self.daily["admissions"].loc[day - pd.Timedelta(days=28): day - pd.Timedelta(days=1)]
        return {"date": C.POTENTIALLY_INCOMPLETE_DAY, "admissions": int(self.daily.at[day, "admissions"]),
                "median_previous_28_days": float(prev.median()), "status": "potentially incomplete"}

    def dev_repeat_rate_pct(self) -> float:
        dev = self.daily.loc["2017-04-01":"2018-12-31"]
        return float(100 * (1 - dev["admissions_dedup_mrd_doa"].sum() / dev["admissions"].sum()))

    def series(self, key: str, start=None, end=None) -> dict:
        col, label = SERIES[key]
        s = self.daily.loc[start or C.DATA_START: end or C.DATA_END]
        pts = [{"date": d.strftime("%Y-%m-%d"), "value": int(v),
                "low_count_flag": bool(s.at[d, "low_count_flag"]),
                "potentially_incomplete": d.strftime("%Y-%m-%d") == C.POTENTIALLY_INCOMPLETE_DAY}
               for d, v in s[col].items()]
        return {"series": key, "column": col, "label": label, "provenance": "observed",
                "is_scored_target": key == "raw", "points": pts}

    def seasonal_naive(self, cutoff) -> list:
        """Exact harness baseline: same weekday in the last observed week (cut-off-7 .. cut-off-1)."""
        c = pd.Timestamp(cutoff)
        start = c - pd.Timedelta(days=7)
        out = []
        for h in range(14):
            d = c + pd.Timedelta(days=h)
            src = start + pd.Timedelta(days=(d - start).days % 7)
            out.append(float(self.daily.at[src, "admissions"]))
        return out

    @cached_property
    def holdout(self) -> dict:
        """Saved Step 5.3/5.4 holdout tables, read as-is (never recomputed, never re-scored)."""
        t = self._root / "outputs" / "holdout" / "tables"
        metric_cols = ["MAE", "RMSE", "WAPE", "bias", "MAE_h1_7", "MAE_h8_14", "MAE_ratio_vs_B0", "coverage80",
                       "coverage80_h1_7", "coverage80_h8_14", "width80", "n_cutoffs", "n_points"]
        read = lambda n: pd.read_csv(t / n)  # noqa: E731
        metrics = {}
        for key, name in (("primary", "metrics_primary_grid.csv"), ("daily", "metrics_daily_sensitivity.csv"),
                          ("daily_excl_last_day", "metrics_daily_excluding_2019-03-31.csv")):
            m = pd.read_csv(t / name, index_col=0)
            metrics[key] = {"model": m.loc["M4 (forecast.py)", metric_cols].astype(float).to_dict(),
                            "baseline": m.loc["Seasonal naive (harness baseline)", metric_cols].astype(float).to_dict()}
        summary = json.loads((t / "summary.json").read_text())
        return {
            "metrics": metrics,
            "ratio_ci": {"primary": summary["ci_primary"], "daily": summary["ci_daily"]},
            "forecast_py_sha256": summary["forecast_py_sha256"],
            "by_horizon": read("by_horizon_primary.csv").to_dict("records"),
            "by_month": read("by_cutoff_month_daily.csv").to_dict("records"),
            "per_cutoff": read("per_cutoff_primary.csv").to_dict("records"),
            "stages": read("context_earlier_stages.csv").to_dict("records"),
            "diagnosis_by_month": read("diagnosis_by_month.csv").rename(columns={"target month": "month"}).to_dict("records"),
        }

    @cached_property
    def interval_evidence(self) -> dict:
        """How the 80% interval was calibrated and how it performed - read from saved Step 4/5 outputs only.

        Construction: central 85% of past log residuals (7.5th-92.5th percentiles), chosen on 2017-18 tuning
        cut-offs as the narrowest setting reaching >= 80% empirical coverage. Target coverage: 80% (HARNESS.md).
        """
        out = self._root / "outputs"
        cal = pd.read_csv(out / "backtest" / "tables" / "interval_calibration_tuning.csv")
        q_lo, q_hi, pool = (float(self.interval_m4.q_lo), float(self.interval_m4.q_hi), int(self.interval_m4.pool))
        row = cal[(cal.model == "M4_glm_Nall_K2_T0_L0") & np.isclose(cal.q_lo, q_lo) & (cal.pool == pool)].iloc[0]
        nominal80 = cal[(cal.model == "M4_glm_Nall_K2_T0_L0") & np.isclose(cal.q_lo, 0.10)]
        p = pd.read_csv(out / "holdout" / "tables" / "forecasts_primary_grid.csv")
        above, below = float((p.y > p.hi).mean()), float((p.y < p.lo).mean())
        return {
            "target_coverage": 0.80,
            "residual_quantiles": {"lower": q_lo, "upper": q_hi, "central_range": round(q_hi - q_lo, 3)},
            "calibration_origins": pool,
            "construction": "Empirical: the 7.5th and 92.5th percentiles (central 85% range) of the model's own log errors "
                            "on its 90 most recent completed 14-day forecasts, per horizon bucket.",
            "selection": "Chosen on 2017-18 development (tuning) cut-offs as the narrowest setting reaching at least 80% "
                         "empirical coverage. A calibration target, not a guarantee for future periods.",
            "nominal_80_tuning_coverage_range": [float(nominal80.tune_coverage.min()), float(nominal80.tune_coverage.max())],
            "coverage_history": [
                {"stage": "tuning", "label": "Tuning (2017-07 to 2018-06, selection data)",
                 "coverage": float(row.tune_coverage), "points": int(row.tune_points)},
                {"stage": "validation", "label": "Validation (2018-07 to 2018-12)",
                 "coverage": float(self.validation_m4["coverage80"]), "points": None},
                {"stage": "holdout", "label": "Jan-Mar 2019 (2019-01 to 2019-03), previously viewed in the M4 holdout context",
                 "coverage": float(self.holdout_m4["coverage80"]), "points": int(len(p))},
            ],
            "holdout_misses": {"above_upper": above, "below_lower": below, "inside": 1 - above - below,
                               "points": int(len(p)), "source": "outputs/holdout/tables/forecasts_primary_grid.csv"},
        }

    def weekday_means(self) -> dict:
        a = self.daily["admissions"]
        m = a.groupby(a.index.dayofweek).mean()
        names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        return {names[i]: float(v) for i, v in m.items()}


def round_floats(obj, nd=3):
    if isinstance(obj, float):
        return None if not np.isfinite(obj) else round(obj, nd)
    if isinstance(obj, dict):
        return {k: round_floats(v, nd) for k, v in obj.items()}
    if isinstance(obj, list):
        return [round_floats(v, nd) for v in obj]
    return obj
