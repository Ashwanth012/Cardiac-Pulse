"""Tests for Build step 4: baseline exactness, leakage control, intervals, scoring rules,
model mechanics, dependency restrictions, saved outputs and integrity of earlier outputs.

Run from the project root:
    PYTHONDONTWRITEBYTECODE=1 python -m pytest validation -v -p no:cacheprovider
"""
import ast
import copy
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from src import backtest as B  # noqa: E402
from src import forecasting as F  # noqa: E402
from test_eda import STEP2_SHA256  # noqa: E402
from test_loaders import ORIGINAL_SHA256, sha256  # noqa: E402

BT = os.path.join(ROOT, "outputs", "backtest")


@pytest.fixture(scope="module")
def store():
    return B.DataStore(ROOT)


def _toy_ctx(cutoff="2018-03-13"):
    idx = pd.date_range("2017-04-01", pd.Timestamp(cutoff) - pd.Timedelta(days=1))
    y = pd.Series(np.arange(len(idx), dtype=float), index=idx)
    cal = pd.DataFrame(False, index=pd.date_range("2017-01-01", "2019-12-31"),
                       columns=["holiday", "holiday_eve", "holiday_after"])
    return F.Context(y=y, env=pd.DataFrame(index=idx[:0]), cal=cal, cutoff=pd.Timestamp(cutoff))


# ===================================================================== baseline
class TestBaseline:
    def test_same_weekday_last_observed_week(self):
        ctx = _toy_ctx("2018-03-13")                     # a Tuesday
        f = F.seasonal_naive(ctx)
        y = ctx.y
        for h, d in enumerate(ctx.dates):
            src = y[(y.index >= ctx.cutoff - pd.Timedelta(days=7)) & (y.index.dayofweek == d.dayofweek)]
            assert len(src) == 1 and f[h] == src.iloc[0]
        assert f[0] == y[ctx.cutoff - pd.Timedelta(days=7)] and f[6] == y.iloc[-1]
        assert (f[:7] == f[7:]).all()

    @pytest.mark.parametrize("cut", ["2017-07-01", "2018-01-10", "2018-12-18"])
    def test_real_data_baseline(self, store, cut):
        ctx = store.context(cut)
        f = F.seasonal_naive(ctx)
        for h, d in enumerate(ctx.dates):
            ref = store.counts[d - pd.Timedelta(days=7 * ((h // 7) + 1))]
            assert f[h] == ref


# ================================================================ leakage control
@pytest.fixture(scope="module")
def perturbed(store):
    cut = pd.Timestamp("2018-10-02")
    s2 = copy.copy(store)
    s2.counts = store.counts.copy()
    s2.counts[s2.counts.index >= cut] *= 10
    s2.weather = store.weather.copy()
    s2.weather.loc[s2.weather.index >= cut] = 999.0
    pr = store.pollution_raw.copy()
    num = [c for c in pr.columns if c not in ("DATE", "date", "date_method", "PROMINENT POLLUTENT")]
    pr.loc[pr["date"] >= cut, num] = "450"
    s2.pollution_raw = pr
    s2._env, s2._full_env = {}, None
    return cut, s2



class TestLeakage:
    def test_context_is_truncated(self, store):
        for cut in B.make_cutoffs()[::10]:
            ctx = store.context(cut, with_env=True)
            assert ctx.y.index.max() == cut - pd.Timedelta(days=1)
            assert ctx.env.index.max() < cut

    def test_pollution_cleaned_on_truncated_file(self, store):
        # NO2 on 2017-09-21/22 is invalid and interpolated in the full file using 2017-09-23.
        # At cut-off 2017-09-23 that neighbour is not yet known, so the value must stay missing.
        early = store.env_at(pd.Timestamp("2017-09-23"))
        assert np.isnan(early.loc["2017-09-22", "no2"])
        assert np.isfinite(store.full_env().loc["2017-09-22", "no2"])

    MODELS = {
        "seasonal_naive": F.seasonal_naive,
        "rolling": lambda c: F.rolling_mean(c, 28),
        "weekday": lambda c: F.weekday_level(c, 56, 364),
        "weekday_cal": lambda c: F.weekday_level(c, 56, 364, calendar=True),
        "year_ago": lambda c: F.year_ago(c, 56),
        "glm": lambda c: F.glm_calendar(c, None, 2, False, 0),
        "glm_trend_level": lambda c: F.glm_calendar(c, 364, 2, True, 28),
        "holt_winters": lambda c: F.holt_winters(c, 182),
        "env_lag14": lambda c: F.glm_calendar(c, None, 2, False, 0, env_cols=("co", "pm25", "temp_mean_c"), env_lag=14),
    }

    @pytest.mark.parametrize("name", list(MODELS))
    def test_future_data_cannot_change_forecast(self, store, perturbed, name):
        cut, s2 = perturbed
        fn = self.MODELS[name]
        a = fn(store.context(cut, with_env=True))
        b = fn(s2.context(cut, with_env=True))
        assert np.allclose(a, b), name

    def test_oracle_does_change(self, store, perturbed):
        """Sanity check that the perturbation test can detect leakage at all."""
        cut, s2 = perturbed
        a = F.glm_calendar(store.context(cut), None, 2, False, 0, env_cols=("pm25",), env_lag=0, oracle_env=store.full_env())
        b = F.glm_calendar(s2.context(cut), None, 2, False, 0, env_cols=("pm25",), env_lag=0, oracle_env=s2.full_env())
        assert not np.allclose(a, b)

    def test_short_env_lag_refused(self, store):
        with pytest.raises(AssertionError):
            F.glm_calendar(store.context("2018-10-02", with_env=True), env_cols=("pm25",), env_lag=7)

    def test_cutoffs_inside_dev(self):
        c = B.make_cutoffs()
        assert c.min() == pd.Timestamp("2017-07-01") and c.max() == pd.Timestamp("2018-12-15")
        assert (c + pd.Timedelta(days=13) <= B.DEV_END).all()
        assert set(np.diff(c).astype("timedelta64[D]").astype(int)) == {4}
        assert set(c.dayofweek) == set(range(7))


# =================================================================== intervals
def _fake_fc(n_cut=30, seed=0):
    rng = np.random.default_rng(seed)
    cuts = pd.date_range("2018-01-01", periods=n_cut, freq="4D")
    rows = []
    for c in cuts:
        for h in range(1, 15):
            m = 20.0
            rows.append(("m", c, h, c + pd.Timedelta(days=h - 1), float(rng.poisson(m)), m, 0.0))
    return pd.DataFrame(rows, columns=["model", "cutoff", "h", "date", "y", "mean", "seconds"])


class TestIntervals:
    def test_only_completed_past_forecasts_used(self):
        fc = _fake_fc()
        base = B.add_intervals(fc, min_pool=5)
        late = fc.copy()
        last = late.cutoff.max()
        late.loc[late.cutoff == last, "y"] = 500.0           # change the most recent window's truth
        alt = B.add_intervals(late, min_pool=5)
        keep = base.cutoff <= last                           # no cut-off can see it before last+14
        assert np.allclose(base.loc[keep, "lo"].fillna(-1), alt.loc[keep, "lo"].fillna(-1))
        assert np.allclose(base.loc[keep, "hi"].fillna(-1), alt.loc[keep, "hi"].fillna(-1))

    def test_pool_counts_respect_14_day_rule(self):
        iv = B.add_intervals(_fake_fc(), min_pool=1)
        for c, g in iv.groupby("cutoff"):
            expect = sum(1 for o in iv.cutoff.unique() if o <= c - pd.Timedelta(days=14))
            assert g["pool"].iloc[0] == min(expect, 45)

    def test_interval_contains_mean_and_nonnegative(self):
        iv = B.add_intervals(_fake_fc(), min_pool=5).dropna(subset=["lo"])
        assert (iv["lo"] <= iv["mean"]).all() and (iv["mean"] <= iv["hi"]).all() and (iv["lo"] >= 0).all()

    def test_calibrated_on_stationary_data(self):
        iv = B.add_intervals(_fake_fc(120, seed=3), 0.10, 0.90, pool=45, min_pool=20).dropna(subset=["lo"])
        cov = ((iv.y >= iv.lo) & (iv.y <= iv.hi)).mean()
        assert 0.75 <= cov <= 0.92


# ============================================================ metrics & scoring
class TestScoring:
    def test_harness_error_points(self):
        assert B.error_points(1.0) == 12.5 and B.error_points(0.5) == 18.75
        assert B.error_points(2.0) == 0 and B.error_points(0.0) == 25

    def test_harness_coverage_points(self):
        assert B.coverage_points(0.80) == 15 and B.coverage_points(0.95) == 15
        assert B.coverage_points(0.60) == pytest.approx(7.5) and B.coverage_points(0.40) == 0

    def test_width_penalty_approximation(self):
        assert B.width_penalty_factor(30, 20) == 1.0
        assert B.width_penalty_factor(80, 20) == pytest.approx(0.5)

    def test_metrics_values(self):
        df = pd.DataFrame({"cutoff": [1, 1, 1, 1], "h": [1, 2, 8, 9], "y": [10, 10, 20, 20],
                           "mean": [12, 8, 20, 25], "lo": [9, 9, 19, 19], "hi": [11, 11, 21, 21]})
        m = B.metrics(df)
        assert m["MAE"] == pytest.approx(9 / 4) and m["WAPE"] == pytest.approx(9 / 60)
        assert m["RMSE"] == pytest.approx(np.sqrt(33 / 4))
        assert m["MAE_h1_7"] == 2 and m["MAE_h8_14"] == 2.5 and m["coverage80"] == 1.0


# =============================================================== model mechanics
class TestModels:
    def test_poisson_irls_recovers_coefficients(self):
        rng = np.random.default_rng(0)
        X = np.column_stack([np.ones(3000), rng.normal(size=3000), rng.integers(0, 2, 3000)])
        b = np.array([2.5, 0.3, -0.4])
        y = rng.poisson(np.exp(X @ b))
        assert np.allclose(F.poisson_irls(X, y, ridge=1e-6), b, atol=0.03)

    def test_weekday_level_flat_series(self):
        ctx = _toy_ctx()
        ctx = F.Context(y=pd.Series(20.0, index=ctx.y.index), env=ctx.env, cal=ctx.cal, cutoff=ctx.cutoff)
        assert np.allclose(F.weekday_level(ctx, 28, 91), 20.0)
        assert np.allclose(F.glm_calendar(ctx, 364, 0, False, 0), 20.0, atol=0.01)

    def test_all_models_return_14_finite(self, store):
        ctx = store.context("2018-02-01", with_env=True)
        for name, fn in TestLeakage.MODELS.items():
            out = fn(ctx)
            assert out.shape == (14,) and np.isfinite(out).all() and (out >= 0).all(), name

    def test_forecasting_module_uses_only_pandas_numpy(self):
        tree = ast.parse(open(os.path.join(ROOT, "src", "forecasting.py"), encoding="utf-8").read())
        mods = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                mods |= {a.name.split(".")[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom):
                mods.add((node.module or "").split(".")[0])
        assert mods <= {"numpy", "pandas", "__future__", "dataclasses"}, mods


# ================================================================ saved outputs
class TestOutputs:
    def test_report_and_tables_exist(self):
        assert os.path.isfile(os.path.join(BT, "backtest_report.md"))
        for t in ["validation_family_winners.csv", "tuning_all_configs.csv", "environment_vs_same_glm.csv",
                  "interval_calibration_tuning.csv", "stability_by_season.csv", "simulated_harness_score_validation.csv"]:
            assert os.path.isfile(os.path.join(BT, "tables", t)), t
        assert len([f for f in os.listdir(os.path.join(BT, "figures")) if f.endswith(".png")]) == 7

    def test_no_holdout_in_archive(self):
        fc = pd.read_csv(os.path.join(BT, "forecasts_selected_models.csv.gz"), parse_dates=["cutoff", "date"])
        assert fc["date"].max() <= pd.Timestamp("2018-12-31")
        assert fc["cutoff"].max() <= pd.Timestamp("2018-12-15")

    def test_validation_table_beats_baseline(self):
        v = pd.read_csv(os.path.join(BT, "tables", "validation_family_winners.csv"), index_col=0)
        assert v.loc["B0", "ratio_vs_B0"] == pytest.approx(1.0)
        assert (v.drop(index=["B0"])["MAE"] > 0).all()


# ==================================================================== integrity
class TestIntegrity:
    def test_original_files_unchanged(self):
        for name, h in ORIGINAL_SHA256.items():
            assert sha256(os.path.join(ROOT, name)) == h, name

    def test_step2_outputs_unchanged(self):
        for name, h in STEP2_SHA256.items():
            assert sha256(os.path.join(ROOT, "outputs", name)) == h, name

    def test_step3_outputs_unchanged(self):
        ref = json.load(open(os.path.join(HERE, "fixtures", "step3_outputs_sha256.json")))
        assert len(ref) == 37
        for path, h in ref.items():
            assert hashlib.sha256(open(os.path.join(ROOT, path), "rb").read()).hexdigest() == h, path
