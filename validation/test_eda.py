"""Tests for Build step 3 (EDA): lag construction, statistics helpers, holdout exclusion,
generated outputs, and integrity of the original data and Step 1/2 outputs.

Run from the project root:
    PYTHONDONTWRITEBYTECODE=1 python -m pytest validation -v -p no:cacheprovider
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src import eda as E  # noqa: E402
from test_loaders import ORIGINAL_SHA256, sha256  # noqa: E402

EDA = os.path.join(ROOT, "outputs", "eda")
STEP2_SHA256 = {  # recorded after Step 2 approval, before Step 3 ran
    "clean_daily.csv": "1d4638c58111cb38d6656de7058b6e9b20dbafce59727d0d911b0a2bb42f7fb9",
    "clean_daily_columns.csv": "07b2d27f2e707f0c80ffa791a71a133f778f730ae7cfdf8bdbbd00b85d4fe864",
    "cleaning_log.csv": "ec6fb97cef267370c2fbeda0e0d52c10b31960a1c3264deffcb882adc91fbff0",
    "daily_admissions.csv": "1f0ad40f2e6267234343a68ef6f0b3c965c04e2f7fb740b6d9221c1af93b20d1",
    "data_quality_report.md": "a20cf7d29f3576f549a860f95ad80b57efb6fb8a0246df8b3003542c13302e13",
    "duplicate_rows.csv": "4287b09820eb24ca74da68cb7c39293737932cb8acf2f47c8e331f8d6ce5fc5e",
    "duplicates_by_month.csv": "54cdc4e05f4db41b6ac9ad6a926cc1a80a05f400800f1ab36dc9962216c50534",
    "environment_daily_extended.csv": "c8742357a4afbf8166b449556086cf498e1477372167dba951d8238274d3965c",
}


@pytest.fixture(scope="module")
def inputs():
    return E.load_inputs(os.path.join(ROOT, "outputs"))


@pytest.fixture(scope="module")
def frame(inputs):
    return E.model_frame(*inputs)


# ================================================================ lag features
class TestLags:
    def test_lag_uses_past_values(self):
        idx = pd.date_range("2018-01-01", periods=10)
        s = pd.Series(np.arange(10.0), index=idx)
        lag3 = E.lagged(s, 3)
        assert lag3["2018-01-04"] == s["2018-01-01"]
        assert lag3.iloc[:3].isna().all()

    def test_moving_average_includes_same_day(self):
        idx = pd.date_range("2018-01-01", periods=10)
        s = pd.Series(np.arange(10.0), index=idx)
        ma = E.moving_avg(s, 2)
        assert ma["2018-01-05"] == pytest.approx((2 + 3 + 4) / 3)
        assert ma.iloc[:2].isna().all()

    def test_gap_in_index_rejected(self):
        idx = pd.to_datetime(["2018-01-01", "2018-01-02", "2018-01-04"])
        with pytest.raises(ValueError):
            E.lagged(pd.Series([1.0, 2.0, 3.0], index=idx), 1)

    def test_frame_lags_match_environment(self, inputs, frame):
        _, env = inputs
        for d in ["2017-04-15", "2018-06-30", "2018-12-31"]:
            t = pd.Timestamp(d)
            assert frame.at[t, "pm25__lag7"] == env.at[t - pd.Timedelta(days=7), "pm25"]
            assert frame.at[t, "temp_mean_c__lag14"] == env.at[t - pd.Timedelta(days=14), "temp_mean_c"]
            assert frame.at[t, "co__ma0_7"] == pytest.approx(env.loc[t - pd.Timedelta(days=7):t, "co"].mean())


# ============================================================ holdout exclusion
class TestHoldoutExcluded:
    def test_dev_frame_window(self, frame):
        assert frame.index.min() == pd.Timestamp("2017-04-01")
        assert frame.index.max() == pd.Timestamp("2018-12-31")
        assert len(frame) == 640 and (frame.index < E.HOLDOUT_START).all()

    def test_saved_inference_tables_have_no_holdout(self):
        lag = pd.read_csv(os.path.join(EDA, "tables", "lag_associations.csv"))
        assert lag["n"].max() <= 640
        for name in ["confounding_ladder.csv", "associations_by_target.csv", "influence_2017_smog_window.csv",
                     "season_adjustment_sensitivity.csv", "two_pollutant.csv"]:
            assert pd.read_csv(os.path.join(EDA, "tables", name))["n"].max() <= 640, name
        dm = pd.read_csv(os.path.join(EDA, "tables", "duplicates_monthly.csv"), index_col=0, parse_dates=True)
        assert dm.index.max() < E.HOLDOUT_START
        opd = pd.read_csv(os.path.join(EDA, "tables", "opd_share_monthly.csv"), index_col=0, parse_dates=True)
        assert opd.index.max() < E.HOLDOUT_START


# ============================================================ statistics helpers
class TestStats:
    @pytest.mark.parametrize("per_year", [2, 3, 4, 6, 8, 12])
    def test_season_design_full_rank(self, frame, per_year):
        """Regression test: an uncentred cr() spline is collinear with the intercept."""
        import patsy
        rhs = E.base_formula(season_df=E.spline_df(frame, per_year)).split("~")[1]
        X = np.asarray(patsy.dmatrix(rhs, frame))
        assert np.linalg.matrix_rank(X) == X.shape[1]
        assert np.linalg.cond(X) < 1e6

    def test_analysis_script_uses_centred_splines_only(self):
        import re
        src = open(os.path.join(ROOT, "analysis", "step3_eda.py"), encoding="utf-8").read()
        src += open(os.path.join(ROOT, "src", "eda.py"), encoding="utf-8").read()
        calls = re.findall(r"cr\(t, df=[^)]*\)", src)
        assert calls and all("constraints='center'" in c for c in calls), calls

    def test_bh_matches_statsmodels(self):
        from statsmodels.stats.multitest import multipletests
        rng = np.random.default_rng(0)
        p = np.r_[rng.uniform(0, 0.01, 5), rng.uniform(0, 1, 45)]
        assert np.allclose(E.bh_qvalues(p), multipletests(p, method="fdr_bh")[1])

    def test_association_recovers_known_effect(self):
        """Over 20 simulated datasets: unbiased log-RR and close-to-nominal 95% CI coverage."""
        n, beta, est, covered = 700, 0.10, [], 0
        idx = pd.date_range("2017-04-01", periods=n)
        wd = idx.dayofweek.to_numpy()
        for seed in range(20):
            rng = np.random.default_rng(seed)
            x = rng.normal(0, 1, n)
            mu = np.exp(np.log(20) + beta * x - 0.4 * (wd == 6))
            f = pd.DataFrame({"admissions": rng.poisson(mu), "x": x, "weekday": wd,
                              "holiday": (np.arange(n) % 30 == 0).astype(int), "t": np.arange(n, dtype=float)}, index=idx)
            r = E.association(f, "x", E.base_formula(season_df=5), unit=1.0)
            est.append(np.log(r["rr"]))
            covered += r["lo"] < np.exp(beta) < r["hi"]
            assert r["p"] < 0.001
        assert abs(np.mean(est) - beta) < 0.01
        assert covered >= 16  # >= 80% of nominal-95% intervals contain the truth

    def test_null_effect_not_significant_on_average(self):
        rng = np.random.default_rng(7)
        n = 640
        idx = pd.date_range("2017-04-01", periods=n)
        hits = 0
        for _ in range(20):
            f = pd.DataFrame({"admissions": rng.poisson(20, n), "x": rng.normal(size=n),
                              "weekday": idx.dayofweek, "holiday": (np.arange(n) % 30 == 0).astype(int), "t": np.arange(n, dtype=float)}, index=idx)
            hits += E.association(f, "x", E.base_formula(season_df=5), unit=1.0)["p"] < 0.05
        assert hits <= 4  # nominal 5% false-positive rate (with HAC slack)

    def test_vif_independent_near_one(self):
        rng = np.random.default_rng(1)
        f = pd.DataFrame(rng.normal(size=(500, 3)), columns=list("abc"))
        assert (E.vif(f, list("abc")) < 1.1).all()

    def test_acf_lag0_is_one(self):
        assert E.acf(pd.Series(np.random.default_rng(3).normal(size=200)), 5)[0] == pytest.approx(1.0)


# ================================================================ outputs
class TestOutputs:
    def test_report_tables_figures_exist(self):
        assert os.path.isfile(os.path.join(EDA, "eda_report.md"))
        assert len([f for f in os.listdir(os.path.join(EDA, "tables")) if f.endswith(".csv")]) == 23
        figs = [f for f in os.listdir(os.path.join(EDA, "figures")) if f.endswith(".png")]
        assert len(figs) == 13
        for f in figs:
            assert os.path.getsize(os.path.join(EDA, "figures", f)) > 10_000, f

    def test_lag_table_complete(self):
        lag = pd.read_csv(os.path.join(EDA, "tables", "lag_associations.csv"))
        assert len(lag) == len(E.EXPOSURES) * (len(E.LAGS) + len(E.MOVING)) == 126
        assert lag["q_bh"].between(0, 1).all() and (lag["q_bh"] >= lag["p"] - 1e-12).all()

    def test_report_mentions_key_caveats(self):
        txt = open(os.path.join(EDA, "eda_report.md"), encoding="utf-8").read()
        for phrase in ["not causal", "Jan-Mar 2019 is never used", "Influence check", "Limitations"]:
            assert phrase in txt, phrase


# ================================================================ integrity
class TestIntegrity:
    def test_original_files_unchanged(self):
        for name, h in ORIGINAL_SHA256.items():
            assert sha256(os.path.join(ROOT, name)) == h, name

    def test_step2_outputs_unchanged(self):
        for name, h in STEP2_SHA256.items():
            assert sha256(os.path.join(ROOT, "outputs", name)) == h, name
