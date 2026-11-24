"""Leakage and data-contract tests for the Stage A feature builder (real project data only).

Run from the project root:
    PYTHONDONTWRITEBYTECODE=1 python -m pytest experiments/multivar -q -p no:cacheprovider
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from features import ENV_CUT, ENV_OR, ENV_VARS, FEATURE_SETS, HIST, build_rows  # noqa: E402


@pytest.fixture(scope="module")
def data():
    d = pd.read_csv(ROOT / "outputs" / "clean_daily.csv", parse_dates=["date"]).set_index("date")
    return d["admissions"], d[list(ENV_VARS)], pd.read_csv(ROOT / "festival_calendar_punjab.csv")


ORIGINS = [pd.Timestamp("2018-03-10"), pd.Timestamp("2018-10-02")]


def test_target_is_raw_admissions(data):
    y, _, _ = data
    daily = pd.read_csv(ROOT / "outputs" / "daily_admissions.csv", parse_dates=["date"]).set_index("date")["admissions"]
    assert y.equals(daily) and int(y.sum()) == 15757


def test_history_and_cutoff_env_ignore_the_future(data):
    y, env, fest = data
    base = build_rows(y, env, fest, ORIGINS)
    for o in ORIGINS:
        y2, env2 = y.copy(), env.copy()
        y2.loc[o:] += 1000                                   # perturb everything on/after the cut-off
        env2.loc[o:] += 1000
        alt = build_rows(y2, env2, fest, [o])
        a, b = base[base.origin == o].reset_index(drop=True), alt.reset_index(drop=True)
        cols = [c for c in HIST + ENV_CUT + ["h", "dow", "holiday", "hol_eve", "hol_after", "doy_s1"]]
        pd.testing.assert_frame_equal(a[cols], b[cols])     # deployable features unchanged
        assert not np.allclose(a[ENV_OR].to_numpy(), b[ENV_OR].to_numpy())   # oracle reads the future by design


def test_oracle_is_target_date_value_and_cut_is_7_days_before(data):
    y, env, fest = data
    r = build_rows(y, env, fest, [ORIGINS[1]])
    row = r[r.h == 5].iloc[0]
    assert row.date == ORIGINS[1] + pd.Timedelta(days=4)
    assert row.or_pm25 == pytest.approx(env.at[row.date, "pm25"])
    w = env.loc[ORIGINS[1] - pd.Timedelta(days=7): ORIGINS[1] - pd.Timedelta(days=1)]
    assert row.cut_pm25 == pytest.approx(w.pm25.mean()) and row.cut_precip_mm == pytest.approx(w.precip_mm.sum())
    assert row.lvl28 == pytest.approx(y.loc[ORIGINS[1] - pd.Timedelta(days=28): ORIGINS[1] - pd.Timedelta(days=1)].mean())


def test_calendar_flags_match_official_calendar(data):
    y, env, fest = data
    r = build_rows(y, env, fest, [pd.Timestamp("2018-11-01")]).set_index("date")
    assert r.at[pd.Timestamp("2018-11-07"), "holiday"] == 1               # Diwali 2018
    assert r.at[pd.Timestamp("2018-11-06"), "hol_eve"] == 1 and r.at[pd.Timestamp("2018-11-08"), "hol_after"] == 1


def test_feature_sets_separate_deployable_from_oracle():
    assert not set(FEATURE_SETS["H"]) & set(ENV_CUT + ENV_OR)
    assert set(ENV_CUT) <= set(FEATURE_SETS["HE"]) and not set(ENV_OR) & set(FEATURE_SETS["HE"])
    assert set(ENV_OR) <= set(FEATURE_SETS["HX"])
