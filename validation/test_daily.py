"""Tests for Build step 2: daily counts, coverage, duplicate analysis, cleaning, integrity.

Run from the project root:
    PYTHONDONTWRITEBYTECODE=1 python -m pytest validation -v -p no:cacheprovider
"""
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src import daily as D  # noqa: E402
from src import loaders as L  # noqa: E402
from test_loaders import ORIGINAL_SHA256, sha256  # noqa: E402

OUT = os.path.join(ROOT, "outputs")
ALLOWED_ACTIONS = {"corrected", "set_missing", "imputed", "flagged"}


@pytest.fixture(scope="module")
def built():
    return D.build_all(ROOT)


@pytest.fixture(scope="module")
def daily(built):
    return built["daily"]


# ================================================================ daily coverage
class TestCoverage:
    def test_every_date_exactly_once(self, daily):
        expected = pd.date_range("2017-04-01", "2019-03-31", freq="D")
        assert len(daily) == 730
        assert daily.index.is_unique and daily.index.is_monotonic_increasing
        assert daily.index.equals(expected)

    def test_no_missing_target(self, daily):
        assert daily["admissions"].notna().all()
        assert daily["admissions"].dtype.kind == "i"


# ================================================================== daily counts
class TestDailyCounts:
    def test_raw_counts_not_deduplicated(self, daily):
        assert daily["admissions"].sum() == 15757  # every row in the file
        assert daily["admissions"].sum() > daily["admissions_dedup_exact"].sum()

    def test_matches_independent_groupby(self, daily):
        raw = L.read_raw_csv(os.path.join(ROOT, L.ADMISSION_FILE))
        my = pd.to_datetime(raw["month year"], format="%b-%y").dt.to_period("M")
        mdy = pd.to_datetime(raw["D.O.A"], format="%m/%d/%Y", errors="coerce")
        dmy = pd.to_datetime(raw["D.O.A"], format="%d/%m/%Y", errors="coerce")
        doa = mdy.where(mdy.dt.to_period("M") == my, dmy)
        ref = doa.value_counts().sort_index()
        assert (ref.reindex(daily.index, fill_value=0).to_numpy() == daily["admissions"].to_numpy()).all()

    def test_emergency_plus_opd_equals_total(self, daily):
        assert (daily["admissions_emergency"] + daily["admissions_opd"] == daily["admissions"]).all()
        assert "admissions_type_other" not in daily.columns

    def test_zero_days_preserved_and_window_enforced(self):
        adm = pd.DataFrame({"doa": pd.to_datetime(["2018-01-01", "2018-01-01", "2018-01-03", "2017-12-31", "2018-01-04"]),
                            "TYPE OF ADMISSION-EMERGENCY/OPD": ["E", "O", "E", "E", "E"]})
        out = D.daily_admission_counts(adm, start="2018-01-01", end="2018-01-03")
        assert list(out.index) == list(pd.date_range("2018-01-01", "2018-01-03"))
        assert list(out["admissions"]) == [2, 0, 1]          # genuine zero kept, not dropped
        assert list(out["zero_count_day"]) == [False, True, False]
        assert list(out["admissions_opd"]) == [1, 0, 0]

    def test_real_data_has_no_zero_days(self, daily):
        assert daily["zero_count_day"].sum() == 0 and daily["admissions"].min() == 2

    def test_dedup_columns_consistent(self, daily, built):
        s = built["dup_summary"]
        assert daily["admissions_dedup_exact"].sum() == 15757 - s["exact_extra_copies"]
        assert daily["admissions_dedup_mrd_doa"].sum() == 15757 - s["mrd_doa_extra_rows"]
        assert (daily["admissions_dedup_mrd_doa"] >= 1).all()


# ============================================================ duplicate analysis
def _toy_admissions():
    rows = [  # SNO, MRD, DOA, DOD, month year, DUR, X
        ("1", "100", "4/1/2018", "4/3/2018", "Apr-18", "3", "a"),     # original
        ("2", "100", "4/1/2018", "4/3/2018", "Apr-18", "3", "a"),     # exact copy (same strings)
        ("3", "100", "01/04/2018", "03/04/2018", "Apr-18", "3", "a"),  # exact copy, other date order
        ("4", "100", "4/1/2018", "4/3/2018", "Apr-18", "3", "b"),     # near duplicate (X differs)
        ("5", "200", "4/1/2018", "4/10/2018", "Apr-18", "10", "a"),   # unique
        ("6", "200", "4/5/2018", "4/8/2018", "Apr-18", "4", "a"),     # overlapping stay (5 Apr < 10 Apr)
        ("7", "300", "5/2/2018", "5/4/2018", "May-18", "3", "a"),     # unique
    ]
    raw = pd.DataFrame(rows, columns=["SNO", "MRD No.", "D.O.A", "D.O.D", "month year", "DURATION OF STAY", "X"])
    adm, _ = L.load_admissions(raw=raw)
    return adm


class TestDuplicates:
    def test_toy_classification(self):
        f, s, by_month, diff = D.duplicate_analysis(_toy_admissions())
        assert s["exact_raw_string_extra_copies"] == 1
        assert s["exact_extra_copies"] == 2           # date-order variant is caught
        assert s["mrd_doa_extra_rows"] == 3
        assert s["near_extra_rows"] == 1
        assert s["overlapping_stays"] == 1
        assert list(f["near_extra"]) == [False, False, False, True, False, False, False]
        assert f.loc[3, "differing_columns"].startswith("X")
        assert "D.O.A (format only)" in f.loc[0, "differing_columns"]
        assert by_month.loc["2018-04", "rows"] == 6

    def test_toy_counts_not_removed(self):
        adm = _toy_admissions()
        flags = D.duplicate_analysis(adm)[0]
        out = D.daily_admission_counts(adm, flags, start="2018-04-01", end="2018-04-05")
        assert out.loc["2018-04-01", "admissions"] == 5          # all rows counted
        assert out.loc["2018-04-01", "admissions_dedup_exact"] == 3
        assert out.loc["2018-04-01", "admissions_dedup_mrd_doa"] == 2

    def test_real_summary(self, built):
        s = built["dup_summary"]
        assert s["exact_raw_string_extra_copies"] == 882   # matches the inspection report
        assert s["exact_extra_copies"] == 885
        assert s["mrd_doa_extra_rows"] == 1358
        assert s["near_extra_rows"] == s["mrd_doa_extra_rows"] - s["exact_extra_copies"]
        assert s["rows"] == 15757

    def test_exact_extras_are_subset_of_mrd_doa(self, built):
        f = built["dup_flags"]
        assert not (f["exact_extra"] & ~f["mrd_doa_extra"]).any()

    def test_by_month_totals(self, built):
        bm = built["dup_by_month"]
        assert bm["rows"].sum() == 15757 and len(bm) == 24
        assert bm["exact_extra"].sum() == built["dup_summary"]["exact_extra_copies"]


# ============================================================ predictor hygiene
class TestPredictorHygiene:
    FORBIDDEN = ["dod", "d.o.d", "duration", "outcome", "icu", "age", "gender", "rural", "hb", "tlc",
                 "platelet", "glucose", "urea", "creatinine", "bnp", "ef", "smoking", "alcohol"]

    def test_no_patient_level_columns(self, daily):
        for c in daily.columns:
            tokens = c.lower().replace("_", " ").split()
            assert not any(t in self.FORBIDDEN for t in tokens), c

    def test_every_column_has_a_role(self, daily):
        assert set(daily.columns) <= set(D.COLUMN_ROLES)

    def test_predictor_candidates_only_from_allowed_sources(self):
        for c, (src, role, _) in D.COLUMN_ROLES.items():
            if role == "predictor_candidate":
                assert src in {"calendar", "weather", "pollution", "festival"}, c
        assert D.COLUMN_ROLES["admissions"][1] == "TARGET"
        for c in ["admissions_emergency", "admissions_opd", "admissions_dedup_exact", "admissions_dedup_mrd_doa"]:
            assert D.COLUMN_ROLES[c][1] in {"analysis_only", "sensitivity_only"}


# =========================================================== cleaning decisions
class TestCleaning:
    def test_log_actions_valid(self, built):
        log = built["cleaning_log"]
        assert set(log["action"]) <= ALLOWED_ACTIONS
        assert set(log.loc[log.source == "admissions", "action"]) == {"flagged"}  # target untouched

    def test_corrected_value(self, built):
        log = built["cleaning_log"]
        row = log[(log.action == "corrected")]
        assert len(row) == 1 and row.iloc[0]["raw_value"] == "`150" and row.iloc[0]["new_value"] == "150"

    def test_set_missing_then_imputed(self, built, daily):
        log = built["cleaning_log"]
        sm = log[(log.action == "set_missing") & (log.column == "NO2 AVG")]
        assert set(sm["date"]) == {"2017-09-21", "2017-09-22", "2019-03-05"}
        assert daily.loc["2017-09-21", "no2_imputed"] and daily.loc["2017-09-21", "no2"] < 400
        assert daily.loc["2017-11-09", "pm25_imputed"] and daily.loc["2017-11-09", "pm25"] != 30

    def test_flagged_values_unchanged(self, built, daily):
        raw = L.load_pollution(ROOT).set_index("date")
        log = built["cleaning_log"]
        flagged = log[(log.action == "flagged") & (log.column == "PM10 AVG")]
        assert len(flagged) > 0
        for d in flagged["date"]:
            assert daily.loc[d, "pm10"] == float(raw.loc[d, "PM10 AVG"])

    def test_imputed_flags_match_log(self, built, daily):
        log = built["cleaning_log"]
        n_log = (log.action == "imputed").sum()
        n_flag = sum(int(daily[c].sum()) for c in daily.columns if c.endswith("_imputed"))
        assert n_log == n_flag == 10

    def test_main_environment_complete(self, daily):
        for c in ["temp_min_c", "temp_max_c", "rel_humidity_mean_pct", "pm25", "pm10", "aqi", "no2"]:
            assert daily[c].notna().all(), c

    def test_festival_one_row_per_day(self, daily):
        assert daily.loc["2019-01-13", "holiday_count"] == 2
        assert "Lohri" in daily.loc["2019-01-13", "holiday_names"]
        assert daily.loc["2018-11-07", "holiday_major_festival"]  # Diwali 2018

    def test_extended_environment_window(self, built):
        e = built["env_extended"]
        assert e.index.min() == pd.Timestamp("2017-03-01") and e.index.max() == pd.Timestamp("2019-03-31")
        assert e.index.is_unique and e["temp_min_c"].notna().all()


# =========================================================== saved outputs
class TestSavedOutputs:
    FILES = ["clean_daily.csv", "daily_admissions.csv", "clean_daily_columns.csv", "cleaning_log.csv",
             "duplicate_rows.csv", "duplicates_by_month.csv", "environment_daily_extended.csv",
             "data_quality_report.md"]

    def test_files_exist(self):
        for f in self.FILES:
            assert os.path.isfile(os.path.join(OUT, f)), f

    def test_saved_daily_matches_fresh_build(self, daily):
        saved = pd.read_csv(os.path.join(OUT, "clean_daily.csv"), index_col="date", parse_dates=["date"])
        assert saved.index.equals(daily.index)
        assert (saved["admissions"] == daily["admissions"]).all()
        for c in ["pm25", "temp_min_c", "aqi"]:
            assert np.allclose(saved[c], daily[c])

    def test_saved_duplicates_match(self, built):
        dup = pd.read_csv(os.path.join(OUT, "duplicate_rows.csv"))
        s = built["dup_summary"]
        assert dup["exact_extra"].sum() == s["exact_extra_copies"]
        assert dup["near_extra"].sum() == s["near_extra_rows"]


# ============================================================ source integrity
class TestSourceIntegrity:
    def test_hashes_unchanged_after_build(self, built):
        for name, expected in ORIGINAL_SHA256.items():
            assert sha256(os.path.join(ROOT, name)) == expected, name

    def test_outputs_do_not_shadow_sources(self):
        assert not set(os.listdir(OUT)) & set(ORIGINAL_SHA256)

    def test_git_sees_no_change_to_originals(self):
        out = subprocess.run(["git", "status", "--porcelain", "--", *ORIGINAL_SHA256],
                             cwd=ROOT, capture_output=True, text=True)
        if out.returncode != 0:
            pytest.skip("git not available")
        assert out.stdout.strip() == ""
