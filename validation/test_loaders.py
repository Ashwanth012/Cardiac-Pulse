"""Self-tests for src/loaders.py (Build step 1).

Run from the project root:
    PYTHONDONTWRITEBYTECODE=1 python -m pytest validation/test_loaders.py -v -p no:cacheprovider
"""
import hashlib
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src import loaders as L  # noqa: E402

# SHA-256 of the files as published (recorded before any project code existed).
ORIGINAL_SHA256 = {
    "festival_calendar_punjab.csv": "897d7178d90ef104b501691a7e237a1ebfc6f5f25f7e0f62de97f8baba72184d",
    "HDHI Admission data.csv": "f6eb2c53090aad5df7b6c368f7d1b9aaeef4d77a630ccc43aad7a8cf2395f00a",
    "HDHI Mortality Data.csv": "825964a20d661441241970770ebb4fda8054bd1c4705730da5c6c5eeed3a435e",
    "HDHI Pollution Data.csv": "132f781b092037c3d3052f061e16c824cdb906ceecf40875563b94808e50114b",
    "table_headings.csv": "7110900f4330559f4e92f8387ab8f36b29ece73c29c0685cf44f86f63e5bcec9",
    "weather_ludhiana_daily.csv": "361ad06d00894cb36176b0b3acff216e1ab1c28db0443e8cb16d54579905311d",
    "HARNESS.md": "c5ff560d8d0eade8ca556f5138cb1fd36d8c0069082cd0052eb3f57696209e2a",
    "README.md": "5e4510db39a63f7b4d3afe1587536efc83558df1af517076570a88a0c9f1feb0",
    "team_stub.py": "34ee609ffb57374059856d70a4ed46c08c63a9ef72580c8b4b0f3fd0d55a1172",
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _parse(dates, month_year=None):
    s = pd.Series(dates, dtype="object")
    my = None if month_year is None else pd.Series(month_year, dtype="object")
    return L.parse_mixed_dates(s, my)


@pytest.fixture(scope="module")
def raw_adm():
    return L.read_raw_csv(os.path.join(ROOT, L.ADMISSION_FILE))


@pytest.fixture(scope="module")
def adm(raw_adm):
    return L.load_admissions(raw=raw_adm)


# ============================================================ synthetic: D.O.A
class TestMixedDatesSynthetic:
    def test_month_year_picks_month_first(self):
        r = _parse(["4/1/2017"], ["Apr-17"])
        assert r.date[0] == pd.Timestamp("2017-04-01") and r.method[0] == "month_year"

    def test_month_year_picks_day_first(self):
        r = _parse(["4/1/2017"], ["Jan-17"])
        assert r.date[0] == pd.Timestamp("2017-01-04") and r.method[0] == "month_year"

    def test_zero_padded_day_first(self):
        r = _parse(["03/02/2019"], ["Feb-19"])
        assert r.date[0] == pd.Timestamp("2019-02-03")

    def test_day_equals_month(self):
        r = _parse(["05/05/2018"], ["May-18"])
        assert r.date[0] == pd.Timestamp("2018-05-05") and r.method[0] == "month_year_equal"

    def test_part_above_12_with_month_year(self):
        r = _parse(["13/02/2018", "2/13/2018"], ["Feb-18", "Feb-18"])
        assert list(r.date) == [pd.Timestamp("2018-02-13")] * 2
        assert list(r.method) == ["month_year", "month_year"]

    def test_whitespace_tolerated(self):
        r = _parse(["  4/1/2017 "], [" apr-17 "])
        assert r.date[0] == pd.Timestamp("2017-04-01")

    def test_missing_values(self):
        r = _parse(["", "   ", None, np.nan], ["Apr-17"] * 4)
        assert r.date.isna().all() and set(r.method) == {"missing"}

    @pytest.mark.parametrize("bad", ["2-1217", "abc", "4/1/17", "2017-04-01", "4//2017", "EMPTY", "1/2/3/2017"])
    def test_malformed(self, bad):
        r = _parse([bad], ["Apr-17"])
        assert pd.isna(r.date[0]) and r.method[0] == "malformed"

    @pytest.mark.parametrize("bad", ["31/02/2018", "13/13/2018", "0/5/2018", "2/30/2018"])
    def test_impossible_calendar_dates(self, bad):
        r = _parse([bad], ["Feb-18"])
        assert pd.isna(r.date[0]) and r.method[0] == "invalid"

    def test_month_year_conflict_falls_back_and_is_flagged(self):
        # 13/2/2018 can only be 13 Feb, but month-year says March -> conflict flagged.
        r = _parse(["13/2/2018"], ["Mar-18"])
        assert r.date[0] == pd.Timestamp("2018-02-13")
        assert r.method[0] == "unambiguous" and bool(r.month_year_conflict[0])
        assert r.month_year_ok[0] == False  # noqa: E712

    def test_conflict_and_ambiguous_without_neighbours_is_unresolved(self):
        r = _parse(["3/4/2018"], ["Dec-18"])
        assert pd.isna(r.date[0]) and r.method[0] == "ambiguous"

    def test_invalid_month_year_label_uses_fallbacks(self):
        r = _parse(["13/2/2018"], ["garbage"])
        assert r.date[0] == pd.Timestamp("2018-02-13") and r.method[0] == "unambiguous"

    def test_no_month_year_unambiguous(self):
        r = _parse(["13/2/2018", "2/13/2018", "7/7/2018"])
        assert list(r.date) == [pd.Timestamp("2018-02-13")] * 2 + [pd.Timestamp("2018-07-07")]
        assert list(r.method) == ["unambiguous", "unambiguous", "equal"]

    def test_no_month_year_ambiguous_resolved_by_neighbours(self):
        # Neighbours are early April 2018 -> '3/4/2018' must be 3 April, not 4 March.
        dates = ["01/04/2018", "02/04/2018", "3/4/2018", "15/04/2018", "16/04/2018"]
        r = _parse(dates)
        assert r.date[2] == pd.Timestamp("2018-04-03") and r.method[2] == "neighbour"
        # and the opposite context -> 4 March
        r2 = _parse(["3/1/2018", "3/2/2018", "3/4/2018", "3/13/2018", "3/14/2018"])
        assert r2.date[2] == pd.Timestamp("2018-03-04")

    def test_no_month_year_no_evidence_is_ambiguous(self):
        r = _parse(["3/4/2018"])
        assert pd.isna(r.date[0]) and r.method[0] == "ambiguous"

    def test_neighbour_too_far_is_ambiguous(self):
        r = _parse(["13/12/2018", "3/4/2018", "14/12/2018"])
        assert r.method[1] == "ambiguous"

    def test_input_series_not_mutated_and_index_preserved(self):
        s = pd.Series(["4/1/2017", "", "2-1217"], index=[10, 20, 30], dtype="object")
        my = pd.Series(["Apr-17", "Apr-17", "Apr-17"], index=[10, 20, 30])
        before = s.copy()
        r = L.parse_mixed_dates(s, my)
        pd.testing.assert_series_equal(s, before)
        assert list(r.index) == [10, 20, 30]


# ============================================================ synthetic: D.O.D
class TestDischargeDatesSynthetic:
    def _p(self, dod, doa, dur):
        return L.parse_discharge_dates(
            pd.Series(dod, dtype="object"), pd.to_datetime(pd.Series(doa)), pd.Series(dur, dtype="object"))

    def test_duration_resolves_ambiguity(self):
        r = self._p(["4/3/2017", "03/04/2017"], ["2017-04-01", "2017-04-01"], ["3", "3"])
        assert list(r.date) == [pd.Timestamp("2017-04-03")] * 2 and set(r.method) == {"duration"}

    def test_blank_discharge_is_missing(self):  # harness blanks D.O.D for in-patients
        r = self._p(["", None], ["2019-01-30", "2019-01-30"], ["", None])
        assert r.date.isna().all() and set(r.method) == {"missing"}

    def test_malformed_discharge(self):
        r = self._p(["2-1217"], ["2017-11-27"], ["6"])
        assert r.method[0] == "malformed" and pd.isna(r.date[0])

    def test_discharge_before_admission_is_inconsistent(self):
        r = self._p(["3/19/2017"], ["2017-04-19"], ["1"])
        assert r.method[0] == "inconsistent" and pd.isna(r.date[0])


# ======================================================== single-order columns
class TestSingleOrder:
    def test_infer_mdy(self):
        assert L.infer_numeric_date_order(pd.Series(["3/25/2017", "4/1/2017"])) == "mdy"

    def test_infer_dmy(self):
        assert L.infer_numeric_date_order(pd.Series(["25/3/2017", "1/4/2017"])) == "dmy"

    def test_mixed_raises(self):
        with pytest.raises(ValueError):
            L.infer_numeric_date_order(pd.Series(["25/3/2017", "3/25/2017"]))

    def test_iso_strict(self):
        r = L.parse_iso_dates(pd.Series(["2019-01-13", "13/01/2019", ""]))
        assert list(r.method) == ["iso", "malformed", "missing"]


# ============================================================ real admission file
class TestRealAdmissions:
    def test_every_doa_parsed(self, adm):
        df, rep = adm
        assert rep["rows_in_file"] == 15757 and len(df) == 15757
        assert rep["doa_unresolved"] == 0 and df["doa"].notna().all()

    def test_every_doa_matches_month_year(self, adm):
        df, rep = adm
        assert df["doa_month_year_ok"].eq(True).all()
        assert rep["doa_month_year_disagree"] == 0 and rep["doa_month_year_conflicts"] == 0
        assert set(df["doa_method"]) <= {"month_year", "month_year_equal"}

    def test_both_readings_agree_count(self, adm):
        df, _ = adm
        assert (df["doa_method"] == "month_year_equal").sum() == 534

    def test_range_and_coverage(self, adm):
        df, _ = adm
        assert df["doa"].min() == pd.Timestamp("2017-04-01")
        assert df["doa"].max() == pd.Timestamp("2019-03-31")
        days = pd.date_range("2017-04-01", "2019-03-31")
        assert set(days) == set(df["doa"])  # every one of the 730 days present

    def test_dates_non_decreasing_in_file_order(self, adm):
        df, _ = adm
        sno = pd.to_numeric(df[L.find_col(df, "SNO")])
        assert sno.is_monotonic_increasing
        assert df["doa"].is_monotonic_increasing

    def test_independent_cross_check(self, raw_adm, adm):
        """Check against a naive, separate implementation using to_datetime."""
        df, _ = adm
        my = pd.to_datetime(raw_adm["month year"], format="%b-%y").dt.to_period("M")
        mdy = pd.to_datetime(raw_adm["D.O.A"], format="%m/%d/%Y", errors="coerce")
        dmy = pd.to_datetime(raw_adm["D.O.A"], format="%d/%m/%Y", errors="coerce")
        ref = mdy.where(mdy.dt.to_period("M") == my, dmy)
        assert (ref == df["doa"]).all()

    def test_fallback_without_month_year_column(self, raw_adm, adm):
        """Robustness: drop `month year` and check the fallback agrees with it."""
        df, _ = adm
        r = L.parse_mixed_dates(raw_adm["D.O.A"], None)
        resolved = r["date"].notna()
        agree = (r.loc[resolved, "date"] == df.loc[resolved, "doa"]).mean()
        print(f"\n  fallback (no month-year): resolved {resolved.mean():.2%}, agreement {agree:.4%}, "
              f"methods {r['method'].value_counts().to_dict()}")
        assert resolved.mean() > 0.99 and agree > 0.99

    def test_discharge_parsing(self, adm):
        df, rep = adm
        m = rep["dod_methods"]
        print(f"\n  D.O.D methods: {m}")
        assert m.get("malformed", 0) == 1  # '2-1217'
        assert df.loc[df["dod"].notna(), "dod"].ge(df.loc[df["dod"].notna(), "doa"]).all()

    def test_cutoff_filter(self, raw_adm):
        df, rep = L.load_admissions(raw=raw_adm, cutoff="2019-02-01")
        assert df["doa"].max() == pd.Timestamp("2019-01-31")
        assert rep["rows_returned"] == len(df) < 15757

    def test_harness_like_truncated_file(self, raw_adm):
        """Simulate the harness: rows before cut-off, D.O.D blank for in-patients."""
        full, _ = L.load_admissions(raw=raw_adm)
        cut = pd.Timestamp("2019-02-15")
        keep = full["doa"] < cut
        t = raw_adm[keep.to_numpy()].copy()
        inpatient = (full.loc[keep, "dod"].isna() | (full.loc[keep, "dod"] >= cut)).to_numpy()
        for c in ["D.O.D", "DURATION OF STAY", "OUTCOME"]:
            t.loc[inpatient, c] = ""
        df, rep = L.load_admissions(raw=t, cutoff=cut)
        assert rep["doa_unresolved"] == 0 and len(df) == keep.sum()
        assert (df["dod_method"] == "missing").sum() >= inpatient.sum()

    def test_column_lookup_tolerates_bom_and_spaces(self):
        df = pd.DataFrame(columns=["﻿SNO", "SMOKING ", "GENDER "])
        assert L.find_col(df, "SNO") == "﻿SNO"
        assert L.find_col(df, "smoking") == "SMOKING "


# ======================================================== other files' dates
class TestOtherFiles:
    def test_pollution(self):
        p = L.load_pollution(ROOT)
        assert set(p["date_method"]) == {"mdy"} and p["date"].is_unique
        assert (p["date"].min(), p["date"].max()) == (pd.Timestamp("2017-03-25"), pd.Timestamp("2019-03-31"))
        assert len(pd.date_range(p["date"].min(), p["date"].max()).difference(p["date"])) == 0

    def test_weather(self):
        w = L.load_weather(ROOT)
        assert set(w["date_method"]) == {"iso"} and w["date"].is_unique and len(w) == 761

    def test_mortality(self):
        m = L.load_mortality(ROOT)
        assert set(m["date_method"]) == {"mdy"} and len(m) == 359

    def test_festivals(self):
        f = L.load_festivals(ROOT)
        assert set(f["date_method"]) == {"iso"} and len(f) == 117

    def test_glossary(self):
        g = L.load_glossary(ROOT)
        assert len(g) == 58

    def test_cutoff_applies_to_env_files(self):
        c = "2019-01-15"
        assert L.load_pollution(ROOT, c)["date"].max() == pd.Timestamp("2019-01-14")
        assert L.load_weather(ROOT, c)["date"].max() == pd.Timestamp("2019-01-14")


# ===================================================== raw data never modified
class TestRawDataUnchanged:
    def test_loader_returns_copy_and_does_not_mutate_input(self, raw_adm):
        before = raw_adm.copy(deep=True)
        df, _ = L.load_admissions(raw=raw_adm, cutoff="2018-06-01")
        df.iloc[0, 0] = "MUTATED"
        pd.testing.assert_frame_equal(raw_adm, before)
        assert list(raw_adm.columns) == list(before.columns)

    def test_original_columns_preserved_verbatim(self, raw_adm, adm):
        df, _ = adm
        pd.testing.assert_frame_equal(df[raw_adm.columns], raw_adm)

    def test_files_hash_unchanged_after_all_loaders(self):
        L.load_admissions(ROOT); L.load_pollution(ROOT); L.load_weather(ROOT)
        L.load_mortality(ROOT); L.load_festivals(ROOT); L.load_glossary(ROOT)
        for name, expected in ORIGINAL_SHA256.items():
            assert sha256(os.path.join(ROOT, name)) == expected, name

    def test_git_reports_original_files_unmodified(self):
        out = subprocess.run(["git", "status", "--porcelain", "--", *ORIGINAL_SHA256],
                             cwd=ROOT, capture_output=True, text=True)
        if out.returncode != 0:
            pytest.skip("git not available")
        assert out.stdout.strip() == ""
