"""Read-only loaders for the PS-06 files, with robust date parsing.

Dependencies: pandas and numpy only, so this code can later be inlined into the
self-contained scored file ``forecast.py``.

Nothing in this module writes to disk. Every loader reads the CSV as strings
(``dtype=str``) and returns a *new* DataFrame; the raw values are kept as-is and
parsed dates are added as extra columns.

Admission dates (``D.O.A`` / ``D.O.D``) are mixed per row between M/D/YYYY and
DD/MM/YYYY. ``D.O.A`` is resolved with the ``month year`` column; ``D.O.D`` is
resolved with ``D.O.A`` + ``DURATION OF STAY``.
"""
from __future__ import annotations

import os
import re

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- files
ADMISSION_FILE = "HDHI Admission data.csv"
POLLUTION_FILE = "HDHI Pollution Data.csv"
MORTALITY_FILE = "HDHI Mortality Data.csv"
WEATHER_FILE = "weather_ludhiana_daily.csv"
FESTIVAL_FILE = "festival_calendar_punjab.csv"
GLOSSARY_FILE = "table_headings.csv"

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Parse statuses (column ``*_method``). Resolved statuses have a valid date.
RESOLVED = (
    "month_year",        # exactly one reading falls in the `month year` month
    "month_year_equal",  # both readings fall in it (day == month): same date
    "unambiguous",       # only one reading is a valid calendar date
    "equal",             # both readings are valid and identical (day == month)
    "neighbour",         # two valid readings; picked the one closest to nearby rows
    "duration",          # D.O.D: reading equals D.O.A + duration - 1 (or + duration)
    "after_admission",   # D.O.D: the only reading on/after D.O.A
    "closest_expected",  # D.O.D: both on/after D.O.A, closest to expected date
    "iso",               # fixed-format column parsed
)
UNRESOLVED = (
    "missing",           # blank / NaN
    "malformed",         # does not look like d/m/yyyy
    "invalid",           # looks like a date but no reading is a real calendar date
    "ambiguous",         # two valid readings, no evidence to choose
    "inconsistent",      # D.O.D: no reading on/after D.O.A
)

_DMY_RE = re.compile(r"^\s*(\d{1,2})\s*[/\-.]\s*(\d{1,2})\s*[/\-.]\s*(\d{4})\s*$")


# ------------------------------------------------------------------------ helpers
def data_dir_default() -> str:
    """PS06_DATA_DIR if set (harness), otherwise the project root (development)."""
    return os.environ.get("PS06_DATA_DIR", _PROJECT_ROOT)


def norm_col(name: str) -> str:
    """Normalise a column name: drop BOM, collapse whitespace, upper-case."""
    return re.sub(r"\s+", " ", str(name).replace("﻿", "")).strip().upper()


def find_col(df: pd.DataFrame, *candidates: str, required: bool = True):
    """Return the actual column name matching any candidate after normalisation."""
    lookup = {norm_col(c): c for c in df.columns}
    for cand in candidates:
        hit = lookup.get(norm_col(cand))
        if hit is not None:
            return hit
    if required:
        raise KeyError(f"none of {candidates} found in columns {list(df.columns)}")
    return None


def read_raw_csv(path: str) -> pd.DataFrame:
    """Read a CSV exactly as text: no type inference, no NA conversion, BOM stripped."""
    return pd.read_csv(path, dtype=str, encoding="utf-8-sig", keep_default_na=False)


def _split_numeric_date(s: pd.Series) -> pd.DataFrame:
    """Split 'a/b/yyyy' strings into integer parts; non-matching rows get NaN."""
    text = s.astype("string").fillna("")
    parts = text.str.extract(_DMY_RE.pattern)
    out = parts.apply(pd.to_numeric, errors="coerce")
    out.columns = ["p1", "p2", "year"]
    out["blank"] = text.str.strip().eq("").to_numpy()
    return out


def _make_date(year: pd.Series, month: pd.Series, day: pd.Series) -> pd.Series:
    """Build datetimes from components; impossible dates (13th month, 31 Feb) -> NaT."""
    ok = year.notna() & month.between(1, 12) & day.between(1, 31)
    txt = pd.Series(pd.NA, index=year.index, dtype="string")
    txt[ok] = (
        year[ok].astype(int).astype(str).str.zfill(4) + "-"
        + month[ok].astype(int).astype(str).str.zfill(2) + "-"
        + day[ok].astype(int).astype(str).str.zfill(2)
    )
    return pd.to_datetime(txt, format="%Y-%m-%d", errors="coerce")


def _candidates(s: pd.Series):
    """Return (parts, month-first reading, day-first reading)."""
    p = _split_numeric_date(s)
    mdy = _make_date(p["year"], p["p1"], p["p2"])
    dmy = _make_date(p["year"], p["p2"], p["p1"])
    return p, mdy, dmy


def parse_month_year(s: pd.Series) -> pd.Series:
    """Parse 'Apr-17' style labels to month periods (NaT when unparseable)."""
    text = s.astype("string").str.strip().str.title()
    dt = pd.to_datetime(text, format="%b-%y", errors="coerce")
    dt = dt.fillna(pd.to_datetime(text, format="%b-%Y", errors="coerce"))
    return dt.dt.to_period("M")


def _neighbour_reference(resolved: pd.Series, window: int) -> pd.Series:
    """Rolling median (in file order) of already-resolved dates, as a reference."""
    days = (resolved - pd.Timestamp("1970-01-01")).dt.days.astype("float")
    ref = days.rolling(2 * window + 1, center=True, min_periods=1).median()
    ref = ref.ffill().bfill()
    return ref


# ------------------------------------------------------------- mixed date parsing
def parse_mixed_dates(
    dates: pd.Series,
    month_year: pd.Series | None = None,
    neighbour_window: int = 10,
    neighbour_max_days: int = 45,
) -> pd.DataFrame:
    """Resolve per-row mixed M/D/YYYY vs DD/MM/YYYY dates.

    Priority per row:
      1. `month_year` evidence: keep the reading that falls in that month.
      2. Only one reading is a valid calendar date -> use it.
      3. Both readings valid and identical (day == month) -> use it.
      4. Two different valid readings -> the one closest to the rolling median of
         resolved neighbouring rows (file order), if within `neighbour_max_days`.
      5. Otherwise NaT with an explicit status.

    Returns a DataFrame aligned to `dates.index` with columns
      date, method, month_year_ok (True/False/NA), month_year_conflict (bool).
    The input Series is never modified.
    """
    idx = dates.index
    p, mdy, dmy = _candidates(dates)

    date = pd.Series(pd.NaT, index=idx, dtype="datetime64[ns]")
    method = pd.Series(pd.NA, index=idx, dtype="object")

    method[p["blank"].to_numpy()] = "missing"
    shaped = p["year"].notna()
    method[~shaped & method.isna()] = "malformed"
    valid_m, valid_d = mdy.notna(), dmy.notna()
    method[shaped & ~valid_m & ~valid_d] = "invalid"

    # 1. month-year evidence
    conflict = pd.Series(False, index=idx)
    if month_year is not None:
        my = parse_month_year(month_year.reindex(idx))
        has_my = my.notna()
        m_ok = valid_m & has_my & (mdy.dt.to_period("M") == my)
        d_ok = valid_d & has_my & (dmy.dt.to_period("M") == my)
        both = m_ok & d_ok & method.isna()
        only_m = m_ok & ~d_ok & method.isna()
        only_d = d_ok & ~m_ok & method.isna()
        date[both], method[both] = mdy[both], "month_year_equal"
        date[only_m], method[only_m] = mdy[only_m], "month_year"
        date[only_d], method[only_d] = dmy[only_d], "month_year"
        conflict = has_my & (valid_m | valid_d) & ~m_ok & ~d_ok

    # 2./3. intrinsic evidence
    todo = method.isna()
    one_m = todo & valid_m & ~valid_d
    one_d = todo & valid_d & ~valid_m
    same = todo & valid_m & valid_d & (mdy == dmy)
    date[one_m], method[one_m] = mdy[one_m], "unambiguous"
    date[one_d], method[one_d] = dmy[one_d], "unambiguous"
    date[same], method[same] = mdy[same], "equal"

    # 4. neighbour evidence for genuinely ambiguous rows
    amb = method.isna() & valid_m & valid_d
    if amb.any():
        ref = _neighbour_reference(date, neighbour_window)
        if ref.notna().any():
            epoch = pd.Timestamp("1970-01-01")
            dist_m = ((mdy - epoch).dt.days - ref).abs()
            dist_d = ((dmy - epoch).dt.days - ref).abs()
            pick_m = amb & (dist_m < dist_d) & (dist_m <= neighbour_max_days)
            pick_d = amb & (dist_d < dist_m) & (dist_d <= neighbour_max_days)
            date[pick_m], method[pick_m] = mdy[pick_m], "neighbour"
            date[pick_d], method[pick_d] = dmy[pick_d], "neighbour"
    method[method.isna()] = "ambiguous"
    date[~method.isin(RESOLVED)] = pd.NaT

    # month-year agreement of the final date (NA where no month-year available)
    if month_year is not None:
        my = parse_month_year(month_year.reindex(idx))
        agree = pd.Series(pd.NA, index=idx, dtype="boolean")
        chk = my.notna() & date.notna()
        agree[chk] = (date[chk].dt.to_period("M") == my[chk]).to_numpy()
    else:
        agree = pd.Series(pd.NA, index=idx, dtype="boolean")

    return pd.DataFrame(
        {"date": date, "method": method.astype(str),
         "month_year_ok": agree, "month_year_conflict": conflict.astype(bool)},
        index=idx,
    )


def parse_discharge_dates(
    dod: pd.Series, doa: pd.Series, duration: pd.Series
) -> pd.DataFrame:
    """Resolve mixed-format discharge dates using admission date and length of stay.

    Blank discharge dates (patients still in hospital at a harness cut-off) give
    NaT / 'missing' without error. Returns columns date, method.
    """
    idx = dod.index
    p, mdy, dmy = _candidates(dod)
    doa = pd.to_datetime(doa.reindex(idx), errors="coerce")
    dur = pd.to_numeric(duration.reindex(idx), errors="coerce")
    exp_incl = doa + pd.to_timedelta(dur - 1, unit="D")   # dataset convention
    exp_excl = doa + pd.to_timedelta(dur, unit="D")

    date = pd.Series(pd.NaT, index=idx, dtype="datetime64[ns]")
    method = pd.Series(pd.NA, index=idx, dtype="object")
    method[p["blank"].to_numpy()] = "missing"
    method[p["year"].isna() & method.isna()] = "malformed"
    vm, vd = mdy.notna(), dmy.notna()
    method[method.isna() & ~vm & ~vd] = "invalid"

    for expected in (exp_incl, exp_excl):
        todo = method.isna()
        hm = todo & vm & (mdy == expected)
        hd = todo & vd & (dmy == expected) & ~hm
        date[hm], method[hm] = mdy[hm], "duration"
        date[hd], method[hd] = dmy[hd], "duration"

    todo = method.isna()
    am = vm & (mdy >= doa)
    ad = vd & (dmy >= doa)
    only_m = todo & am & ~ad
    only_d = todo & ad & ~am
    date[only_m], method[only_m] = mdy[only_m], "after_admission"
    date[only_d], method[only_d] = dmy[only_d], "after_admission"
    both = todo & am & ad
    if both.any():
        closer_m = (mdy - exp_incl).abs() <= (dmy - exp_incl).abs()
        bm, bd = both & closer_m, both & ~closer_m
        date[bm], method[bm] = mdy[bm], "closest_expected"
        date[bd], method[bd] = dmy[bd], "closest_expected"
    method[method.isna()] = "inconsistent"
    date[~method.isin(RESOLVED)] = pd.NaT
    return pd.DataFrame({"date": date, "method": method.astype(str)}, index=idx)


def infer_numeric_date_order(dates: pd.Series) -> str:
    """For a column claimed to use one numeric format, infer 'mdy' or 'dmy'.

    Raises ValueError if the column contains both orders (then use
    ``parse_mixed_dates`` instead) or gives no evidence at all.
    """
    p = _split_numeric_date(dates)
    first_gt12 = int((p["p1"] > 12).sum())
    second_gt12 = int((p["p2"] > 12).sum())
    if first_gt12 and second_gt12:
        raise ValueError(f"mixed date order ({first_gt12} day-first, {second_gt12} month-first)")
    if second_gt12:
        return "mdy"
    if first_gt12:
        return "dmy"
    raise ValueError("no row disambiguates the date order")


def parse_single_order_dates(dates: pd.Series) -> pd.DataFrame:
    """Parse a numeric date column whose order is consistent (inferred, not assumed)."""
    order = infer_numeric_date_order(dates)
    p, mdy, dmy = _candidates(dates)
    date = mdy if order == "mdy" else dmy
    method = pd.Series(order, index=dates.index, dtype="object")
    method[p["year"].isna()] = "malformed"
    method[p["blank"].to_numpy()] = "missing"
    method[p["year"].notna() & date.isna()] = "invalid"
    return pd.DataFrame({"date": date.where(method == order), "method": method})


def parse_iso_dates(dates: pd.Series) -> pd.DataFrame:
    """Parse YYYY-MM-DD strictly."""
    text = dates.astype("string").str.strip()
    date = pd.to_datetime(text, format="%Y-%m-%d", errors="coerce")
    method = pd.Series("iso", index=dates.index, dtype="object")
    method[date.isna()] = "malformed"
    method[text.fillna("").eq("")] = "missing"
    return pd.DataFrame({"date": date, "method": method})


def _apply_cutoff(df: pd.DataFrame, date_col: str, cutoff) -> pd.DataFrame:
    if cutoff is None:
        return df
    c = pd.Timestamp(cutoff)
    return df[df[date_col].notna() & (df[date_col] < c)].copy()


# ----------------------------------------------------------------------- loaders
def load_admissions(data_dir: str | None = None, cutoff=None, raw: pd.DataFrame | None = None):
    """Load admissions with resolved dates.

    Adds: doa, doa_method, doa_month_year_ok, doa_month_year_conflict, dod, dod_method.
    Original columns are kept untouched as strings. With `cutoff`, keeps only rows
    whose resolved admission date is before it (unresolved rows are dropped and
    counted in the report).

    Returns (DataFrame, report dict).
    """
    if raw is None:
        raw = read_raw_csv(os.path.join(data_dir or data_dir_default(), ADMISSION_FILE))
    df = raw.copy(deep=True)

    c_doa = find_col(df, "D.O.A")
    c_my = find_col(df, "month year", required=False)
    c_dod = find_col(df, "D.O.D", required=False)
    c_dur = find_col(df, "DURATION OF STAY", required=False)

    a = parse_mixed_dates(df[c_doa], df[c_my] if c_my else None)
    df["doa"] = a["date"]
    df["doa_method"] = a["method"]
    df["doa_month_year_ok"] = a["month_year_ok"]
    df["doa_month_year_conflict"] = a["month_year_conflict"]

    if c_dod is not None and c_dur is not None:
        d = parse_discharge_dates(df[c_dod], df["doa"], df[c_dur])
        df["dod"], df["dod_method"] = d["date"], d["method"]
    else:
        df["dod"], df["dod_method"] = pd.NaT, "missing"

    report = {
        "rows_in_file": int(len(df)),
        "doa_methods": df["doa_method"].value_counts().to_dict(),
        "doa_unresolved": int(df["doa"].isna().sum()),
        "doa_month_year_disagree": int((df["doa_month_year_ok"] == False).sum()),  # noqa: E712
        "doa_month_year_conflicts": int(df["doa_month_year_conflict"].sum()),
        "doa_min": df["doa"].min(),
        "doa_max": df["doa"].max(),
        "dod_methods": df["dod_method"].value_counts().to_dict(),
    }
    out = _apply_cutoff(df, "doa", cutoff)
    report["rows_returned"] = int(len(out))
    report["rows_dropped_unresolved"] = int(df["doa"].isna().sum()) if cutoff is not None else 0
    return out, report


def load_pollution(data_dir: str | None = None, cutoff=None):
    """Pollution file with `date` added (order inferred and validated, not assumed)."""
    df = read_raw_csv(os.path.join(data_dir or data_dir_default(), POLLUTION_FILE))
    parsed = parse_single_order_dates(df[find_col(df, "DATE")])
    df["date"], df["date_method"] = parsed["date"], parsed["method"]
    return _apply_cutoff(df, "date", cutoff)


def load_mortality(data_dir: str | None = None, cutoff=None):
    """Brought-dead file with `date` added (order inferred and validated)."""
    df = read_raw_csv(os.path.join(data_dir or data_dir_default(), MORTALITY_FILE))
    parsed = parse_single_order_dates(df[find_col(df, "DATE OF BROUGHT DEAD")])
    df["date"], df["date_method"] = parsed["date"], parsed["method"]
    return _apply_cutoff(df, "date", cutoff)


def load_weather(data_dir: str | None = None, cutoff=None):
    """Open-Meteo daily weather with `date` parsed strictly as ISO."""
    df = read_raw_csv(os.path.join(data_dir or data_dir_default(), WEATHER_FILE))
    parsed = parse_iso_dates(df[find_col(df, "date")])
    df["date"], df["date_method"] = parsed["date"], parsed["method"]
    return _apply_cutoff(df, "date", cutoff)


def load_festivals(data_dir: str | None = None):
    """Festival calendar (complete in the harness; never cut off). One row per entry."""
    df = read_raw_csv(os.path.join(data_dir or data_dir_default(), FESTIVAL_FILE))
    parsed = parse_iso_dates(df[find_col(df, "date")])
    df["date"], df["date_method"] = parsed["date"], parsed["method"]
    return df


def load_glossary(data_dir: str | None = None):
    """Column glossary with the trailing blank rows removed."""
    df = read_raw_csv(os.path.join(data_dir or data_dir_default(), GLOSSARY_FILE))
    keep = df.apply(lambda col: col.str.strip().ne("")).any(axis=1)
    return df[keep].reset_index(drop=True)
