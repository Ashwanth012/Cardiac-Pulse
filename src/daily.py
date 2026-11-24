"""Step 2: daily raw admission counts, duplicate analysis, cleaned daily dataset.

Pure functions (no file writes); ``analysis/step2_build_daily.py`` saves outputs.
pandas + numpy only.

Rules fixed by the project brief:
* The forecasting target is the RAW number of admission rows per D.O.A. Duplicates
  are analysed and reported, never removed from the target.
* D.O.D, length of stay, outcome, laboratory values and patient characteristics are
  NOT predictors. Emergency/OPD split counts and de-duplicated counts are carried
  only as analysis / sensitivity columns (see COLUMN_ROLES).
* Source files are never modified.

Cleaning-log actions:
  corrected   - value changed to a confidently-recovered value
  set_missing - value judged invalid and replaced by NaN
  imputed     - NaN filled by interpolation (always after set_missing or a blank)
  flagged     - value kept unchanged, recorded for investigation
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import loaders as L

START = pd.Timestamp("2017-04-01")
END = pd.Timestamp("2019-03-31")

POLLUTANTS = ["PM2.5", "PM10", "NO2", "NH3", "SO2", "CO", "OZONE"]
POLL_SHORT = {"PM2.5": "pm25", "PM10": "pm10", "NO2": "no2", "NH3": "nh3",
              "SO2": "so2", "CO": "co", "OZONE": "ozone"}
MAX_INTERP_GAP = 3            # days; longer gaps stay NaN
NO2_AVG_CEILING = 400         # NO2 AVG above this is treated as invalid (see report)
AVG_FAR_OUTSIDE = 2.0         # AVG < MIN/2 or AVG > 2*MAX -> set missing
LOW_COUNT_RATIO = 0.40        # daily count < 40% of 29-day rolling median -> flagged
SPIKE_Z = 5.0                 # robust z (log scale) for pollution spike flags

MAJOR_FESTIVALS = {
    "Diwali", "Lohri", "Maghi", "Holi", "Dussehra", "Guru Nanak's Birthday",
    "New Year's Day", "Vaisakhi", "Karva Chauth", "Raksha Bandhan",
    "Janmashtami (Vaishnava)", "Eid al-Fitr", "Eid al-Adha", "Christmas",
}

LOG_COLUMNS = ["source", "date", "row_ref", "column", "raw_value", "new_value",
               "action", "rule", "reason"]


def _log(entries, source, date, row_ref, column, raw, new, action, rule, reason):
    entries.append(dict(source=source, date=None if pd.isna(date) else pd.Timestamp(date).date().isoformat(),
                        row_ref=row_ref, column=column, raw_value=raw, new_value=new,
                        action=action, rule=rule, reason=reason))


# ===================================================================== duplicates
def duplicate_analysis(adm: pd.DataFrame):
    """Classify admission rows; nothing is removed.

    * exact duplicate  : identical to an earlier row in every original column except SNO
    * MRD+D.O.A repeat : same MRD No. and same resolved admission date as an earlier row
    * near duplicate   : MRD+D.O.A repeat that is NOT an exact copy of an earlier row
    * overlapping stay : same MRD, later admission date that falls inside the
                         previous stay (D.O.A_next <= D.O.D_prev) - flagged only

    Returns (row_flags, summary dict, by_month DataFrame, differing-column counts).
    """
    sno, mrd = L.find_col(adm, "SNO"), L.find_col(adm, "MRD No.")
    orig = [c for c in adm.columns if c in _original_columns(adm)]
    content = [c for c in orig if c != sno]
    # The same record is sometimes typed with D.O.A/D.O.D in the other date order,
    # so "exact" compares RESOLVED dates instead of the raw date strings.
    raw_date_cols = {L.find_col(adm, "D.O.A"), L.find_col(adm, "D.O.D"), L.find_col(adm, "month year")}
    content_norm = [c for c in content if c not in raw_date_cols] + ["doa", "dod"]

    f = pd.DataFrame(index=adm.index)
    f["SNO"] = adm[sno]
    f["MRD"] = adm[mrd]
    f["doa"] = adm["doa"]
    f["exact_raw_extra"] = adm.duplicated(subset=content, keep="first")
    f["exact_extra"] = adm.duplicated(subset=content_norm, keep="first")
    f["exact_any"] = adm.duplicated(subset=content_norm, keep=False)
    f["mrd_doa_extra"] = adm.duplicated(subset=[mrd, "doa"], keep="first")
    f["mrd_doa_any"] = adm.duplicated(subset=[mrd, "doa"], keep=False)
    f["near_extra"] = f["mrd_doa_extra"] & ~f["exact_extra"]

    grp_key = adm[mrd].astype(str) + "|" + adm["doa"].dt.strftime("%Y-%m-%d")
    f["group_id"] = np.where(f["mrd_doa_any"], grp_key, "")
    f["group_size"] = grp_key.map(grp_key.value_counts()).where(f["mrd_doa_any"], 1).astype(int)

    # columns that differ inside each MRD+DOA group
    diff_cols = {}
    differing = pd.Series("", index=adm.index, dtype=object)
    sub = adm[f["mrd_doa_any"]]
    for key, g in sub.groupby(grp_key[f["mrd_doa_any"]]):
        cols = [c for c in content_norm if g[c].nunique(dropna=False) > 1]
        cols += [f"{c} (format only)" for c in content if c in raw_date_cols
                 and g[c].nunique() > 1 and c not in ("month year",)
                 and not (c == L.find_col(adm, "D.O.D") and g["dod"].nunique(dropna=False) > 1)]
        for c in cols:
            diff_cols[c] = diff_cols.get(c, 0) + 1
        differing[g.index] = ";".join(cols)
    f["differing_columns"] = differing

    # SNO distance between a repeat and the first row of its group
    first_sno = pd.to_numeric(adm[sno]).groupby(grp_key).transform("min")
    f["sno_gap_to_first"] = (pd.to_numeric(adm[sno]) - first_sno).where(f["mrd_doa_extra"])

    # overlapping stays for the same patient (distinct admission dates)
    o = adm[[mrd, "doa", "dod"]].copy()
    o = o[~f["mrd_doa_extra"]].sort_values([mrd, "doa"])
    prev_dod = o.groupby(mrd)["dod"].shift()
    overlap_idx = o.index[(o["doa"] <= prev_dod)]
    f["overlapping_stay"] = f.index.isin(overlap_idx)

    month = adm["doa"].dt.to_period("M")
    by_month = pd.DataFrame({
        "rows": adm.groupby(month).size(),
        "exact_extra": f.groupby(month)["exact_extra"].sum(),
        "near_extra": f.groupby(month)["near_extra"].sum(),
        "mrd_doa_extra": f.groupby(month)["mrd_doa_extra"].sum(),
    })
    by_month["exact_extra_pct"] = (100 * by_month["exact_extra"] / by_month["rows"]).round(2)
    by_month["mrd_doa_extra_pct"] = (100 * by_month["mrd_doa_extra"] / by_month["rows"]).round(2)
    by_month.index = by_month.index.astype(str)
    by_month.index.name = "month"

    summary = {
        "rows": int(len(adm)),
        "exact_raw_string_extra_copies": int(f["exact_raw_extra"].sum()),
        "exact_extra_copies": int(f["exact_extra"].sum()),
        "rows_in_exact_groups": int(f["exact_any"].sum()),
        "mrd_doa_extra_rows": int(f["mrd_doa_extra"].sum()),
        "rows_in_mrd_doa_groups": int(f["mrd_doa_any"].sum()),
        "mrd_doa_group_sizes": f.loc[f["mrd_doa_any"]].groupby("group_id").size().value_counts().sort_index().to_dict(),
        "near_extra_rows": int(f["near_extra"].sum()),
        "overlapping_stays": int(f["overlapping_stay"].sum()),
        "unique_patients": int(adm[mrd].nunique()),
        "patients_multiple_rows": int((adm[mrd].value_counts() > 1).sum()),
        "sno_gap_median": float(f["sno_gap_to_first"].median()),
        "sno_gap_p90": float(f["sno_gap_to_first"].quantile(0.9)),
        "sno_gap_max": float(f["sno_gap_to_first"].max()),
    }
    diff_counts = pd.Series(diff_cols, dtype=int).sort_values(ascending=False)
    return f, summary, by_month, diff_counts


def _original_columns(adm):
    added = {"doa", "doa_method", "doa_month_year_ok", "doa_month_year_conflict", "dod", "dod_method"}
    return [c for c in adm.columns if c not in added]


# =================================================================== daily counts
def daily_admission_counts(adm: pd.DataFrame, dup_flags: pd.DataFrame | None = None,
                           start=START, end=END) -> pd.DataFrame:
    """One row per calendar day in [start, end]; days without rows get 0.

    `admissions` is the RAW row count (forecasting target). Other count columns
    are for analysis/sensitivity only.
    """
    days = pd.date_range(pd.Timestamp(start), pd.Timestamp(end), freq="D")
    inwin = adm[(adm["doa"] >= days[0]) & (adm["doa"] <= days[-1])]
    out = pd.DataFrame(index=days)
    out.index.name = "date"
    out["admissions"] = inwin.groupby("doa").size().reindex(days, fill_value=0).astype(int)

    c_type = L.find_col(adm, "TYPE OF ADMISSION-EMERGENCY/OPD", required=False)
    if c_type is not None:
        t = inwin[c_type].str.strip().str.upper()
        out["admissions_emergency"] = inwin[t == "E"].groupby("doa").size().reindex(days, fill_value=0).astype(int)
        out["admissions_opd"] = inwin[t == "O"].groupby("doa").size().reindex(days, fill_value=0).astype(int)
        other = inwin[~t.isin(["E", "O"])].groupby("doa").size().reindex(days, fill_value=0)
        if other.sum():
            out["admissions_type_other"] = other.astype(int)

    if dup_flags is not None:
        fl = dup_flags.loc[inwin.index]
        out["extra_exact_copies"] = fl[fl["exact_extra"]].groupby("doa").size().reindex(days, fill_value=0).astype(int)
        out["extra_near_copies"] = fl[fl["near_extra"]].groupby("doa").size().reindex(days, fill_value=0).astype(int)
        out["admissions_dedup_exact"] = out["admissions"] - out["extra_exact_copies"]
        out["admissions_dedup_mrd_doa"] = out["admissions"] - out["extra_exact_copies"] - out["extra_near_copies"]

    roll = out["admissions"].rolling(29, center=True, min_periods=8).median()
    out["low_count_flag"] = out["admissions"] < LOW_COUNT_RATIO * roll
    out["zero_count_day"] = out["admissions"] == 0
    return out


def admission_quality_flags(adm: pd.DataFrame, log: list) -> dict:
    """Record patient-level oddities (flag only; none of these fields is a predictor)."""
    sno = L.find_col(adm, "SNO")
    ref = lambda i: f"SNO {adm.at[i, sno]}"  # noqa: E731
    for i in adm.index[adm["dod_method"].isin(["malformed", "invalid", "inconsistent"])]:
        _log(log, "admissions", adm.at[i, "doa"], ref(i), "D.O.D", adm.at[i, L.find_col(adm, "D.O.D")], None,
             "flagged", "A1", f"discharge date {adm.at[i, 'dod_method']} (analysis-only field)")
    for i in adm.index[adm["dod_method"].isin(["after_admission", "closest_expected"])]:
        _log(log, "admissions", adm.at[i, "doa"], ref(i), "D.O.D", adm.at[i, L.find_col(adm, "D.O.D")],
             adm.at[i, "dod"].date().isoformat(), "flagged", "A2",
             "discharge date does not match DURATION OF STAY; resolved by order only")
    age = pd.to_numeric(adm[L.find_col(adm, "AGE")], errors="coerce")
    for i in adm.index[(age < 18) | (age > 100)]:
        _log(log, "admissions", adm.at[i, "doa"], ref(i), "AGE", adm.at[i, L.find_col(adm, "AGE")], None,
             "flagged", "A3", "age outside 18-100 in an adult cardiology unit")
    icu = pd.to_numeric(adm[L.find_col(adm, "duration of intensive unit stay")], errors="coerce")
    los = pd.to_numeric(adm[L.find_col(adm, "DURATION OF STAY")], errors="coerce")
    for i in adm.index[icu > los]:
        _log(log, "admissions", adm.at[i, "doa"], ref(i), "duration of intensive unit stay",
             adm.at[i, L.find_col(adm, "duration of intensive unit stay")], None, "flagged", "A4",
             f"ICU days exceed total stay ({los[i]:.0f})")
    binary = [c for c in adm.columns[adm.columns.get_loc(L.find_col(adm, "SMOKING")):]
              if c in _original_columns(adm) and set(adm[c].unique()) <= {"0", "1", "\\", ""}]
    for c in binary:
        for i in adm.index[~adm[c].isin(["0", "1"])]:
            _log(log, "admissions", adm.at[i, "doa"], ref(i), c, adm.at[i, c], None, "flagged", "A5",
                 "binary field holds a non 0/1 value")
    labs = ["HB", "TLC", "PLATELETS", "GLUCOSE", "UREA", "CREATININE", "BNP", "EF"]
    lab_missing = {}
    for c in labs:
        s = adm[L.find_col(adm, c)].str.strip()
        lab_missing[c] = {"blank": int((s == "").sum()), "EMPTY": int((s.str.upper() == "EMPTY").sum())}
    return {"lab_missing": lab_missing}


# ====================================================================== pollution
def clean_pollution(pol: pd.DataFrame, log: list) -> pd.DataFrame:
    """Numeric, rule-cleaned pollution table indexed by date (all station days kept)."""
    p = pol.set_index("date").sort_index()
    out = pd.DataFrame(index=p.index)

    # R1: non-numeric cells
    num_cols = ["AQI"] + [f"{v} {s}" for v in POLLUTANTS for s in ("AVG", "MIN", "MAX")] + \
               ["MAX TEMP", "MIN TEMP", "HUMIDITY"]
    for c in num_cols:
        raw = p[L.find_col(p, c)].astype(str).str.strip()
        val = pd.to_numeric(raw, errors="coerce")
        for d in raw.index[val.isna()]:
            r = raw[d]
            if r == "":
                _log(log, "pollution", d, None, c, r, None, "set_missing", "P1", "blank cell")
                continue
            digits = r.strip("`'\" ")
            if digits.isdigit() and digits != r:
                val[d] = float(digits)
                _log(log, "pollution", d, None, c, r, digits, "corrected", "P1",
                     "stray punctuation around a plain number removed")
            else:
                _log(log, "pollution", d, None, c, r, None, "set_missing", "P1",
                     "non-numeric value with no unambiguous reading")
        out[c] = val

    # R2: NO2 AVG impossible (equals AQI, MAX at the 900 cap)
    no2 = out["NO2 AVG"]
    for d in no2.index[no2 > NO2_AVG_CEILING]:
        _log(log, "pollution", d, None, "NO2 AVG", int(no2[d]), None, "set_missing", "P2",
             f"NO2 AVG > {NO2_AVG_CEILING}, equals the reported AQI and NO2 MAX is 900 (instrument cap)")
        if out.at[d, "AQI"] == no2[d]:
            _log(log, "pollution", d, None, "AQI", int(out.at[d, "AQI"]), None, "set_missing", "P2",
                 "AQI was driven by the invalid NO2 value")
            out.at[d, "AQI"] = np.nan
        out.at[d, "NO2 AVG"] = np.nan
    for d in out.index[out["NO2 MAX"] >= 900]:
        _log(log, "pollution", d, None, "NO2 MAX", int(out.at[d, "NO2 MAX"]), None, "flagged", "P2",
             "NO2 MAX at 900 looks like an instrument cap")

    # R3: AVG versus its own MIN/MAX
    for v in POLLUTANTS:
        a, mn, mx = out[f"{v} AVG"], out[f"{v} MIN"], out[f"{v} MAX"]
        far = (a < mn / AVG_FAR_OUTSIDE) | (a > AVG_FAR_OUTSIDE * mx)
        near = ((a < mn) | (a > mx)) & ~far
        for d in a.index[far.fillna(False)]:
            _log(log, "pollution", d, None, f"{v} AVG", a[d], None, "set_missing", "P3",
                 f"AVG far outside the day's reported [MIN, MAX] = [{mn[d]:g}, {mx[d]:g}]")
            out.at[d, f"{v} AVG"] = np.nan
        for d in a.index[near.fillna(False)]:
            _log(log, "pollution", d, None, f"{v} AVG", a[d], None, "flagged", "P3",
                 f"AVG slightly outside the day's range [{mn[d]:g}, {mx[d]:g}]")
        for d in a.index[(mn > mx).fillna(False)]:
            _log(log, "pollution", d, None, f"{v} MIN/MAX", f"{mn[d]:g}/{mx[d]:g}", None, "flagged", "P3",
                 "MIN greater than MAX")

    # R4: AQI should equal the max pollutant sub-index
    sub = out[[f"{v} AVG" for v in POLLUTANTS]].max(axis=1)
    for d in out.index[(out["AQI"].notna()) & (sub.notna()) & (out["AQI"] != sub)]:
        _log(log, "pollution", d, None, "AQI", out.at[d, "AQI"], None, "flagged", "P4",
             f"AQI differs from max pollutant AVG ({sub[d]:g})")

    # R5: robust spike screen on log AVG (flag only; real smog episodes are expected)
    for v in ["PM2.5", "PM10", "AQI"]:
        col = f"{v} AVG" if v != "AQI" else "AQI"
        x = np.log1p(out[col])
        med = x.rolling(15, center=True, min_periods=5).median()
        mad = (x - med).abs().rolling(61, center=True, min_periods=15).median() * 1.4826
        z = (x - med) / mad.replace(0, np.nan)
        for d in z.index[(z.abs() > SPIKE_Z).fillna(False)]:
            _log(log, "pollution", d, None, col, out.at[d, col], None, "flagged", "P5",
                 f"robust spike z={z[d]:.1f} vs 15-day median (kept; may be a genuine episode)")

    # assemble short names; interpolate short gaps of the main daily values
    clean = pd.DataFrame(index=out.index)
    clean["aqi"] = out["AQI"]
    for v in POLLUTANTS:
        clean[POLL_SHORT[v]] = out[f"{v} AVG"]
    for c in list(clean.columns):
        missing = clean[c].isna()
        filled = clean[c].interpolate(method="linear", limit=MAX_INTERP_GAP, limit_area="inside")
        for d in clean.index[missing & filled.notna()]:
            _log(log, "pollution", d, None, c, None, round(float(filled[d]), 2), "imputed", "P6",
                 f"linear interpolation (gap <= {MAX_INTERP_GAP} days)")
        clean[c] = filled
        clean[f"{c}_imputed"] = missing & filled.notna()
    clean["prominent_pollutant"] = p[L.find_col(p, "PROMINENT POLLUTENT")].str.strip()
    clean["station_tmax_c"] = out["MAX TEMP"]
    clean["station_tmin_c"] = out["MIN TEMP"]
    clean["station_humidity_pct"] = out["HUMIDITY"]  # not imputed: cross-check only
    return clean


# ======================================================================== weather
def check_weather(w: pd.DataFrame, log: list) -> pd.DataFrame:
    x = w.set_index("date").sort_index()
    cols = ["temp_max_c", "temp_min_c", "temp_mean_c", "precip_mm",
            "rel_humidity_mean_pct", "wind_max_kmh", "wind_mean_kmh"]
    out = x[cols].apply(pd.to_numeric, errors="coerce")
    for c in cols:
        for d in out.index[out[c].isna()]:
            _log(log, "weather", d, None, c, x.at[d, c], None, "set_missing", "W1", "non-numeric or blank")
    bad = (out.temp_min_c > out.temp_mean_c) | (out.temp_mean_c > out.temp_max_c)
    for d in out.index[bad]:
        _log(log, "weather", d, None, "temp_*", None, None, "flagged", "W2", "min <= mean <= max violated")
    for d in out.index[(out.rel_humidity_mean_pct < 0) | (out.rel_humidity_mean_pct > 100) | (out.precip_mm < 0)]:
        _log(log, "weather", d, None, "humidity/precip", None, None, "flagged", "W3", "physically impossible value")
    return out


# ====================================================================== festivals
def festival_daily(fest: pd.DataFrame, days: pd.DatetimeIndex) -> pd.DataFrame:
    """Collapse the calendar to one row per day (several holidays can share a date)."""
    f = fest.copy()
    f["type"] = f["type"].str.strip().str.lower()
    g = f.groupby("date")
    out = pd.DataFrame(index=days)
    out["holiday_count"] = g.size().reindex(days, fill_value=0).astype(int)
    out["holiday_any"] = out["holiday_count"] > 0
    for t in ["national", "state", "festival"]:
        out[f"holiday_{t}"] = f[f["type"] == t].groupby("date").size().reindex(days, fill_value=0).gt(0)
    out["holiday_major_festival"] = f[f["name"].isin(MAJOR_FESTIVALS)].groupby("date").size() \
        .reindex(days, fill_value=0).gt(0)
    out["holiday_names"] = g["name"].agg(lambda s: "; ".join(sorted(s))).reindex(days).fillna("")
    return out


# ======================================================================= assemble
COLUMN_ROLES = {
    # column: (source, role, description)
    "date": ("calendar", "key", "calendar day, 2017-04-01 .. 2019-03-31"),
    "weekday": ("calendar", "predictor_candidate", "0=Mon .. 6=Sun"),
    "admissions": ("admissions", "TARGET", "raw admission rows with this D.O.A (duplicates NOT removed)"),
    "admissions_emergency": ("admissions", "analysis_only", "raw rows with type E (not a predictor)"),
    "admissions_opd": ("admissions", "analysis_only", "raw rows with type O (not a predictor)"),
    "extra_exact_copies": ("admissions", "data_quality", "rows that exactly repeat an earlier row (excl. SNO)"),
    "extra_near_copies": ("admissions", "data_quality", "MRD+D.O.A repeats that are not exact copies"),
    "admissions_dedup_exact": ("admissions", "sensitivity_only", "admissions minus exact copies"),
    "admissions_dedup_mrd_doa": ("admissions", "sensitivity_only", "one row per MRD+D.O.A"),
    "low_count_flag": ("admissions", "data_quality", f"count < {LOW_COUNT_RATIO:.0%} of 29-day rolling median"),
    "zero_count_day": ("admissions", "data_quality", "no admission rows (kept as genuine 0)"),
    "brought_dead": ("mortality", "analysis_only", "patients brought dead (separate population)"),
    "temp_max_c": ("weather", "predictor_candidate", "Open-Meteo daily max temperature"),
    "temp_min_c": ("weather", "predictor_candidate", "Open-Meteo daily min temperature"),
    "temp_mean_c": ("weather", "predictor_candidate", "Open-Meteo daily mean temperature"),
    "precip_mm": ("weather", "predictor_candidate", "Open-Meteo precipitation"),
    "rel_humidity_mean_pct": ("weather", "predictor_candidate", "Open-Meteo mean relative humidity"),
    "wind_max_kmh": ("weather", "predictor_candidate", "Open-Meteo max wind"),
    "wind_mean_kmh": ("weather", "predictor_candidate", "Open-Meteo mean wind"),
    "aqi": ("pollution", "predictor_candidate", "reported AQI (cleaned, short gaps interpolated)"),
    "pm25": ("pollution", "predictor_candidate", "PM2.5 AVG (cleaned; units uncertain - see report)"),
    "pm10": ("pollution", "predictor_candidate", "PM10 AVG (cleaned)"),
    "no2": ("pollution", "predictor_candidate", "NO2 AVG (cleaned)"),
    "nh3": ("pollution", "predictor_candidate", "NH3 AVG (cleaned)"),
    "so2": ("pollution", "predictor_candidate", "SO2 AVG (cleaned)"),
    "co": ("pollution", "predictor_candidate", "CO AVG (cleaned)"),
    "ozone": ("pollution", "predictor_candidate", "OZONE AVG (cleaned)"),
    "prominent_pollutant": ("pollution", "analysis_only", "station's prominent pollutant label"),
    "station_tmax_c": ("pollution", "cross_check_only", "pollution-station max temperature"),
    "station_tmin_c": ("pollution", "cross_check_only", "pollution-station min temperature"),
    "station_humidity_pct": ("pollution", "cross_check_only", "pollution-station humidity (not imputed)"),
    "holiday_count": ("festival", "predictor_candidate", "number of calendar entries on the day"),
    "holiday_any": ("festival", "predictor_candidate", "any holiday/festival"),
    "holiday_national": ("festival", "predictor_candidate", "national holiday"),
    "holiday_state": ("festival", "predictor_candidate", "Punjab state holiday"),
    "holiday_festival": ("festival", "predictor_candidate", "calendar type 'festival'"),
    "holiday_major_festival": ("festival", "predictor_candidate", "one of MAJOR_FESTIVALS"),
    "holiday_names": ("festival", "label", "names joined with '; '"),
}
for _c in ["aqi", "pm25", "pm10", "no2", "nh3", "so2", "co", "ozone"]:
    COLUMN_ROLES[f"{_c}_imputed"] = ("pollution", "data_quality", f"{_c} filled by interpolation")


def build_all(data_dir: str | None = None) -> dict:
    """Run step 2 in memory. Returns every table needed for the saved outputs."""
    log: list = []
    adm, adm_report = L.load_admissions(data_dir)
    flags, dup_summary, dup_by_month, diff_counts = duplicate_analysis(adm)
    daily_adm = daily_admission_counts(adm, flags)
    adm_extra = admission_quality_flags(adm, log)
    for d in daily_adm.index[daily_adm["low_count_flag"]]:
        _log(log, "admissions", d, None, "admissions (daily)", int(daily_adm.at[d, "admissions"]), None,
             "flagged", "A6", f"daily count below {LOW_COUNT_RATIO:.0%} of 29-day rolling median "
             "(possible under-recording; kept)")

    pol = clean_pollution(L.load_pollution(data_dir), log)
    wea = check_weather(L.load_weather(data_dir), log)
    mort = L.load_mortality(data_dir)
    fest = L.load_festivals(data_dir)

    days = daily_adm.index
    daily = daily_adm.copy()
    daily.insert(0, "weekday", days.dayofweek)
    daily["brought_dead"] = mort.groupby("date").size().reindex(days, fill_value=0).astype(int)
    daily = daily.join(wea, how="left").join(pol, how="left").join(festival_daily(fest, days), how="left")
    daily.index.name = "date"

    env_ext_days = pd.date_range(min(wea.index.min(), pol.index.min()), END)
    env_ext = pd.DataFrame(index=env_ext_days).join(wea).join(pol)
    env_ext.index.name = "date"

    log_df = pd.DataFrame(log, columns=LOG_COLUMNS)
    return dict(daily=daily, daily_admissions=daily_adm, dup_flags=flags, dup_summary=dup_summary,
                dup_by_month=dup_by_month, dup_diff_columns=diff_counts, cleaning_log=log_df,
                adm_report=adm_report, adm_extra=adm_extra, env_extended=env_ext,
                pollution=pol, weather=wea, festivals=fest)
