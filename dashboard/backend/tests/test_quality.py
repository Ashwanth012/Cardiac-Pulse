"""Phase 3A (Data Quality & Monitoring) backend tests.

Every expected value is recomputed independently from the saved Step 2 outputs.
Run from the project root:
    PYTHONDONTWRITEBYTECODE=1 python -m pytest dashboard/backend/tests -v -p no:cacheprovider
"""
import ast
import hashlib
import json
import re
import sys
from pathlib import Path

import pandas as pd
import pytest

BACKEND = Path(__file__).resolve().parents[1]
ROOT = BACKEND.parents[1]
sys.path.insert(0, str(BACKEND))

from fastapi.testclient import TestClient  # noqa: E402

from app import config as C  # noqa: E402
from app import forecast_runner as FR  # noqa: E402
from app import quality as Q  # noqa: E402
from app.datastore import DataStore, assert_no_pii  # noqa: E402
from app.main import app  # noqa: E402

OUT = ROOT / "outputs"
URLS = ["/api/quality/summary", "/api/quality/duplicates", "/api/quality/missing", "/api/quality/pollution-log",
        "/api/quality/case-mix"]
PAGE = ROOT / "dashboard" / "frontend" / "src" / "pages" / "DataQuality.tsx"


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def daily():
    return pd.read_csv(OUT / "daily_admissions.csv", parse_dates=["date"]).set_index("date")


@pytest.fixture(scope="module")
def log():
    return pd.read_csv(OUT / "cleaning_log.csv", dtype=str)


def _strings(obj):
    if isinstance(obj, dict):
        for v in obj.values():
            yield from _strings(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _strings(v)
    elif isinstance(obj, str):
        yield obj


# ========================================================================= contracts & PII
@pytest.mark.parametrize("url", URLS)
def test_endpoints_ok_and_pii_free(client, url):
    r = client.get(url)
    assert r.status_code == 200, r.text
    assert_no_pii(r.json())


@pytest.mark.parametrize("url", URLS)
def test_get_only(client, url):
    assert client.post(url).status_code == 405


def test_no_admission_level_details_anywhere(client, log):
    """No serial numbers, no admission-file values or reasons, no patient fields in any response."""
    adm = log[log.source == "admissions"]

    def strip_calendar(o):            # calendar-day fields legitimately hold every date in the data window
        if isinstance(o, dict):
            return {k: strip_calendar(v) for k, v in o.items() if k not in ("date", "start", "end")}
        if isinstance(o, list):
            return [strip_calendar(v) for v in o]
        return o
    blob = json.dumps([strip_calendar(client.get(u).json()) for u in URLS])
    assert not re.search(r"\bSNO\b|\bS\.NO\b|row_ref", blob, re.I)
    for col in ("raw_value", "new_value", "reason"):
        for v in adm[col].dropna().unique():
            if len(str(v)) >= 4:                       # skip tiny tokens that could collide with ordinary text
                assert str(v) not in blob, (col, v)
    for field in ("D.O.D", "DURATION OF STAY", "CHEST INFECTION", "GENDER", "OUTCOME"):
        assert field not in blob, field


def test_cleaning_log_identifier_column_never_read(monkeypatch):
    """The guarded loaders never request the per-record reference column (usecols whitelist)."""
    seen = []
    real = pd.read_csv

    def spy(path, *a, **k):
        if Path(str(path)).name == "cleaning_log.csv":
            seen.append(k.get("usecols"))
        return real(path, *a, **k)
    monkeypatch.setattr(Q.pd, "read_csv", spy)
    qd = Q.QualityData(DataStore())
    qd.summary, qd.pollution_log
    assert seen and all(u is not None and "row_ref" not in u for u in seen)
    assert "row_ref" not in Q.LOG_COUNT_COLUMNS + Q.LOG_POLLUTION_COLUMNS


def test_quality_module_loads_no_patient_level_source():
    tree = ast.parse((BACKEND / "app" / "quality.py").read_text(encoding="utf-8"))
    docs = {id(n.body[0].value) for n in ast.walk(tree)
            if isinstance(n, (ast.Module, ast.FunctionDef, ast.ClassDef)) and n.body
            and isinstance(n.body[0], ast.Expr) and isinstance(n.body[0].value, ast.Constant)}
    lits = [n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in docs]
    for name in ("HDHI Admission data", "HDHI Mortality", "duplicate_rows", "brought_dead", "row_ref"):
        assert not any(name in s for s in lits), name
    assert "brought_dead" not in Q.CLEAN_DAILY_COLUMNS and not any("admission" in c for c in Q.CLEAN_DAILY_COLUMNS)


def test_incomplete_only_ever_potentially_incomplete(client):
    for url in URLS:
        for s in _strings(client.get(url).json()):
            for m in re.finditer(r"incomplete", s, re.I):
                assert re.search(r"potentially[ _-]$", s[max(0, m.start() - 12):m.start()].lower()), (url, s)


# ========================================================================= summary values
def test_summary_record_counts(client, daily):
    r = client.get("/api/quality/summary").json()["records"]
    raw, ex, mrd = daily.admissions.sum(), daily.admissions_dedup_exact.sum(), daily.admissions_dedup_mrd_doa.sum()
    assert (r["raw"], r["dedup_exact"], r["dedup_mrd"]) == (raw, ex, mrd) == (15757, 14872, 14399)
    assert (r["exact_extra"], r["mrd_extra"], r["near_extra"]) == (885, 1358, int(daily.extra_near_copies.sum())) == (885, 1358, 473)
    assert r["exact_extra_rate"] == pytest.approx(885 / 15757, abs=6e-4)
    assert r["mrd_extra_rate"] == pytest.approx(1358 / 15757, abs=6e-4)


def test_summary_days(client, daily):
    d = client.get("/api/quality/summary").json()["days"]
    low = daily[daily.low_count_flag]
    assert d["zero_count"] == int(daily.zero_count_day.sum()) == 0
    assert d["low_count"] == len(low) == 14
    assert [x["date"] for x in d["low_count_days"]] == low.index.strftime("%Y-%m-%d").tolist()
    assert [x["admissions"] for x in d["low_count_days"]] == low.admissions.tolist()
    flagged = [x for x in d["low_count_days"] if x["potentially_incomplete"]]
    assert [x["date"] for x in flagged] == [C.POTENTIALLY_INCOMPLETE_DAY]
    assert d["potentially_incomplete"]["status"] == "potentially incomplete"
    assert "Diwali" in next(x for x in d["low_count_days"] if x["date"] == "2017-10-19")["holidays"]


def test_summary_cleaning_counts(client, log):
    c = client.get("/api/quality/summary").json()
    assert c["cleaning"]["total"] == len(log) == 249
    got = {(x["source"], x["action"]): x["entries"] for x in c["cleaning"]["by_source_action"]}
    assert got == {k: int(v) for k, v in log.groupby(["source", "action"]).size().items()}
    assert got == {("admissions", "flagged"): 177, ("pollution", "corrected"): 1, ("pollution", "flagged"): 45,
                   ("pollution", "imputed"): 10, ("pollution", "set_missing"): 16}
    rules = {x["rule"]: x for x in c["cleaning"]["by_rule"]}
    assert {r: x["entries"] for r, x in rules.items()} == {k: int(v) for k, v in log.groupby("rule").size().items()}
    assert all(x["description"] and not x["description"].startswith("Rule ") for x in rules.values())
    assert all("fields" not in x for x in rules.values() if x["source"] == "admissions")
    assert c["environment_cells"] == {"corrected": 1, "set_missing": 16, "imputed": 10, "flagged": 45}
    assert "no admission record was removed or changed" in c["cleaning"]["admissions_note"]


# ========================================================================= duplicates
def test_duplicates_monthly_matches_saved_table(client):
    got = client.get("/api/quality/duplicates").json()["monthly"]
    ref = pd.read_csv(OUT / "duplicates_by_month.csv")
    assert [g["month"] for g in got] == ref.month.tolist() and len(got) == 24
    for g, (_, r) in zip(got, ref.iterrows()):
        assert (g["rows"], g["exact_extra"], g["mrd_extra"], g["near_extra"]) == (r.rows, r.exact_extra, r.mrd_doa_extra, r.near_extra)
        assert g["exact_rate"] * 100 == pytest.approx(r.exact_extra_pct, abs=0.06)
        assert g["mrd_rate"] * 100 == pytest.approx(r.mrd_doa_extra_pct, abs=0.06)


def test_duplicates_daily_series_separate_and_exact(client, daily):
    d = client.get("/api/quality/duplicates").json()
    assert d["scored_series"] == "raw" and d["provenance"] == "sensitivity"
    rows = pd.DataFrame(d["daily"]).set_index("date")
    assert len(rows) == 730
    assert rows.raw.tolist() == daily.admissions.tolist()
    assert rows.dedup_exact.tolist() == daily.admissions_dedup_exact.tolist()
    assert rows.dedup_mrd.tolist() == daily.admissions_dedup_mrd_doa.tolist()
    assert (rows.extra_exact == rows.raw - rows.dedup_exact).all() and (rows.extra_mrd == rows.raw - rows.dedup_mrd).all()
    assert (rows.extra_exact.values == daily.extra_exact_copies.values).all()
    assert rows.potentially_incomplete.sum() == 1 and rows.potentially_incomplete.iloc[-1]


def test_duplicates_weekday_and_periods(client, daily):
    d = client.get("/api/quality/duplicates").json()
    wd = daily.groupby(daily.index.dayofweek)
    for g, (_, f) in zip(d["weekday"], wd):
        assert g["rows"] == f.admissions.sum()
        assert g["mrd_rate"] == pytest.approx(1 - f.admissions_dedup_mrd_doa.sum() / f.admissions.sum(), abs=6e-4)
    assert [p["label"] for p in d["periods"]] == ["Apr-Dec 2017", "Jan-Dec 2018", "Jan-Mar 2019"]
    assert sum(p["rows"] for p in d["periods"]) == 15757 and sum(p["mrd_extra"] for p in d["periods"]) == 1358
    p19 = daily.loc["2019-01-01":"2019-03-31"]
    assert d["periods"][2]["exact_rate"] == pytest.approx(1 - p19.admissions_dedup_exact.sum() / p19.admissions.sum(), abs=6e-4)


# ========================================================================= completeness & logs
def test_missing_matrix_recomputes(client):
    m = client.get("/api/quality/missing").json()
    e = pd.read_csv(OUT / "clean_daily.csv", parse_dates=["date"]).set_index("date").loc[C.DATA_START:C.DATA_END]
    assert m["period"]["days"] == 730 and len(m["months"]) == 24 and m["admissions"]["missing_days"] == 0
    for g in m["groups"]:
        for v in g["variables"]:
            col = v["variable"]
            assert v["missing"] == int(e[col].isna().sum()), col
            imp = int(e[f"{col}_imputed"].sum()) if f"{col}_imputed" in e else 0
            assert v["imputed"] == imp, col
            assert sum(x["missing"] for x in v["months"]) == v["missing"] and sum(x["days"] for x in v["months"]) == 730
    totals = {v["variable"]: (v["missing"], v["imputed"]) for g in m["groups"] for v in g["variables"]}
    assert sum(i for _, i in totals.values()) == 10 and totals["station_humidity_pct"] == (4, 0)


def test_pollution_log_lists_pollution_entries_only(client, log):
    p = client.get("/api/quality/pollution-log").json()
    pol = log[log.source == "pollution"]
    assert p["entries"] == len(p["rows"]) == len(pol) == 72
    assert all(set(r) == {"date", "field", "original_value", "cleaned_value", "action", "rule", "rule_description", "reason"} for r in p["rows"])
    assert all(r["rule"].startswith("P") for r in p["rows"])
    assert sorted((r["date"], r["field"], r["action"]) for r in p["rows"]) == sorted(zip(pol.date, pol.column, pol.action))
    corrected = next(r for r in p["rows"] if r["action"] == "corrected")
    assert (corrected["field"], corrected["cleaned_value"]) == ("PM10 MAX", "150")


def test_case_mix_recomputes(client, daily):
    c = client.get("/api/quality/case-mix").json()
    m = daily[["admissions", "admissions_opd"]].resample("MS").sum()
    assert [x["opd_share"] for x in c["months"]] == pytest.approx((m.admissions_opd / m.admissions).tolist(), abs=6e-4)
    assert [x["month"] for x in c["months"] if x["anomaly"]] == ["2017-11", "2017-12"]
    assert c["overall"]["opd_share"] == pytest.approx(daily.admissions_opd.sum() / daily.admissions.sum(), abs=6e-4)
    assert c["other_development_months_opd_share_range"] == pytest.approx([0.230, 0.374], abs=1e-3)
    assert "not a forecasting target" in c["note"]


# ========================================================================= read-only & integrity
def test_quality_endpoints_never_run_model_or_write(client, monkeypatch):
    def forbidden(*a, **k):
        raise AssertionError("data-quality endpoints must not run the model")
    monkeypatch.setattr(FR, "run_forecast", forbidden)
    monkeypatch.setattr(FR, "run_explain", forbidden)
    files = [p for p in OUT.rglob("*") if p.is_file()] + [ROOT / "forecast.py"]
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    for u in URLS:
        assert client.get(u).status_code == 200
    assert {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in files} == before
    assert before[ROOT / "forecast.py"] == C.FROZEN_FORECAST_SHA256


def test_phase1_and_phase2_endpoints_unchanged_contract(client):
    m = client.get("/api/meta").json()
    assert "weather" in m["model"]["not_used"] and m["model"]["sha256"] == C.FROZEN_FORECAST_SHA256
    rows = client.get("/api/forecast/historical").json()["forecast"]
    assert all(set(r) == {"date", "mean", "lower_80", "upper_80", "weekday", "holidays"} for r in rows)


# ========================================================================= frontend wording & route
def test_frontend_page_wording_and_route():
    src = re.sub(r"\s+", " ", PAGE.read_text(encoding="utf-8"))
    for phrase in ("Nothing has been removed", "sensitivity", "potentially incomplete", "no patient-level information",
                   "not used by the forecast model", "context only"):
        assert phrase.lower() in src.lower(), phrase
    for m in re.finditer(r"incomplete", src, re.I):
        assert re.search(r"potentially[ _-]$", src[max(0, m.start() - 12):m.start()].lower()), src[m.start() - 40:m.start() + 20]
    for banned in ("bed", "staff", "occupancy", "capacity"):
        assert not re.search(rf"\b{banned}", src, re.I), banned
    router = (ROOT / "dashboard" / "frontend" / "src" / "lib" / "router.tsx").read_text(encoding="utf-8")
    assert re.search(r'path: "/quality".*?ready: true', router, re.S)
    for other in ("/conditions", "/calendar", "/insights"):          # built in Stages 2-3
        assert re.search(rf'path: "{other}"[^\n]*\n?[^\n]*ready: true', router), other
