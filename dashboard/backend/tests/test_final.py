"""Final verification: official forecast.py contract under harness emulation, dashboard parity, client/internal separation.

The harness emulation reuses validation/test_forecast.py (files truncated to days < cut-off, fresh process,
PS06_DATA_DIR = working folder). Runs at Jan-Mar 2019 cut-offs only check the output contract; nothing is scored.
Run from the project root:
    PYTHONDONTWRITEBYTECODE=1 python -m pytest dashboard/backend/tests/test_final.py -v -p no:cacheprovider
"""
import ast
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

BACKEND = Path(__file__).resolve().parents[1]
ROOT = BACKEND.parents[1]
sys.path.insert(0, str(BACKEND))
sys.path.insert(0, str(ROOT / "validation"))

from fastapi.testclient import TestClient  # noqa: E402

from app import config as C  # noqa: E402
from app.main import app  # noqa: E402
from test_forecast import _harness_folder, _run_in_fresh_process  # noqa: E402

SRC = ROOT / "dashboard" / "frontend" / "src"
FROZEN = "aa4809046af56166ceed6f3daab1521be093878a7d9c0ffaa9d93b784462c866"
CUTOFFS = ["2019-01-15", "2019-02-05", "2019-02-26", "2019-03-19", "2019-04-01"]


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def harness_runs(tmp_path_factory):
    out = {}
    for cut in CUTOFFS:
        folder = _harness_folder(tmp_path_factory.mktemp(f"h{cut}"), cut)
        out[cut] = _run_in_fresh_process(folder, cut)
    return out


# ============================================================================ official submission file
def test_forecast_py_is_frozen_single_file():
    p = ROOT / "forecast.py"
    assert hashlib.sha256(p.read_bytes()).hexdigest() == FROZEN == C.FROZEN_FORECAST_SHA256
    tree = ast.parse(p.read_text(encoding="utf-8"))
    mods = {a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    mods |= {n.module.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
    assert mods <= {"__future__", "os", "re", "sys", "numpy", "pandas"}, mods      # no local modules, no network, no ML packages
    fn = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "forecast"]
    assert len(fn) == 1 and [a.arg for a in fn[0].args.args] == ["cutoff_date"]


def test_forecast_py_reads_only_harness_files():
    src = (ROOT / "forecast.py").read_text(encoding="utf-8")
    code = src.split('"""', 2)[2]                                                     # skip the module docstring
    assert 'os.environ["PS06_DATA_DIR"]' in code or "PS06_DATA_DIR" in code
    assert re.findall(r'^(ADMISSION_FILE|FESTIVAL_FILE)\s*=', code, re.M) == ["ADMISSION_FILE", "FESTIVAL_FILE"]
    for forbidden in ("to_csv(", "open(", "urllib", "requests", "httpx", "socket", "weather_ludhiana", "Pollution Data"):
        assert forbidden not in code, forbidden


@pytest.mark.parametrize("cut", CUTOFFS)
def test_harness_output_contract(harness_runs, cut):
    df, stderr, seconds = harness_runs[cut]
    assert list(df.columns) == ["date", "mean", "lower_80", "upper_80"]                # exactly four, in order
    assert len(df) == 14
    assert list(df["date"]) == pd.date_range(cut, periods=14).strftime("%Y-%m-%d").tolist()
    v = df[["mean", "lower_80", "upper_80"]].to_numpy(float)
    assert np.isfinite(v).all() and (v[:, 1] <= v[:, 0]).all() and (v[:, 0] <= v[:, 2]).all() and (v >= 0).all()
    assert "FALLBACK" not in stderr and seconds < 900


def test_dashboard_serves_the_harness_output(client, harness_runs):
    df = harness_runs["2019-04-01"][0]
    run = client.get("/api/run").json()
    assert [r["date"] for r in run["official"]] == list(df["date"])
    for col in ("mean", "lower_80", "upper_80"):
        assert np.allclose([r[col] for r in run["official"]], df[col], atol=6e-4), col
    assert run["manifest"]["sha256_matches_frozen"] is True and run["mode"] == "demonstration"


def test_dashboard_refuses_current_dates(client):
    r = client.get("/api/run?as_of=2026-11-24")
    assert r.status_code == 422 and "Demonstration mode" in r.json()["detail"]


# ============================================================================ client / internal separation
CLIENT_FILES = ["pages/CommandCentre.tsx", "pages/Forecast14.tsx", "pages/Conditions.tsx", "components/charts/EnvCharts.tsx"]
METRIC = re.compile(r"\b(MAE|RMSE|WAPE|bias|coverage|validation|holdout|backtest|bootstrap|ReliabilityCard|IntervalMeaning|"
                    r"BiasDisclosure|MlComparison|CoverageHistory|/performance)\b", re.I)


@pytest.mark.parametrize("f", CLIENT_FILES)
def test_client_pages_expose_no_diagnostics_or_internal_links(f):
    src = (SRC / f).read_text(encoding="utf-8")
    hits = [m.group(0) for m in METRIC.finditer(src)]
    assert hits == [], (f, hits)


def test_client_components_used_by_client_pages_are_metric_free():
    rp = (SRC / "components" / "forecast" / "RunParts.tsx").read_text(encoding="utf-8")
    for name in ("RunBanner", "ReadingGuide", "DaysToWatch", "UpcomingHolidays", "IndicatorExplainer", "ForecastStrip", "OfficialTable", "NotModelInputs"):
        start = rp.index(f"export function {name}")
        end = rp.find("\nexport function", start + 1)
        body = rp[start:end if end > 0 else None]
        assert not re.search(r"\b(MAE|RMSE|WAPE|coverage80|holdout|/performance)\b", body), name


def test_overview_text_on_client_command_centre_has_no_metrics(client):
    o = client.get("/api/overview").json()
    text = json.dumps([o["alerts"], o["insights"]])
    assert not re.search(r"MAE|RMSE|WAPE|coverage|ratio", text, re.I)
    bias = next(a for a in o["alerts"] if a["id"] == "model_bias")
    assert "previously viewed" in bias["detail"]


def test_internal_page_keeps_diagnostics_and_previously_viewed_label():
    mp = re.sub(r"\s+", " ", (SRC / "pages" / "ModelPerformance.tsx").read_text(encoding="utf-8"))
    for part in ("<MlComparisonPanel />", "<InternalReliability />", "<IntervalMeaning />", "<BiasDisclosure />", "previously viewed",
                 "function HoldoutView", "function RetroView"):
        assert part in mp, part
    side = (SRC / "components" / "layout" / "Sidebar.tsx").read_text(encoding="utf-8")
    assert "Internal · data & model" in side


def test_meta_holdout_label_previously_viewed(client):
    h = [s for s in client.get("/api/meta").json()["model"]["interval"]["coverage_history"] if s["stage"] == "holdout"][0]
    assert "previously viewed" in h["label"] and "Untouched" not in h["label"]


def test_client_wording_demonstration_and_context_only():
    cc = re.sub(r"\s+", " ", (SRC / "pages" / "CommandCentre.tsx").read_text(encoding="utf-8"))
    rp = re.sub(r"\s+", " ", (SRC / "components" / "forecast" / "RunParts.tsx").read_text(encoding="utf-8"))
    cond = re.sub(r"\s+", " ", (SRC / "pages" / "Conditions.tsx").read_text(encoding="utf-8"))
    assert "not a live hospital forecast" in rp and "They are not a forecast for today" in rp
    assert "never feed into the April 2019 admission forecast" in cc
    assert "not used by the admissions model" in cond and "US AQI" in cond
