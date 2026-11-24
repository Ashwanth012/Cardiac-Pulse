"""Phase 2 wording clarification: the 80% interval (target) vs the central 85% residual-quantile range (construction).

Every value is recomputed independently from saved outputs; wording is checked in the API and the frontend source.

Run from the project root:
    PYTHONDONTWRITEBYTECODE=1 python -m pytest dashboard/backend/tests -v -p no:cacheprovider
"""
import hashlib
import sys
from pathlib import Path

import pandas as pd
import pytest

BACKEND = Path(__file__).resolve().parents[1]
ROOT = BACKEND.parents[1]
sys.path.insert(0, str(BACKEND))

from fastapi.testclient import TestClient  # noqa: E402

from app import config as C  # noqa: E402
from app.main import app  # noqa: E402

# Stage 1 redesign: the Phase 2 Studio was split; the interval wording now lives in a shared component used by the
# 14-Day Forecast and Model Performance pages, and the holdout miss breakdown in the Model Performance page.
STUDIO_FILES = [ROOT / "dashboard" / "frontend" / "src" / "components" / "forecast" / "ForecastNotes.tsx",
                ROOT / "dashboard" / "frontend" / "src" / "pages" / "ModelPerformance.tsx"]


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def expected():
    cal = pd.read_csv(ROOT / "outputs" / "backtest" / "tables" / "interval_calibration_tuning.csv")
    m4 = cal[cal.model == "M4_glm_Nall_K2_T0_L0"]
    tune = m4[(m4.q_lo == 0.075) & (m4.pool == 90)].tune_coverage.iloc[0]
    val = pd.read_csv(ROOT / "outputs" / "backtest" / "tables" / "validation_family_winners.csv", index_col=0).loc["M4", "coverage80"]
    p = pd.read_csv(ROOT / "outputs" / "holdout" / "tables" / "forecasts_primary_grid.csv")
    return {"tuning": tune, "validation": val, "holdout": ((p.y >= p.lo) & (p.y <= p.hi)).mean(),
            "above": (p.y > p.hi).mean(), "below": (p.y < p.lo).mean(), "points": len(p),
            "nominal80": (m4[m4.q_lo == 0.10].tune_coverage.min(), m4[m4.q_lo == 0.10].tune_coverage.max())}


def test_forecast_py_quantiles_unchanged():
    src = (ROOT / "forecast.py").read_text(encoding="utf-8")
    assert "Q_LO, Q_HI = 0.075, 0.925" in src and "POOL = 90" in src
    assert hashlib.sha256((ROOT / "forecast.py").read_bytes()).hexdigest() == C.FROZEN_FORECAST_SHA256


def test_meta_interval_construction_vs_target(client):
    iv = client.get("/api/meta").json()["model"]["interval"]
    assert iv["level"] == "80%" and iv["target_coverage"] == 0.8
    assert iv["residual_quantiles"] == {"lower": 0.075, "upper": 0.925, "central_range": 0.85}
    assert iv["calibration_origins"] == 90
    for phrase in ("central 85%", "at least 80%", "80% prediction interval"):
        assert phrase in iv["meaning"] + iv["construction"] + iv["selection"], phrase
    assert "not a guarantee" in iv["selection"] and "development" in iv["selection"]


def test_coverage_history_values(client, expected):
    iv = client.get("/api/meta").json()["model"]["interval"]
    hist = {h["stage"]: h["coverage"] for h in iv["coverage_history"]}
    assert list(hist) == ["tuning", "validation", "holdout"]
    assert hist["tuning"] == pytest.approx(expected["tuning"], abs=6e-4) == pytest.approx(0.834, abs=6e-4)
    assert hist["validation"] == pytest.approx(expected["validation"], abs=6e-4) == pytest.approx(0.923, abs=6e-4)
    assert hist["holdout"] == pytest.approx(expected["holdout"], abs=6e-4) == pytest.approx(0.842, abs=6e-4)
    lo, hi = iv["nominal_80_tuning_coverage_range"]
    assert (lo, hi) == pytest.approx(expected["nominal80"], abs=6e-4) and hi < 0.80


def test_holdout_miss_breakdown(client, expected):
    for m in (client.get("/api/meta").json()["model"]["interval"]["holdout_misses"],
              client.get("/api/evaluation/holdout").json()["interval_evidence"]["holdout_misses"]):
        assert m["above_upper"] == pytest.approx(expected["above"], abs=6e-4) == pytest.approx(0.139, abs=6e-4)
        assert m["below_lower"] == pytest.approx(expected["below"], abs=6e-4) == pytest.approx(0.019, abs=6e-4)
        assert m["above_upper"] + m["below_lower"] + m["inside"] == pytest.approx(1.0, abs=2e-3)
        assert m["points"] == expected["points"] == 266
        assert m["source"].endswith("forecasts_primary_grid.csv")


def test_frontend_wording():
    src = "\n".join(f.read_text(encoding="utf-8") for f in STUDIO_FILES)
    for phrase in ("80% prediction interval", "at least 80%", "central 85%", "7.5th and 92.5th percentiles",
                   "not a guarantee", "development data", "above</strong>", "below</strong>"):
        assert phrase in src, phrase
    assert "8 of every 10 days" not in src and "expects about" not in src
    # all coverage numbers come from the API, none hard-coded in the explanation
    for hard in ("83.4%", "92.3%", "84.2%", "13.9%", "1.9%"):
        assert hard not in src, hard
