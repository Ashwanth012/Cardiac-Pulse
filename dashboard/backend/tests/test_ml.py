"""Stage A multi-variable comparison: served values, the pre-registered decision, and honesty constraints.

Run from the project root:
    PYTHONDONTWRITEBYTECODE=1 python -m pytest dashboard/backend/tests -v -p no:cacheprovider
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

BACKEND = Path(__file__).resolve().parents[1]
ROOT = BACKEND.parents[1]
sys.path.insert(0, str(BACKEND))
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient  # noqa: E402

from app import config as C  # noqa: E402
from app import forecast_runner as FR  # noqa: E402
from app.datastore import assert_no_pii  # noqa: E402
from app.main import app  # noqa: E402
from src import backtest as B  # noqa: E402

RES = ROOT / "experiments" / "multivar" / "results"
SRC = ROOT / "dashboard" / "frontend" / "src"


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def comp(client):
    r = client.get("/api/ml/comparison")
    assert r.status_code == 200, r.text
    return r.json()


@pytest.fixture(scope="module")
def fc():
    return pd.read_csv(RES / "forecasts_dev_grid.csv.gz", parse_dates=["cutoff", "date"])


def test_endpoint_pii_free_and_get_only(client, comp):
    assert_no_pii(comp)
    assert client.post("/api/ml/comparison").status_code == 405


def test_rows_equal_saved_results(comp):
    t = pd.read_csv(RES / "comparison_validation.csv")
    assert [r["model"] for r in comp["rows"]] == t.model.tolist() and len(t) == 14
    for r, (_, s) in zip(comp["rows"], t.iterrows()):
        assert r["val_MAE"] == pytest.approx(s.val_MAE, abs=1e-3) and r["val_coverage80"] == pytest.approx(s.val_coverage80, abs=1e-3)


def test_m4_reference_equals_step4_validation(fc):
    val = fc[fc.cutoff >= "2018-07-01"]
    m4 = B.metrics(val[val.model == "M4_glm_Nall_K2_T0_L0"])
    saved = pd.read_csv(ROOT / "outputs" / "backtest" / "tables" / "validation_family_winners.csv", index_col=0).loc["M4"]
    assert m4["MAE"] == pytest.approx(saved.MAE, abs=1e-3) == pytest.approx(4.497, abs=1e-3)
    assert m4["coverage80"] == pytest.approx(saved.coverage80, abs=1e-3) and m4["n_cutoffs"] == 42 and m4["n_points"] == 588


def test_decision_recomputed_from_forecasts(fc, comp):
    val = fc[fc.cutoff >= "2018-07-01"]
    deploy = [m for m in val.model.unique() if re.match(r"^(GLMH|LGBM|XGB|RF)_(H|HE)$", m)]
    assert len(deploy) == 8
    maes = {m: B.metrics(val[val.model == m])["MAE"] for m in deploy}
    challenger = min(maes, key=maes.get)
    point, lo, hi = B.block_bootstrap_ratio(val, challenger, "M4_glm_Nall_K2_T0_L0", val.cutoff.unique())
    cov = B.metrics(val[val.model == challenger])["coverage80"]
    passed = hi < 1 and cov >= 0.80
    d = comp["decision"]
    assert d["challenger"] == challenger and d["rule_passed"] is passed is False
    assert d["selected"] == "M4_glm_Nall_K2_T0_L0"
    assert d["ratio_ci_95"] == pytest.approx([lo, hi], abs=1e-3) and hi >= 1


def test_oracle_rows_labelled_and_never_selectable(comp):
    kinds = {r["model"]: r["kind"] for r in comp["rows"]}
    assert all(kinds[m] == "oracle" for m in kinds if m.endswith("_HX"))
    assert not comp["decision"]["challenger"].endswith("_HX")
    assert "oracle" in comp["selection_rule"].lower() and "cannot be selected" in comp["selection_rule"]


def test_no_new_model_scored_on_previously_viewed_period(comp):
    assert comp["jan_mar_2019_previously_viewed"] is None
    assert not (RES / "forecasts_jan_mar_2019_selected.csv").exists()
    fcd = pd.read_csv(RES / "forecasts_dev_grid.csv.gz", parse_dates=["cutoff"])
    assert fcd.cutoff.max() <= pd.Timestamp("2018-12-15")


def test_experiment_leakage_guard_and_grid(fc):
    cand = fc[fc.model == "GLMH_H"]
    assert sorted(cand.cutoff.unique()) == sorted(B.make_cutoffs())
    assert (cand.date >= cand.cutoff).all() and (cand.date <= cand.cutoff + pd.Timedelta(days=13)).all()


def test_selected_model_serves_the_official_forecast(client):
    run = client.get("/api/run").json()
    direct = FR.run_forecast(C.HISTORICAL_CUTOFF)["rows"]
    assert [o["mean"] for o in run["official"]] == pytest.approx([r["mean"] for r in direct], abs=6e-4)
    assert run["manifest"]["sha256_matches_frozen"] is True


def test_client_page_wording_and_internal_labels():
    f14 = re.sub(r"\s+", " ", (SRC / "pages" / "Forecast14.tsx").read_text(encoding="utf-8"))
    assert "none forecast admissions more accurately" in f14 and "<NotModelInputs" in f14
    mlc = (SRC / "components" / "forecast" / "MlComparison.tsx").read_text(encoding="utf-8")
    assert "Oracle – not deployable" in mlc and "cannot be validated honestly" in re.sub(r"\s+", " ", mlc)
    mp = (SRC / "pages" / "ModelPerformance.tsx").read_text(encoding="utf-8")
    assert "<MlComparisonPanel />" in mp
