"""Stage A results for the internal Model Performance area (read-only).

Serves the saved multi-variable model comparison from experiments/multivar/results/. Nothing is trained or
re-scored here, and nothing in outputs/ is touched. Client-facing pages do not use this endpoint.
"""
from __future__ import annotations

import json
from typing import Callable

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException

from . import config as C

RESULTS = C.ROOT / "experiments" / "multivar" / "results"
FAMILY = {"GLMH": "Poisson GLM + recent history (numpy)", "LGBM": "LightGBM (Poisson)", "XGB": "XGBoost (Poisson)",
          "RF": "Random forest", "M4": "M4 Poisson GLM (frozen forecast.py)", "B0": "Seasonal naive (harness baseline)"}
FSET = {"H": "History + calendar", "HE": "History + calendar + environment known at the cut-off (deployable)",
        "HX": "History + calendar + ACTUAL environment on the target date (oracle upper bound, not deployable)"}


def _label(model: str) -> dict:
    if model.startswith("M4_"):
        return {"family": FAMILY["M4"], "features": "Weekday, holidays, annual seasonality", "kind": "benchmark"}
    if model.startswith("B0_"):
        return {"family": FAMILY["B0"], "features": "Same weekday last week", "kind": "baseline"}
    fam, fs = model.split("_")
    return {"family": FAMILY[fam], "features": FSET[fs], "kind": "oracle" if fs == "HX" else "candidate"}


def load_comparison() -> dict:
    p, d = RESULTS / "comparison_validation.csv", RESULTS / "decision.json"
    if not (p.exists() and d.exists()):
        raise FileNotFoundError("Stage A results not found; run experiments/multivar/run_stage_a.py")
    t = pd.read_csv(p)
    rows = []
    for r in t.to_dict("records"):
        ci = r.get("ci_vs_M4")
        ci = json.loads(ci.replace("(", "[").replace(")", "]")) if isinstance(ci, str) else None
        rows.append({**{k: (None if isinstance(v, float) and not np.isfinite(v) else v) for k, v in r.items() if k != "ci_vs_M4"},
                     "ci_vs_M4": ci, **_label(r["model"])})
    dec = json.loads(d.read_text())
    return {"provenance": "evaluation", "validation_period": "Cut-offs 2018-07-01 .. 2018-12-15 (42 x 14 days)",
            "tuning_period": "Cut-offs 2017-07-01 .. 2018-06-17 (interval quantile choice only)",
            "selection_rule": ("Lowest validation MAE among deployable candidates replaces M4 only if its block-bootstrap 95% CI of "
                               "the MAE ratio vs M4 is entirely below 1 and its validation 80% coverage is at least 80%. Oracle "
                               "(actual future environment) variants are upper bounds and cannot be selected."),
            "rows": rows, **dec}


def build_router(safe: Callable) -> APIRouter:
    router = APIRouter(prefix="/api/ml", tags=["model comparison (internal)"])

    @router.get("/comparison")
    def comparison():
        try:
            return safe(load_comparison())
        except FileNotFoundError as e:
            raise HTTPException(503, str(e)) from e

    return router
