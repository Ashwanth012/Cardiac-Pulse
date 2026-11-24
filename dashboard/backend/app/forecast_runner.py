"""Runs the frozen root-level forecast.py in an isolated subprocess (never imported into the API process).

* Refuses to run if forecast.py's SHA-256 differs from the frozen hash.
* `python -I -B`: isolated mode (project not on sys.path), no bytecode written next to forecast.py.
* PS06_DATA_DIR = project root; forecast.py itself uses only admission rows dated before the cut-off.
* Output contract is re-validated here; stderr FALLBACK lines become a `fallback_used` flag.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
import sys
import threading
from datetime import date, timedelta

from . import config as C

_RUNNER = (
    "import importlib.util, sys\n"
    "spec = importlib.util.spec_from_file_location('ps06_submission', sys.argv[1])\n"
    "m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)\n"
    "df = m.forecast(sys.argv[2])\n"
    "sys.stdout.write(df.to_json(orient='records', double_precision=15))\n"
)
_cache: dict[str, dict] = {}
_lock = threading.Lock()


class ForecastError(RuntimeError):
    pass


def forecast_sha256() -> str:
    return hashlib.sha256(C.FORECAST_PY.read_bytes()).hexdigest()


def validate_rows(rows: list, cutoff: str) -> list:
    if not isinstance(rows, list) or len(rows) != 14:
        raise ForecastError(f"expected 14 rows, got {len(rows) if isinstance(rows, list) else type(rows)}")
    start = date.fromisoformat(cutoff)
    out = []
    for i, r in enumerate(rows):
        if set(r) != {"date", "mean", "lower_80", "upper_80"}:
            raise ForecastError(f"unexpected columns {sorted(r)}")
        if r["date"] != (start + timedelta(days=i)).isoformat():
            raise ForecastError(f"row {i}: date {r['date']} is not cutoff+{i}")
        lo, m, hi = (float(r[k]) for k in ("lower_80", "mean", "upper_80"))
        if not all(math.isfinite(v) for v in (lo, m, hi)) or not (lo <= m <= hi):
            raise ForecastError(f"row {i}: invalid values {lo}, {m}, {hi}")
        out.append({"date": r["date"], "mean": m, "lower_80": lo, "upper_80": hi})
    return out


def run_forecast(cutoff: str, timeout: float = C.FORECAST_TIMEOUT_S) -> dict:
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", cutoff or ""):
        raise ForecastError("cutoff must be YYYY-MM-DD")
    date.fromisoformat(cutoff)
    with _lock:
        if cutoff in _cache:
            return _cache[cutoff]
    sha = forecast_sha256()
    if sha != C.FROZEN_FORECAST_SHA256:
        raise ForecastError("forecast.py differs from the frozen submission; refusing to run")
    env = {k: v for k, v in os.environ.items() if not k.upper().startswith("PYTHON")}
    env["PS06_DATA_DIR"] = str(C.ROOT)
    try:
        p = subprocess.run([sys.executable, "-I", "-B", "-c", _RUNNER, str(C.FORECAST_PY), cutoff],
                           cwd=str(C.ROOT), env=env, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as e:
        raise ForecastError(f"forecast.py timed out after {timeout:.0f} s") from e
    if p.returncode != 0:
        raise ForecastError("forecast.py exited with an error")
    rows = validate_rows(json.loads(p.stdout), cutoff)
    fallback_messages = [ln.split("FALLBACK:", 1)[1].strip() for ln in p.stderr.splitlines() if "FALLBACK:" in ln]
    result = {"cutoff": cutoff, "rows": rows, "fallback_used": bool(fallback_messages),
              "fallback_messages": fallback_messages, "forecast_py_sha256": sha}
    with _lock:
        _cache[cutoff] = result
    return result


def clear_cache():
    with _lock:
        _cache.clear()
        _explain_cache.clear()


# ------------------------------------------------------------------ model explanation
# Uses forecast.py's OWN functions (load_counts, load_calendar, design, poisson_irls) to report the fitted
# coefficients behind a forecast. Nothing is modified or retrained differently: this is the same fit
# forecast() performs, and `mean_check` must reproduce forecast()'s mean (verified by the API and tests).
_EXPLAIN = (
    "import importlib.util, sys, json\n"
    "import numpy as np, pandas as pd\n"
    "spec = importlib.util.spec_from_file_location('ps06_submission', sys.argv[1])\n"
    "m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)\n"
    "cutoff = pd.Timestamp(sys.argv[2]); root = sys.argv[3]\n"
    "y = m.load_counts(root, cutoff); cal = m.load_calendar(root, cutoff, y.index.min())\n"
    "fourier = m.M4_FOURIER if (y.index.max() - y.index.min()).days >= m.M4_MIN_SPAN_FOR_FOURIER else 0\n"
    "b = m.poisson_irls(m.design(y.index, cal, fourier), y.to_numpy(dtype=float))\n"
    "dates = pd.date_range(cutoff, periods=m.H, freq='D')\n"
    "mean = np.exp(np.clip(m.design(dates, cal, fourier) @ b, -20, 20))\n"
    "sys.stdout.write(json.dumps({'coef': b.tolist(), 'fourier': int(fourier), 'ridge': m.M4_RIDGE,\n"
    "  'train_start': str(y.index.min().date()), 'train_end': str(y.index.max().date()), 'train_days': int(len(y)),\n"
    "  'mean_check': mean.tolist()}))\n"
)
_explain_cache: dict[str, dict] = {}


def run_explain(cutoff: str, timeout: float = C.FORECAST_TIMEOUT_S) -> dict:
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", cutoff or ""):
        raise ForecastError("cutoff must be YYYY-MM-DD")
    date.fromisoformat(cutoff)
    with _lock:
        if cutoff in _explain_cache:
            return _explain_cache[cutoff]
    if forecast_sha256() != C.FROZEN_FORECAST_SHA256:
        raise ForecastError("forecast.py differs from the frozen submission; refusing to run")
    env = {k: v for k, v in os.environ.items() if not k.upper().startswith("PYTHON")}
    try:
        p = subprocess.run([sys.executable, "-I", "-B", "-c", _EXPLAIN, str(C.FORECAST_PY), cutoff, str(C.ROOT)],
                           cwd=str(C.ROOT), env=env, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as e:
        raise ForecastError(f"model explanation timed out after {timeout:.0f} s") from e
    if p.returncode != 0:
        raise ForecastError("model explanation failed")
    out = json.loads(p.stdout)
    fc = run_forecast(cutoff)
    if max(abs(a - r["mean"]) for a, r in zip(out["mean_check"], fc["rows"])) > 1e-6:
        raise ForecastError("explanation does not reproduce forecast.py's mean; refusing to report it")
    with _lock:
        _explain_cache[cutoff] = out
    return out
