"""Static configuration. All paths point at existing project files, which are only ever read."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]          # project root
OUTPUTS = ROOT / "outputs"
FORECAST_PY = ROOT / "forecast.py"
FROZEN_FORECAST_SHA256 = "aa4809046af56166ceed6f3daab1521be093878a7d9c0ffaa9d93b784462c866"

DATA_START = "2017-04-01"
DATA_END = "2019-03-31"
POTENTIALLY_INCOMPLETE_DAY = "2019-03-31"
HISTORICAL_CUTOFF = "2019-04-01"                    # forecasts 2019-04-01 .. 2019-04-14 (historical, not live)
FORECAST_TIMEOUT_S = 60

# Retrospective explorer: development-period cut-offs only. The last allowed cut-off's 14-day window ends
# 2018-12-28, so no explored forecast can reach the Jan-Mar 2019 holdout.
RETRO_MIN_CUTOFF = "2017-07-01"
RETRO_MAX_CUTOFF = "2018-12-15"
HOLDOUT_START = "2019-01-01"

ALLOWED_ORIGINS = [
    "http://localhost:5173", "http://127.0.0.1:5173",   # vite dev
    "http://localhost:4173", "http://127.0.0.1:4173",   # vite preview
]
