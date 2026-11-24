# CardioPulse - PS-06 Cardiac Admissions Intelligence (dashboard)

Redesign Stage 1: a 14-day admission forecasting product. Primary pages: Command Centre and 14-Day Forecast
(built), Weather & AQI (Stage 2), Calendar and Prediction Insights (Stage 3). Secondary: Model performance (the
Phase 2 retrospective explorer and holdout evaluation) and Data quality (Phase 3A). Old links (`#/environment`,
`#/workload`, `?view=retro|holdout#/forecast`) redirect to their replacements.

The admission forecast is a **historical hackathon demonstration** for 1-14 Apr 2019 (cut-off 2019-04-01, the day
after the data end), served by `GET /api/run?as_of=2019-04-01` from the unchanged `forecast.py`. Weather and air
quality are not model inputs; live forecasts for the current period (Stage 2) will be shown separately.

Everything here is **read-only** with respect to the project: it reads aggregate outputs under `outputs/`
and runs the frozen root `forecast.py` in an isolated subprocess (hash-checked). Nothing in the forecasting
submission depends on this folder.

## Stage 2: live weather & air quality (context only)

`GET /api/conditions` returns live Open-Meteo forecasts for Ludhiana (30.901 N, 75.8573 E, Asia/Kolkata):
weather (daily, up to 14 days from today) and air quality (hourly CAMS data aggregated to daily US AQI maximum and
pollutant means; about 4-5 days). The backend is the only component that calls Open-Meteo; requests carry only
coordinates, timezone and variable names. Responses are validated (units, ranges, array lengths), cached for 60 min in
`dashboard/backend/.cache/` (git-ignored), served as "stale" for up to 24 h if a refresh fails, and otherwise reported
as unavailable. Missing dates are never filled. US AQI only - not India's National AQI. The admissions model does not
use any of this data; `provider/` and `conditions.py` are not imported by the forecasting code.
Attribution: Open-Meteo.com (CC BY 4.0); air quality based on Copernicus Atmosphere Monitoring Service information.

## Run

```bash
# 1. API (from dashboard/backend; uses already-installed FastAPI/uvicorn)
cd dashboard/backend
PYTHONDONTWRITEBYTECODE=1 python -m uvicorn app.main:app --host 127.0.0.1 --port 8000

# 2a. Frontend, development (from dashboard/frontend)
cd dashboard/frontend
npm ci            # exact versions from package-lock.json
npm run dev       # http://127.0.0.1:5173  (proxies /api to :8000)

# 2b. or production build + preview
npm run build && npm run preview   # http://127.0.0.1:4173
```

Theme: toggle in the top bar, or append `?theme=light` / `?theme=dark` to the URL.

## Checks

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest dashboard/backend/tests -p no:cacheprovider   # API, PII guard, parity, integrity
python dashboard/tools/color_check.py                                                     # WCAG contrast + colour-vision separation
cd dashboard/frontend && npm run build                                                    # type-check + production build
```

## Data rules enforced

* Aggregate daily counts only; patient-level files are never loaded and every response passes a PII guard.
* Raw records (the forecast target) and de-duplicated counts (sensitivity) are separate series, never mixed.
* The default forecast is a **historical** forecast (cut-off 1 Apr 2019 -> 1-14 Apr 2019); it contains no actual outcomes.
* 31 Mar 2019 is labelled **potentially incomplete**.
* Data Quality (`/api/quality/*`): reads `outputs/cleaning_log.csv` through a column whitelist that never includes the
  per-record reference column. Admission-file entries are returned only as counts per rule (with fixed generic
  descriptions); only pollution-station entries are listed, with values renamed `original_value` / `cleaned_value`.
  Repeats are measured, never removed. None of these endpoints run the forecasting model.

## Known dependency advisories

`npm audit` reports two advisories for Vite 5.4.21 / esbuild (dev-server path traversal and request forwarding).
They are fixed only in a newer Vite major and affect only the local dev/preview servers, not the built files.
Mitigation: both servers are bound to `127.0.0.1` with `strictPort`. React Router was removed (a minimal hash
router is used instead) to avoid its open-redirect advisories.
