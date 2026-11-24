# Stage A – multi-variable model comparison

Separate experiment; nothing here changes `forecast.py` or anything in `outputs/`.

```bash
# from the project root
PYTHONDONTWRITEBYTECODE=1 python experiments/multivar/run_stage_a.py              # ~6 min on 14 workers
PYTHONDONTWRITEBYTECODE=1 python -m pytest experiments/multivar -q -p no:cacheprovider   # leakage tests
```

* Data: `outputs/clean_daily.csv` (raw admissions target, Open-Meteo weather, station pollution) + `festival_calendar_punjab.csv`.
* Cut-offs: the Step 4 grid (tuning <= 2018-06-17, validation 2018-07-01 .. 2018-12-15); same metric/interval/bootstrap code (`src/backtest.py`).
* Models: Poisson GLM + recent history (numpy), LightGBM, XGBoost, Random Forest; fixed hyper-parameters.
* Feature sets: `H` history + calendar; `HE` + environment observed in the 7 days before the cut-off (deployable);
  `HX` + actual environment on the target date (ORACLE upper bound – never selectable).
* Pre-registered selection rule: see the docstring of `run_stage_a.py`.

Result (`results/decision.json`): no candidate met the rule; **M4 (frozen `forecast.py`) is kept**.
Best deployable challenger GLMH_H: validation MAE 4.79 vs M4 4.50 (ratio 1.07, 95% CI 0.98–1.17).
Environment known at the cut-off did not help any model; oracle environment did not beat M4 either.
Archived weather/AQI forecasts for 2017–19 do not exist, so forecast-driven environmental models cannot be validated.
Jan–Mar 2019 (already viewed in Step 5) was not used for any new model.
