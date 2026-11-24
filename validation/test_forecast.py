"""Tests for the scored submission forecast.py (Build step 5.2).

Covers: output contract, interval ordering, parity with the Step 4 M4 backtest, cut-off leakage,
calibration-origin restrictions, harness emulation in a fresh process, fallback behaviour,
dependency/data-access restrictions and integrity of approved files.

No test reads Jan-Mar 2019 admission truth; no score is computed for the holdout.

Run from the project root:
    PYTHONDONTWRITEBYTECODE=1 python -m pytest validation -v -p no:cacheprovider
"""
import ast
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import forecast as FC  # noqa: E402
from src import loaders as L  # noqa: E402
from test_eda import STEP2_SHA256  # noqa: E402
from test_loaders import ORIGINAL_SHA256, sha256  # noqa: E402

M4 = "M4_glm_Nall_K2_T0_L0"
DEV_LAST_INPUT = pd.Timestamp("2018-12-31")      # tests never load admissions on/after this day


@pytest.fixture(scope="module")
def hist():
    """Admission history up to 2018-12-30 and the calendar (DEV only)."""
    y = FC.load_counts(ROOT, DEV_LAST_INPUT)
    cal = FC.load_calendar(ROOT, DEV_LAST_INPUT, y.index.min())
    return y, cal


@pytest.fixture(scope="module")
def archive():
    a = pd.read_csv(os.path.join(ROOT, "outputs", "backtest", "forecasts_selected_models.csv.gz"),
                    parse_dates=["cutoff", "date"])
    return a[a.model == M4]


def _check_contract(df, cutoff):
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["date", "mean", "lower_80", "upper_80"]
    assert len(df) == 14
    exp = pd.date_range(cutoff, periods=14).strftime("%Y-%m-%d").tolist()
    assert df["date"].tolist() == exp and all(isinstance(d, str) for d in df["date"])
    v = df[["mean", "lower_80", "upper_80"]].to_numpy(dtype=float)
    assert np.isfinite(v).all()
    assert (v[:, 1] <= v[:, 0]).all() and (v[:, 0] <= v[:, 2]).all()
    assert (v >= 0).all()


def _harness_folder(tmp, cutoff):
    """Emulate HARNESS.md step 1: files truncated to days < cutoff, discharge blank for in-patients."""
    cutoff = pd.Timestamp(cutoff)
    raw = L.read_raw_csv(os.path.join(ROOT, L.ADMISSION_FILE))
    adm, _ = L.load_admissions(raw=raw)
    keep = (adm["doa"] < cutoff).to_numpy()
    t = raw[keep].copy()
    inpat = (adm.loc[keep, "dod"].isna() | (adm.loc[keep, "dod"] >= cutoff)).to_numpy()
    for c in ["D.O.D", "DURATION OF STAY", "duration of intensive unit stay", "OUTCOME"]:
        t.loc[inpat, c] = ""
    t.to_csv(os.path.join(tmp, L.ADMISSION_FILE), index=False, encoding="utf-8-sig")
    pol = L.read_raw_csv(os.path.join(ROOT, L.POLLUTION_FILE))
    pol[(L.load_pollution(ROOT)["date"] < cutoff).to_numpy()].to_csv(
        os.path.join(tmp, L.POLLUTION_FILE), index=False, encoding="utf-8-sig")
    w = L.read_raw_csv(os.path.join(ROOT, L.WEATHER_FILE))
    w[pd.to_datetime(w["date"]) < cutoff].to_csv(os.path.join(tmp, L.WEATHER_FILE), index=False)
    for f in (L.FESTIVAL_FILE, L.GLOSSARY_FILE, L.MORTALITY_FILE):
        shutil.copy(os.path.join(ROOT, f), tmp)
    return tmp


def _run_in_fresh_process(folder, cutoff):
    code = ("import importlib.util, sys\n"
            f"spec = importlib.util.spec_from_file_location('sub', r'{os.path.join(ROOT, 'forecast.py')}')\n"
            "m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)\n"
            f"df = m.forecast('{cutoff}')\n"
            "sys.stdout.write(df.to_json(orient='split', double_precision=15))\n")
    env = dict(os.environ, PS06_DATA_DIR=str(folder), PYTHONDONTWRITEBYTECODE="1")
    t0 = time.perf_counter()
    p = subprocess.run([sys.executable, "-c", code], cwd=str(folder), env=env, capture_output=True, text=True, timeout=900)
    dt = time.perf_counter() - t0
    assert p.returncode == 0, p.stderr
    from io import StringIO
    return pd.read_json(StringIO(p.stdout), orient="split", dtype={"date": str}), p.stderr, dt


# ============================================================================ contract
class TestContract:
    @pytest.mark.parametrize("cut", ["2017-10-03", "2018-03-15", "2018-08-01", "2018-12-15"])
    def test_output_format_and_ordering(self, hist, cut):
        y, cal = hist
        _check_contract(FC.forecast_from_history(y, cal, pd.Timestamp(cut)), cut)

    def test_entry_point_signature(self):
        import inspect
        sig = inspect.signature(FC.forecast)
        assert list(sig.parameters) == ["cutoff_date"]

    def test_public_forecast_on_dev_cutoff(self, monkeypatch):
        monkeypatch.setenv("PS06_DATA_DIR", ROOT)
        _check_contract(FC.forecast("2018-11-20"), "2018-11-20")

    @pytest.mark.parametrize("bad", ["2019/01/15", "15-01-2019", "2019-1-5", "", "tomorrow"])
    def test_bad_cutoff_string_rejected(self, bad):
        with pytest.raises(ValueError):
            FC.forecast(bad)


# ======================================================================= Step 4 parity
class TestParityWithStep4:
    def test_all_validation_cutoffs_match_backtest(self, hist, archive):
        y, cal = hist
        cuts = sorted(archive[archive.cutoff >= "2018-07-01"].cutoff.unique())
        assert len(cuts) == 42
        for c in cuts:
            out = FC.forecast_from_history(y, cal, pd.Timestamp(c))
            ref = archive[archive.cutoff == c].sort_values("h")
            assert np.allclose(out["mean"], ref["mean"], rtol=0, atol=1e-9), c
            assert np.allclose(out["lower_80"], ref["lo"], rtol=0, atol=1e-9), c
            assert np.allclose(out["upper_80"], ref["hi"], rtol=0, atol=1e-9), c

    def test_tuning_cutoff_means_match(self, hist, archive):
        y, cal = hist
        for c in sorted(archive[archive.cutoff <= "2018-06-14"].cutoff.unique())[::9]:
            out = FC.m4_forecast(y[y.index < c], cal, pd.Timestamp(c))
            assert np.allclose(out, archive[archive.cutoff == c].sort_values("h")["mean"], atol=1e-9)

    def test_counts_match_backtest_datastore(self, hist):
        from src import backtest as B
        y, _ = hist
        s = B.DataStore(ROOT)
        ref = s.counts[s.counts.index < DEV_LAST_INPUT]
        assert y.index.equals(ref.index) and np.array_equal(y.to_numpy(), ref.to_numpy())

    def test_calendar_matches_backtest(self, hist):
        from src import backtest as B
        _, cal = hist
        ref = B.DataStore(ROOT).cal
        assert cal.reindex(ref.index).astype(bool).equals(ref.astype(bool))

    def test_frozen_configuration(self):
        assert (FC.M4_FOURIER, FC.M4_RIDGE, FC.IRLS_ITERS, FC.IRLS_TOL) == (2, 1.0, 50, 1e-8)
        assert (FC.Q_LO, FC.Q_HI, FC.POOL) == (0.075, 0.925, 90)
        assert FC.BUCKETS == ((1, 3), (4, 7), (8, 14))
        assert FC.GRID_ANCHOR == pd.Timestamp("2017-07-01") and FC.GRID_STEP_DAYS == 4
        chosen = pd.read_csv(os.path.join(ROOT, "outputs", "backtest", "tables", "interval_settings_chosen.csv"))
        row = chosen[chosen.model == M4].iloc[0]
        assert (row.q_lo, row.q_hi, int(row.pool)) == (FC.Q_LO, FC.Q_HI, FC.POOL)


# ============================================================================= leakage
class TestLeakage:
    def test_future_history_values_ignored(self, hist):
        y, cal = hist
        c = pd.Timestamp("2018-10-02")
        a = FC.forecast_from_history(y, cal, c)
        y2 = y.copy()
        y2[y2.index >= c] = 999.0
        b = FC.forecast_from_history(y2, cal, c)
        pd.testing.assert_frame_equal(a, b)

    def test_rows_after_cutoff_in_file_are_ignored(self, tmp_path):
        cut = "2018-09-14"
        _harness_folder(tmp_path, cut)
        base, _, _ = _run_in_fresh_process(tmp_path, cut)
        # append fake admissions dated after the cut-off (as if truncation had failed)
        adm = L.read_raw_csv(os.path.join(tmp_path, L.ADMISSION_FILE))
        extra = adm.tail(300).copy()
        extra["D.O.A"], extra["month year"] = "20/09/2018", "Sep-18"
        pd.concat([adm, extra]).to_csv(os.path.join(tmp_path, L.ADMISSION_FILE), index=False, encoding="utf-8-sig")
        leak, _, _ = _run_in_fresh_process(tmp_path, cut)
        pd.testing.assert_frame_equal(base, leak)

    def test_m4_refuses_data_on_or_after_cutoff(self, hist):
        y, cal = hist
        with pytest.raises(AssertionError):
            FC.m4_forecast(y, cal, pd.Timestamp("2018-06-01"))

    def test_weather_and_pollution_never_opened(self, tmp_path):
        _harness_folder(tmp_path, "2018-11-05")
        os.remove(os.path.join(tmp_path, L.POLLUTION_FILE))
        os.remove(os.path.join(tmp_path, L.WEATHER_FILE))
        out, err, _ = _run_in_fresh_process(tmp_path, "2018-11-05")
        _check_contract(out, "2018-11-05")
        assert "FALLBACK" not in err
        src = open(os.path.join(ROOT, "forecast.py"), encoding="utf-8").read()
        assert "Pollution" not in src.split('"""', 2)[2] and "weather_ludhiana" not in src


# ============================================================ calibration origins
class TestCalibrationOrigins:
    @pytest.mark.parametrize("cut", ["2017-08-30", "2018-03-01", "2018-12-15", "2019-02-20"])
    def test_origins_on_grid_window_closed_and_capped(self, cut):
        c = pd.Timestamp(cut)
        o = FC.calibration_origins(c, c - pd.Timedelta(days=1))
        assert len(o) <= 90
        assert ((o - FC.GRID_ANCHOR).days % 4 == 0).all() and (o >= FC.GRID_ANCHOR).all()
        assert (o + pd.Timedelta(days=13) < c).all()           # whole window ended before the cut-off
        full = pd.date_range(FC.GRID_ANCHOR, c - pd.Timedelta(days=14), freq="4D")
        assert list(o) == list(full[-90:])                     # the most recent eligible origins

    def test_origins_shrink_when_history_ends_early(self):
        c = pd.Timestamp("2018-12-15")
        full = FC.calibration_origins(c, c - pd.Timedelta(days=1))
        short = FC.calibration_origins(c, c - pd.Timedelta(days=10))
        assert (short + pd.Timedelta(days=13) <= c - pd.Timedelta(days=10)).all()
        assert short.max() < full.max()

    def test_each_refit_sees_only_data_before_its_origin(self, hist):
        y, cal = hist
        c = pd.Timestamp("2018-12-15")
        seen = []

        def spy(h, cal_, o):
            seen.append((h.index.max(), o))
            return FC.m4_forecast(h, cal_, o)
        res = FC.calibration_residuals(y[y.index < c], cal, c, model=spy)
        assert len(seen) == 90 and all(last < o for last, o in seen)
        assert res["origin"].nunique() == 90 and len(res) == 90 * 14
        assert list(pd.DatetimeIndex([o for _, o in seen])) == list(FC.calibration_origins(c, c - pd.Timedelta(days=1)))


# ============================================================ harness emulation
class TestHarnessEmulation:
    @pytest.mark.parametrize("cut", ["2018-08-21", "2018-12-11"])
    def test_fresh_process_matches_in_process(self, tmp_path, hist, cut):
        y, cal = hist
        _harness_folder(tmp_path, cut)
        before = sorted(os.listdir(tmp_path))
        out, err, dt = _run_in_fresh_process(tmp_path, cut)
        _check_contract(out, cut)
        ref = FC.forecast_from_history(y, cal, pd.Timestamp(cut))
        assert np.allclose(out[["mean", "lower_80", "upper_80"]], ref[["mean", "lower_80", "upper_80"]], atol=1e-9)
        assert sorted(os.listdir(tmp_path)) == before            # writes nothing
        assert "FALLBACK" not in err
        assert dt < 120                                          # limit is 900 s

    def test_holdout_shaped_cutoff_runs(self, tmp_path):
        """Structure/runtime check for a Jan-Mar 2019 cut-off. Only history before the cut-off is supplied
        (as the harness does); no 2019 truth is read and nothing is scored."""
        cut = "2019-02-01"
        _harness_folder(tmp_path, cut)
        out, err, dt = _run_in_fresh_process(tmp_path, cut)
        _check_contract(out, cut)
        assert "FALLBACK" not in err and dt < 120


# ============================================================================ fallbacks
class TestFallbacks:
    def test_m4_exception_falls_back(self, hist, monkeypatch, capsys):
        y, cal = hist
        monkeypatch.setattr(FC, "m4_forecast", lambda *a, **k: (_ for _ in ()).throw(np.linalg.LinAlgError("singular")))
        out = FC.forecast_from_history(y, cal, pd.Timestamp("2018-09-01"))
        _check_contract(out, "2018-09-01")
        err = capsys.readouterr().err
        assert "FALLBACK: M4 failed" in err and "weekday-level fallback" in err

    def test_m4_nan_falls_back(self, hist, monkeypatch, capsys):
        y, cal = hist
        monkeypatch.setattr(FC, "m4_forecast", lambda *a, **k: np.full(14, np.nan))
        _check_contract(FC.forecast_from_history(y, cal, pd.Timestamp("2018-09-01")), "2018-09-01")
        assert "non-finite" in capsys.readouterr().err

    def test_all_means_fail_gives_constant(self, tmp_path, monkeypatch, capsys):
        _harness_folder(tmp_path, "2018-09-01")
        monkeypatch.setenv("PS06_DATA_DIR", str(tmp_path))
        boom = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x"))  # noqa: E731
        for f in ("m4_forecast", "weekday_level_forecast", "seasonal_naive_forecast"):
            monkeypatch.setattr(FC, f, boom)
        _check_contract(FC.forecast("2018-09-01"), "2018-09-01")
        assert "constant forecast" in capsys.readouterr().err

    def test_interval_failure_uses_band(self, hist, monkeypatch, capsys):
        y, cal = hist
        monkeypatch.setattr(FC, "calibration_residuals", lambda *a, **k: (_ for _ in ()).throw(ValueError("x")))
        _check_contract(FC.forecast_from_history(y, cal, pd.Timestamp("2018-09-01")), "2018-09-01")
        assert "seasonal-naive residual band" in capsys.readouterr().err

    def test_short_history_pool_too_small(self, hist, capsys):
        y, cal = hist
        out = FC.forecast_from_history(y, cal, pd.Timestamp("2017-07-20"))   # origins 07-01, 07-05 only
        _check_contract(out, "2017-07-20")
        err = capsys.readouterr().err
        assert "interval pool too small (2 origins)" in err and "seasonal-naive residual band" in err

    def test_short_history_degraded_pool(self, hist, capsys):
        y, cal = hist
        out = FC.forecast_from_history(y, cal, pd.Timestamp("2017-07-25"))   # origins 07-01, 07-05, 07-09
        _check_contract(out, "2017-07-25")
        err = capsys.readouterr().err
        assert "only 3 origins" in err and "FALLBACK" not in err

    def test_missing_admission_file(self, tmp_path, monkeypatch, capsys):
        shutil.copy(os.path.join(ROOT, L.FESTIVAL_FILE), tmp_path)
        monkeypatch.setenv("PS06_DATA_DIR", str(tmp_path))
        _check_contract(FC.forecast("2019-01-15"), "2019-01-15")
        assert "FALLBACK" in capsys.readouterr().err

    def test_missing_festival_file(self, tmp_path, monkeypatch, capsys):
        _harness_folder(tmp_path, "2018-09-01")
        os.remove(os.path.join(tmp_path, L.FESTIVAL_FILE))
        monkeypatch.setenv("PS06_DATA_DIR", str(tmp_path))
        _check_contract(FC.forecast("2018-09-01"), "2018-09-01")
        assert "calendar unreadable" in capsys.readouterr().err

    def test_garbage_rows_tolerated(self, tmp_path, monkeypatch, capsys):
        _harness_folder(tmp_path, "2018-09-01")
        p = os.path.join(tmp_path, L.ADMISSION_FILE)
        adm = L.read_raw_csv(p)
        bad = adm.head(3).copy()
        bad["D.O.A"] = ["", "2-1217", "31/02/2018"]
        pd.concat([adm, bad]).to_csv(p, index=False, encoding="utf-8-sig")
        monkeypatch.setenv("PS06_DATA_DIR", str(tmp_path))
        _check_contract(FC.forecast("2018-09-01"), "2018-09-01")
        assert "3 admission rows with unresolvable D.O.A ignored" in capsys.readouterr().err


# ===================================================================== restrictions
class TestRestrictions:
    def test_imports_only_pandas_numpy_stdlib(self):
        tree = ast.parse(open(os.path.join(ROOT, "forecast.py"), encoding="utf-8").read())
        mods = set()
        for n in ast.walk(tree):
            if isinstance(n, ast.Import):
                mods |= {a.name.split(".")[0] for a in n.names}
            elif isinstance(n, ast.ImportFrom):
                mods.add((n.module or "").split(".")[0])
        assert mods <= {"numpy", "pandas", "os", "re", "sys", "__future__"}, mods

    def test_no_file_writes_or_network_in_source(self):
        src = open(os.path.join(ROOT, "forecast.py"), encoding="utf-8").read()
        for bad in ("to_csv", "open(", "urllib", "requests", "socket", "pickle", "random"):
            assert bad not in src, bad


# ======================================================================= integrity
class TestIntegrity:
    def test_original_files_unchanged(self):
        for name, h in ORIGINAL_SHA256.items():
            assert sha256(os.path.join(ROOT, name)) == h, name

    def test_step2_outputs_unchanged(self):
        for name, h in STEP2_SHA256.items():
            assert sha256(os.path.join(ROOT, "outputs", name)) == h, name

    @pytest.mark.parametrize("fixture", ["step3_outputs_sha256.json", "src_and_step4_sha256.json"])
    def test_approved_outputs_and_src_unchanged(self, fixture):
        ref = json.load(open(os.path.join(HERE, "fixtures", fixture)))
        for path, h in ref.items():
            assert hashlib.sha256(open(os.path.join(ROOT, path), "rb").read()).hexdigest() == h, path
