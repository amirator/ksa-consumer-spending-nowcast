import json

import numpy as np
import pandas as pd
import pytest

from nowcast import config as C
from nowcast.calendar_features import month_features
from nowcast.data import _holiday_week, fix_outliers, weekly_sector_series, weekly_to_monthly
from nowcast.metrics import mape, smape
from nowcast.models import seasonal_naive


# ---------- calendar (Hijri) features, checked against known Umm al-Qura dates
def test_ramadan_2025_month_counts():
    # Ramadan 1446 = 1 Mar 2025 .. 29 Mar 2025; Eid al-Fitr 30 Mar 2025
    f = month_features(pd.Timestamp("2025-03-01"))
    assert f["ramadan_days"] == 29 and f["eid_fitr_days"] == 2
    assert month_features(pd.Timestamp("2025-04-01"))["eid_fitr_days"] == 1


def test_eid_adha_2024():
    # Eid al-Adha 1445 = 16 Jun 2024 (Arafah 15 Jun)
    assert month_features(pd.Timestamp("2024-06-01"))["eid_adha_days"] == 5
    assert month_features(pd.Timestamp("2024-01-01"))["ramadan_days"] == 0


def test_holiday_week_guard():
    assert _holiday_week(pd.Timestamp("2025-03-30"))      # Eid al-Fitr week
    assert not _holiday_week(pd.Timestamp("2026-02-01"))  # ordinary week


# ---------- data
@pytest.fixture(scope="module")
def weekly():
    return weekly_sector_series()


def test_raw_sources_have_manifest():
    man = json.loads((C.RAW / "manifest.json").read_text())
    for src in C.SOURCES.values():
        assert any(v["file"] == src["file"] for v in man.values())
        assert (C.RAW / src["file"]).exists()


def test_splice_only_where_bulletins_agree(weekly):
    w, audit = weekly
    assert set(audit.query("spliced").sector) == {"Total", "Beverage and Food", "Gas Stations", "Hotels", "Public Utilities"}
    assert (audit.query("spliced").max_rel_gap_pct <= 100 * C.SPLICE_TOLERANCE).all()


def test_sectors_sum_to_total(weekly):
    w, _ = weekly
    a = w[w.week_start <= "2025-07-06"]
    tot = a[a.sector == "Total"].set_index("week_start").value
    parts = a[a.sector != "Total"].groupby("week_start").value.sum()
    assert ((parts / tot - 1).abs() < 0.01).mean() > 0.97


def test_monthly_conserves_value():
    w = pd.DataFrame({"sector": "X", "week_start": pd.date_range("2024-01-07", periods=8, freq="7D"), "value": 7.0e6})
    m = weekly_to_monthly(w)
    # only Feb 2024 (29 days) is fully covered: 29 days x 1e6 k SAR = 29 SAR bn
    assert list(m.ds) == [pd.Timestamp("2024-02-01")] and m.y.iloc[0] == pytest.approx(29.0)


def test_outlier_fix_flags_dropped_digit_only():
    w = pd.DataFrame({"sector": "X", "week_start": pd.date_range("2026-01-04", periods=9, freq="7D"),
                      "value": [100, 102, 98, 101, 7, 99, 103, 100, 101]}).astype({"value": float})
    fixed, log = fix_outliers(w)
    assert len(log) == 1 and fixed.value.iloc[4] == pytest.approx(100.0)


# ---------- metrics and baseline
def test_metrics():
    assert mape([100, 200], [110, 180]) == pytest.approx(10.0)
    assert smape([100], [100]) == 0
    assert smape([100], [0]) == pytest.approx(200.0)


def test_seasonal_naive_repeats_last_year():
    y = pd.Series(np.arange(1, 25, dtype=float), index=pd.date_range("2022-01-01", periods=24, freq="MS"))
    assert list(seasonal_naive(y, 3)) == [13.0, 14.0, 15.0]


# ---------- outputs of `make all`
def test_results_outputs():
    r = json.loads((C.PROCESSED / "results.json").read_text())
    assert r["n_series"] == 18
    sc = pd.read_csv(C.PROCESSED / "backtest_scores.csv")
    assert {"SeasonalNaive", "ETS", "SARIMAX", "LightGBM"} <= set(sc.model)
    assert sc[["MAPE", "sMAPE"]].notna().all().all()
    fc = pd.read_csv(C.PROCESSED / "forecast_best_model.csv")
    assert fc.groupby("unique_id").size().eq(C.HORIZON).all()
    assert (fc.lo80 <= fc.yhat).all() and (fc.yhat <= fc.hi80).all() and (fc.yhat > 0).all()
    for k, v in r["best_by_sector"].items():
        assert v["MAPE"] == pytest.approx(sc[sc.unique_id == k].query("model != 'SARIMA_noHijri'").MAPE.min(), abs=0.01)
