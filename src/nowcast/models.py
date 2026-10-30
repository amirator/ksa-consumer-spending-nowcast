"""Forecasting models. Each local model maps a training series (monthly, SAR bn) to h point forecasts."""
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

from . import config as C

EXOG = ["ramadan_days", "eid_fitr_days", "eid_adha_days"]


def seasonal_naive(y: pd.Series, h: int, **_) -> np.ndarray:
    last = y.values[-C.SEASON:]
    return np.array([last[i % C.SEASON] for i in range(h)])


def ets(y: pd.Series, h: int, **_) -> np.ndarray:
    """Additive Holt-Winters on log values with damped trend (multiplicative in levels)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = ExponentialSmoothing(np.log(y.values), trend="add", damped_trend=True,
                                   seasonal="add", seasonal_periods=C.SEASON,
                                   initialization_method="estimated").fit()
    return np.exp(fit.forecast(h))


def sarimax(y: pd.Series, h: int, cal: pd.DataFrame, **_) -> np.ndarray:
    """Airline-type SARIMA(0,1,1)(0,1,1)12 on log values with Hijri holiday regressors."""
    c = cal.set_index("ds")[EXOG].astype(float)
    future = pd.date_range(y.index[-1] + pd.offsets.MonthBegin(1), periods=h, freq="MS")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = SARIMAX(np.log(y.values), exog=c.loc[y.index].values, order=(0, 1, 1),
                      seasonal_order=(0, 1, 1, C.SEASON), enforce_stationarity=False,
                      enforce_invertibility=False).fit(disp=False)
    return np.exp(fit.forecast(h, exog=c.loc[future].values))


def sarima_no_hijri(y: pd.Series, h: int, **_) -> np.ndarray:
    """Ablation: the same SARIMA without the Hijri holiday regressors."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = SARIMAX(np.log(y.values), order=(0, 1, 1), seasonal_order=(0, 1, 1, C.SEASON),
                      enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
    return np.exp(fit.forecast(h))


LOCAL_MODELS = {"SeasonalNaive": seasonal_naive, "ETS": ets, "SARIMAX": sarimax,
                "SARIMA_noHijri": sarima_no_hijri}
ABLATIONS = {"SARIMA_noHijri"}  # scored for comparison, never selected as the production model

# ---------------------------------------------------------------- global LightGBM (direct strategy)
CAL_COLS = ["ramadan_days", "ramadan_last10_days", "eid_fitr_days", "eid_adha_days", "weekend_days", "days_in_month"]
GBM_FEATURES = (["h", "month", "sector_code", "yoy_0", "yoy_1", "yoy_2", "yoy_mean3", "yoy_mean12", "mom_0"]
                + [f"{c}" for c in CAL_COLS] + [f"d_{c}" for c in CAL_COLS])
GBM_PARAMS = dict(objective="l1", n_estimators=400, learning_rate=0.03, num_leaves=15,
                  min_child_samples=10, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                  random_state=C.SEED, verbose=-1)


def _anchor_features(logy: pd.Series) -> pd.DataFrame:
    """Features known at anchor month t (lag/rolling features of log YoY growth)."""
    yoy = logy - logy.shift(C.SEASON)
    return pd.DataFrame({"yoy_0": yoy, "yoy_1": yoy.shift(1), "yoy_2": yoy.shift(2),
                         "yoy_mean3": yoy.rolling(3).mean(), "yoy_mean12": yoy.rolling(12).mean(),
                         "mom_0": logy - logy.shift(1)}, index=logy.index)


def gbm_rows(df: pd.DataFrame, cal: pd.DataFrame, origin: pd.Timestamp, codes: dict, train: bool):
    """Direct multi-horizon design matrix. Target = log y[t+h] - log y[t+h-12] (h <= 12, so the
    base month is always observed). For train=True rows with t+h <= origin; otherwise anchor = origin."""
    calx = cal.set_index("ds")[CAL_COLS]
    rows = []
    for uid, g in df[df.ds <= origin].groupby("unique_id"):
        logy = np.log(g.set_index("ds")["y"])
        feats = _anchor_features(logy)
        anchors = feats.index if train else [logy.index[-1]]
        for t in anchors:
            if pd.isna(feats.loc[t, "yoy_mean3"]):
                continue
            for h in range(1, C.HORIZON + 1):
                tgt = t + pd.DateOffset(months=h)
                base = tgt - pd.DateOffset(months=C.SEASON)
                if base not in logy.index:
                    continue
                if train and (tgt not in logy.index or tgt < pd.Timestamp(C.GBM_MIN_TARGET)):
                    continue
                r = {"unique_id": uid, "anchor": t, "ds": tgt, "h": h, "month": tgt.month,
                     "sector_code": codes[uid], "base_log": logy[base], **feats.loc[t].to_dict()}
                for c in CAL_COLS:
                    r[c] = calx.loc[tgt, c]
                    r[f"d_{c}"] = calx.loc[tgt, c] - calx.loc[base, c]
                if train:
                    r["target"] = logy[tgt] - logy[base]
                rows.append(r)
    return pd.DataFrame(rows)


def gbm_forecast(df: pd.DataFrame, cal: pd.DataFrame, origin: pd.Timestamp, codes: dict,
                 series: list | None = None) -> pd.DataFrame:
    """Fit one global model on data up to `origin`, forecast h=1..HORIZON for series ending at origin."""
    tr = gbm_rows(df, cal, origin, codes, train=True)
    model = lgb.LGBMRegressor(**GBM_PARAMS)
    model.fit(tr[GBM_FEATURES], tr["target"], categorical_feature=["sector_code"])
    te = gbm_rows(df, cal, origin, codes, train=False)
    te = te[te.anchor == origin]
    if series is not None:
        te = te[te.unique_id.isin(series)]
    te["yhat"] = np.exp(te["base_log"] + model.predict(te[GBM_FEATURES]))
    return te[["unique_id", "ds", "h", "yhat"]]
