"""Final forecasts from the backtest-selected model per sector, with conformal-style intervals."""
import numpy as np
import pandas as pd

from . import config as C
from .models import ABLATIONS, LOCAL_MODELS, gbm_forecast


def all_model_forecasts(df, cal, codes) -> pd.DataFrame:
    out = []
    for uid, g in df.groupby("unique_id"):
        y = g.set_index("ds")["y"].asfreq("MS")
        idx = pd.date_range(y.index[-1] + pd.offsets.MonthBegin(1), periods=C.HORIZON, freq="MS")
        for name, fn in LOCAL_MODELS.items():
            out.append(pd.DataFrame({"unique_id": uid, "model": name, "ds": idx,
                                     "h": range(1, C.HORIZON + 1), "yhat": fn(y, C.HORIZON, cal=cal)}))
    ends = df.groupby("unique_id").ds.max()
    for o, uids in ends.groupby(ends):
        f = gbm_forecast(df, cal, o, codes, series=list(uids.index))
        out.append(f.assign(model="LightGBM"))
    return pd.concat(out, ignore_index=True)


def interval_table(bt: pd.DataFrame) -> pd.DataFrame:
    """Quantiles of backtest log errors log(y/yhat) per series, model and horizon."""
    bt = bt.assign(le=np.log(bt.y / bt.yhat))
    q = bt.groupby(["unique_id", "model", "h"]).le.quantile(list(C.INTERVAL)).unstack()
    q.columns = ["q_lo", "q_hi"]
    return q.reset_index()


def final_forecast(fc_all, bt, scores) -> pd.DataFrame:
    scores = scores[~scores.model.isin(ABLATIONS)]
    best = scores.loc[scores.groupby("unique_id").MAPE.idxmin(), ["unique_id", "model"]]
    f = fc_all.merge(best, on=["unique_id", "model"]).merge(interval_table(bt), on=["unique_id", "model", "h"], how="left")
    # widen nothing silently: horizons never scored (none expected) fall back to the widest scored horizon
    f[["q_lo", "q_hi"]] = f.groupby("unique_id")[["q_lo", "q_hi"]].ffill()
    f["lo80"] = f.yhat * np.exp(np.minimum(f.q_lo, 0))
    f["hi80"] = f.yhat * np.exp(np.maximum(f.q_hi, 0))
    return f[["unique_id", "model", "ds", "h", "yhat", "lo80", "hi80"]]
