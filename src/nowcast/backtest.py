"""Rolling-origin (expanding window) backtest, refit at every origin."""
import pandas as pd

from . import config as C
from .metrics import mape, smape
from .models import LOCAL_MODELS, gbm_forecast


def origins_for(g: pd.DataFrame) -> list:
    ds = sorted(g.ds)
    return ds[-(C.N_ORIGINS + 1):-1]  # last origin still has >= 1 month to score


def run_backtest(df: pd.DataFrame, cal: pd.DataFrame, codes: dict) -> pd.DataFrame:
    out = []
    by_origin = {}
    for uid, g in df.groupby("unique_id"):
        y = g.set_index("ds")["y"].asfreq("MS")
        for o in origins_for(g):
            by_origin.setdefault(o, []).append(uid)
            train = y.loc[:o]
            future = y.loc[o + pd.offsets.MonthBegin(1):].iloc[: C.HORIZON]
            for name, fn in LOCAL_MODELS.items():
                pred = fn(train, len(future), cal=cal)
                out += [{"unique_id": uid, "model": name, "origin": o, "ds": d, "h": i + 1,
                         "y": v, "yhat": p} for i, (d, v, p) in enumerate(zip(future.index, future.values, pred))]
    actual = df.set_index(["unique_id", "ds"])["y"]
    for o, uids in sorted(by_origin.items()):
        f = gbm_forecast(df, cal, o, codes, series=uids)
        f = f[f.set_index(["unique_id", "ds"]).index.isin(actual.index)]
        for r in f.itertuples():
            out.append({"unique_id": r.unique_id, "model": "LightGBM", "origin": o, "ds": r.ds,
                        "h": r.h, "y": actual[(r.unique_id, r.ds)], "yhat": r.yhat})
    return pd.DataFrame(out)


def score(bt: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (uid, m), g in bt.groupby(["unique_id", "model"]):
        g1 = g[g.h == 1]
        rows.append({"unique_id": uid, "model": m, "n_forecasts": len(g),
                     "MAPE": mape(g.y, g.yhat), "sMAPE": smape(g.y, g.yhat),
                     "MAPE_h1": mape(g1.y, g1.yhat)})
    return pd.DataFrame(rows)
