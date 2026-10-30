"""End-to-end: data -> features -> backtest -> forecasts -> figures -> results.json.
Run with `python -m nowcast.pipeline` (from src/) or `make all`."""
import argparse
import json

import numpy as np
import pandas as pd

from . import config as C
from .backtest import run_backtest, score
from .calendar_features import calendar_frame
from .data import build, download
from .forecast import all_model_forecasts, final_forecast


def main(refresh: bool = False):
    np.random.seed(C.SEED)
    download(force=refresh)
    d = build()
    df, partial = d["monthly"], d["partial"]
    cal = calendar_frame()
    cal.to_csv(C.PROCESSED / "calendar_features.csv", index=False)
    codes = {u: i for i, u in enumerate(sorted(df.unique_id.unique()))}

    print(f"series={df.unique_id.nunique()} months={df.ds.min():%Y-%m}..{df.ds.max():%Y-%m}")
    bt = run_backtest(df, cal, codes)
    bt.to_csv(C.PROCESSED / "backtest_forecasts.csv", index=False)
    sc = score(bt)
    sc.to_csv(C.PROCESSED / "backtest_scores.csv", index=False)

    fc_all = all_model_forecasts(df, cal, codes)
    fc_all.to_csv(C.PROCESSED / "forecast_all_models.csv", index=False)
    fc = final_forecast(fc_all, bt, sc)
    fc.to_csv(C.PROCESSED / "forecast_best_model.csv", index=False)

    # Nowcast check for the partially observed month (data ends mid-month)
    nc = partial.merge(fc[fc.h == 1][["unique_id", "ds", "yhat", "lo80", "hi80", "model"]], on=["unique_id", "ds"], how="inner")
    nc["observed_share_of_forecast_pct"] = 100 * nc.y_obs / nc.yhat
    nc.to_csv(C.PROCESSED / "nowcast_partial_month.csv", index=False)

    from .models import ABLATIONS
    abl = sc[sc.model.isin(["SARIMAX", "SARIMA_noHijri"])].pivot(index="unique_id", columns="model", values="MAPE")
    sc_all, sc = sc, sc[~sc.model.isin(ABLATIONS)]
    best = sc.loc[sc.groupby("unique_id").MAPE.idxmin()]
    pivot = sc.pivot(index="unique_id", columns="model", values="MAPE")
    snaive = pivot["SeasonalNaive"]
    overall = sc.groupby("model")[["MAPE", "sMAPE", "MAPE_h1"]].mean()
    ends = df.groupby("unique_id").ds.agg(["min", "max", "count"])
    results = {
        "n_series": int(df.unique_id.nunique()),
        "series_start": f"{df.ds.min():%Y-%m}",
        "series_end_by_sector": {u: f"{e:%Y-%m}" for u, e in ends["max"].items()},
        "months_by_sector": {u: int(n) for u, n in ends["count"].items()},
        "spliced_sectors": d["audit"].query("spliced").sector.tolist(),
        "n_origins": C.N_ORIGINS, "horizon": C.HORIZON,
        "mean_over_sectors": overall.round(2).to_dict(orient="index"),
        "wins_by_model": best.model.value_counts().to_dict(),
        "best_by_sector": {r.unique_id: {"model": r.model, "MAPE": round(r.MAPE, 2), "sMAPE": round(r.sMAPE, 2),
                                         "MAPE_h1": round(r.MAPE_h1, 2),
                                         "SeasonalNaive_MAPE": round(snaive[r.unique_id], 2)}
                           for r in best.itertuples()},
        "sectors_beating_baseline": int((best.MAPE < snaive[best.unique_id].values - 1e-9).sum()),
        "hijri_ablation": {"SARIMAX_mean_MAPE": round(abl.SARIMAX.mean(), 2),
                           "SARIMA_noHijri_mean_MAPE": round(abl.SARIMA_noHijri.mean(), 2),
                           "sectors_where_hijri_regressors_help": int((abl.SARIMAX < abl.SARIMA_noHijri).sum()),
                           "by_sector": abl.round(2).to_dict(orient="index")},
        "nowcast": nc.round(3).astype({"ds": str}).to_dict(orient="records"),
    }
    (C.PROCESSED / "results.json").write_text(json.dumps(results, indent=2, default=str))

    from .charts import make_all
    make_all()
    from .thumbnail import make_thumbnail
    make_thumbnail()
    print(json.dumps({k: results[k] for k in ["hijri_ablation", "mean_over_sectors", "wins_by_model", "sectors_beating_baseline"]}, indent=1))
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true", help="re-download SAMA data from KAPSARC")
    main(ap.parse_args().refresh)
