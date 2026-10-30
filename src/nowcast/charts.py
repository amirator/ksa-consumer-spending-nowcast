"""PNG charts from the processed outputs."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config as C

NAVY, BLUE, ORANGE, GREY = "#0B3D5C", "#2E86C1", "#E67E22", "#7F8C8D"
MODEL_COLORS = {"SeasonalNaive": GREY, "ETS": "#27AE60", "SARIMAX": BLUE, "LightGBM": ORANGE}
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titleweight": "bold", "font.size": 10})


def _load():
    p = C.PROCESSED
    return (pd.read_csv(p / "monthly_sector_value.csv", parse_dates=["ds"]),
            pd.read_csv(p / "backtest_forecasts.csv", parse_dates=["ds", "origin"]),
            pd.read_csv(p / "backtest_scores.csv"),
            pd.read_csv(p / "forecast_best_model.csv", parse_dates=["ds"]),
            pd.read_csv(p / "calendar_features.csv", parse_dates=["ds"]),
            json.loads((p / "results.json").read_text()))


def _ramadan_shade(ax, cal, lo, hi):
    for r in cal[(cal.ds >= lo) & (cal.ds <= hi) & (cal.ramadan_days > 0)].itertuples():
        ax.axvspan(r.ds, r.ds + pd.offsets.MonthEnd(0), color="#F4D03F", alpha=0.12 + 0.01 * r.ramadan_days, lw=0)


def fig_history(df, cal):
    tot = df[df.unique_id == "Total"].set_index("ds").y
    fig, ax = plt.subplots(figsize=(11, 4.5))
    _ramadan_shade(ax, cal, tot.index.min(), tot.index.max())
    ax.plot(tot.index, tot.values, color=NAVY, lw=2)
    ax.set_title("Monthly POS card spending in Saudi Arabia, all sectors (SAR bn)")
    ax.set_ylabel("SAR bn per month")
    ax.text(0.01, 0.95, "Shaded: months containing Ramadan days (darker = more days)", transform=ax.transAxes, fontsize=9, color=GREY)
    ax.text(0.99, -0.13, "Source: SAMA weekly POS bulletin via KAPSARC Data Portal; weekly values spread to calendar months",
            transform=ax.transAxes, ha="right", fontsize=8, color=GREY)
    fig.tight_layout(); fig.savefig(C.FIGURES / "01_total_monthly_history.png"); plt.close(fig)


def fig_sectors(df):
    secs = [s for s in df.groupby("unique_id").y.mean().sort_values(ascending=False).index if s != "Total"]
    fig, axes = plt.subplots(6, 3, figsize=(13, 15), sharex=True)
    for ax, s in zip(axes.flat, secs):
        g = df[df.unique_id == s]
        ax.plot(g.ds, g.y, color=BLUE, lw=1.4)
        ax.set_title(s, fontsize=10)
    for ax in axes.flat[len(secs):]:
        ax.axis("off")
    fig.suptitle("Monthly POS spending by sector (SAR bn), SAMA", fontweight="bold", y=0.995)
    fig.tight_layout(); fig.savefig(C.FIGURES / "02_sector_small_multiples.png"); plt.close(fig)


def fig_heatmap(sc):
    sc = sc[sc.model != "SARIMA_noHijri"]
    pv = sc.pivot(index="unique_id", columns="model", values="MAPE")[["SeasonalNaive", "ETS", "SARIMAX", "LightGBM"]]
    pv = pv.loc[pv.min(axis=1).sort_values().index]
    fig, ax = plt.subplots(figsize=(8.5, 8.5))
    im = ax.imshow(pv.values, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=np.nanpercentile(pv.values, 95))
    ax.set_xticks(range(pv.shape[1]), pv.columns); ax.set_yticks(range(pv.shape[0]), pv.index)
    for i in range(pv.shape[0]):
        best = np.nanargmin(pv.values[i])
        for j in range(pv.shape[1]):
            ax.text(j, i, f"{pv.values[i, j]:.1f}", ha="center", va="center", fontsize=9,
                    fontweight="bold" if j == best else "normal")
    ax.set_title(f"Backtest MAPE % (12 rolling origins, 1-{C.HORIZON} months ahead)\nbold = best per sector")
    fig.colorbar(im, ax=ax, shrink=0.6, label="MAPE %")
    fig.tight_layout(); fig.savefig(C.FIGURES / "03_backtest_mape_heatmap.png"); plt.close(fig)


def fig_backtest_total(df, bt):
    tot = df[df.unique_id == "Total"].set_index("ds").y
    b = bt[(bt.unique_id == "Total") & (bt.h == 1) & bt.model.isin(list(MODEL_COLORS))]
    fig, ax = plt.subplots(figsize=(11, 4.5))
    hist = tot[tot.index >= b.ds.min() - pd.DateOffset(months=18)]
    ax.plot(hist.index, hist.values, color="black", lw=2.2, label="Actual")
    for m, g in b.groupby("model"):
        ax.plot(g.ds, g.yhat, marker="o", ms=4, lw=1.2, color=MODEL_COLORS[m], label=f"{m} (1-month-ahead)")
    ax.set_title("Total POS spending: one-month-ahead backtest forecasts vs actual (SAR bn)")
    ax.legend(ncol=3, fontsize=8, frameon=False)
    fig.tight_layout(); fig.savefig(C.FIGURES / "04_backtest_total_h1.png"); plt.close(fig)


def fig_forecasts(df, fc, cal):
    secs = ["Total", "Beverage and Food", "Restaurants & Café", "Gas Stations", "Clothing and Footwear", "Hotels"]
    fig, axes = plt.subplots(3, 2, figsize=(13, 11))
    for ax, s in zip(axes.flat, secs):
        g = df[(df.unique_id == s) & (df.ds >= "2023-01-01")]
        f = fc[fc.unique_id == s]
        _ramadan_shade(ax, cal, g.ds.min(), f.ds.max())
        ax.plot(g.ds, g.y, color=NAVY, lw=1.8, label="Actual")
        ax.plot(f.ds, f.yhat, color=ORANGE, lw=2, marker="o", ms=3, label=f"Forecast ({f.model.iloc[0]})")
        ax.fill_between(f.ds, f.lo80, f.hi80, color=ORANGE, alpha=0.25, label="80% interval")
        ax.set_title(s); ax.legend(fontsize=8, frameon=False, loc="upper left")
    fig.suptitle(f"{C.HORIZON}-month forecasts with 80% intervals (best backtest model per sector), SAR bn",
                 fontweight="bold")
    fig.tight_layout(); fig.savefig(C.FIGURES / "05_forecast_fans.png"); plt.close(fig)


def fig_ramadan(df, cal):
    """Log YoY growth vs change in Ramadan days in the month: why Hijri features matter."""
    d = df.merge(cal, on="ds")
    d = d.sort_values("ds")
    d["yoy"] = d.groupby("unique_id").y.transform(lambda s: np.log(s / s.shift(12))) * 100
    d["d_ram"] = d.groupby("unique_id").ramadan_days.transform(lambda s: s - s.shift(12))
    d = d[d.ds >= "2022-01-01"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, s in zip(axes, ["Total", "Clothing and Footwear"]):
        g = d[d.unique_id == s]
        ax.scatter(g.d_ram, g.yoy, color=BLUE, alpha=0.7)
        ax.axhline(0, color=GREY, lw=0.8)
        ax.set_xlabel("Ramadan days this month minus same month last year")
        ax.set_ylabel("YoY growth, % (log)")
        ax.set_title(s)
    fig.suptitle("Ramadan moves ~11 days a year, so month-on-same-month comparisons swing with it (2022 onward)",
                 fontweight="bold", fontsize=10)
    fig.tight_layout(); fig.savefig(C.FIGURES / "06_ramadan_shift_effect.png"); plt.close(fig)


def fig_thumb(df, fc, cal):
    g = df[(df.unique_id == "Total") & (df.ds >= "2022-01-01")]
    f = fc[fc.unique_id == "Total"]
    fig, ax = plt.subplots(figsize=(11.5, 4.9), dpi=100)
    _ramadan_shade(ax, cal, g.ds.min(), f.ds.max())
    ax.plot(g.ds, g.y, color=NAVY, lw=2.4, label="Actual monthly POS spending (SAMA)")
    ax.plot(f.ds, f.yhat, color=ORANGE, lw=2.6, marker="o", label=f"{C.HORIZON}-month forecast ({f.model.iloc[0]})")
    ax.fill_between(f.ds, f.lo80, f.hi80, color=ORANGE, alpha=0.25, label="80% interval")
    ax.set_ylabel("SAR bn per month", fontsize=12)
    ax.set_title("Saudi card spending at points of sale: forecast with Ramadan & Eid features", fontsize=14)
    ax.legend(frameon=False, fontsize=11, loc="upper left")
    ax.tick_params(labelsize=11)
    fig.tight_layout(); fig.savefig(C.FIGURES / "00_thumbnail_chart.png"); plt.close(fig)


def make_all():
    C.FIGURES.mkdir(parents=True, exist_ok=True)
    df, bt, sc, fc, cal, _ = _load()
    fig_history(df, cal); fig_sectors(df); fig_heatmap(sc)
    fig_backtest_total(df, bt); fig_forecasts(df, fc, cal); fig_ramadan(df, cal); fig_thumb(df, fc, cal)
