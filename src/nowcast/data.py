"""Load SAMA weekly POS bulletins, splice where consistent, convert to calendar months."""
import hashlib
import json
import urllib.request
from datetime import datetime, timezone

import pandas as pd

from . import config as C


def download(force: bool = False) -> dict:
    """Fetch the two KAPSARC exports of the SAMA bulletin; write a manifest with sha256."""
    C.RAW.mkdir(parents=True, exist_ok=True)
    man_path = C.RAW / "manifest.json"
    manifest = json.loads(man_path.read_text()) if man_path.exists() else {}
    for key, src in C.SOURCES.items():
        path = C.RAW / src["file"]
        if path.exists() and not force:
            continue
        urllib.request.urlretrieve(src["url"], path)
        manifest[f"pos_{key.lower()}"] = {
            "file": src["file"], "download_url": src["url"], "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "downloaded_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
    man_path.write_text(json.dumps(manifest, indent=2))
    return manifest


def _read(key: str) -> pd.DataFrame:
    df = pd.read_csv(C.RAW / C.SOURCES[key]["file"], encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    df = df[(df["number_value_change_transactions"] == C.VALUE_MEASURE) & (df["city"] == "Total")]
    out = df[["starting_date", "sectors", "value"]].rename(columns={"starting_date": "week_start", "sectors": "sector"})
    out["week_start"] = pd.to_datetime(out["week_start"]).dt.tz_localize(None)
    out["value"] = out["value"].astype(float)
    return out.drop_duplicates(["week_start", "sector"]).reset_index(drop=True)


def weekly_sector_series() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return weekly national value by sector (SAR thousands) and a splice audit table."""
    a, b = _read("A"), _read("B")
    overlap = sorted(set(a.week_start) & set(b.week_start))
    audit, pieces = [], [a]
    for sec_a, sec_b in C.B_CANDIDATES.items():
        va = a[a.sector == sec_a].set_index("week_start").value.reindex(overlap)
        vb = b[b.sector == sec_b].set_index("week_start").value.reindex(overlap)
        gap = float((vb / va - 1).abs().max())
        ok = bool(gap <= C.SPLICE_TOLERANCE)
        audit.append({"sector": sec_a, "b_sector": sec_b, "overlap_weeks": len(overlap),
                      "max_rel_gap_pct": round(100 * gap, 3), "spliced": ok})
        if ok:
            ext = b[(b.sector == sec_b) & (b.week_start > a.week_start.max())].assign(sector=sec_a)
            pieces.append(ext)
    w = pd.concat(pieces, ignore_index=True).sort_values(["sector", "week_start"]).reset_index(drop=True)
    return w, pd.DataFrame(audit)


def _holiday_week(week_start: pd.Timestamp) -> bool:
    """True if the Sun-Sat week touches the end of Ramadan, the Eid al-Fitr week or the Hajj/Eid al-Adha
    period. Genuine closures (e.g. schools) make some sectors collapse in these weeks, so they are never 'fixed'."""
    from hijridate import Gregorian
    for k in range(7):
        d = week_start + pd.Timedelta(days=k)
        h = Gregorian(d.year, d.month, d.day).to_hijri()
        if (h.month == 9 and h.day >= 20) or (h.month == 10 and h.day <= 7) or (h.month == 12 and 8 <= h.day <= 15):
            return True
    return False


def fix_outliers(w: pd.DataFrame, ratio: float = 0.25) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Flag isolated weekly values below `ratio` x the median of the 4 weeks either side
    (e.g. a dropped digit in the source), outside Ramadan/Eid/Hajj weeks, and replace them with the mean of the two adjacent weeks.
    Every replacement is logged so it can be audited."""
    w = w.copy()
    fixes = []
    for sec, g in w.groupby("sector"):
        v = g["value"]
        med = v.rolling(9, center=True, min_periods=5).median()
        bad = (v < ratio * med) & ~g["week_start"].map(_holiday_week)
        for i in g.index[bad]:
            pos = g.index.get_loc(i)
            nb = [v.iloc[j] for j in (pos - 1, pos + 1) if 0 <= j < len(v)]
            new = float(sum(nb) / len(nb))
            fixes.append({"sector": sec, "week_start": g.loc[i, "week_start"].date(),
                          "reported_k_sar": v[i], "neighbour_median_k_sar": med[i], "replaced_with_k_sar": new})
            w.loc[i, "value"] = new
    return w, pd.DataFrame(fixes, columns=["sector", "week_start", "reported_k_sar",
                                           "neighbour_median_k_sar", "replaced_with_k_sar"])


def weekly_to_monthly(w: pd.DataFrame) -> pd.DataFrame:
    """Spread each Sun-Sat week evenly over its 7 days, sum by calendar month,
    keep only months fully covered by data."""
    days = w.loc[w.index.repeat(7)].copy()
    days["date"] = days["week_start"] + pd.to_timedelta(days.groupby(level=0).cumcount(), unit="D")
    days["value"] = days["value"] / 7.0
    days["month"] = days["date"].dt.to_period("M").dt.to_timestamp()
    m = days.groupby(["sector", "month"]).agg(value=("value", "sum"), days=("date", "nunique")).reset_index()
    m = m[m["days"] == m["month"].dt.days_in_month]
    m["y"] = m["value"] / 1e6  # SAR billions
    return m[["sector", "month", "y"]].rename(columns={"sector": "unique_id", "month": "ds"}).reset_index(drop=True)


def partial_months(w: pd.DataFrame) -> pd.DataFrame:
    """Observed SAR bn and day coverage for the last, incomplete month of each series."""
    days = w.loc[w.index.repeat(7)].copy()
    days["date"] = days["week_start"] + pd.to_timedelta(days.groupby(level=0).cumcount(), unit="D")
    days["value"] = days["value"] / 7.0
    days["ds"] = days["date"].dt.to_period("M").dt.to_timestamp()
    g = days.groupby(["sector", "ds"]).agg(y_obs=("value", "sum"), days=("date", "nunique"),
                                           last_day=("date", "max")).reset_index()
    g = g[g["days"] < g["ds"].dt.days_in_month]
    g = g.sort_values("ds").groupby("sector").tail(1)
    g = g[g["ds"] > pd.Timestamp("2020-06-01")]
    g["y_obs"] = g["y_obs"] / 1e6
    return g.rename(columns={"sector": "unique_id"}).reset_index(drop=True)


def build() -> dict:
    w, audit = weekly_sector_series()
    w, fixes = fix_outliers(w)
    m = weekly_to_monthly(w)
    pm = partial_months(w)
    C.PROCESSED.mkdir(parents=True, exist_ok=True)
    m.to_csv(C.PROCESSED / "monthly_sector_value.csv", index=False)
    audit.to_csv(C.PROCESSED / "splice_audit.csv", index=False)
    pm.to_csv(C.PROCESSED / "partial_months.csv", index=False)
    fixes.to_csv(C.PROCESSED / "data_corrections.csv", index=False)
    return {"fixes": fixes, "weekly": w, "monthly": m, "audit": audit, "partial": pm}
