"""Monthly calendar features from the Hijri (Umm al-Qura) calendar via `hijridate`
(the maintained successor of `hijri-converter`, same author and API)."""
from functools import lru_cache

import pandas as pd
from hijridate import Gregorian


@lru_cache(maxsize=None)
def _hijri(d: pd.Timestamp):
    h = Gregorian(d.year, d.month, d.day).to_hijri()
    return h.month, h.day


def month_features(ds: pd.Timestamp) -> dict:
    """Counts of special days in a Gregorian month."""
    days = pd.date_range(ds, ds + pd.offsets.MonthEnd(0), freq="D")
    ram = fitr = adha = pre_eid = wknd = 0
    for d in days:
        hm, hd = _hijri(d)
        ram += hm == 9
        pre_eid += (hm == 9 and hd >= 20)          # last ten nights of Ramadan (Eid shopping)
        fitr += (hm == 10 and hd <= 3)             # Eid al-Fitr
        adha += (hm == 12 and 9 <= hd <= 13)       # Arafah + Eid al-Adha + Tashreeq
        wknd += d.day_of_week in (4, 5)              # Fri/Sat weekend
    return {"ramadan_days": ram, "ramadan_last10_days": pre_eid, "eid_fitr_days": fitr,
            "eid_adha_days": adha, "weekend_days": wknd, "days_in_month": len(days),
            "ramadan_flag": int(ram > 0), "eid_flag": int(fitr + adha > 0)}


def calendar_frame(start="2019-01-01", end="2027-12-01") -> pd.DataFrame:
    months = pd.date_range(start, end, freq="MS")
    return pd.DataFrame([{"ds": m, **month_features(m)} for m in months])
