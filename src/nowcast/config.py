from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
FIGURES = ROOT / "reports" / "figures"

SOURCES = {
    "A": {
        "file": "point-of-sale-transactions-by-sector-and-city.csv",
        "url": "https://datasource.kapsarc.org/api/explore/v2.1/catalog/datasets/point-of-sale-transactions-by-sector-and-city/exports/csv?delimiter=%2C",
    },
    "B": {
        "file": "detailed-point-of-sale-transactions-by-sector-and-city.csv",
        "url": "https://datasource.kapsarc.org/api/explore/v2.1/catalog/datasets/detailed-point-of-sale-transactions-by-sector-and-city/exports/csv?delimiter=%2C",
    },
}
VALUE_MEASURE = "Value of Transactions (In Thousand SAR)"

# Sector names in the detailed (B) bulletin that may continue an A-series.
# A sector is only spliced if the two bulletins agree in the overlap weeks (see data.py).
B_CANDIDATES = {
    "Total": "20.Total",
    "Beverage and Food": "6.Food & Beverages",
    "Gas Stations": "17.Gas Stations",
    "Hotels": "5.Hotels",
    "Public Utilities": "16.Public Utilities & Services",
    "Transportation": "1. Transportation",
    "Health": "2.Health",
    "Restaurants & Café": "3.Restaurants & Café",
    "Clothing and Footwear": "7.Apparel, Clothing & Accessories",
    "Jewelry": "13.Jewelry",
    "Furniture": "11.Furniture & Home Supplies",
    "Electronic & Electric Devices": "10.Electronic & Electric Devices",
    "Construction & Building Materials": "12.Construction & Building Materials",
    "Telecommunication": "14.Telecommunication",
    "Education": "15.Education",
    "Recreation and Culture": "8.Recreation & Culture",
    "Other": "19.Others",
}
SPLICE_TOLERANCE = 0.01  # max relative gap in each overlap week

SEASON = 12
HORIZON = 6          # forecast and backtest horizon (months)
N_ORIGINS = 12       # rolling origins per series
GBM_MIN_TARGET = "2022-01-01"  # GBM targets before this compare against the 2020/21 COVID base
INTERVAL = (0.10, 0.90)  # 80% interval
SEED = 42
