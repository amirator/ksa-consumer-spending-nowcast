# Data sources

Both files are the Saudi Central Bank (SAMA) weekly point-of-sale (POS) bulletin, republished in machine-readable form on the KAPSARC Data Portal. Copied from Amir Khan's [ksa-pos-spending-dashboard](https://github.com/amirator/ksa-pos-spending-dashboard) project (same files, same SHA-256 in `manifest.json`, downloaded 17 Oct 2026). `make refresh` re-downloads them from the URLs below. On 30 Oct 2026 the portal's latest week was still 22 Mar 2026, so no newer data was available.

| Key | Dataset | Download URL (public, no key) | Landing page | Coverage used |
|---|---|---|---|---|
| A | Point of Sale Transactions by Sector and City | https://datasource.kapsarc.org/api/explore/v2.1/catalog/datasets/point-of-sale-transactions-by-sector-and-city/exports/csv?delimiter=%2C | https://datasource.kapsarc.org/explore/dataset/point-of-sale-transactions-by-sector-and-city/ | Weekly, 10 May 2020 to 6 Jul 2025, 17 sectors + total, national ("Total" city) |
| B | Detailed Point of Sale Transactions by Sector and City | https://datasource.kapsarc.org/api/explore/v2.1/catalog/datasets/detailed-point-of-sale-transactions-by-sector-and-city/exports/csv?delimiter=%2C | https://datasource.kapsarc.org/explore/dataset/detailed-point-of-sale-transactions-by-sector-and-city/ | Weekly, 29 Jun 2025 to 22 Mar 2026, new 2025 sector list |

License: labelled Public Domain on the KAPSARC Data Portal; source copyright SAMA. Units: value of transactions in SAR thousands (converted to SAR bn per calendar month by the pipeline).
