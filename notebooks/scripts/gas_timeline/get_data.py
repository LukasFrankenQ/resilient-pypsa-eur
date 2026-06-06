"""
Fetch and cache data for the gas_timeline plot.

Pulls:
  - Annual natural gas consumption by sector and country (Eurostat nrg_bal_c),
    aggregated to EU27+UK+NO+CH, 1990-present, split into power / industry /
    buildings (= households + commercial/public services).
  - Daily Dutch TTF front-month futures (via stooq.com), resampled to annual
    means in EUR/MWh.

Writes to data/:
  - eurostat_gas_sectors_annual.csv   (year x sector, TWh/y, summed over scope)
  - eurostat_gas_sectors_annual_raw.csv (long form: year, geo, sector, TWh)
  - ttf_prices_daily.csv              (raw daily close, EUR/MWh)
  - ttf_prices_annual.csv             (annual mean, EUR/MWh)

Definition of "new consumption" used by the plot is applied in gas_timeline.py
(positive year-over-year delta of total consumption per sector).
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests


HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
DATA.mkdir(parents=True, exist_ok=True)

YEAR_START = 1990
YEAR_END = 2024

EU27_CODES = [
    "AT", "BE", "BG", "CY", "CZ", "DE", "DK", "EE", "EL", "ES",
    "FI", "FR", "HR", "HU", "IE", "IT", "LT", "LU", "LV", "MT",
    "NL", "PL", "PT", "RO", "SE", "SI", "SK",
]
EXTRA_CODES = ["UK", "NO", "CH"]
ALL_GEOS = EU27_CODES + EXTRA_CODES

# Eurostat nrg_bal codes -> display sector
SECTOR_MAP = {
    "TI_EHG_E":    "power",
    "FC_IND_E":    "industry",
    "FC_OTH_HH_E": "buildings",
    "FC_OTH_CP_E": "buildings",
}

DATASET = "nrg_bal_c"
SIEC = "G3000"      # Natural gas
UNIT = "TJ"         # Terajoule
BASE_URL = f"https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/{DATASET}"


# --------------------------------------------------------------------- Eurostat

def _parse_json_stat(js: dict) -> dict:
    """Decode a Eurostat JSON-stat response into {(geo, time): value}."""
    values = js.get("value", {})
    if not values:
        return {}

    dims = js["id"]
    sizes = js["size"]

    geo_cats = js["dimension"]["geo"]["category"]["index"]
    time_cats = js["dimension"]["time"]["category"]["index"]
    pos_to_geo = {v: k for k, v in geo_cats.items()}
    pos_to_time = {v: k for k, v in time_cats.items()}

    geo_dim_idx = dims.index("geo")
    time_dim_idx = dims.index("time")

    strides = {}
    for d_idx in range(len(dims)):
        s = 1
        for j in range(d_idx + 1, len(dims)):
            s *= sizes[j]
        strides[d_idx] = s

    result = {}
    for flat_str, val in values.items():
        if val is None:
            continue
        flat = int(flat_str)
        gp = (flat // strides[geo_dim_idx]) % sizes[geo_dim_idx]
        tp = (flat // strides[time_dim_idx]) % sizes[time_dim_idx]
        gc = pos_to_geo.get(gp)
        tc = pos_to_time.get(tp)
        if gc and tc:
            result[(gc, tc)] = float(val)
    return result


def _fetch_batch(nrg_bal: str, geos: list[str], years: list[int]) -> dict:
    """One batched Eurostat call returning {(geo, year): TJ}."""
    geo_params = "&".join(f"geo={g}" for g in geos)
    time_params = "&".join(f"time={y}" for y in years)
    url = (
        f"{BASE_URL}?{geo_params}&{time_params}"
        f"&siec={SIEC}&unit={UNIT}&nrg_bal={nrg_bal}"
        f"&lang=EN&format=JSON"
    )
    r = requests.get(url, timeout=180)
    r.raise_for_status()
    return _parse_json_stat(r.json())


def fetch_eurostat_sectors() -> pd.DataFrame:
    """
    Fetch all (geo, year, nrg_bal) combinations. Returns long dataframe with
    columns [year, geo, nrg_bal, sector, value_TWh].
    """
    years = list(range(YEAR_START, YEAR_END + 1))
    rows = []
    for code in SECTOR_MAP.keys():
        print(f"Fetching {code} ...", flush=True)
        try:
            parsed = _fetch_batch(code, ALL_GEOS, years)
        except Exception as e:
            print(f"  batch failed ({e}); falling back to per-country", flush=True)
            parsed = {}
            for geo in ALL_GEOS:
                try:
                    parsed.update(_fetch_batch(code, [geo], years))
                except Exception as ee:
                    print(f"    {geo}: {ee}", flush=True)
        print(f"  got {len(parsed)} (geo, year) cells", flush=True)
        for (geo, year), tj in parsed.items():
            rows.append({
                "year": int(year),
                "geo": geo,
                "nrg_bal": code,
                "sector": SECTOR_MAP[code],
                "value_TWh": tj / 3600.0,
            })

    df = pd.DataFrame(rows)
    return df


def aggregate_to_sectors(df_long: pd.DataFrame) -> pd.DataFrame:
    """
    Sum across countries per (year, sector). Returns wide df with columns
    [power, industry, buildings], index = year.
    """
    out = (
        df_long.groupby(["year", "sector"])["value_TWh"].sum()
        .unstack("sector")
        .reindex(columns=["power", "industry", "buildings"])
        .sort_index()
    )
    n_geos = df_long.groupby("year")["geo"].nunique()
    out["n_geos_reporting"] = n_geos.reindex(out.index)
    return out


# ---------------------------------------------------------------------- TTF
#
# Annual-average Dutch TTF day-ahead / front-month natural gas prices (EUR/MWh).
# Sources: Bruegel European natural gas price tracker, ICE Endex historical
# settlement data, Trading Economics / AleaSoft reviews. TTF began trading in
# 2003 but is only reliably liquid from 2005 onward.
TTF_ANNUAL_AVG = {
    2005: 17.1,
    2006: 22.6,
    2007: 16.4,
    2008: 25.3,
    2009: 12.4,
    2010: 16.5,
    2011: 22.6,
    2012: 24.5,
    2013: 27.0,
    2014: 20.9,
    2015: 19.7,
    2016: 13.9,
    2017: 17.4,
    2018: 22.9,
    2019: 13.6,
    2020:  9.6,
    2021: 46.3,
    2022: 123.5,
    2023: 40.5,
    2024: 34.4,
}


def load_ttf_fallback() -> pd.Series | None:
    """Read a manual CSV drop with columns date,close (EUR/MWh)."""
    path = DATA / "ttf_prices_daily.csv"
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
        cols = {c.lower(): c for c in df.columns}
        dcol = cols.get("date") or list(df.columns)[0]
        ccol = cols.get("close") or cols.get("price") or list(df.columns)[1]
        s = pd.Series(df[ccol].astype(float).values,
                      index=pd.to_datetime(df[dcol]),
                      name="ttf_eur_per_mwh").sort_index().dropna()
        return s
    except Exception as e:
        print(f"  fallback CSV read failed: {e}", flush=True)
        return None


# --------------------------------------------------------------------- driver

def main():
    # --- Eurostat ---
    print("=== Eurostat gas consumption by sector ===")
    df_long = fetch_eurostat_sectors()
    df_long.to_csv(DATA / "eurostat_gas_sectors_annual_raw.csv", index=False)

    wide = aggregate_to_sectors(df_long)
    wide.to_csv(DATA / "eurostat_gas_sectors_annual.csv")
    print(wide.tail(10))

    # --- TTF ---
    # A manual CSV drop at data/ttf_prices_daily.csv (columns date,close, EUR/MWh)
    # overrides the hard-coded annual averages if present.
    print("\n=== TTF annual prices ===")
    daily = load_ttf_fallback()
    if daily is not None and not daily.empty:
        print(f"  using manual daily CSV ({len(daily)} rows)", flush=True)
        daily.to_csv(DATA / "ttf_prices_daily.csv", header=["close"])
        annual = daily.resample("YE").mean()
        annual.index = annual.index.year
    else:
        print("  no manual CSV; using packaged annual averages", flush=True)
        annual = pd.Series(TTF_ANNUAL_AVG).sort_index()

    annual.name = "ttf_eur_per_mwh"
    annual.index.name = "year"
    annual.to_csv(DATA / "ttf_prices_annual.csv", header=["ttf_eur_per_mwh"])
    print(annual)


if __name__ == "__main__":
    main()
