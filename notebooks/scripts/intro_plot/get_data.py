import requests
import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path
import matplotlib.pyplot as plt

####################### Gas consumption by country ############################################

"""
European natural gas consumption 2024 by country (TWh).

Fetches from Eurostat API (nrg_cb_gasm dataset).
Sums monthly values (Jan-Dec) to get annual totals.

Source: Eurostat, dataset nrg_cb_gasm
        Inland consumption, observed (IC_OBS)
        Natural gas (G3000), TJ_GCV -> converted to TWh (/3600)
"""

# -- Config --
DATASET = "nrg_cb_gasm"
SIEC = "G3000"
UNIT = "TJ_GCV"
NRG_BAL = "IC_OBS"
YEAR = 2024
BASE_URL = f"https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/{DATASET}"

EU27_CODES = [
    "AT", "BE", "BG", "CY", "CZ", "DE", "DK", "EE", "EL", "ES",
    "FI", "FR", "HR", "HU", "IE", "IT", "LT", "LU", "LV", "MT",
    "NL", "PL", "PT", "RO", "SE", "SI", "SK",
]
EXTRA_CODES = ["UK", "NO", "CH"]

COUNTRY_NAMES = {
    "AT": "Austria", "BE": "Belgium", "BG": "Bulgaria", "CY": "Cyprus",
    "CZ": "Czechia", "DE": "Germany", "DK": "Denmark", "EE": "Estonia",
    "EL": "Greece", "ES": "Spain", "FI": "Finland", "FR": "France",
    "HR": "Croatia", "HU": "Hungary", "IE": "Ireland", "IT": "Italy",
    "LT": "Lithuania", "LU": "Luxembourg", "LV": "Latvia", "MT": "Malta",
    "NL": "Netherlands", "PL": "Poland", "PT": "Portugal", "RO": "Romania",
    "SE": "Sweden", "SI": "Slovenia", "SK": "Slovakia",
    "UK": "United Kingdom", "NO": "Norway", "CH": "Switzerland",
}


def _parse_json_stat(js: dict) -> dict:
    """
    Parse Eurostat JSON-stat response.
    Returns {(geo_code, time_code): TJ_value}.
    """
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

    # Compute strides for each dimension
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


def _fetch_country_annual(geo: str, months: list) -> tuple:
    """Fetch all months for one country. Returns (total_TJ, n_months) or (None, 0)."""
    time_params = "&".join(f"time={t}" for t in months)
    url = (
        f"{BASE_URL}?geo={geo}&{time_params}"
        f"&siec={SIEC}&unit={UNIT}&nrg_bal={NRG_BAL}"
        f"&lang=EN&format=JSON"
    )
    try:
        r = requests.get(url, timeout=60)
        if r.status_code == 200:
            parsed = _parse_json_stat(r.json())
            total = sum(parsed.values())
            n = len(parsed)
            if total > 0:
                return total, n
        return None, 0
    except Exception:
        return None, 0


def fetch_from_eurostat(year: int = 2024) -> dict:
    """
    Fetch country-level natural gas inland consumption from Eurostat.
    Sums all 12 monthly values to get annual total.
    Returns {country_name: TWh}.
    """
    # Eurostat API time format: "YYYY-MM"
    months = [f"{year}-{m:02d}" for m in range(1, 13)]
    time_params = "&".join(f"time={t}" for t in months)

    results = {}  # geo_code -> annual TJ

    # -- Step 1: Batch fetch EU27 (all countries x 12 months) --
    geo_params = "&".join(f"geo={c}" for c in EU27_CODES)
    url = (
        f"{BASE_URL}?{geo_params}&{time_params}"
        f"&siec={SIEC}&unit={UNIT}&nrg_bal={NRG_BAL}"
        f"&lang=EN&format=JSON"
    )
    print(f"Fetching EU27 ({len(EU27_CODES)} countries x 12 months)...")
    batch_ok = False
    try:
        r = requests.get(url, timeout=120)
        if r.status_code == 200:
            parsed = _parse_json_stat(r.json())
            for (gc, tc), val in parsed.items():
                results[gc] = results.get(gc, 0.0) + val
            print(f"  OK: got data for {len(results)} countries")
            for gc in EU27_CODES:
                n_months = sum(1 for (g, t) in parsed if g == gc)
                if 0 < n_months < 12:
                    print(f"  Warning: {COUNTRY_NAMES[gc]} only {n_months}/12 months")
            batch_ok = True
        else:
            print(f"  Failed ({r.status_code}), falling back to per-country...")
    except Exception as e:
        print(f"  Exception: {e}, falling back to per-country...")

    if not batch_ok:
        for code in EU27_CODES:
            total, n = _fetch_country_annual(code, months)
            if total:
                results[code] = total
                print(f"  {COUNTRY_NAMES[code]:20s} {total:>12,.0f} TJ = {total/3600:>8.1f} TWh ({n}m)")
            else:
                print(f"  {COUNTRY_NAMES[code]:20s} no data")

    # -- Step 2: Fetch non-EU countries individually --
    print(f"\nFetching non-EU countries (UK, NO, CH)...")
    for code in EXTRA_CODES:
        total, n = _fetch_country_annual(code, months)
        if total:
            results[code] = total
            print(f"  {COUNTRY_NAMES[code]:20s} {total:>12,.0f} TJ = {total/3600:>8.1f} TWh ({n}m)")
        else:
            print(f"  {COUNTRY_NAMES[code]:20s} no data in nrg_cb_gasm")

    # -- Convert to TWh --
    out = {}
    for code, tj in results.items():
        name = COUNTRY_NAMES.get(code, code)
        out[name] = round(tj / 3600.0, 1)

    return out


# -- Run --
if __name__ == "__main__":
    print(f"European natural gas inland consumption {YEAR}")
    print(f"Source: Eurostat nrg_cb_gasm, IC_OBS, G3000, TJ_GCV")
    print("=" * 55)
    print()

    data = fetch_from_eurostat(YEAR)

    if data:
        eu27 = {k: v for k, v in data.items()
                if k not in ("United Kingdom", "Norway", "Switzerland")}
        extra = {k: v for k, v in data.items()
                 if k in ("United Kingdom", "Norway", "Switzerland")}

        print(f"\n{'='*55}")
        print(f"{'Country':<25} {'TWh':>8}")
        print(f"{'-'*55}")

        print("  EU27:")
        for k, v in sorted(eu27.items(), key=lambda x: -x[1]):
            print(f"  {k:<23} {v:>8.1f}")
        print(f"  {'-'*53}")
        print(f"  {'EU27 Total':<23} {sum(eu27.values()):>8.1f}")

        if extra:
            print(f"\n  Non-EU:")
            for k, v in sorted(extra.items(), key=lambda x: -x[1]):
                print(f"  {k:<23} {v:>8.1f}")

        print(f"\n  {'Grand Total':<23} {sum(data.values()):>8.1f}")

        # Print copy-paste dict
        print(f"\n{'='*55}")
        print("# Copy-paste Python dict:")
        print(f"# Source: Eurostat nrg_cb_gasm, IC_OBS, {YEAR}, TJ_GCV -> TWh")
        print(f"european_gas_consumption_{YEAR}_twh = {{")
        for k, v in sorted(data.items(), key=lambda x: -x[1]):
            print(f'    "{k}": {v},')
        print("}")
    else:
        print("No data retrieved.")

df_countries = pd.Series(data)
df_countries.loc['United Kingdom'] = 206 + 345 + 85
df_countries.loc['Switzerland'] = 26.5

df_countries.to_csv('gas_countries.csv')


####################### Gas supply ############################################

# Conversion: 1 bcm ≈ 10.55 TWh (gross calorific value)
BCM_TO_TWH = 10.55

# =============================================================================
# DF1: EU-27 + UK gas supply sources, 2024 annual totals
# Source: OIES Quarterly Gas Review Issue 29, Figures 11–12
# =============================================================================
df_supply = pd.DataFrame([
    {"type": "Domestic production", "origin": "UK",           "bcm_2024": 27.1, "twh_2024": 285.9, "lat": 54.00, "lon": -2.00, "coord_note": "UK centroid"},
    {"type": "Domestic production", "origin": "Netherlands",  "bcm_2024":  9.6, "twh_2024": 101.3, "lat": 52.13, "lon":  5.29, "coord_note": "Netherlands centroid"},
    {"type": "Domestic production", "origin": "Romania",      "bcm_2024":  8.6, "twh_2024":  90.7, "lat": 45.94, "lon": 24.97, "coord_note": "Romania centroid"},
    {"type": "Domestic production", "origin": "Rest of EU",   "bcm_2024": 13.2, "twh_2024": 139.3, "lat": 48.00, "lon": 10.00, "coord_note": "Approx. EU27 centroid"},
    {"type": "Pipeline import",    "origin": "Norway",        "bcm_2024": 121.8,"twh_2024": 1285.0,"lat": 53.60, "lon":  7.15, "coord_note": "Dornum, DE (Europipe receiving terminal)"},
    {"type": "Pipeline import",    "origin": "Russia",        "bcm_2024": 32.0, "twh_2024": 337.6, "lat": 42.10, "lon": 27.83, "coord_note": "Strandzha, BG (TurkStream EU entry)"},
    {"type": "Pipeline import",    "origin": "North Africa",  "bcm_2024": 32.2, "twh_2024": 339.7, "lat": 37.65, "lon": 12.59, "coord_note": "Mazara del Vallo, IT (TransMed landfall)"},
    {"type": "Pipeline import",    "origin": "Azerbaijan",    "bcm_2024": 12.3, "twh_2024": 129.8, "lat": 41.08, "lon": 26.36, "coord_note": "Kipoi, GR (TAP/TANAP border crossing)"},
    {"type": "Pipeline import",    "origin": "Other",         "bcm_2024":  0.4, "twh_2024":   4.2, "lat": 48.00, "lon": 10.00, "coord_note": "Generic EU location"},
    {"type": "LNG",                "origin": "Various",       "bcm_2024": 116.3,"twh_2024": 1227.0,"lat": 51.96, "lon":  4.02, "coord_note": "Gate Terminal, Rotterdam, NL"},
])

df_supply.to_csv('gas_supply.csv')

####################### Gas consumption by sector ############################################

# =============================================================================
# DF2: EU-27 + UK gas demand by sector, 2024 annual totals
# Source: OIES Quarterly Gas Review Issue 29, Figures 20–22
# =============================================================================
#
# DATASET improved on by the methods thereafter
#
# df_demand = pd.DataFrame([
#     ("Residential & commercial", 152.0),
#     ("Industry",                  83.2),
#     ("Power",                    117.9),
# ], columns=["sector", "bcm_2024"])

# df_demand = df_demand.set_index('sector')['twh_2024']
# df_demand["twh_2024"] = (df_demand["bcm_2024"] * BCM_TO_TWH).round(1)

DATASET = "nrg_bal_c"
GEO = "EU27_2020"
TIME = "2024"
SIEC = "G3000"      # Natural gas :contentReference[oaicite:3]{index=3}
UNIT = "TJ"         # Terajoule (energy content basis as provided by the balance)

BCM_TO_TWH = 10.5

# Eurostat balance items (nrg_bal) for your sectoral split :contentReference[oaicite:4]{index=4}
SECTORS = [
    ("Power sector (input to electricity & heat generation)", "TI_EHG_E"),
    ("Residential heating (households)", "FC_OTH_HH_E"),
    ("Services heating (commercial & public services)", "FC_OTH_CP_E"),
    ("Industry heating (energy use)", "FC_IND_E"),
    ("Industry feedstock (non-energy use)", "FC_IND_NE"),
]

BASE_URL = f"https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/{DATASET}"

def fetch_one(nrg_bal_code: str) -> float:
    params = {
        "geo": GEO,
        "time": TIME,
        "siec": SIEC,
        "unit": UNIT,
        "nrg_bal": nrg_bal_code,
        "lang": "EN",
        "format": "JSON",
    }
    r = requests.get(BASE_URL, params=params, timeout=60)
    r.raise_for_status()
    js = r.json()

    # JSON-stat: values are in js["value"], typically a dict {index: value}
    values = js.get("value", {})
    if not values:
        return float("nan")

    # There should be exactly 1 observation given the tight filters
    v = next(iter(values.values()))
    return float(v) if v is not None else float("nan")

rows = []
for sector_label, code in SECTORS:
    tj = fetch_one(code)
    rows.append({
        "sector": sector_label,
        "nrg_bal": code,
        "geo": GEO,
        "time": int(TIME),
        "unit": UNIT,
        "siec": SIEC,
        "value_TJ": tj,
        "value_TWh": tj / 3600.0,   # 1 TWh = 3600 TJ
    })

df = pd.DataFrame(rows)

# Convenience aggregates matching your requested buckets
df_agg_eu = pd.DataFrame([
    {
        "sector": "Power sector",
        "value_TJ": df.loc[df["nrg_bal"].eq("TI_EHG_E"), "value_TJ"].sum(),
    },
    {
        "sector": "Residential + Services heating",
        "value_TJ": df.loc[df["nrg_bal"].isin(["FC_OTH_HH_E", "FC_OTH_CP_E"]), "value_TJ"].sum(),
    },
    {
        "sector": "Industry heating",
        "value_TJ": df.loc[df["nrg_bal"].eq("FC_IND_E"), "value_TJ"].sum(),
    },
    {
        "sector": "Industry feedstock",
        "value_TJ": df.loc[df["nrg_bal"].eq("FC_IND_NE"), "value_TJ"].sum(),
    },
])
df_agg_eu["value_TWh"] = df_agg_eu["value_TJ"] / 3600.0
df_agg_eu["geo"] = GEO
df_agg_eu["time"] = int(TIME)
df_agg_eu["unit"] = "TJ/TWh"
df_agg_eu

# You now have:
# - df: raw-by-balance-item view (with Eurostat codes)
# - df_agg: exactly your 4 sector buckets

# %%
# Hard-coded dataframe with the specified values
df_agg_uk = pd.DataFrame([
    {
        "sector": "Power sector",
        "value_TJ": np.nan,
        "value_TWh": 206.,
        "geo": "UK",
        "time": 2024,
        "unit": "TJ/TWh"
    },
    {
        "sector": "Residential + Services heating",
        "value_TJ": np.nan,
        "value_TWh": 345,
        "geo": "UK",
        "time": 2024,
        "unit": "TJ/TWh"
    },
    {
        "sector": "Industry heating",
        "value_TJ": np.nan,
        "value_TWh": 85.,
        "geo": "UK",
        "time": 2024,
        "unit": "TJ/TWh"
    },
    {
        "sector": "Industry feedstock",
        "value_TJ": np.nan,
        "value_TWh": 0.,
        "geo": "UK",
        "time": 2024,
        "unit": "TJ/TWh"
    }
])

df_agg_uk

# %%
df_agg_no = pd.DataFrame([
    {
        "sector": "Power sector",
        "value_TJ": np.nan,
        "value_TWh": 0.,  # Not separately reported in SSB data
        "geo": "Norway",
        "time": 2024,
        "unit": "TJ/TWh"
    },
    {
        "sector": "Residential + Services heating",
        "value_TJ": np.nan,
        "value_TWh": 0.,  # Not separately reported in SSB data
        "geo": "Norway",
        "time": 2024,
        "unit": "TJ/TWh"
    },
    {
        "sector": "Industry heating",
        "value_TJ": np.nan,
        "value_TWh": 2.5,  # Manufacturing etc.
        "geo": "Norway",
        "time": 2024,
        "unit": "TJ/TWh"
    },
    {
        "sector": "Industry feedstock",
        "value_TJ": np.nan,
        "value_TWh": 1.5,  # Non energy consumption
        "geo": "Norway",
        "time": 2024,
        "unit": "TJ/TWh"
    }
])

df_agg_ch = pd.DataFrame([
    {
        "sector": "Total",
        "value_TJ": np.nan,
        "value_TWh": 26.5,
        "geo": "Switzerland",
        "time": 2024,
        "unit": "TJ/TWh"
    },
])

# Source notes:
# NORWAY (SSB energy balance table excerpt)
# Natural gas:
#   - Final consumption (11+12): 5.4 TWh
#   - Non energy consumption:    1.5 TWh
#   - Final energy consumption:  3.9 TWh
#   - Manufacturing etc.:        2.5 TWh
#   - Transport:                1.2 TWh
# (All from SSB "Supply and use of energy in Norway by energy product, Energy balance. TWh" for 2024)
# https://www.ssb.no/en/energi-og-industri/energi/statistikk/produksjon-og-forbruk-av-energi-energibalanse-og-energiregnskap  (StatBank table 11561)

# SWITZERLAND (SFOE extract, "Energy Consumption in Switzerland 2024")
# Gas final consumption (2024): 26.522 TWh (95'480 TJ)
# Gas used for district heating & electricity production: 1.825 TWh
# Imports of natural gas (2024): 27.911 TWh (100'480 TJ)
# https://pubdb.bfe.admin.ch/en/publication/download/12195

# %%
df_demand = pd.concat([df_agg_eu, df_agg_uk, df_agg_no, df_agg_ch])

df_demand.to_csv('gas_demand.csv')




