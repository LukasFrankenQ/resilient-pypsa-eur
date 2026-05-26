# %%
import sys
import requests
import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from frontend_export import export_frontend_data  # noqa: E402

world = gpd.read_file(Path(__file__).resolve().parent / "countries.json").set_index('sovereignt')

european_gas_consumption_2024_twh = {
    # --- EU27 countries (source: Eurostat nrg_cb_gasm, 2024) ---
    "Germany":      838.0,   # 3,016,962 TJ (exact from Eurostat)
    "Italy":        654.8,   # 2,357,157 TJ (exact from Eurostat)
    "France":       363.7,   # 1,309,178 TJ (exact from Eurostat)
    "Netherlands":  310.0,   # ~1,116,000 TJ (Eurostat database)
    "Spain":        310.0,   # ~1,116,000 TJ (Eurostat database)
    "Poland":       207.0,   # ~745,000 TJ
    "Belgium":      155.0,   # ~558,000 TJ
    "Romania":      105.0,   # ~378,000 TJ
    "Hungary":       90.0,   # ~324,000 TJ
    "Czechia":       78.0,   # ~281,000 TJ
    "Austria":       77.0,   # ~277,000 TJ
    "Greece":        68.0,   # ~245,000 TJ (↑31.3% YoY per Eurostat)
    "Ireland":       53.0,   # ~191,000 TJ
    "Slovakia":      49.0,   # ~176,000 TJ
    "Portugal":      44.0,   # ~158,000 TJ (↓16.7% YoY per Eurostat)
    "Denmark":       28.0,   # ~101,000 TJ
    "Bulgaria":      28.0,   # ~101,000 TJ
    "Finland":       18.0,   # ~65,000 TJ (↑9.5% YoY per Eurostat)
    "Croatia":       25.0,   # ~90,000 TJ (↓5.4% YoY per Eurostat)
    "Lithuania":     20.0,   # ~72,000 TJ (↑9.2% YoY per Eurostat)
    "Sweden":        11.0,   # ~40,000 TJ
    "Latvia":        11.0,   # ~40,000 TJ
    "Luxembourg":     7.0,   # ~25,000 TJ
    "Slovenia":       8.5,   # ~31,000 TJ
    "Estonia":        4.5,   # ~16,000 TJ
    "Malta":          0.7,   # ~2,500 TJ (↓7.1% YoY per Eurostat)
    "Cyprus":         0.0,   # no significant gas consumption
    # --- Non-EU European countries ---
    "United Kingdom": 690.0, # ~2,484,000 TJ (Bruegel / DESNZ Energy Trends)
    "Norway":          5.0,  # domestic consumption only (major exporter)
    "Switzerland":    33.0,  # ~119,000 TJ (Swiss Federal Office of Energy)
}

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

# %%
df_countries = pd.Series(data)
# UK total = Power 206 + Heat 345 + Industry 85 + Other 54 = 690 TWh (DESNZ, GCV)
df_countries.loc['United Kingdom'] = 206 + 345 + 85 + 54
df_countries.loc['Switzerland'] = 26.5
df_countries

# %%
import pandas as pd

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
    {"type": "Domestic production", "origin": "Norway",       "bcm_2024": 121.8,"twh_2024": 1285.0,"lat": 60.50, "lon":  8.50, "coord_note": "Norway centroid"},
    {"type": "Pipeline import",    "origin": "Russia",        "bcm_2024": 32.0, "twh_2024": 337.6, "lat": 42.10, "lon": 27.83, "coord_note": "Strandzha, BG (TurkStream EU entry)"},
    {"type": "Pipeline import",    "origin": "North Africa",  "bcm_2024": 32.2, "twh_2024": 339.7, "lat": 37.65, "lon": 12.59, "coord_note": "Mazara del Vallo, IT (TransMed landfall)"},
    {"type": "Pipeline import",    "origin": "Azerbaijan",    "bcm_2024": 12.3, "twh_2024": 129.8, "lat": 41.08, "lon": 26.36, "coord_note": "Kipoi, GR (TAP/TANAP border crossing)"},
    {"type": "Pipeline import",    "origin": "Other",         "bcm_2024":  0.4, "twh_2024":   4.2, "lat": 48.00, "lon": 10.00, "coord_note": "Generic EU location"},
    {"type": "LNG",                "origin": "LNG",           "bcm_2024": 116.3,"twh_2024": 1227.0,"lat": 45.50, "lon": -8.00, "coord_note": "Atlantic, W of France / NW of Spain"},
    # 2024 EU27 underground storage net drawdown: +152 TWh NCV (Eurostat nrg_bal_c, STK_CHG)
    # Converted to GCV: 152 × 1.108 ≈ 168 TWh. Not a geographic point — excluded from map.
    {"type": "Storage drawdown",   "origin": "Storage drawdown","bcm_2024": 16.0, "twh_2024": 168.0, "lat": 0.0,   "lon": 0.0,   "coord_note": "Aggregate EU27 storage; not mapped"},
])
# =============================================================================
# DF2: EU-27 + UK gas demand by sector, 2024 annual totals
# Source: OIES Quarterly Gas Review Issue 29, Figures 20–22
# =============================================================================
df_demand = pd.DataFrame([
    ("Residential & commercial", 152.0),
    ("Industry",                  83.2),
    ("Power",                    117.9),
], columns=["sector", "bcm_2024"])

df_demand["twh_2024"] = (df_demand["bcm_2024"] * BCM_TO_TWH).round(1)
df_demand = df_demand.set_index('sector')['twh_2024']

# %% [markdown]
# #### Getting sectoral consumption data

# %%
# --- Eurostat query settings ---
DATASET = "nrg_bal_c"
GEO = "EU27_2020"
TIME = "2024"
SIEC = "G3000"      # Natural gas :contentReference[oaicite:3]{index=3}
UNIT = "TJ"         # Terajoule (energy content basis as provided by the balance)

BCM_TO_TWH = 10.5

# nrg_bal_c reports natural-gas energy on NCV basis ('TJ' unit). The country
# bar (df_countries) uses nrg_cb_gasm with TJ_GCV. Multiply the sectoral split
# by this factor to put both bars on the same calorific basis.
GCV_NCV_RATIO = 1.108

# Eurostat balance items (nrg_bal) for the sectoral split. The first 5 codes
# are the headline gas-demand sectors; the remaining codes capture the
# residual (energy sector own use, transport, agriculture, distribution
# losses, statistical differences) that previously fell outside the chart.
SECTORS = [
    ("Power sector", "TI_EHG_E"),
    ("Residential + Services heating", "FC_OTH_HH_E"),
    ("Residential + Services heating", "FC_OTH_CP_E"),
    ("Industry heating", "FC_IND_E"),
    ("Industry feedstock", "FC_IND_NE"),
    ("Other", "FC_TRA_E"),
    ("Other", "FC_OTH_AF_E"),
    ("Other", "FC_OTH_FISH_E"),
    ("Other", "FC_OTH_NSP_E"),
    ("Other", "NRG_E"),
    ("Other", "DL"),
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
    {
        "sector": "Other",
        "value_TJ": df.loc[df["nrg_bal"].isin(
            ["FC_TRA_E", "FC_OTH_AF_E", "FC_OTH_FISH_E", "FC_OTH_NSP_E", "NRG_E", "DL"]
        ), "value_TJ"].sum(),
    },
])
df_agg_eu["value_TWh"] = df_agg_eu["value_TJ"] / 3600.0 * GCV_NCV_RATIO
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
    },
    {
        # Residual to reconcile with DESNZ UK total of ~690 TWh
        # (transport, energy own use, agriculture, distribution losses).
        "sector": "Other",
        "value_TJ": np.nan,
        "value_TWh": 54.,
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
df = pd.concat([df_agg_eu, df_agg_uk, df_agg_no, df_agg_ch])


sector_colors = {
    "Power sector": "#006400",           # Dark green
    "Residential + Services heating": "#228B22",  # Forest green
    "Industry heating": "#32CD32",       # Lime green
    "Industry feedstock": "#90EE90",     # Light green
    "Total": "#98FB98",                  # Pale green
}

sector_nicenames = {
    "Power sector": "Power\nsector",
    "Residential + Services heating": "Building\nheating",
    "Industry heating": "Industry\nheating",
    "Industry feedstock": "Industry\nfeedstock",
    "Other": "Other",
    "Total": "Total"
}

# %%
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import geopandas as gpd
from matplotlib.colors import to_rgba

# =============================================================================
# DATA — replace these with your real data
# =============================================================================

# --- Assume gdf is your GeoDataFrame with country geometries ---
# It needs at minimum a 'geometry' column and a 'name' column.
# For demo purposes we load Natural Earth:
gdf = world.copy()
# gdf = gdf[gdf['name'] != 'Russia']

# --- Scatter points (dummy) ---
point_colors_map = ['#e63946', '#457b9d', '#e9c46a']  # red, blue, gold
color_labels = ['Category A', 'Category B', 'Category C']

np.random.seed(42)
pts_lon = np.array([2.3, -3.7, 10.4, 14.4, 25.0, 18.1, -1.2, 12.5, 21.0, 28.9])
pts_lat = np.array([48.8, 40.4, 51.2, 50.1, 46.8, 59.3, 53.5, 41.9, 64.1, 41.0])
pts_sizes = np.array([60, 180, 120, 300, 90, 250, 150, 200, 100, 220])
pts_cat = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])

# --- Bar chart data (dummy) ---
bar_labels = ['Source', 'Country\nConsumption', 'Sectoral\nConsumption']
total = 100

bar1_vals = [35, 40, 25]
bar1_colors = point_colors_map
bar1_texts = ['Segment X', 'Segment Y', 'Segment Z']

blues = plt.cm.Blues(np.linspace(0.3, 0.7, 4))
bar2_vals = [20, 30, 25, 25]
bar2_colors = blues
bar2_texts = ['Block P', 'Block Q', 'Block R', 'Block S']

greens = plt.cm.Greens(np.linspace(0.3, 0.7, 5))
bar3_vals = [10, 20, 25, 30, 15]
bar3_colors = greens
bar3_texts = ['Part 1', 'Part 2', 'Part 3', 'Part 4', 'Part 5']

# --- Sea labels ---
seas = {
    'Atlantic\nOcean': (-13, 43),
    'North\nSea': (3, 56),
    'Mediterranean Sea': (3, 38.2),   # just south of Mallorca
    'Baltic\nSea': (20, 58),
    'Black\nSea': (34, 43.5),
}

# =============================================================================
# FIGURE
# =============================================================================

# Reproject to EPSG:3035 (LAEA Europe) for a properly-shaped European map that
# is naturally wider than tall (vertical compression compared to plain lon/lat).
from pyproj import Transformer
_MAP_CRS = "EPSG:3035"
_TO_MAP = Transformer.from_crs("EPSG:4326", _MAP_CRS, always_xy=True).transform

def _proj_xy(lon, lat):
    return _TO_MAP(lon, lat)

gdf = gdf.to_crs(_MAP_CRS)

# Slightly smaller and less tall than before so text appears a bit bigger and
# the overall layout is vertically compressed. wspace is minimal so the bar
# chart sits right next to the map.
fig, (ax_map, ax_bar) = plt.subplots(
    1, 2, figsize=(16, 6.8),
    gridspec_kw={'width_ratios': [2.2, 1], 'wspace': 0.07}
)

# =============================================================================
# LEFT: Map of Europe
# =============================================================================

gdf.plot(ax=ax_map, color='white', edgecolor='#cccccc', linewidth=0.6)

# Extent in EPSG:3035 metres — chosen to cover gas-relevant Europe including
# Norway, and to roughly match the map axis aspect (keeps set_aspect('equal')
# without big blank margins).
ax_map.set_xlim(1800000, 6750000)
ax_map.set_ylim(1250000, 4500000)
ax_map.set_facecolor('#e8e8e8')
ax_map.set_aspect('equal')
ax_map.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
# for spine in ax_map.spines.values():
    # spine.set_visible(False)

import pycountry

# Country code labels at centroids
for _, row in gdf.iterrows():
    pt = row.geometry.representative_point()
    country_name = row['name']
    # Find the alpha_2 code using pycountry, fallback to name if not found
    try:
        # Some names may not be a direct match; try catch those
        country = pycountry.countries.lookup(country_name)
        label = country.alpha_2
    except LookupError:
        continue
        # label = country_name  # fallback to name if not resolved
    if ax_map.get_xlim()[0] < pt.x < ax_map.get_xlim()[1] and \
       ax_map.get_ylim()[0] < pt.y < ax_map.get_ylim()[1]:
        ax_map.text(pt.x, pt.y, label,
                    ha='center', va='center', fontsize=7, color='#999999', zorder=3)

# Sea labels (reprojected to map CRS)
for name, (sx, sy) in seas.items():
    px, py = _proj_xy(sx, sy)
    ax_map.text(px, py, name, ha='center', va='center',
                fontsize=7.5, color='#666666', style='italic', zorder=3)

# -----------------------------------------------------------------------------
# Gas pipeline infrastructure (SciGrid-gas), inspired by aegis.py
# -----------------------------------------------------------------------------
import json as _json

def _load_scigrid_pipelines(fn):
    df = gpd.read_file(fn)
    param = df.param.apply(_json.loads).apply(pd.Series)
    cols = ["diameter_mm", "max_cap_M_m3_per_d"]
    method = df.method.apply(_json.loads).apply(pd.Series)[cols]
    method.columns = method.columns + "_method"
    df = pd.concat([df, param, method], axis=1)
    to_drop = df.columns.intersection(["param", "uncertainty", "method", "tags"])
    df.drop(to_drop, axis=1, inplace=True)
    return df

# Resolve infrastructure paths relative to this script so the layer renders
# regardless of the user's CWD. Plot at low alpha so it sits in the background.
_repo_root = Path(__file__).resolve().parents[3]
_pipelines_fn = _repo_root / "data/gas_network/scigrid-gas/data/IGGIELGN_PipeSegments.geojson"
_pipeline_centroids = {}
if _pipelines_fn.exists():
    gas_pipes = _load_scigrid_pipelines(_pipelines_fn)
    gas_pipes = gas_pipes.query("length_km < 1000").to_crs(_MAP_CRS)

    # Clip the SciGrid network to European countries so non-EU portions
    # (Russia, Turkey, North Africa) don't show under the orange import-route
    # overlay we draw next.
    _EU_LIKE = {
        'Austria','Belgium','Bulgaria','Croatia','Cyprus','Czechia','Denmark',
        'Estonia','Finland','France','Germany','Greece','Hungary','Ireland',
        'Italy','Latvia','Lithuania','Luxembourg','Malta','Netherlands','Poland',
        'Portugal','Romania','Slovakia','Slovenia','Spain','Sweden',
        'United Kingdom','Norway','Switzerland','Iceland',
        'Albania','Bosnia and Herz.','Kosovo','Montenegro','North Macedonia','Serbia',
        'Belarus','Moldova','Ukraine',
        'Liechtenstein','San Marino','Monaco','Andorra','Faroe Is.','Isle of Man',
    }
    _europe_polygon = gdf[gdf['name'].isin(_EU_LIKE)].geometry.union_all()
    gas_pipes = gpd.clip(gas_pipes, _europe_polygon)

    _pipe_lw = (gas_pipes["max_cap_M_m3_per_d"].fillna(0) / 60).clip(0.2, 2.0)
    gas_pipes.plot(ax=ax_map, color='#8B5A2B', linestyle='-',
                   linewidth=_pipe_lw, alpha=0.25, zorder=2)

    # -------------------------------------------------------------------------
    # Custom geometries for the OUTSIDE-Europe portions of the actual 2024
    # EU+UK import pipelines, drawn from the source country up to the European
    # border. Inside Europe, the (gray) SciGrid network already shows the route.
    # Routes from Bruegel European Gas Tracker, OIES Q29, ENTSOG flow maps.
    # -------------------------------------------------------------------------
    from shapely.geometry import LineString as _LS
    _corridors = {
        "TurkStream": _LS([
            (41.00, 45.30),    # deeper into RU (Krasnodar Krai)
            (39.50, 45.10),
            (37.50, 44.70),    # Russkaya (RU export, Anapa)
            (33.00, 43.50),    # mid Black Sea
            (28.50, 41.70),    # Kıyıköy (TR Black Sea coast)
            (27.83, 42.05),    # Strandzha (TR-BG entry)
        ]),
        "Brotherhood (UA transit)": _LS([
            (44.00, 52.50),    # deeper into RU (Volga region)
            (41.00, 51.80),
            (38.00, 51.00),    # RU-UA border (near Sudzha)
            (35.00, 50.40),
            (30.50, 50.40),    # Kyiv
            (26.50, 49.50),
            (24.00, 48.90),
            (22.10, 48.62),    # Velke Kapusany (UA-SK entry)
        ]),
        "TransMed": _LS([
            (3.30, 32.90),     # Hassi R'Mel gas field, DZ (interior)
            (5.50, 34.50),
            (7.00, 36.00),
            (8.30, 36.85),     # El Haouaria, TN coast
            (10.20, 36.80),    # Cap Bon, TN
            (11.20, 37.20),    # mid Sicilian channel
            (12.59, 37.65),    # Mazara del Vallo (TN-IT landfall, EU border)
        ]),
        "Medgaz": _LS([
            (3.30, 32.90),     # Hassi R'Mel gas field, DZ (interior)
            (1.50, 34.20),
            (-0.30, 35.70),    # Beni Saf, DZ (export point)
            (-1.20, 36.00),    # offshore Mediterranean
            (-2.27, 36.78),    # Almería (DZ-ES landfall, EU border)
        ]),
        "TAP / TANAP": _LS([
            (41.50, 40.40),    # NE Turkey, near GE border (TANAP entry)
            (39.50, 39.90),
            (37.00, 39.50),    # eastern TR (TANAP route)
            (34.50, 39.80),
            (32.00, 40.50),    # central TR
            (29.50, 40.80),
            (28.00, 41.10),    # Edirne TR
            (26.80, 41.05),
            (26.31, 41.00),    # Kipoi (TR-GR entry)
        ]),
    }
    # Frozen scatter-dot positions (lon, lat) for each origin — kept constant
    # so the labels don't move when the pipelines are extended deeper into
    # the source regions. These are the OLD avg-of-starts positions.
    _FIXED_DOT_LONLAT = {
        "Russia":       (37.75, 47.85),
        "North Africa": (4.00, 36.275),
        "Azerbaijan":   (32.00, 40.50),
    }
    _origin_to_corridors = {
        "Russia":       ["TurkStream", "Brotherhood (UA transit)"],
        "North Africa": ["TransMed", "Medgaz"],
        "Azerbaijan":   ["TAP / TANAP"],
    }

    _orange = "#d97a1a"
    _corridor_starts_3035 = {}  # origin (lon, lat) of each pipeline start
    _corridor_lines_3035 = {}   # full LineString (EPSG:3035) for nearest-point
    for _cname, _line in _corridors.items():
        _line_3035 = gpd.GeoSeries([_line], crs="EPSG:4326").to_crs(_MAP_CRS)
        _line_3035.plot(ax=ax_map, color=_orange, linewidth=2.0,
                        alpha=0.85, zorder=4)
        # First vertex of the LineString = source-side starting point.
        _start = _line_3035.iloc[0].coords[0]
        _corridor_starts_3035[_cname] = _start
        _corridor_lines_3035[_cname] = _line_3035.iloc[0]

    # Pipelines that physically connect to Europe but delivered no gas in 2024
    # (Nord Stream 1+2 sabotaged 2022, Yamal-Europe halted by RU 2022,
    # Maghreb-Europe closed by DZ Nov 2021). Same outside-Europe → border
    # treatment as the active corridors, but dashed and dark grey.
    _inactive_corridors = {
        "Nord Stream": _LS([
            (33.00, 60.80),    # deeper into RU (Karelia / Lake Ladoga area)
            (30.50, 60.50),
            (28.40, 60.30),    # Vyborg, RU (NS1 onshore start)
            (24.00, 59.50),
            (20.00, 58.00),
            (16.00, 55.80),
            (13.60, 54.14),    # Greifswald-Lubmin, DE landfall
        ]),
        "Yamal-Europe": _LS([
            (42.00, 56.50),    # deeper into RU (Nizhny Novgorod region)
            (38.00, 55.50),
            (35.00, 54.50),    # Torzhok area RU
            (30.00, 53.50),    # Belarus
            (24.00, 52.80),    # eastern Poland
            (18.00, 52.50),    # central Poland
            (14.45, 52.34),    # Mallnow (PL-DE crossing)
        ]),
        "Maghreb-Europe": _LS([
            (3.30, 32.90),     # Hassi R'Mel gas field, DZ (interior)
            (1.00, 33.50),
            (-1.50, 35.00),    # west feeder
            (-3.50, 35.50),    # Algeria-Morocco border
            (-5.50, 35.60),    # Tangier MA
            (-5.60, 36.00),    # Tarifa ES landfall
        ]),
    }
    for _cname, _line in _inactive_corridors.items():
        _line_3035 = gpd.GeoSeries([_line], crs="EPSG:4326").to_crs(_MAP_CRS)
        _line_3035.plot(ax=ax_map, color='#444444', linewidth=2.0,
                        linestyle='--', alpha=0.7, zorder=3.5)

    # Place each origin's scatter dot at the FROZEN position (independent of
    # the corridor extension). For multi-pipeline origins draw a dotted leader
    # from the dot to the nearest point on each pipeline. Per request, North
    # Africa uses a single horizontal leader going right until it intersects
    # the corridor (instead of two slanted leaders to each pipeline).
    from shapely.geometry import Point as _Pt, LineString as _LSeg
    from shapely.ops import nearest_points as _nearest
    for _origin, _cnames in _origin_to_corridors.items():
        if _origin not in _FIXED_DOT_LONLAT:
            continue
        _lon_c, _lat_c = _FIXED_DOT_LONLAT[_origin]
        _ax, _ay = _proj_xy(_lon_c, _lat_c)
        _mask = df_supply["origin"] == _origin
        df_supply.loc[_mask, "lon"] = _lon_c
        df_supply.loc[_mask, "lat"] = _lat_c
        _lines = [_corridor_lines_3035[c] for c in _cnames if c in _corridor_lines_3035]
        if len(_lines) <= 1:
            continue
        if _origin == "North Africa":
            _ray = _LSeg([(_ax, _ay), (_ax + 5_000_000, _ay)])
            _hits = []
            for _ln in _lines:
                _it = _ray.intersection(_ln)
                if _it.is_empty:
                    continue
                if hasattr(_it, "geoms"):
                    _hits.extend(list(_it.geoms))
                else:
                    _hits.append(_it)
            if _hits:
                _closest = min(_hits, key=lambda p: p.x - _ax)
                ax_map.plot([_ax, _closest.x], [_ay, _ay],
                            color='black', lw=0.9, alpha=0.7,
                            linestyle=':', zorder=4.5)
        else:
            _dot = _Pt(_ax, _ay)
            for _ln in _lines:
                _, _np = _nearest(_dot, _ln)
                ax_map.plot([_ax, _np.x], [_ay, _np.y],
                            color='black', lw=0.9, alpha=0.7,
                            linestyle=':', zorder=4.5)

# -----------------------------------------------------------------------------
# Industrial sites (gas-consuming subsectors), from Industrial_Database.csv
# -----------------------------------------------------------------------------
_industry_fn = _repo_root / "data/Industrial_Database.csv"
if _industry_fn.exists():
    _ind = pd.read_csv(_industry_fn, sep=";", index_col=0)
    _ind[["srid", "coordinates"]] = _ind["geom"].str.split(";", expand=True)
    _ind = _ind.dropna(subset=["coordinates"]).copy()
    _gas_subsectors = ["Chemical industry", "Refineries", "Iron and steel", "Cement"]
    _ind = _ind[_ind["Subsector"].isin(_gas_subsectors)]
    _ind["coordinates"] = gpd.GeoSeries.from_wkt(_ind["coordinates"])
    industry_gdf = gpd.GeoDataFrame(_ind, geometry="coordinates", crs="EPSG:4326").to_crs(_MAP_CRS)
    _ind_size = (industry_gdf["Emissions_ETS_2014"].fillna(0) / 50_000).clip(4, 40)
    ax_map.scatter(
        industry_gdf.geometry.x, industry_gdf.geometry.y,
        s=_ind_size, marker='s', c='#555555', alpha=0.25,
        edgecolors='none', zorder=3,
    )

# Color mapping for source types
source_palette = {
    "Domestic production": "#4287f5",
    "Pipeline import": "#f59b42",
    "LNG": "#22a060",
    "Storage drawdown": "#888888",
}
df_supply["color"] = df_supply["type"].map(source_palette)

# Marker size proportional to TWh (scatter s = area in points²).
# This makes the 1000-TWh circle exactly 10x the area of the 100-TWh circle.
import numpy as np
_SIZE_PER_TWH = 0.55
df_supply["marker_size"] = df_supply["twh_2024"] * _SIZE_PER_TWH

# Scatter points: position by lon/lat (reprojected to map CRS), color by type, size by twh_2024
# Also attach a small framed callout label naming the origin (country / route / LNG).
# Per-point offsets (in projected metres) to nudge callouts away from neighbours.
_LABEL_OFFSETS = {
    # origin: (dx, dy) in metres — tight offsets so labels stay close to points.
    # Russia (42.1°N) sits slightly north of Azerbaijan (41.08°N), so its label
    # is placed above-right and Azerbaijan's below-right, preserving vertical
    # order between point and label.
    "UK":           (-160_000,  -70_000),
    "Netherlands":  ( 160_000,  -90_000),
    "Romania":      ( 130_000,  110_000),
    "Rest of EU":   (-120_000, -160_000),
    "Norway":       ( 180_000,   60_000),
    "Russia":       ( 150_000,  110_000),
    "Azerbaijan":   ( 150_000, -110_000),
    "North Africa": (-150_000,   80_000),
    "LNG":          (-150_000,   80_000),
}
_callout_bbox = dict(boxstyle="round,pad=0.25", fc="white", ec="#555555", lw=0.6, alpha=0.95)

for _, row in df_supply.iterrows():
    if row["twh_2024"] < 5:
        continue  # skip tiny 'Other' category — its (48°N, 10°E) overlaps Rest of EU
    if row["type"] == "Storage drawdown":
        continue  # aggregate quantity, not a geographic source
    _px, _py = _proj_xy(row["lon"], row["lat"])
    ax_map.scatter(
        _px, _py,
        s=row["marker_size"],
        c=row["color"],
        edgecolors='white',
        linewidths=0.8,
        zorder=5, alpha=0.85,
        label=row["type"]  # legend deduplication will be handled next
    )
    dx, dy = _LABEL_OFFSETS.get(row["origin"], (300_000, 200_000))
    # Origins whose points are large enough that the label can overlap the
    # marker directly — no leader line needed. Russia / North Africa /
    # Azerbaijan also drop the leader-to-text since their dot is already
    # connected to the pipeline by its own black line.
    _no_leader = {"LNG", "Norway", "Netherlands", "North Africa",
                  "Russia", "Azerbaijan"}
    _arrow = None if row["origin"] in _no_leader else dict(
        arrowstyle="-", color="black", lw=1.2, shrinkA=0, shrinkB=3,
    )
    ax_map.annotate(
        row["origin"],
        xy=(_px, _py),
        xytext=(_px + dx, _py + dy),
        fontsize=7.5, color="#222222", weight="medium",
        ha="center", va="center",
        bbox=_callout_bbox,
        arrowprops=_arrow,
        zorder=6,
    )
    # Optional: annotate with origin
    # ax_map.text(row["lon"], row["lat"], row["origin"], fontsize=7, ha='center', va='bottom', color='black' if row["type"] != "Pipeline import" else "#444444", zorder=6)

# Legend — source types (domestic / pipeline / LNG)
from matplotlib.lines import Line2D
source_handles = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor=source_palette[t],
           markersize=9, label=t, markeredgecolor='white', lw=0.5)
    for t in df_supply["type"].unique()
]
leg_source = ax_map.legend(handles=source_handles, loc='upper left', frameon=True,
                           framealpha=0.9, edgecolor='#cccccc', fontsize=8,
                           title='Source type', title_fontsize=8.5,
                           bbox_to_anchor=(0.01, 0.985))
ax_map.add_artist(leg_source)

# Legend — infrastructure (pipeline + industry) directly below Source type
infra_handles = [
    Line2D([0], [0], color='#8B5A2B', lw=1.8, alpha=0.7, label='Gas pipeline'),
    Line2D([0], [0], marker='s', color='w', markerfacecolor='#555555',
           markersize=7, alpha=0.7, label='Industrial site', lw=0),
    Line2D([0], [0], color="#d97a1a", lw=2.0, alpha=0.85,
           label='Importing pipeline (2024)'),
    Line2D([0], [0], color='#444444', lw=2.0, linestyle='--', alpha=0.7,
           label='Inactive pipeline'),
]
leg_infra = ax_map.legend(handles=infra_handles, loc='upper left', frameon=True,
                          framealpha=0.9, edgecolor='#cccccc', fontsize=8,
                          title='Infrastructure', title_fontsize=8.5,
                          bbox_to_anchor=(0.01, 0.7925))
ax_map.add_artist(leg_infra)

# Legend — sizes (TWh order-of-magnitude)
import matplotlib as mpl
import math
twh_vals_for_legend = [100, 500]
size_scalings_for_legend = [v * _SIZE_PER_TWH for v in twh_vals_for_legend]
size_labels_for_legend = [f"{int(v)}" for v in twh_vals_for_legend]
size_handles = [
    ax_map.scatter([], [], s=s, c='gray', edgecolors='white',
                   linewidths=0.7, label=l)
    for s, l in zip(size_scalings_for_legend, size_labels_for_legend)
]
ax_map.legend(handles=size_handles, loc='upper left', frameon=True,
              framealpha=0.9, edgecolor='#cccccc', fontsize=8,
              bbox_to_anchor=(0.01, 0.62), handletextpad=1.3, scatterpoints=1,
              title='2024 gas supply (TWh)', title_fontsize=8.5)

# =============================================================================
# RIGHT: Vertical stacked bar charts
# =============================================================================

all_bars = [
    (bar1_vals, bar1_colors, bar1_texts),
    (bar2_vals, bar2_colors, bar2_texts),
    (bar3_vals, bar3_colors, bar3_texts),
]

x_positions = [0]
bar_width = 0.55

# -----------------------------------------------------------------------------
# Per-bar distinct colormaps (light -> dark single-hue ramps)
# Each bar has its own colormap so segments within a bar are related by
# interpolation but the three bars are visually distinct.
# -----------------------------------------------------------------------------
from matplotlib.colors import to_rgba

def _shade(cmap, i, n):
    """Pick a shade in [0.25, 0.95] to avoid near-white (bad label contrast)."""
    if n <= 1:
        return cmap(0.7)
    return cmap(0.25 + 0.7 * i / (n - 1))

def _text_color_for(rgba):
    lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
    return 'white' if lum < 0.55 else '#333333'

width = 0.6

# --- Bar 2 (x=2): sector demand, Greens ramp ---
_cmap_sector = plt.get_cmap('Greens')
_sector_series = (df.replace('Total', 'Industry heating')
                    .groupby("sector")['value_TWh'].sum()
                    .sort_values(ascending=True))
bottom = 0
for i, (name, value) in enumerate(_sector_series.items()):
    color = _shade(_cmap_sector, i, len(_sector_series))
    ax_bar.bar(2, value, bottom=bottom, color=color, width=width, edgecolor='k')
    if name == 'Industry feedstock':
        # Segment is too thin for an in-place label — float it to the left
        # with a longer leader line so the callout is clearly readable.
        ax_bar.annotate(
            sector_nicenames[name],
            xy=(2 - width / 2, bottom + value / 2),
            xytext=(2 - 0.75, bottom + value + 280),
            ha='center', va='center',
            fontsize=7.5, color='k', weight='medium',
            bbox=dict(boxstyle='round,pad=0.2', fc='white',
                      ec='#555555', lw=0.6, alpha=0.9),
            arrowprops=dict(arrowstyle='-', color='k', lw=0.9),
            zorder=5,
        )
    else:
        ax_bar.text(2, bottom + value / 2, sector_nicenames[name],
                    ha='center', va='center',
                    fontsize=7.5, color=_text_color_for(color), weight='medium')
    bottom += value


# --- Bar 1 (x=1): country consumption, Purples ramp ---
import pycountry
_cmap_country = plt.get_cmap('Purples')
_country_series = df_countries.sort_values()
_n_country = len(_country_series)
bottom = 0
for i, (name, value) in enumerate(_country_series.items()):
    color = _shade(_cmap_country, i, _n_country)
    ax_bar.bar(1, value, bottom=bottom, color=color, width=width, edgecolor='k')
    if value > 100:
        try:
            alt_names = {"Czechia": "Czech Republic",
                         "United Kingdom": "United Kingdom"}
            country_obj = pycountry.countries.lookup(alt_names.get(name, name))
            country_code = country_obj.alpha_2
        except Exception:
            country_code = name
        ax_bar.text(1, bottom + value / 2, country_code,
                    ha='center', va='center',
                    fontsize=7.5, color=_text_color_for(to_rgba(color)),
                    weight='medium')
    bottom += value


# --- Bar 0 (x=0): supply by origin, Blues ramp ---
# Custom stacking (bottom → top):
#   Domestic production  (ascending within group, so Norway sits just above UK)
#   Pipeline imports     (Azerbaijan, N. Africa, Russia — Russia now above N.Africa)
#   LNG on top
_cmap_supply = plt.get_cmap('Blues')
_supply_pool = df_supply[df_supply['twh_2024'] >= 5].copy()
_domestic = _supply_pool[_supply_pool['type'] == 'Domestic production'] \
              .sort_values('twh_2024', ascending=True)
_pipeline = _supply_pool[_supply_pool['type'] == 'Pipeline import'] \
              .set_index('origin').reindex(['Azerbaijan', 'North Africa', 'Russia']).dropna() \
              .reset_index()
_lng = _supply_pool[_supply_pool['type'] == 'LNG']
_storage = _supply_pool[_supply_pool['type'] == 'Storage drawdown']
_supply_sorted = pd.concat([_domestic, _pipeline, _lng, _storage], ignore_index=True)
bottom = 0
_n_blue = len(_supply_sorted) - len(_storage)  # storage gets its own grey
for i, row in _supply_sorted.iterrows():
    value = row['twh_2024']
    is_storage = row['type'] == 'Storage drawdown'
    if is_storage:
        color = source_palette['Storage drawdown']
    else:
        color = _shade(_cmap_supply, i, _n_blue)
    ax_bar.bar(0, value, bottom=bottom, color=color, width=width, edgecolor='k')
    if is_storage:
        # Storage segment is too thin and sits at the top of the bar — float
        # the label to the right with a leader line, like Industry feedstock.
        ax_bar.annotate(
            'Storage\ndrawdown',
            xy=(0 + width / 2, bottom + value / 2),
            xytext=(0 + 0.55, bottom + value / 2),
            ha='center', va='center',
            fontsize=7.5, color='k', weight='medium',
            bbox=dict(boxstyle='round,pad=0.2', fc='white',
                      ec='#555555', lw=0.6, alpha=0.9),
            arrowprops=dict(arrowstyle='-', color='k', lw=0.9),
            zorder=5,
        )
    elif value > 50:
        ax_bar.text(0, bottom + value / 2, row['origin'],
                    ha='center', va='center',
                    fontsize=7.5, color=_text_color_for(to_rgba(color)),
                    weight='medium')
    bottom += value

# -----------------------------------------------------------------------------
# Threshold lines on top of the bars: No-LNG (2750 TWh) and Autarky (2000 TWh).
# Framed labels float to the left and may overlap the map (clip_on=False).
# -----------------------------------------------------------------------------
# Left end pulled past the axis spine so labels + lines shift ~half a label
# width to the left (over the map); clip_on=False lets them render off-axis.
_line_x_left = -0.55
for _y, _lbl in [(2750, "No-LNG"), (2000, "Autarky")]:
    ax_bar.plot([_line_x_left, 2.55], [_y, _y], color='red', linestyle='--',
                linewidth=1.3, zorder=6, clip_on=False)
    ax_bar.text(_line_x_left, _y, _lbl, ha='right', va='center',
                fontsize=9, color='red', weight='medium',
                bbox=dict(boxstyle='round,pad=0.3', fc='white',
                          ec='red', lw=0.8, alpha=0.95),
                zorder=7, clip_on=False)

'''
for xi, (vals, colors, texts) in zip(x_positions, all_bars):
    bottom = 0
    for v, c, t in zip(vals, colors, texts):
        ax_bar.bar(xi, v, bottom=bottom, width=bar_width, color=c,
                   # edgecolor='white',
                   # linewidth=1.2
                   )
        rgba = to_rgba(c)
        lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
        tc = 'white' if lum < 0.55 else '#333333'
        if v >= 12:
            ax_bar.text(xi, bottom + v / 2, t, ha='center', va='center',
                        fontsize=7.5, color=tc, weight='medium')
        bottom += v
'''

# Place title inside the bar axes, just below the top spine of the map.
ax_bar.text(0.5, 0.97, "2024 European Gas Balance",
            transform=ax_bar.transAxes,
            ha='center', va='top',
            fontsize=12, weight='bold')
ax_bar.set_xticks(list(range(3)))
ax_bar.set_xticklabels(bar_labels, fontsize=10)
ax_bar.set_ylim(0, 4800)
# Tight x-limits so the first bar sits flush against the map on the left.
ax_bar.set_xlim(-0.35, 2.55)
ax_bar.spines['left'].set_visible(False)
ax_bar.spines['top'].set_visible(False)

# Move yticks (and labels) to the right spine
ax_bar.yaxis.tick_right()
ax_bar.set_ylabel('Gas Supply and Demand [TWh]')
ax_bar.yaxis.set_label_position("right")
ax_bar.spines['right'].set_visible(True)
# Optionally, lighten right spine
ax_bar.spines['right'].set_color('#aaaaaa')
# ax_bar.tick_params(left=False, labelleft=False)
# ax_bar.spines['bottom'].set_color('#cccccc')

## plt.tight_layout()
# plt.savefig('/mnt/user-data/outputs/europe_figure.png', dpi=180, bbox_inches='tight')
plt.savefig('intro_plot.pdf', dpi=150, bbox_inches='tight')
# Also save a copy to the sibling gas_resilience repo directory
import shutil
_sibling_dir = Path(__file__).resolve().parents[3].parent / 'gas_resilience' / 'imgs'
_sibling_dir.mkdir(parents=True, exist_ok=True)
shutil.copy('intro_plot.pdf', _sibling_dir / 'intro_plot.pdf')

# --- Frontend export ---
_country_consumption_TWh = {
    str(_k): float(_v) for _k, _v in df_countries.sort_values(ascending=False).items()
}

_supply_records = (
    _supply_sorted[["type", "origin", "bcm_2024", "twh_2024", "lat", "lon"]]
    .assign(stack_order=lambda d: range(len(d)))
    .to_dict(orient="records")
)

_sector_records = [
    {"sector": str(_s), "value_TWh": float(_v)}
    for _s, _v in _sector_series.items()
]

_palette = {
    "supply_type_colors": source_palette,
    "sector_colors": sector_colors,
}

# Country geometries: emit minimal GeoJSON-style centroid + bounds so a
# frontend can render a basic choropleth without re-deriving them.
_geo = []
for _idx, _row in gdf.iterrows():
    try:
        _pt = _row.geometry.representative_point()
    except Exception:
        continue
    _bounds = _row.geometry.bounds  # in EPSG:3035 metres
    _geo.append({
        "name": _row.get("name", _idx),
        "rep_point_3035": [float(_pt.x), float(_pt.y)],
        "bounds_3035": [float(b) for b in _bounds],
    })

export_frontend_data("intro_plot", {
    "title": "European gas supply, country consumption and sectoral demand (2024)",
    "map": {
        "crs": "EPSG:3035",
        "extent_3035": {
            "xlim": list(ax_map.get_xlim()),
            "ylim": list(ax_map.get_ylim()),
        },
        "country_geometry": _geo,
        "supply_points": _supply_records,
        "sea_labels_lonlat": {k: list(v) for k, v in seas.items()},
    },
    "bars": {
        "labels": bar_labels,
        "supply_TWh": _supply_records,
        "country_consumption_TWh": _country_consumption_TWh,
        "sectoral_demand_TWh": _sector_records,
    },
    "colors": _palette,
})

plt.show()
print("Done.")

# %%
df_supply

# %%
df_demand.sum()

# %%



