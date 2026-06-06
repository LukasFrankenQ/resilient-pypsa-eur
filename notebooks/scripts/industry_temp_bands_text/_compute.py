"""
Offline reproduction of the industry-heat temperature-band pipeline used in
``scripts/build_industry_sector_ratios_endogenous.py``.

This module mirrors the build rule (no snakemake needed) so the figures in the
methods section reflect exactly what the model ingests. It exposes the
quantities the four methods figures are built from:

  - ``eu_production()``        EU-total industry production per sector [Mt/a]
  - ``heat_share_by_sector()`` share of each sector's energy demand that is
                               generic heat (solid biomass + gas + waste heat)
  - ``band_heat_twh()``        EU-total generic heat per temperature band,
                               split into the part met by gas (hottest X%) and
                               the part met by other carriers [TWh/a]

The heat carriers treated as generic ("heat", "biomass", "methane") and the
band-share sources (Fleiter et al. 2025 + backup dict) are taken verbatim from
the build script.
"""

from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]

idx = pd.IndexSlice

BANDS = ["heat<100", "heat100-200", "heat200-500", "heat>500"]
HEAT_CARRIERS = ["heat", "biomass", "methane"]

BAND_COLORS = {
    "heat<100":    "#3b82bd",
    "heat100-200": "#86c5dc",
    "heat200-500": "#f3a260",
    "heat>500":    "#c4453c",
}
BAND_LABELS = {
    "heat<100":    "<100 °C",
    "heat100-200": "100-200 °C",
    "heat200-500": "200-500 °C",
    "heat>500":    ">500 °C",
}

# Processes whose heating is handled elsewhere (steel, ammonia-route chemicals).
SKIP_PROCESSES = [
    "Electric arc",
    "Integrated steelworks",
    "Methanol",
    "Other chemicals",
]

# --- verbatim from scripts/build_industry_sector_ratios_endogenous.py ---------
PROCESS_TEMPERATURE_BAND_MAPPER = {
    "Aluminium - secondary production": idx["Non-ferrous metals", "NE metals", ["Aluminium, secondary"]],
    "Aluminium - primary production":   idx["Non-ferrous metals", "NE metals", ["Aluminium, primary"]],
    "Other non-ferrous metals":         idx["Non-ferrous metals", "NE metals",
                                            ["Copper, primary", "Copper, secondary", "Copper further treatment",
                                             "Zinc, primary", "Zinc, secondary"]],
    "HVC":                              idx[:, :, ["Adipic acid", "Calcium carbide", "Carbon black",
                                                   "Nitric acid", "Soda ash", "TDI", "Titanium dioxide"]],
    "Cement":                           idx["Non-metallic mineral products", "Clinker", :],
    "Ceramics & other NMM":             idx["Non-metallic mineral products", "Ceramics", :],
    "Glass production":                 idx["Non-metallic mineral products", "Glass", :],
    "Pulp production":                  idx["Paper and printing", "Paper products", ["Chemical pulp", "Mechanical pulp"]],
    "Paper production":                 idx["Paper and printing", "Paper products", ["Paper", "Paper Electrical"]],
    "Printing and media reproduction":  idx["Paper and printing", "Paper products", ["Chemical pulp", "Mechanical pulp"]],
    "Food, beverages and tobacco":      idx["Food, drink and tobacco", :, :],
}

BACKUP_TEMPERATURE_BAND_SHARES = {
    "Alumina production":           {"heat<100": 0.0, "heat100-200": 0.0,  "heat200-500": 0.0,  "heat>500": 1.0},
    "Pharmaceutical products etc.": {"heat<100": 0.3, "heat100-200": 0.6,  "heat200-500": 0.1,  "heat>500": 0.0},
    "Other industrial sectors":     {"heat<100": 0.1, "heat100-200": 0.65, "heat200-500": 0.25, "heat>500": 0.0},
    "Transport equipment":          {"heat<100": 0.25,"heat100-200": 0.6,  "heat200-500": 0.15, "heat>500": 0.0},
    "Machinery equipment":          {"heat<100": 0.25,"heat100-200": 0.6,  "heat200-500": 0.15, "heat>500": 0.0},
    "Wood and wood products":       {"heat<100": 0.65,"heat100-200": 0.35, "heat200-500": 0.0,  "heat>500": 0.0},
    "Textiles and leather":         {"heat<100": 0.7, "heat100-200": 0.2,  "heat200-500": 0.1,  "heat>500": 0.0},
}
# -----------------------------------------------------------------------------


def take_cumulative_amount(obj, y, *, from_start=True, clip_negative=True):
    """Verbatim port from the build script. Take the first ``y`` of cumulative
    mass from each column of a DataFrame, from the start (cold) or the end
    (hot)."""

    def _take_array(a, y_val):
        a = a.astype(float, copy=True)
        if clip_negative:
            a = np.maximum(a, 0.0)
        if y_val <= 0.0:
            return np.zeros_like(a)
        total = float(a.sum())
        if y_val >= total:
            return a
        if from_start:
            remaining_before = y_val - np.cumsum(np.r_[0.0, a[:-1]])
            return np.clip(remaining_before, 0.0, a)
        ar = a[::-1]
        remaining_before = y_val - np.cumsum(np.r_[0.0, ar[:-1]])
        outr = np.clip(remaining_before, 0.0, ar)
        return outr[::-1]

    y_aligned = y.reindex(obj.columns).astype(float)
    out = obj.copy()
    for col in obj.columns:
        out[col] = _take_array(obj[col].to_numpy(), float(y_aligned.loc[col]))
    return out


def _load_production_per_country():
    """Production per (country, process) in Mt/a."""
    prod = (pd.read_csv(REPO_ROOT / "resources" / "industrial_production_per_country.csv",
                        index_col=0) / 1e3).stack()
    prod.index.names = [None, None]
    return prod


def _load_today_sector_ratios():
    """Energy demand per tonne (MWh/t), carriers x (country, process), with the
    build script's carrier renaming/grouping applied."""
    demand = pd.read_csv(
        REPO_ROOT / "resources" / "industrial_energy_demand_per_country_today.csv",
        header=[0, 1], index_col=0,
    )
    production = _load_production_per_country()
    tsr = demand.div(production, axis=1).replace([np.inf, -np.inf], 0)
    tsr.dropna(how="all", axis=1, inplace=True)
    rename = {"waste": "biomass", "electricity": "elec", "solid": "coke",
              "gas": "methane", "other": "biomass", "liquid": "naphtha"}
    return tsr.rename(rename).groupby(level=0).sum()


def _load_process_temperature_bands():
    """Fleiter et al. 2025 band shares with the Mechanical-pulp override."""
    ptb = pd.read_excel(
        REPO_ROOT / "data" / "ente202300981-sup-0001-suppdata-s1.xlsx",
        sheet_name="Industry_ESC_FEC", index_col=[0, 1, 2], header=[0, 1],
    )
    ptb.columns = ptb.columns.droplevel(0)
    ptb.rename(columns={
        "<\xa0100\xa0°C": "heat<100", "<\xa0100–200\xa0°C": "heat100-200",
        "200–500\xa0°C": "heat200-500", "500–1000\xa0°C": "heat500-1000",
        ">\xa01000\xa0°C": "heat>1000",
    }, inplace=True)
    ptb["heat>500"] = ptb["heat500-1000"] + ptb["heat>1000"]
    ptb = ptb[BANDS]
    ptb.loc[idx["Paper and printing", "Paper products", "Mechanical pulp"], :] = [0.0, 1.0, 0.0, 0.0]
    return ptb


def _band_shares_per_process():
    """Per-process 4-band heat-share vector, combining the Fleiter mapper rows
    (mean) and the backup dict (normalised to 1)."""
    ptb = _load_process_temperature_bands()
    rows = {}
    for process, key in PROCESS_TEMPERATURE_BAND_MAPPER.items():
        sub = ptb.loc[key]
        shares = sub if isinstance(sub, pd.Series) else sub.mean()
        rows[process] = shares.reindex(BANDS).astype(float)
    for process, shares in BACKUP_TEMPERATURE_BAND_SHARES.items():
        rows[process] = pd.Series(shares).reindex(BANDS).astype(float)
    df = pd.DataFrame(rows).T
    return df.div(df.sum(axis=1), axis=0)


def eu_production():
    """EU-total industry production per sector [Mt/a], descending."""
    prod = _load_production_per_country()
    return prod.groupby(level=1).sum().sort_values(ascending=False)


def heat_share_by_sector():
    """Per endogenised sector, share of total energy demand that is generic
    heat (solid biomass + gas + waste heat). EU production-weighted."""
    tsr = _load_today_sector_ratios()
    prod = _load_production_per_country()
    processes = list(PROCESS_TEMPERATURE_BAND_MAPPER) + list(BACKUP_TEMPERATURE_BAND_SHARES)
    out = {}
    for process in processes:
        if process in SKIP_PROCESSES:
            continue
        cols = tsr.columns[tsr.columns.get_level_values(1) == process]
        cols = cols.intersection(prod.index)
        if len(cols) == 0:
            continue
        w = prod.loc[cols].values                       # Mt/a per country
        heat = (tsr.loc[HEAT_CARRIERS, cols].sum().values * w).sum()  # TWh heat
        total = (tsr.loc[:, cols].sum().values * w).sum()             # TWh total
        if total > 0:
            out[process] = heat / total
    return pd.Series(out).sort_values(ascending=False)


def band_heat_twh():
    """EU-total generic heat per temperature band [TWh/a], split into the part
    met by gas (hottest cumulative X%) and the remainder met by other carriers.

    Returns a DataFrame indexed by BANDS with columns ['gas', 'other'].
    """
    tsr = _load_today_sector_ratios()
    prod = _load_production_per_country()
    shares = _band_shares_per_process()

    full = pd.Series(0.0, index=BANDS)   # all generic heat
    gas = pd.Series(0.0, index=BANDS)    # part met by gas (hottest X%)

    for process in shares.index:
        if process in SKIP_PROCESSES:
            continue
        cols = tsr.columns[tsr.columns.get_level_values(1) == process]
        cols = cols.intersection(prod.index)
        if len(cols) == 0:
            continue

        as_fuels = tsr.loc[HEAT_CARRIERS, cols]                 # MWh/t per country
        generic_heat = as_fuels.sum()                          # MWh/t total heat
        methane = as_fuels.loc["methane"]                      # MWh/t gas

        as_heat = pd.DataFrame(
            np.outer(shares.loc[process].values, generic_heat.values),
            index=BANDS, columns=cols,
        )
        gas_heat = take_cumulative_amount(as_heat, methane, from_start=False)

        w = prod.loc[cols].values                              # Mt/a -> MWh/t*Mt = TWh
        full += pd.Series((as_heat.values * w).sum(axis=1), index=BANDS)
        gas += pd.Series((gas_heat.values * w).sum(axis=1), index=BANDS)

    return pd.DataFrame({"gas": gas, "other": full - gas})


def model_band_twh():
    """EU-total endogenised industry heat per temperature band [TWh/a], read
    from the build rule's output (``industry_sector_ratios_endogenous.csv``).

    These are the band volumes the model actually ingests, i.e. the gas-met
    heat split per band with the residual low-grade ``heat`` carrier folded into
    the <100 °C band, exactly as the build script writes it.
    """
    ratios = pd.read_csv(
        REPO_ROOT / "resources" / "industry_sector_ratios_endogenous.csv",
        header=[0, 1], index_col=0,
    )
    prod = _load_production_per_country()
    cols = ratios.columns.intersection(prod.index)
    w = prod.loc[cols].values
    band = ratios.loc[BANDS, cols]
    return pd.Series((band.values * w).sum(axis=1), index=BANDS)
