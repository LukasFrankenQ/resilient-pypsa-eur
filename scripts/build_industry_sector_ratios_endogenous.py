# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
Build endogenous industry sector ratios.

Description
-------

This rule first builds energy needs in MWh/t for different industry sectors and production routes.

These energy needs are expressed in primary energy carriers methane, biomass, low-grade heat etc.

However, a large share of that primary energy is used to produce heat which, depending on the
temperature requirement, can be substituted by other heat carriers.

To the end of endogenising this decision in the model, the heat-producing input energies are here
translated and split into the respective temperature bands, depending on industry sector and process.

These splits are mostly taken from Fleiter et al. 2025.
"Hydrogen Infrastructure in the Future CO2-Neutral European Energy System -
How Does the Demand for Hydrogen Affect the Need for Infrastructure?"

The script does not cover the steel sector. Its endogenisation is taken care of elsewhere.
"""

import logging

import numpy as np
import pandas as pd

from scripts._helpers import configure_logging, set_scenario_config
from _tyndp_helpers import _extract_scenario_values, _extract_scenario_values_rowwise

def methane_industry_energetic(path):
    return _extract_scenario_values(path, sheet_name='8-', row_label='Industry')


logger = logging.getLogger(__name__)

idx = pd.IndexSlice


def take_cumulative_amount(obj, y, *, from_start=True, clip_negative=True):
    """
    Take the first `y` of cumulative mass from a non-negative Series (or each column of a DataFrame),
    either from the start (left) or from the end (right), returning an object of the same shape.

    `y` can be:
      - float/int (applied to the whole Series, or to every column of a DataFrame), or
      - pd.Series indexed by DataFrame columns (per-column y).

    Example:
        s = pd.Series([1,2,3])
        take_cumulative_amount(s, 2.5, from_start=True)  -> [1, 1.5, 0]
        take_cumulative_amount(s, 2.5, from_start=False) -> [0, 0, 2.5]
    """

    def _take_array(a: np.ndarray, y_val: float) -> np.ndarray:
        a = a.astype(float, copy=True)
        if clip_negative:
            a = np.maximum(a, 0.0)

        if not np.isfinite(y_val):
            raise ValueError("y must be a finite number (or a Series of finite numbers).")
        if y_val <= 0.0:
            return np.zeros_like(a)

        total = float(a.sum())
        if y_val >= total:
            return a

        if from_start:
            remaining_before = y_val - np.cumsum(np.r_[0.0, a[:-1]])
            return np.clip(remaining_before, 0.0, a)
        else:
            ar = a[::-1]
            remaining_before = y_val - np.cumsum(np.r_[0.0, ar[:-1]])
            outr = np.clip(remaining_before, 0.0, ar)
            return outr[::-1]

    # --- Series input ---
    if isinstance(obj, pd.Series):
        y_val = float(y)  # must be scalar for Series input
        out = _take_array(obj.to_numpy(), y_val)
        return pd.Series(out, index=obj.index, name=obj.name)

    # --- DataFrame input ---
    if isinstance(obj, pd.DataFrame):
        if isinstance(y, pd.Series):
            # align to columns; require all columns present
            y_aligned = y.reindex(obj.columns)
            missing = y_aligned.isna()
            if missing.any():
                missing_cols = list(obj.columns[missing.to_numpy()])
                raise ValueError(
                    f"y is missing values for columns: {missing_cols}. "
                    "Provide a y Series indexed by all DataFrame columns."
                )
            y_map = y_aligned.astype(float)
        else:
            y_scalar = float(y)
            if not np.isfinite(y_scalar):
                raise ValueError("y must be a finite number (or a Series of finite numbers).")
            y_map = pd.Series(y_scalar, index=obj.columns, dtype=float)

        out = obj.copy()
        for col in obj.columns:
            out[col] = _take_array(obj[col].to_numpy(), float(y_map.loc[col]))
        return out

    raise TypeError("obj must be a pandas Series or DataFrame.")


# maps processes to temperature bands distributions from Fleiter et al. 2025
process_temperature_band_mapper = {
    "Aluminium - secondary production": idx[
        "Non-ferrous metals", "NE metals", ["Aluminium, secondary"]
    ],
    "Aluminium - primary production": idx[
        "Non-ferrous metals", "NE metals", ["Aluminium, primary"]
    ],
    "Other non-ferrous metals": idx[
        "Non-ferrous metals",
        "NE metals",
        [
            "Copper, primary",
            "Copper, secondary",
            "Copper further treatment",
            "Zinc, primary",
            "Zinc, secondary",
        ],
    ],
    "HVC": idx[
        :,
        :,
        [
            "Adipic acid",
            "Calcium carbide",
            "Carbon black",
            "Nitric acid",
            "Soda ash",
            "TDI",
            "Titanium dioxide",
        ],
    ],
    "Cement": idx["Non-metallic mineral products", "Clinker", :],
    "Ceramics & other NMM": idx["Non-metallic mineral products", "Ceramics", :],
    "Glass production": idx["Non-metallic mineral products", "Glass", :],
    "Pulp production": idx[
        "Paper and printing", "Paper products", ["Paper", "Paper Electrical"]
    ],
    "Paper production": idx[
        "Paper and printing", "Paper products", ["Chemical pulp", "Mechanical pulp"]
    ],
    "Printing and media reproduction": idx[
        "Paper and printing", "Paper products", ["Chemical pulp", "Mechanical pulp"]
    ],
    "Food, beverages and tobacco": idx["Food, drink and tobacco", :, :],
}

backup_temperature_band_shares = {
    # from JRC IDEES EU26 Industry
    "Alumina production": {
        "heat<100": 0.0,
        "heat100-200": 0.0,
        "heat200-500": 0.0,
        "heat>500": 1.0,
    },
    "Pharmaceutical products etc.": {
        "heat<100": 0.3,
        "heat100-200": 0.6,
        "heat200-500": 0.1,
        "heat>500": 1.0,
    },
    # from https://energyinnovation.org/data-explorer/overcoming-all-barriers-to-industrial-electrification
    "Other industrial sectors": {
        "heat<100": 0.1,
        "heat100-200": 0.65,
        "heat200-500": 0.25,
        "heat>500": 0.0,
    },
    "Transport equipment": {
        "heat<100": 0.25,
        "heat100-200": 0.6,
        "heat200-500": 0.15,
        "heat>500": 0.0,
    },
    "Machinery equipment": {
        "heat<100": 0.25,
        "heat100-200": 0.6,
        "heat200-500": 0.15,
        "heat>500": 0.0,
    },
    # from https://www.fpl.fs.usda.gov/documents/fplgtr/fplgtr118.pdf
    "Wood and wood products": {
        "heat<100": 0.65,
        "heat100-200": 0.35,
        "heat200-500": 0.0,
        "heat>500": 0.0,
    },
    # from https://www.ifc.org/content/dam/ifc/doc/2000/2007-textiles-manufacturing-ehs-guidelines-en.pdf
    "Textiles and leather": {
        "heat<100": 0.7,
        "heat100-200": 0.2,
        "heat200-500": 0.1,
        "heat>500": 0.0,
    },
}


def build_industry_sector_ratios_endogenous(phaseout):

    # in TWh/a
    demand = pd.read_csv(
        snakemake.input.industrial_energy_demand_per_country_today,
        header=[0, 1],
        index_col=0,
    )

    # in Mt/a
    production = (
        pd.read_csv(snakemake.input.industrial_production_per_country, index_col=0)
        / 1e3
    ).stack()
    production.index.names = [None, None]

    today_sector_ratios = demand.div(production, axis=1).replace([np.inf, -np.inf], 0)

    today_sector_ratios.dropna(how="all", axis=1, inplace=True)

    rename = {
        "waste": "biomass",
        "electricity": "elec",
        "solid": "coke",
        "gas": "methane",
        "other": "biomass",
        "liquid": "naphtha",
    }
    today_sector_ratios = today_sector_ratios.rename(rename).groupby(level=0).sum()

    process_temperature_bands = pd.read_excel(
        snakemake.input.process_temperature_bands,
        sheet_name="Industry_ESC_FEC",
        index_col=[0, 1, 2],
        header=[0, 1],
    )
    process_temperature_bands.columns = process_temperature_bands.columns.droplevel(0)
    process_temperature_bands.rename(
        columns={
            "<\xa0100\xa0°C": "heat<100",
            "<\xa0100–200\xa0°C": "heat100-200",
            "200–500\xa0°C": "heat200-500",
            "500–1000\xa0°C": "heat500-1000",
            ">\xa01000\xa0°C": "heat>1000",
        },
        inplace=True,
    )
    process_temperature_bands.loc[:, "heat>500"] = (
        process_temperature_bands.loc[:, "heat500-1000"]
        + process_temperature_bands.loc[:, "heat>1000"]
    )
    process_temperature_bands = process_temperature_bands[
        ["heat<100", "heat100-200", "heat200-500", "heat>500"]
    ]

    heat_carriers = ["heat", "biomass", "methane"]
    bands = ["heat<100", "heat100-200", "heat200-500", "heat>500"]

    endogenous_sector_ratios = today_sector_ratios.copy()

    endogenous_sector_ratios = endogenous_sector_ratios.reindex(
        endogenous_sector_ratios.index.union(bands)
    ).replace(np.nan, 0)

    skip_processes = [
        'Electric arc',
        'Integrated steelworks',
        'Methanol',
        'Other chemicals',
    ]
    assert pd.Series(skip_processes).isin(endogenous_sector_ratios.columns.unique(1)).all()

    # for each heat-endogenous process, split energy in heat carriers into the respective temperature bands
    for process, key in process_temperature_band_mapper.items():
        if process in skip_processes:
            continue

        as_fuels = endogenous_sector_ratios.loc[heat_carriers, idx[:, process]]

        as_heat = pd.DataFrame(
            np.outer(
                process_temperature_bands.loc[key].mean().values, as_fuels.sum().values
            ),
            index=bands,
            columns=as_fuels.sum().index,
        )

        gas_heat_share = take_cumulative_amount(
            as_heat,
            as_fuels.loc['methane'],
            from_start=False
            )

        decarbonise_heat_share = take_cumulative_amount(
            gas_heat_share,
            as_fuels.loc['methane'] * phaseout,
            from_start=True
            )

        endogenous_sector_ratios.loc[bands, idx[:, process]] = decarbonise_heat_share
        endogenous_sector_ratios.loc[['methane'], idx[:, process]] *= 1 - phaseout
    

    # for processes not covered by Fleiter et al. 2025
    for process, band_shares in backup_temperature_band_shares.items():
        if process in skip_processes:
            continue

        band_shares = pd.Series(band_shares)

        as_fuels = endogenous_sector_ratios.loc[heat_carriers, idx[:, process]]

        as_heat = pd.DataFrame(
            np.outer(band_shares.values, as_fuels.sum().values),
            index=band_shares.index,
            columns=as_fuels.sum().index,
        )

        gas_heat_share = take_cumulative_amount(
            as_heat,
            as_fuels.loc['methane'],
            from_start=False
            )

        decarbonise_heat_share = take_cumulative_amount(
            gas_heat_share,
            as_fuels.loc['methane'] * phaseout,
            from_start=True
            )

        endogenous_sector_ratios.loc[bands, idx[:, process]] = decarbonise_heat_share
        endogenous_sector_ratios.loc[['methane'], idx[:, process]] *= 1 - phaseout

    # heat is only nonzero in processes that are not endogenous here, so this does not produce an error
    endogenous_sector_ratios.loc["heat<100"] += endogenous_sector_ratios.loc["heat"]
    endogenous_sector_ratios.drop("heat", inplace=True)

    endogenous_sector_ratios.index.name = "MWh/t"

    # Ensure all level 1 values are present for all level 0 entries
    # Get all unique level 1 values across the entire dataframe
    all_level1_values = endogenous_sector_ratios.columns.get_level_values(1).unique()

    # For each level 0 entry, ensure all level 1 values exist
    new_columns = []
    for level0 in endogenous_sector_ratios.columns.get_level_values(0).unique():
        existing_level1 = endogenous_sector_ratios[level0].columns
        missing_level1 = all_level1_values.difference(existing_level1)

        if len(missing_level1) > 0:
            # Calculate average values for missing columns across level 0 entries where they exist
            for level1 in missing_level1:
                # Find all level 0 entries that have this level 1 value
                mask = endogenous_sector_ratios.columns.get_level_values(1) == level1
                if mask.any():
                    # Calculate mean across all columns with this level 1 value
                    avg_values = endogenous_sector_ratios.loc[:, mask].mean(axis=1)
                    new_columns.append(((level0, level1), avg_values))

    # Add the new columns to the dataframe
    for (level0, level1), values in new_columns:
        endogenous_sector_ratios[(level0, level1)] = values

    # Sort columns for consistency
    endogenous_sector_ratios = endogenous_sector_ratios.sort_index(axis=1)

    endogenous_sector_ratios.to_csv(snakemake.output.industry_sector_ratios_endogenous)


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake(
            "build_industry_sector_ratios_endogenous",
        )
    configure_logging(snakemake)
    set_scenario_config(snakemake)

    year = int(snakemake.wildcards['planning_horizons'])

    assert year < 2041, f'endogenous industry heating only implemented between 2025 and 2040, but year is {year}'

    heat_gas_demand = methane_industry_energetic(snakemake.input.tyndp_path)

    if year <= 2030:
        phaseout_2030 = 1 - heat_gas_demand['National Trends'][2030] / heat_gas_demand['Reference'][2019]
        weight = (year - 2025) / (2030 - 2025)
        phaseout = weight * phaseout_2030
    else:
        phaseout_2030 = 1 - heat_gas_demand['National Trends'][2030] / heat_gas_demand['Reference'][2019]
        phaseout_2040 = 1 - heat_gas_demand['National Trends'][2040] / heat_gas_demand['Reference'][2019]
        weight = (year - 2030) / (2040 - 2030)
        phaseout = weight * phaseout_2040 + (1 - weight) * phaseout_2030

    build_industry_sector_ratios_endogenous(phaseout)
