# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
Distribute country-level energy demands by population.
"""

import logging

import pandas as pd

from scripts._helpers import configure_logging, get_snapshots, set_scenario_config

idx = pd.IndexSlice

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake(
            "build_population_weighted_energy_totals",
            kind="heat",
            clusters=60,
        )
    configure_logging(snakemake)
    set_scenario_config(snakemake)

    config = snakemake.config["energy"]

    if snakemake.wildcards.kind == "heat":
        snapshots = get_snapshots(
            snakemake.params.snapshots, snakemake.params.drop_leap_day
        )
        data_years = snapshots.year.unique()
    else:
        data_years = int(config["energy_totals_year"])

    pop_layout = pd.read_csv(snakemake.input.clustered_pop_layout, index_col=0)

    totals = pd.read_csv(snakemake.input.energy_totals, index_col=[0, 1])

    years_in_totals = totals.index.get_level_values(1).unique()
    orig_type = type(data_years)
    if isinstance(data_years, int):
        data_years = pd.Index([data_years])
    elif not isinstance(data_years, pd.Index):
        data_years = pd.Index(data_years)

    uncovered_years = data_years.difference(years_in_totals)
    if len(uncovered_years):
        logger.warning('Uncovered years: %s', uncovered_years)
        logger.warning('Taking latest available year')
        max_year = years_in_totals.max()
        data_years = pd.Index([y if y <= max_year else max_year for y in data_years])

    if orig_type == int:
        data_years = data_years[0]

    totals = totals.loc[idx[:, data_years], :].groupby("country").mean()

    nodal_totals = totals.loc[pop_layout.ct].fillna(0.0)
    nodal_totals.index = pop_layout.index
    nodal_totals = nodal_totals.multiply(pop_layout.fraction, axis=0)

    nodal_totals.to_csv(snakemake.output[0])
