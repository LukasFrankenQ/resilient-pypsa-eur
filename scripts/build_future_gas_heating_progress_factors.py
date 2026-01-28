# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
'''
Builds factors that convert existing gas heating capacities to future gas heating capacities.
'''

import pypsa
import pandas as pd
import numpy as np

from scripts._helpers import mock_snakemake

# from TYNDP 2024 Scenario-Report-Data-Figures_240522.xlsx, Sheet 13
tyndp_heating = pd.DataFrame(
    {
        ("Reference", "2019"): {
            "Hybrid heat pump": 0.02, 
            "Electric heat pump": 0.13,
            "Methane boiler": 0.37,
            "Hydrogen boiler": 0.00,
            "Other technologies": 0.48,
        },
        ("Distributed Energy", "2040"): {
            "Hybrid heat pump": 0.05,
            "Electric heat pump": 0.50,
            "Methane boiler": 0.12,
            "Hydrogen boiler": 0.02,
            "Other technologies": 0.31,
        },
        ("Distributed Energy", "2050"): {
            "Hybrid heat pump": 0.07,
            "Electric heat pump": 0.63,
            "Methane boiler": 0.03,
            "Hydrogen boiler": 0.03,
            "Other technologies": 0.25,
        },
        ("Global Ambition", "2040"): {
            "Hybrid heat pump": 0.11,
            "Electric heat pump": 0.37,
            "Methane boiler": 0.14,
            "Hydrogen boiler": 0.05,
            "Other technologies": 0.33,
        },
        ("Global Ambition", "2050"): {
            "Hybrid heat pump": 0.13,
            "Electric heat pump": 0.50,
            "Methane boiler": 0.05,
            "Hydrogen boiler": 0.05,
            "Other technologies": 0.28,
        },
    }
).mul(100.)

tyndp_heating.columns = pd.MultiIndex.from_tuples(
    tyndp_heating.columns, names=["Scenario", "Year"]
    )
tyndp_heating.index.name = "Technology"


def get_efficiency(n, link):

    try:
        if link in (effs := n.links_t.efficiency).columns:
            return effs.loc[:, link].mean()
        elif 'CHP' in link:
            return n.links.loc[link, 'efficiency2']
        else:
            return n.links.loc[link, 'efficiency']
    except KeyError:
        return 0.


def to_shares(n, installed):

    p_nom_rural = installed.loc[:, idx['rural']]

    etas = p_nom_rural.apply(
        lambda col: col.index.map(
            lambda idx: get_efficiency(n, idx + ' rural ' + col.name)
            )
    )
    eff_p_nom_rural = p_nom_rural.mul(etas, axis=0).rename(columns={'ground heat pump': 'heat pump'})

    p_nom_decentral = installed.loc[:, idx['urban decentral']]

    etas = p_nom_decentral.apply(
        lambda col: col.index.map(
            lambda idx: get_efficiency(n, idx + ' urban decentral ' + col.name)
            )
    )
    eff_p_nom_decentral = p_nom_decentral.mul(etas, axis=0).rename(columns={'air heat pump': 'heat pump'})

    eff_p_nom_total = eff_p_nom_rural.add(eff_p_nom_decentral, fill_value=0)

    shares = eff_p_nom_total.div(eff_p_nom_total.sum(axis=1), axis=0)

    return shares.loc[:, shares.replace(np.nan, 0).sum() > 0]


if __name__ == "__main__":
    if "snakemake" not in globals():

        snakemake = mock_snakemake(
            "build_future_gas_heating_shares",
            clusters="50",
            opts="",
            sector_opts="168H-T-H-B-I-A-dist1",
        )

    n = pypsa.Network(
        snakemake.input.network
    )

    idx = pd.IndexSlice
    ex0 = pd.read_csv(
        snakemake.input.existing_heating_distribution,
        header=[0,1], index_col=0
    )

    existing = pd.concat([
        ex0.T.loc[idx[['residential rural', 'services rural']]].groupby(level=1).sum().T,
        ex0.T.loc[idx[['residential urban decentral', 'services urban decentral']]].groupby(level=1).sum().T,
    ],
        axis=1,
        keys=['rural', 'urban decentral']
    )

    existing_shares = to_shares(n, existing)

    target_year = int(snakemake.wildcards.planning_horizons)

    target_capacities = pd.DataFrame(0, index=existing_shares.dropna().index, columns=['rural', 'urban decentral'])

    target_2040 = tyndp_heating.loc['Methane boiler', ('Distributed Energy', '2040')]

    for bus in target_capacities.index:

        current_gas_share = existing_shares.loc[bus, 'gas boiler'] * 100
        year_target = (target_year - 2024) / (2040 - 2024) * (target_2040 - current_gas_share) + current_gas_share

        year_share = year_target

        factor = year_share / current_gas_share if current_gas_share > 0 else 0

        target_capacities.loc[bus, 'rural'] = factor * existing.loc[bus, idx['rural', 'gas boiler']]
        target_capacities.loc[bus, 'urban decentral'] = factor * existing.loc[bus, idx['urban decentral', 'gas boiler']]

    target_capacities.to_csv(snakemake.output.gas_heating_progress_factors)
