import logging

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import pandas as pd

from scripts._helpers import configure_logging, rename_techs, set_scenario_config
from scripts.prepare_sector_network import co2_emissions_year

logger = logging.getLogger(__name__)
plt.style.use("bmh")


preferred_order = pd.Index(
    [
        "transmission lines",
        "hydroelectricity",
        "hydro reservoir",
        "run of river",
        "pumped hydro storage",
        "solid biomass",
        "biogas",
        "onshore wind",
        "offshore wind",
        "offshore wind (AC)",
        "offshore wind (DC)",
        "solar PV",
        "solar thermal",
        "solar rooftop",
        "solar",
        "building retrofitting",
        "ground heat pump",
        "air heat pump",
        "heat pump",
        "resistive heater",
        "power-to-heat",
        "gas-to-power/heat",
        "CHP",
        "OCGT",
        "gas boiler",
        "gas",
        "natural gas",
        "methanation",
        "ammonia",
        "hydrogen storage",
        "power-to-gas",
        "power-to-liquid",
        "battery storage",
        "hot water storage",
        "CO2 sequestration",
    ]
)


def plot_energy_difference(energy_df):
    # energy_df = pd.read_csv(
        # snakemake.input.energy, index_col=list(range(2)), header=list(range(n_header))
    # )
    print(energy_df.head())

    df = energy_df.groupby("carrier").sum()

    # convert MWh to TWh
    df = df / 1e6

    df = df.groupby(df.index.map(rename_techs)).sum()

    to_drop = df.index[
        df.abs().max(axis=1) < snakemake.params.plotting["energy_threshold"]
    ]

    logger.info(
        f"Dropping all technology with energy consumption or production below {snakemake.params['plotting']['energy_threshold']} TWh/a"
    )
    logger.debug(df.loc[to_drop])

    df = df.drop(to_drop)
    
    print('plotting df')
    print(df.head())

    logger.info(f"Total energy of {round(df.sum().iloc[0])} TWh/a")

    if df.empty:
        fig, ax = plt.subplots(figsize=(12, 8))
        fig.savefig(snakemake.output.energy_difference, bbox_inches="tight")
        plt.close(fig)
        return

    new_index = preferred_order.intersection(df.index).append(
        df.index.difference(preferred_order)
    )

    # new_columns = df.columns.sort_values()

    fig, ax = plt.subplots(figsize=(12, 8))

    logger.debug(df.loc[new_index])

    df.loc[new_index].T.plot(
        kind="bar",
        ax=ax,
        stacked=True,
        color=[snakemake.params.plotting["tech_colors"][i] for i in new_index],
    )

    handles, labels = ax.get_legend_handles_labels()

    handles.reverse()
    labels.reverse()

    '''
    ax.set_ylim(
        [
            snakemake.params.plotting["energy_min"],
            snakemake.params.plotting["energy_max"],
        ]
    )
    '''

    ax.set_ylabel("Energy [TWh/a]")

    ax.set_xlabel("")

    ax.grid(axis="x")

    ax.legend(
        handles, labels, ncol=1, loc="upper left", bbox_to_anchor=[1, 1], frameon=False
    )

    fig.savefig(snakemake.output.energy_difference, bbox_inches="tight")
    plt.close(fig)


def plot_balances_difference(balances_df):
    co2_carriers = ["co2", "co2 stored", "process emissions"]

    balances = {k: df for k, df in balances_df.groupby("bus_carrier")}
    balances["energy"] = balances_df.groupby(["component", "carrier"]).sum()

    print('balances')
    # print(balances)
    print(list(balances))

    for bus_carrier, df in balances.items():
        df = df.groupby("carrier").sum()

        # convert MWh to TWh
        df = df / 1e6

        print('df after groupby carrier in bus carrier ', bus_carrier)
        print(df.head())

        df = df.groupby(df.index.map(rename_techs)).sum()

        to_drop = df.index[
            df.abs().max(axis=1) < snakemake.params.plotting["energy_threshold"] / 10
        ]

        units = "MtCO2/a" if bus_carrier in co2_carriers else "TWh/a"
        logger.debug(
            f"Dropping technology energy balance smaller than {snakemake.params['plotting']['energy_threshold'] / 10} {units}"
        )
        logger.debug(df.loc[to_drop])

        df = df.drop(to_drop)

        logger.debug(
            f"Total energy balance for {bus_carrier} of {round(df.sum().iloc[0], 2)} {units}"
        )

        if df.empty:
            print('df is empty for bus carrier ', bus_carrier)

            continue

        new_index = preferred_order.intersection(df.index).append(
            df.index.difference(preferred_order)
        )

        new_columns = df.columns.sort_values()

        fig, ax = plt.subplots(figsize=(12, 8))

        df.loc[new_index, new_columns].T.plot(
            kind="bar",
            ax=ax,
            stacked=True,
            color=[snakemake.params.plotting["tech_colors"][i] for i in new_index],
        )

        handles, labels = ax.get_legend_handles_labels()

        handles.reverse()
        labels.reverse()

        if bus_carrier in co2_carriers:
            ax.set_ylabel("CO2 [MtCO2/a]")
        else:
            ax.set_ylabel("Energy [TWh/a]")

        ax.set_xlabel("")

        ax.grid(axis="x")

        ax.legend(
            handles,
            labels,
            ncol=1,
            loc="upper left",
            bbox_to_anchor=[1, 1],
            frameon=False,
        )

        fig.savefig(
            snakemake.output.balances_difference[:-4] + "_" + bus_carrier + ".svg", bbox_inches="tight"
        )
        plt.close(fig)




if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake("plot_summary")

    configure_logging(snakemake)
    set_scenario_config(snakemake)

    columns = pd.MultiIndex.from_tuples(
        [(
            snakemake.wildcards.clusters,
            snakemake.wildcards.opts,
            snakemake.wildcards.sector_opts,
            snakemake.wildcards.planning_horizons,
            snakemake.wildcards.tyndp_scenario,
            snakemake.wildcards.wiggle,
        )]
    )

    n_header = 5

    energy_df_nohike = pd.read_csv(
        snakemake.input.energy_nohike,
        index_col=list(range(2)),
        )
    energy_df_hike = pd.read_csv(
        snakemake.input.energy_hike,
        index_col=list(range(2)),
        )
    diff = energy_df_hike - energy_df_nohike

    diff.columns = columns

    print(diff.head())

    plot_energy_difference(diff)

    co2_carriers = ["co2", "co2 stored", "process emissions"]

    balances_df_nohike = pd.read_csv(
        snakemake.input.energy_balance_nohike, index_col=list(range(3))
    )
    balances_df_nohike.columns = columns
    print('===========================================================')
    print(balances_df_nohike.head())

    balances_df_hike = pd.read_csv(
        snakemake.input.energy_balance_hike, index_col=list(range(3))
    )
    balances_df_hike.columns = columns
    print('===========================================================')
    print(balances_df_hike.head())

    diff_balances = balances_df_hike - balances_df_nohike

    plot_balances_difference(diff_balances)

    import sys
    sys.exit()