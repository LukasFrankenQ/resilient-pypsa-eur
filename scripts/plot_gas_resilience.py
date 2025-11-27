import logging

import pypsa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

from pathlib import Path
import sys
sys.path.append(str(Path.cwd()))

from scripts._helpers import configure_logging

idx = pd.IndexSlice


def get_gas_demand(n):

    sector_grouping = {
        'Combined-Cycle Gas': 'power',
        'Open-Cycle Gas': 'power',
        'SMR': 'industry',
        'SMR CC': 'industry',
        'gas for industry': 'industry',
        'gas for industry CC': 'industry',
        'rural gas boiler': 'heating',
        'urban central gas CHP': 'heating',
        'urban central gas CHP CC': 'heating',
        'urban central gas boiler': 'heating',
        'urban decentral gas boiler': 'heating'
    }

    eb = n.statistics.energy_balance().loc[idx[:, :, 'gas']] / 1e6
    eb.index = eb.index.droplevel(0)
    return eb.groupby(sector_grouping).sum().mul(-1)


if __name__ == "__main__":

    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake(
            "plot_gas_resilience",
            opts="",
            clusters="50",
            sector_opts="168H-T-H-B-I-A-dist1",
            planning_horizons="2035",
            tyndp_scenario="NT",
            phaseout=0.04,
        )
    configure_logging(snakemake)

    demands = list()
    system_costs = list()

    for n_fn in snakemake.input:

        n = pypsa.Network(n_fn)
        demands.append(get_gas_demand(n).rename(n_fn))
        system_costs.append(n.statistics()[['Capital Expenditure', 'Operational Expenditure']].sum().sum() / 1e9) # in billions of EUR

    demands = pd.concat(demands, axis=1).T
    demands = demands.sum(axis=1)

    system_costs = pd.Series(system_costs, index=demands.index)

    combined = pd.concat([demands.rename('demand'), system_costs.rename('cost')], axis=1)
    combined = combined.sort_values('demand')

    fig, ax = plt.subplots(figsize=(12, 8))

    ax.fill_between(combined['demand'], np.ones(len(combined)) * 900, combined['cost'], color='blue', alpha=0.3)
    ax.plot(combined['demand'], combined['cost'], color='blue', marker='o')

    ax.set_xlabel('Gas Demand [TWh/a]')
    ax.set_ylabel('System Cost [EUR billion]')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_axisbelow(True)

    fig.savefig(snakemake.output[0], bbox_inches='tight')
    plt.close(fig)