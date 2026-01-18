import re
import pypsa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from _tyndp_helpers import _extract_scenario_values, _extract_scenario_values_rowwise
from _helpers import configure_logging

def extract_wiggle(filename):
    """Extract cost variation string (e.g., '+0.00', '-0.3') from network filename"""
    # Find pattern like +0.00.nc or -0.3.nc
    return float(filename.split('.nc')[0][-4:])


# Sort networks by year (2025, 2030, 2035, etc.)
def extract_year(filename):
    """Extract year from network filename"""
    # Find year pattern like _2030_, _2035_, etc.

    assert '_2025_' in filename or '_2030_' in filename or '_2035_' in filename, f"Model must have planning horizon 2025, 2030 or 2035"
    match = re.search(r'_(\d{4})_', filename)
    return int(match.group(1)) if match else 0


countries = [
    "AT", "BE", "BG", "HR", "CY", "CZ", "DK", "EE", "FI", "FR", "DE", "GR", "HU",
    "IE", "IT", "LV", "LT", "LU", "MT", "NL", "PL", "PT", "RO", "SK", "SI", "ES", "SE",
    "GB", "NO", "CH", "ME", "XK", "AL", "MK", "BA", "RS"
]

# values taken from FES 2025, Data Workbook, Sheet F.20, (optimistic) Holistic Transition Scenario
uk_gas_consumption = {
    2030: {
        'Power Generation': 59.1, # TWh/a
        'Residential&Tertiary': 304.6, # TWh/a
        'Industry': 158.7, # TWh/a
    },
    2035: {
        'Power Generation': 58.8, # TWh/a
        'Residential&Tertiary': 231.9, # TWh/a
        'Industry': 113.8, # TWh/a
    },
}

# Norway; values taken from Energy Transition Outlook 2024, Page 33
# https://www.norskindustri.no/siteassets/dokumenter/rapporter-og-brosjyrer/energy-transition-norway/energy-transition-norway-2024.pdf
no_gas_consumption = {
    2030: {
        'Power Generation': 0., # TWh/a (approximately zero)
        'Residential&Tertiary': 0., # TWh/a (approximately zero, dominated by electric and biomass heating)
        'Industry': 250 * 0.278, # TWh/a (0.278 is conversion from PJ to TWh)
    },
    2035: {
        'Power Generation': 0., # TWh/a (approximately zero)
        'Residential&Tertiary': 0., # TWh/a (approximately zero, dominated by electric and biomass heating)
        'Industry': 300 * 0.278, # TWh/a (0.278 is conversion from PJ to TWh)
    },
}

# Switzerland; from Energieperspektiven 2050+, Tabelle 62
# see https://www.bfe.admin.ch/bfe/de/home/politik/energieperspektiven-2050-plus.html
ch_gas_consumption = {
    2030: {
        'Power Generation': 0., # TWh/a
        'Residential&Tertiary': 12., # TWh/a
        'Industry': 12., # TWh/a
    },
    2035: {
        'Power Generation': 0., # TWh/a (approximately zero)
        'Residential&Tertiary': 9., # TWh/a
        'Industry': 90., # TWh/a
    },
}

# Combine into a single DataFrame
data = []
for country, consumption_dict in [('UK', uk_gas_consumption), ('NO', no_gas_consumption), ('CH', ch_gas_consumption)]:
    for year in [2030, 2035]:
        for sector in ['Power Generation', 'Residential&Tertiary', 'Industry']:
            data.append({
                'Country': country,
                'Year': year,
                'Sector': sector,
                'Gas Consumption (TWh/a)': consumption_dict[year][sector]
            })

noneu_df = pd.DataFrame(data)
noneu_df = noneu_df.groupby(['Year', 'Sector'])['Gas Consumption (TWh/a)'].sum()


def get_industry_gas_consumption(n):
    
    fixed_demand = n.loads.loc[n.loads.carrier.isin(['gas for industry', 'gas for industry CC']), ['p_set']]

    fixed_demand = fixed_demand.loc[fixed_demand.index.str.startswith(tuple(countries))]
    w = n.snapshot_weightings['generators']

    fixed_demand = w.sum() * fixed_demand.sum()
    print('Warning! Does not yet consider gas boilers in industry')

    ls = n.links.index[
            n.links.carrier.isin(['SMR', 'SMR CC']) &
            n.links.index.str.startswith(tuple(countries))
        ]
    
    varying_demand = n.links_t.p0.loc[:, ls].sum(axis=1).mul(w, axis=0).sum() / 1e6

    return fixed_demand.sum() / 1e6 + varying_demand


def get_electricity_gas_consumption(n):
    
    w = n.snapshot_weightings['generators']

    ls = n.links.index[n.links.carrier.isin(['CCGT', 'OCGT', 'urban central gas CHP', 'urban central gas CHP CC'])]
    ls = ls[ls.str.startswith(tuple(countries))]

    return n.links_t.p0.loc[:, ls].sum(axis=1).mul(w, axis=0).sum() / 1e6


def get_heating_gas_consumption(n):
    
    w = n.snapshot_weightings['generators']

    ls = n.links.index[(
            n.links.carrier.str.contains('gas boiler') &
            n.links.index.str.startswith(tuple(countries)) &
            n.links.bus1.str.contains('|'.join(['rural', 'urban']))
        )]

    return n.links_t.p0.loc[:, ls].sum(axis=1).mul(w, axis=0).sum() / 1e6


if __name__ == "__main__":

    configure_logging(snakemake)

    networks = snakemake.input
    networks = sorted(networks, key=lambda x: (extract_year(x), extract_wiggle(x)))

    eu27_gas_consumption = {
        "Residential&Tertiary": {2019: 1402, 2030: 990, 2040: 599},
        "Industry": {2019: 1034, 2030: 888, 2040: 596},
        "Power Generation": {2019: 1218, 2030: 972, 2040: 649},
        "Non-energy use": {2019: 178, 2030: 166, 2040: 114},
        "Total": {2019: 3918, 2030: 3368, 2040: 2328},
    }


    fig, ax = plt.subplots(figsize=(len(networks) * 2, 4))

    # Model year - change this to match your model
    model_years = [extract_year(n) for n in networks]

    # Get reference values for the model year
    def get_reference_value(category, year):
        if year in eu27_gas_consumption[category]:
            return eu27_gas_consumption[category][year]
        elif year == 2035:  # Interpolate between 2030 and 2040
            val_2030 = eu27_gas_consumption[category][2030]
            val_2040 = eu27_gas_consumption[category][2040]
            return val_2030 + (val_2040 - val_2030) * 0.5
        else:
            raise ValueError(f"No reference data available for year {year}")

    # Create stacked bar plot
    categories = ['Industry + Non-energy', 'Power Generation', 'Residential & Tertiary']
    nice_names = ['Industry', 'Power Generation', 'Heating']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    bar_width = 0.8

    previous_year = None
    x_position = 0
    x_tick_labels = []
    added_labels = False

    for n_fn, year in zip(networks, model_years):
        print(f'Opening {n_fn}.')

        if year != previous_year:

            # Reference values
            ref_industry = get_reference_value("Industry", year) + get_reference_value("Non-energy use", year)
            ref_power = get_reference_value("Power Generation", year)
            ref_residential = get_reference_value("Residential&Tertiary", year)
            ref_values = [
                ref_industry + noneu_df.loc[year, 'Industry'],
                ref_power + noneu_df.loc[year, 'Power Generation'],
                ref_residential + noneu_df.loc[year, 'Residential&Tertiary']
                ]

            previous_year = year

            x_tick_labels.append(f'TYNDP Reference {year}')

            # Plot stacked bars for reference
            bottom_ref = 0
            for i, (value, color, category) in enumerate(zip(ref_values, colors, categories)):
                ax.bar(x_position, value, bar_width, bottom=bottom_ref, 
                    color=color, alpha=0.4)
                # Add horizontal line extending to model bar
                ax.text(x_position, bottom_ref + value/2, f'{value:.0f} TWh',
                        ha='center', va='center', fontweight='bold', fontsize=9)
                ax.hlines(bottom_ref + value, x_position - bar_width/2, x_position + bar_width/2,
                        colors='k', linestyles='--', alpha=0.6, linewidth=1)
                bottom_ref += value

            x_position += 1
        
        n = pypsa.Network(n_fn)
        wiggle = extract_wiggle(n_fn)

        ind = get_industry_gas_consumption(n)
        elec = get_electricity_gas_consumption(n)
        heat = get_heating_gas_consumption(n)

        model_values = [ind, elec, heat]

        bottom_model = 0
        for i, (value, color, category) in enumerate(zip(model_values, colors, categories)):
            if not added_labels:
                kwargs = {'label': nice_names[i]}
            else:
                kwargs = {}
            bar = ax.bar(x_position, value, bar_width, bottom=bottom_model,
                        color=color, alpha=0.7, **kwargs)
            # Add value label
            ax.text(x_position, bottom_model + value/2, f'{value:.0f} TWh',
                    ha='center', va='center', fontweight='bold', fontsize=9)
            bottom_model += value

        added_labels = True
        
        x_position += 1
        x_tick_labels.append(f'Model {year} {wiggle}')

    # Formatting
    ax.set_xticks(range(len(x_tick_labels)))
    ax.set_xticklabels(x_tick_labels, rotation=45, ha='right')
    ax.set_ylabel('Gas Consumption (TWh/a)')

    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=3, frameon=False)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_xlim(-0.5, len(x_tick_labels) - 0.5)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.savefig(snakemake.output[0], bbox_inches='tight')
    plt.close(fig)