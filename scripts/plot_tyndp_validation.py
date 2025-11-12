import logging

import pypsa
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.backends.backend_pdf import PdfPages

from _tyndp_helpers import _extract_scenario_values, _extract_scenario_values_rowwise
from _helpers import configure_logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Final electricity demand per sector, EU27 (TWh)
# Sheet: '6-'  |  Row labels: 'Residential & Tertiary', 'Transport', 'Industry'
# ---------------------------------------------------------------------------

def final_electricity_residential_tertiary(path):
    return _extract_scenario_values(path, sheet_name='6-', row_label='Residential & Tertiary')

def final_electricity_transport(path):
    return _extract_scenario_values(path, sheet_name='6-', row_label='Transport')

def final_electricity_industry(path):
    return _extract_scenario_values(path, sheet_name='6-', row_label='Industry')


# ---------------------------------------------------------------------------
# Methane demand per sector, EU27 (TWh)
# Sheet: '8-'  |  Row labels (top table): 'Residential&Tertiary', 'Power Generation', 'Industry'
# ---------------------------------------------------------------------------

def methane_residential_tertiary(path):
    # Note: in the sheet, it's "Residential&Tertiary" (no spaces around &)
    return _extract_scenario_values(path, sheet_name='8-', row_label='Residential&Tertiary')

def power_generation_methane(path):
    return _extract_scenario_values(path, sheet_name='8-', row_label='Power Generation')

def methane_industry_energetic(path):
    return _extract_scenario_values(path, sheet_name='8-', row_label='Industry')

def methane_industry_nonenergetic(path):
    return _extract_scenario_values(path, sheet_name='8-', row_label='Non-energy use')


# ---------------------------------------------------------------------------
# Energy sources for District Heating, EU27 (TWh) – Methane only
# Sheet: '12-'  |  Row label: 'Methane'
# ---------------------------------------------------------------------------


def district_heating_methane(path):
    return _extract_scenario_values(path, sheet_name='12-', row_label='Methane')
# ---------------------------------------------------------------------------
# Primary energy supply mix, EU27 (TWh)
# Sheet: '19-'  |  Row label: 'Natural gas****', 'Oil', 'Coal', 'Biomass', 'Nuclear', 'Solar', 'Wind'
# ---------------------------------------------------------------------------

def methane_primary_total(path):
    return _extract_scenario_values(path, sheet_name='19-', row_label='Natural gas****')

def oil_primary_total(path):
    return _extract_scenario_values(path, sheet_name='19-', row_label='Oil')

def coal_primary_total(path):
    return _extract_scenario_values(path, sheet_name='19-', row_label='Coal')

def biomass_primary_total(path):
    return _extract_scenario_values(path, sheet_name='19-', row_label='Biomass')

def nuclear_primary_total(path):
    return _extract_scenario_values(path, sheet_name='19-', row_label='Nuclear')

def solar_primary_total(path):
    return _extract_scenario_values(path, sheet_name='19-', row_label='Solar')

def wind_primary_total(path):
    return _extract_scenario_values(path, sheet_name='19-', row_label='Wind')

# ---------------------------------------------------------------------------
# Power capacity mix, EU27 (GW)
# Sheet: '25-'  |  Row label: Gas, Solar, Wind Onshore, Wind Offshore, Nuclear, Battery
# ---------------------------------------------------------------------------

def gas_power_capacity_mix(path):
    return _extract_scenario_values_rowwise(path, '25-', 'Methane')

def solar_power_capacity_mix(path):
    return _extract_scenario_values_rowwise(path, '25-', 'Solar')

def wind_onshore_power_capacity_mix(path):
    return _extract_scenario_values_rowwise(path, '25-', 'Wind Onshore')

def wind_offshore_power_capacity_mix(path):
    return _extract_scenario_values_rowwise(path, '25-', 'Wind Offshore')

def nuclear_power_capacity_mix(path):
    return _extract_scenario_values_rowwise(path, '25-', 'Nuclear')

def battery_power_capacity_mix(path):
    return _extract_scenario_values_rowwise(path, '25-', 'Battery')

# ---------------------------------------------------------------------------
# Power generation mix, EU27 (TWh)
# Sheet: '26-'  |  Row label: Gas, Solar, Wind Onshore, Wind Offshore, Nuclear, Battery
# ---------------------------------------------------------------------------

def gas_power_generation_mix(path):
    return _extract_scenario_values_rowwise(path, '26-', 'Methane****')

def solar_power_generation_mix(path):
    return _extract_scenario_values_rowwise(path, '26-', 'Solar**')

def wind_onshore_power_generation_mix(path):
    return _extract_scenario_values_rowwise(path, '26-', 'Wind Onshore**')

def wind_offshore_power_generation_mix(path):
    return _extract_scenario_values_rowwise(path, '26-', 'Wind Offshore**')

def nuclear_power_generation_mix(path):
    return _extract_scenario_values_rowwise(path, '26-', 'Nuclear')

def battery_power_generation_mix(path):
    return _extract_scenario_values_rowwise(path, '26-', 'Battery')

# ---------------------------------------------------------------------------
# Domestic gas production, EU27 (TWh)
# Sheet: '31-'  |  Row label: 'Natural gas (unabated)', 'Biomethane', 'Hydrogen (Green)', 'Hydrogen (Blue)', 'Hydrogen (Grey)', 'Synthetic methane'
# ---------------------------------------------------------------------------

def methane_domestic_production(path):
    return _extract_scenario_values(path, sheet_name='31-', row_label='Natural gas (unabated)')

def biomethane_domestic_production(path):
    return _extract_scenario_values(path, sheet_name='31-', row_label='Biomethane')

def green_hydrogen_domestic_production(path):
    return _extract_scenario_values(path, sheet_name='31-', row_label='Hydrogen (Green)')

def blue_hydrogen_domestic_production(path):
    return _extract_scenario_values(path, sheet_name='31-', row_label='Hydrogen (Blue)')

def grey_hydrogen_domestic_production(path):
    return _extract_scenario_values(path, sheet_name='31-', row_label='Hydrogen (Grey)')

def synthetic_gas_domestic_production(path):
    return _extract_scenario_values(path, sheet_name='31-', row_label='Synthetic methane')

# ---------------------------------------------------------------------------
# Methane Supply for EU27 (TWh)
# Sheet: '32-'  |  Row label: 'Natural gas (Import)', 'Natural gas (Domestic)', 'Biomethane (Import)', 'Biomethane (Domestic)', 'Synthetic methane (Import)', 'Synthetic methane (Domestic)'
# ---------------------------------------------------------------------------

def natural_gas_import(path):
    return _extract_scenario_values(path, sheet_name='32-', row_label='Natural gas (Import)')

def natural_gas_domestic(path):
    return _extract_scenario_values(path, sheet_name='32-', row_label='Natural gas (Domestic)')

def biomethane_import(path):
    return _extract_scenario_values(path, sheet_name='32-', row_label='Biomethane (Import)')

def biomethane_domestic(path):
    return _extract_scenario_values(path, sheet_name='32-', row_label='Biomethane (Domestic)')

def synthetic_methane_import(path):
    return _extract_scenario_values(path, sheet_name='32-', row_label='Synthetic methane (Import)')

def synthetic_methane_domestic(path):
    return _extract_scenario_values(path, sheet_name='32-', row_label='Synthetic methane (Domestic)')


def data_to_plot(
    ax,
    years,
    model_data,
    tyndp_data=None,
    title=None,
    ylabel=None,
    ylimmax=None,
    ):

    color = np.random.rand(3,)
    ax.plot(years, model_data, marker='o', color=color)

    gradient = (model_data[-1] - model_data[len(years)//2]) / (years[-1] - years[len(years)//2])

    extrapolation_years = [years[-1], 2040]
    extrapolation = [model_data[-1], model_data[-1] + gradient * (2040 - years[-1])]

    ax.plot(extrapolation_years, extrapolation, linestyle='--', color=color)

    if title is not None:
        ax.set_title(title)
    if ylabel is not None:
        ax.set_ylabel(ylabel)

    used_data = [0]
    if tyndp_data is not None:
        ax.scatter(
            [2030, 2040],
            [tyndp_data['National Trends'][2030], tyndp_data['National Trends'][2040]],
            color=tyndp_colors['National Trends'],
            edgecolors=darken_color(tyndp_colors['National Trends']),
            s=100,
            alpha=0.5,
            )
        ax.scatter(
            [2040],
            [tyndp_data['Distributed Energy'][2040]],
            color=tyndp_colors['Distributed Energy'],
            edgecolors=darken_color(tyndp_colors['Distributed Energy']),
            s=100,
            alpha=0.5,
            )
        ax.scatter(
            [2040],
            [tyndp_data['Global Ambition'][2040]],
            color=tyndp_colors['Global Ambition'],
            edgecolors=darken_color(tyndp_colors['Global Ambition']),
            s=100,
            alpha=0.5,
        )
        used_data = [
            tyndp_data['National Trends'][2030],
            tyndp_data['National Trends'][2040],
            tyndp_data['Distributed Energy'][2040],
            tyndp_data['Global Ambition'][2040],
        ]

    if ylimmax is None:
        ax.set_ylim(0, max(max(model_data), max(used_data)) * 1.1)
    else:
        ax.set_ylim(0, ylimmax)

    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel('Year')

    ax.set_xlim(2024, 2041)
    ax.set_xticks([2025, 2030, 2035, 2040])


def get_ylimmax(args):
    ylimmax = 0

    for arg in args:
        if isinstance(arg, list):
            ylimmax = max(ylimmax, max(arg))
        elif isinstance(arg, dict):
            ylimmax = max(ylimmax, max(
                [arg['National Trends'][2030], arg['National Trends'][2040], arg['Distributed Energy'][2040], arg['Global Ambition'][2040]]
                )
            )
    return ylimmax


def add_tyndp_legend(axs):
    # Add legend for TYNDP scenarios
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', markerfacecolor=tyndp_colors['National Trends'], 
            markeredgecolor=darken_color(tyndp_colors['National Trends']), 
            markersize=10, color='w', label='National Trends', alpha=0.5),
        Line2D([0], [0], marker='o', markerfacecolor=tyndp_colors['Distributed Energy'], 
            markeredgecolor=darken_color(tyndp_colors['Distributed Energy']), 
            markersize=10, color='w', label='Distributed Energy', alpha=0.5),
        Line2D([0], [0], marker='o', markerfacecolor=tyndp_colors['Global Ambition'], 
            markeredgecolor=darken_color(tyndp_colors['Global Ambition']), 
            markersize=10, color='w', label='Global Ambition', alpha=0.5),
    ]
    axs[-1].legend(
        title='TYNDP Scenarios',
        handles=legend_elements,
        loc='center left',
        bbox_to_anchor=(1, 0.5),
        ncol=1,
        frameon=False,
        )

if __name__ == "__main__":

    configure_logging(snakemake)

    valpath = snakemake.input.valpath
    years  = list(range(2025, 2036))

    logger.info(f"Loading networks from {snakemake.input.networks}")
    ns = {
        year: pypsa.Network(fn)
        for year, fn in zip(years, snakemake.input.networks)
    }
    n = ns[2025].copy()

    eu27_countries = [
        "AT",  # Austria
        "BE",  # Belgium
        "BG",  # Bulgaria
        "HR",  # Croatia
        "CY",  # Cyprus
        "CZ",  # Czechia
        "DK",  # Denmark
        "EE",  # Estonia
        "FI",  # Finland
        "FR",  # France
        "DE",  # Germany
        "GR",  # Greece
        "HU",  # Hungary
        "IE",  # Ireland
        "IT",  # Italy
        "LV",  # Latvia
        "LT",  # Lithuania
        "LU",  # Luxembourg
        "MT",  # Malta
        "NL",  # Netherlands
        "PL",  # Poland
        "PT",  # Portugal
        "RO",  # Romania
        "SK",  # Slovakia
        "SI",  # Slovenia
        "ES",  # Spain
        "SE",  # Sweden
    ]
    eu27_buses = ns[2027].buses.index[
        (ns[2027].buses.country.isin(eu27_countries)) &
        (ns[2027].buses.carrier == 'AC')
    ]
    all_eu27_buses = ns[2027].buses.index[
        (ns[2027].buses.country.isin(eu27_countries))
    ]

    tyndp_colors = {
        'National Trends': '#fe4a49',
        'Distributed Energy': '#2ab7ca',
        'Global Ambition': '#fed766',
    }

    def darken_color(color, factor=0.7):
        rgb = mcolors.to_rgb(color)
        return tuple(c * factor for c in rgb)

    logger.info(f'Creating validation pdf {snakemake.output[0]}')

    with PdfPages(snakemake.output[0]) as pdf:

        tyndp_ev_demand = final_electricity_transport(valpath)

        ev_load = list()
        ice_load = list()

        w = n.snapshot_weightings.generators.iloc[0]

        for year in years:

            evs = ns[year].loads.index[(ns[year].loads.carrier == 'land transport EV') & (ns[year].loads.bus.isin(all_eu27_buses))]
            ices = ns[year].loads.index[(ns[year].loads.carrier == 'land transport oil') & (ns[year].loads.bus.isin(all_eu27_buses))]

            ev_load.append(ns[year].loads_t.p_set.loc[:, evs].sum().sum() * w * 1e-6)
            ice_load.append(ns[year].loads_t.p_set.loc[:, ices].sum().sum() * w * 1e-6)
        
        fig, axs = plt.subplots(1, 2, figsize=(8.5, 4))

        ylimmax = get_ylimmax([ev_load, ice_load, tyndp_ev_demand]) * 1.08

        data_to_plot(axs[0], years, ev_load, tyndp_data=tyndp_ev_demand, title='Electricity demand for land transport', ylabel='Demand [TWh]', ylimmax=ylimmax)
        data_to_plot(axs[1], years, ice_load, title='Fossil fuel demand for land transport', ylabel='Demand [TWh]', ylimmax=ylimmax)

        add_tyndp_legend(axs)

        fig.subplots_adjust(hspace=0.3, wspace=0.3)
        pdf.savefig(fig)
        plt.close(fig)


        idx = pd.IndexSlice

        heating_carriers = pd.Index([
            'urban decentral heat',
            'urban central heat',
            'residential urban decentral heat',
            'residential rural heat',
            'rural heat',
        ])

        tyndp_gas_heating = methane_residential_tertiary(valpath)


        indicators = ['gas boiler', 'biomass boiler', 'resistive heater', 'heat pump', 'oil boiler']

        def clean_carrier(carrier):
            if len(matches := [ind for ind in indicators if ind in carrier]) > 0:
                return matches[0]
            else:
                return 'other'


        data = []

        for year in years:
            try:
                eb = ns[year].statistics.energy_balance(groupby=['carrier', 'bus', 'bus_carrier'])
                eb = eb.loc[idx[:, :, all_eu27_buses.intersection(eb.index.get_level_values(2).unique())]]
                eb = eb.groupby(level=[0,1,3]).sum()
                eb = eb.loc[
                    idx[:, :, heating_carriers.intersection(eb.index.get_level_values(2).unique())]
                    ].sort_values()
            except KeyError:
                continue

            grouper = {
                carrier: clean_carrier(carrier)
                for carrier in eb.index.get_level_values(1)
            }

            eb.index = eb.index.get_level_values(1)
            eb = eb.groupby(grouper).sum().drop('other').mul(1e-6)

            data.append(eb.rename(year))

        data = pd.concat([d.to_frame() for d in data], axis=1)

        ylimmax = data.max().max() * 1.1

        fig, axs = plt.subplots(2, 3, figsize=(3 * 4.2, 8))

        for i, (name, row) in enumerate(data.iterrows()):

            ax = axs.flatten()[i]

            if name == 'gas boiler':
                kwargs = {'tyndp_data': tyndp_gas_heating}
            else:
                kwargs = {}

            data_to_plot(ax, row.index, list(row), title=name, ylabel='Energy [TWh]', ylimmax=ylimmax, **kwargs)

        axs.flatten()[-1].set_visible(False)
        add_tyndp_legend(axs.flatten()[:-1])

        pdf.savefig(fig)
        plt.close(fig)


        industry_loads = [
            'gas for industry',
            'H2 for industry',
            'naphtha for industry',
            'coal for industry',
            'low-temperature heat for industry',
            'industry electricity',
            'solid biomass for industry',
            'industry methanol',
        ]

        data = []

        for year in years:

            year_loads = []

            for load in industry_loads:
                index = ns[year].loads.index[(ns[year].loads.carrier == load) & ns[year].loads.bus.isin(all_eu27_buses)]
                
                try:
                    year_loads.append(ns[year].loads_t.p_set.loc[:, index].sum().sum() * w * 1e-6)
                except KeyError:
                    year_loads.append(ns[year].loads.loc[index, 'p_set'].sum() * w * 1e-6 * len(n.snapshots))

            year_loads = pd.DataFrame({year: year_loads}, index=industry_loads)
            data.append(year_loads)

        data = pd.concat(data, axis=1)


        def add_dict_leaves(dict1, dict2):
            """Add all leaf nodes of two dicts with the same structure."""
            result = {}
            for key in dict1:
                if isinstance(dict1[key], dict):
                    result[key] = add_dict_leaves(dict1[key], dict2[key])
                else:
                    result[key] = dict1[key] + dict2[key]
            return result

        gas_industry_energetic = methane_industry_energetic(valpath)
        gas_industry_nonenergetic = methane_industry_nonenergetic(valpath)
        gas_industry = add_dict_leaves(gas_industry_energetic, gas_industry_nonenergetic)

        electricity_industry = final_electricity_industry(valpath)

        ylimmax = get_ylimmax([data.values.flatten(), gas_industry, electricity_industry]) * 1.1

        fig, axs = plt.subplots(2, len(industry_loads)//2, figsize=(len(industry_loads)//2 * 4, 8))

        for i, (name, row) in enumerate(data.iterrows()):
            ax = axs.flatten()[i]

            if name == 'gas for industry':
                kwargs = {'tyndp_data': gas_industry}
            elif name == 'industry electricity':
                kwargs = {'tyndp_data': electricity_industry}
            else:
                kwargs = {}

            data_to_plot(ax, row.index, list(row), title=name, ylabel='Energy [TWh]', ylimmax=ylimmax, **kwargs)


        fig.subplots_adjust(hspace=0.3, wspace=0.3)

        pdf.savefig(fig)
        plt.close(fig)
        
        cs = n.statistics.optimal_capacity().index.get_level_values(1).unique()

        capacities = {
            'offwind': ['Offshore Wind (AC)', 'Offshore Wind (DC)', 'Offshore Wind (Floating)'],
            'onwind': ['Onshore Wind'],
            'solar': ['Solar', 'solar-hsat'],
            'hydro': ['Run of River', 'Reservoir & Dam'],
            'gas': ['Combined-Cycle Gas', 'Open-Cycle Gas'],
            'coal': ['coal', 'lignite'],
            'nuclear': ['nuclear'],
            'battery': ['Battery Storage'],
            'PHS': ['Pumped Hydro Storage'],
            'electrolyser': ['H2 Electrolysis'],
        }

        gas_capacity = gas_power_capacity_mix(valpath)
        solar_capacity = solar_power_capacity_mix(valpath)
        onwind_capacity = wind_onshore_power_capacity_mix(valpath)
        offwind_capacity = wind_offshore_power_capacity_mix(valpath)
        nuclear_capacity = nuclear_power_capacity_mix(valpath)
        battery_capacity = battery_power_capacity_mix(valpath)

        eu_mapping = {
            'Biomass': 'Biomass',
            'Waste': 'Waste',
            'Fossil Brown coal/Lignite': 'Lignite',
            'Fossil Hard coal': 'Coal',
            'Fossil Coal-derived gas': 'Gas',
            'Fossil Gas': 'Gas',
            'Fossil Oil': 'Gas',
            'Fossil Oil shale': 'Gas',
            'Fossil Peat': 'Gas',
            'Geothermal': 'Geothermal',
            'Hydro Pumped Storage': 'Hydro',
            'Hydro Run-of-river and poundage': 'Hydro',
            'Hydro Water Reservoir': 'Hydro',
            'Marine': 'Hydro',
            'Nuclear': 'Nuclear',
            'Other': 'Gas',
            'Other renewable': 'Biomass',
            'Solar': 'Solar',
            'Wind Offshore': 'Wind Offshore',
            'Wind Onshore': 'Wind Onshore',
        }

        # Load 2024 installed capacities from ENTSO-E data
        net_gen_cap = pd.read_csv(
            snakemake.input.existing_capacities,
            sep='\t',
            index_col=[3,4]
        )['ProvidedValue'].unstack().replace(np.nan, 0).groupby(eu_mapping).sum()

        # print(net_gen_cap)
        
        # Map ENTSO-E categories to our capacity categories
        entso_e_2024 = {
            'offwind': net_gen_cap.loc['Wind Offshore'].sum() * 1e-3,  # Convert MW to GW
            'onwind': net_gen_cap.loc['Wind Onshore'].sum() * 1e-3,
            'solar': net_gen_cap.loc['Solar'].sum() * 1e-3,
            'hydro': net_gen_cap.loc['Hydro'].sum() * 1e-3,
            'gas': net_gen_cap.loc['Gas'].sum() * 1e-3,
            'coal': (net_gen_cap.loc['Coal'].sum() + net_gen_cap.loc['Lignite'].sum()) * 1e-3,
            'nuclear': net_gen_cap.loc['Nuclear'].sum() * 1e-3,
            'battery': 0,
            'PHS': 0,
            'electrolyser': 0,  # Not in ENTSO-E data
        }

        data = []

        for year in years:

            year_caps = []
            
            caps = ns[year].statistics.optimal_capacity(groupby=['carrier', 'bus'])
            caps = caps.loc[idx[:, :, caps.index.get_level_values(2).isin(all_eu27_buses.tolist() + ['EU coal', 'EU lignite', 'EU uranium', 'EU gas'])]]
            caps = caps.groupby(level=[0, 1]).sum()

            for name, carriers in capacities.items():
                
                if name in ['coal', 'gas']:
                    year_caps.append(caps.loc[idx['Link', carriers]].sum() * 1e-3)
                elif name in ['nuclear']:
                    year_caps.append(caps.loc[idx['Generator', carriers]].sum() * 1e-3)
                else:
                    year_caps.append(caps.loc[idx[:, carriers]].sum() * 1e-3)

            year_caps = pd.DataFrame({year: year_caps}, index=capacities.keys())
            data.append(year_caps)

        data.insert(0, pd.DataFrame({2024: pd.Series(entso_e_2024)}))

        data = pd.concat(data, axis=1).sort_index(axis=1)
        
        ylimmax = get_ylimmax([
            data.values.flatten(),
            gas_capacity,
            solar_capacity,
            onwind_capacity,
            offwind_capacity,
            nuclear_capacity,
            battery_capacity
            ]) * 1.1
        
        data = data.loc[:, 2025:]


        fig, axs = plt.subplots(2, len(capacities)//2, figsize=(len(capacities)//2 * 4, 8))

        for i, (name, row) in enumerate(data.iterrows()):
            ax = axs.flatten()[i]

            if name+'_capacity' in globals():
                kwargs = {'tyndp_data': globals()[name+'_capacity']}
            else:
                kwargs = {}
            
            data_to_plot(ax, row.index, list(row), title=name, ylimmax=ylimmax, **kwargs)
            
            # Extend x-axis to include 2024
            ax.set_xlim(2023, 2036)

            existing_capacity = entso_e_2024[name]
            ax.scatter(
                [2024],
                [existing_capacity],
                color='w',
                edgecolors='black',
                s=80,
                alpha=0.8,
            )

        for ax in axs[:,0]:
            ax.set_ylabel('Power Capacity [GW]')

        fig.subplots_adjust(hspace=0.3, wspace=0.3)
        pdf.savefig(fig)
        plt.close(fig)

        gas_mix = gas_power_generation_mix(valpath)
        solar_mix = solar_power_generation_mix(valpath)
        onwind_mix = wind_onshore_power_generation_mix(valpath)
        offwind_mix = wind_offshore_power_generation_mix(valpath)
        nuclear_mix = nuclear_power_generation_mix(valpath)
        battery_mix = battery_power_generation_mix(valpath)

        data = []

        for year in years:

            year_mix = []
            # mixes = ns[year].statistics.supply()
            mixes = ns[year].statistics.supply(groupby=['carrier', 'bus']).loc[idx[:, :, eu27_buses]]
            mixes = mixes.groupby(level=[0, 1]).sum()

            for name, carriers in capacities.items():

                if name == 'battery' or name == 'electrolyser':
                    year_mix.append(0)
                    continue


                if name in ['coal', 'gas']:
                    year_mix.append(mixes.loc[idx['Link', carriers]].sum() * 1e-6)
                else:
                    year_mix.append(mixes.loc[idx[:, carriers]].sum() * 1e-6)

            year_mix = pd.DataFrame({year: year_mix}, index=capacities.keys())
            data.append(year_mix)

        data = pd.concat(data, axis=1)
        ylimmax = get_ylimmax([
            data.values.flatten(),
            gas_mix,
            solar_mix,
            onwind_mix,
            offwind_mix,
            nuclear_mix,
            battery_mix
        ]) * 1.1


        fig, axs = plt.subplots(2, len(capacities)//2, figsize=(len(capacities)//2 * 4, 8))

        for i, (name, row) in enumerate(data.iterrows()):
            ax = axs.flatten()[i]

            if name+'_mix' in globals():
                kwargs = {'tyndp_data': globals()[name+'_mix']}
            else:
                kwargs = {}
            
            data_to_plot(ax, row.index, list(row), title=name, ylimmax=ylimmax, **kwargs)

        for ax in axs[:,0]:
            ax.set_ylabel('Power Generation [TWh]')

        fig.subplots_adjust(hspace=0.3, wspace=0.3)

        pdf.savefig(fig)
        plt.close(fig)


        gens = n.generators.index[n.generators.carrier == 'co2-ets']

        emissions = []

        for year in years:
            emissions.append(ns[year].generators_t.p.loc[:, gens].sum().sum() * w * -1e-6)

        emissions = pd.DataFrame({'emissions': emissions}, index=years)

        fig, ax = plt.subplots(1, 1, figsize=(4, 4))

        emissions['emissions'].plot(ax=ax, marker='o', color=np.random.rand(3,))
        ax.set_title('Emissions')

        ax.grid(True, linestyle='--', alpha=0.7)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax.set_xlabel('Year')

        ax.set_ylabel('CO2 Emissions [MtCO2/a]')

        pdf.savefig(fig)
        plt.close(fig)