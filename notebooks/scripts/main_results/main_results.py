"""
Reproduce the main results figure (main_results.pdf) as a standalone script.

Converted from notebooks/plot_total_gas_use.ipynb. Only the cells that feed
the saved figure are kept. Output is written next to this script and copied
to ../../../../gas_resilience/imgs/main_results.pdf, matching the behaviour
of the original notebook.
"""
from __future__ import annotations

import sys
import shutil
from pathlib import Path

import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
NOTEBOOKS = ROOT / "notebooks"
RESULTS = ROOT / "results"
SIBLING_IMGS = ROOT.parent / "gas_resilience" / "imgs"

sys.path.insert(0, str(NOTEBOOKS))
from plotting_globals import GAS_COLOR  # noqa: E402
sys.path.insert(0, str(NOTEBOOKS / "scripts"))
from frontend_export import export_frontend_data  # noqa: E402

idx = pd.IndexSlice


# ============================================================
# TECH COLORS
# ============================================================

with open(ROOT / "config" / "plotting.default.yaml", "r") as f:
    tech_colors = yaml.safe_load(f)["plotting"]["tech_colors"]

tech_colors.update({
    "SMR CC":                        "#1f77b4",
    "heat100-200 industry gas":      "#ff7f0e",
    "heat<100 industry gas":         "#2ca02c",
    "Combined-Cycle Gas":            "#d62728",
    "Open-Cycle Gas":                "#9467bd",
    "heat200-500 industry gas":      "#17becf",
    "H2 electrolysis":               "#e377c2",
    "urban central gas boiler":      "#7f7f7f",
    "urban central gas CHP":         "#bcbd22",
    "rural gas boiler":              "#8c564b",
    "urban decentral gas boiler":    "#aec7e8",
    "urban decentral biomass boiler":"#92648c",
    "rural biomass boiler":          "#94312c",
    "heat>500 industry gas":         "#98df8a",
    "gas DRI":                       "#000000",
    "hydrogen heating":               "magenta",
    # Aggregate sectors coloured from config.basicrun.yaml tech_colors so the
    # site matches the canonical PyPSA-Eur palette. ">500C industry heat and
    # feedstock" has no config equivalent (custom aggregate), so it keeps a
    # distinct teal.
    ">500C industry heat and feedstock": "#4A9BA8",
    "<500C industry heat": "#8f2727",   # config: low-temperature heat for industry
    "biomethane": "#e36311",            # config: biogas to gas
    "building heat": "#cc1f1f",         # config: heat
    "electricity": "#110d63",           # config: electricity
    "hydrogen": "#bf13a0",              # config: hydrogen / H2
})

tech_colors['AC'] = tech_colors['electricity']
tech_colors['residential heating'] = tech_colors['building heat']
tech_colors['low-temperature industry heat'] = tech_colors['<500C industry heat']

tech_colors['residential gas boiler'] = tech_colors['urban decentral gas boiler']
tech_colors['gas turbine'] = tech_colors['Open-Cycle Gas']
tech_colors['wind'] = tech_colors['Onshore Wind']
tech_colors['solar'] = tech_colors['solar-hsat']
tech_colors['biomass buildings'] = tech_colors['urban decentral biomass boiler']
tech_colors['biomass industry'] = tech_colors['heat200-500 industry solid biomass']
tech_colors['building heat pump'] = tech_colors['ground heat pump']
tech_colors['rural ground heat pump'] = tech_colors['air heat pump']
tech_colors['rural oil boiler'] = tech_colors['oil boiler']
tech_colors['urban decentral oil boiler'] = tech_colors['oil boiler']
tech_colors['urban decentral air heat pump'] = tech_colors['air heat pump']
tech_colors['urban decentral resistive heater'] = tech_colors['resistive heater']
tech_colors['rural resistive heater'] = tech_colors['resistive heater']
tech_colors['building electric heating'] = tech_colors['ground heat pump']
tech_colors['industrial heat pump'] = tech_colors['heat<100 industry industrial heat pump medium temperature']


# ============================================================
# LOAD DATA
# ============================================================

mode = '3'
csvs = RESULTS / f"csvs_{mode}_latest"

bal = pd.read_csv(
    csvs / "energy_balance.csv",
    index_col=list(range(3)), header=list(range(5)),
).loc[:, idx[:, :, :, 'free']]
bal.columns = bal.columns.get_level_values(-1).astype(int)

mp = pd.read_csv(
    csvs / "prices.csv",
    index_col=list(range(1)), header=list(range(5)),
).loc[:, idx[:, :, :, 'free']]
mp.columns = mp.columns.get_level_values(-1).astype(int)

commodity_prices = mp.loc[['gas', 'low voltage', 'urban decentral heat',
                           'heat<100 industry', 'solid biomass']]

commidity_nice_names = {
    'gas': 'gas',
    'low voltage': 'electricity',
    'urban decentral heat': 'building heat',
    'heat<100 industry': 'low-temperature\nindustry heat',
    'solid biomass': 'biomass',
}
commidity_colors = {
    'gas': tech_colors['gas'],
    'low voltage': tech_colors['electricity'],     # config: electricity
    'urban decentral heat': tech_colors['building heat'],  # config: heat
    'heat<100 industry': tech_colors['<500C industry heat'],
    'solid biomass': tech_colors['solid biomass'],
}

eb = pd.read_csv(
    csvs / "energy_balance.csv",
    header=list(range(5)), index_col=[0, 1, 2],
)
eb.columns = eb.columns.get_level_values(-1).astype(int)

prices = pd.read_csv(
    csvs / "prices.csv",
    header=list(range(5)), index_col=[0],
)
prices.columns = prices.columns.get_level_values(-1).astype(int)

costs_raw = pd.read_csv(
    csvs / "costs.csv",
    index_col=list(range(3)), header=list(range(5)),
)

# Actual ETS payment per scenario = cost of the 'co2-ets' carrier (whole-system
# carbon price x realised emissions of every emitting carrier, net of capture),
# read straight from the cost summary. Mirrors co2_ets_cost_be() in
# plot_cost_price_surfaces.py rather than synthesising a gas-only adder.
ets_costs = costs_raw.loc[costs_raw.index.get_level_values('carrier') == 'co2-ets']
ets_costs = ets_costs.sum().mul(1e-9)
ets_costs.index = ets_costs.index.get_level_values(-1)

costs = costs_raw.loc[costs_raw.index.get_level_values('carrier') != 'co2-ets']
costs = costs.sum().mul(1e-9)
costs.index = costs.index.get_level_values(-1)
if '1750' in costs.index:
    i = costs.index.get_loc('1750')
    if i > 0 and i < len(costs) - 1:
        prev_val = costs.iloc[i - 1]
        next_val = costs.iloc[i + 1]
        costs.iloc[i] = (prev_val + next_val) / 2
costs.name = 'System Cost [B€]'
costs.index.name = 'Gas Consumption [TWh/a]'
ets_costs = ets_costs.reindex(costs.index)


# ============================================================
# CONSTANTS
# ============================================================

gas_techs_by_bus = {
    "heat<100 industry": ["heat<100 industry gas"],
    "heat100-200 industry": ["heat100-200 industry gas"],
    "heat200-500 industry": ["heat200-500 industry gas"],
    "AC": ["Combined-Cycle Gas", "Open-Cycle Gas", "urban central gas CHP"],
    "rural heat": ["rural gas boiler"],
    "urban central heat": ["urban central gas CHP", "urban central gas boiler"],
    "urban decentral heat": ["urban decentral gas boiler"],
    "Hydrogen Storage": ["SMR"],
    "heat>500 industry": ["heat>500 industry gas"],
}

nice_demand_names = {
    "heat<100 industry": "Industry Heat <100°C",
    "heat100-200 industry": "Industry Heat 100-200°C",
    "heat200-500 industry": "Industry Heat 200-500°C",
    "AC": "Electricity",
    "rural heat": "Rural Heating",
    "urban central heat": "District Heating",
    "urban decentral heat": "Urban Decentral Heating",
    "Hydrogen Storage": "Hydrogen",
    "heat>500 industry": "Industry Heat >500°C",
}

carrier_mapping = {
    'gas for industry': '>500C industry heat and feedstock',
    'heat>500 industry gas': '>500C industry heat and feedstock',
    'heat200-500 industry gas': '<500C industry heat',
    'heat100-200 industry gas': '<500C industry heat',
    'heat<100 industry gas': '<500C industry heat',
    'biogas to gas': 'biomethane',
    'SMR': 'hydrogen',
    'urban decentral gas boiler': 'building heat',
    'urban central gas boiler': 'building heat',
    'urban central gas CHP': 'building heat',
    'rural gas boiler': 'building heat',
    'Open-Cycle Gas': 'electricity',
    'Combined-Cycle Gas': 'electricity',
}

nice_names = {
    'heat<100 industry industrial heat pump medium temperature': 'heat pump',
    'heat<100 industry solid biomass': 'biomass',
    'heat<100 industry gas': 'gas boiler',
    'heat100-200 industry industrial heat pump high temperature': 'heat pump',
    'heat100-200 industry solid biomass': 'biomass',
    'heat100-200 industry gas': 'gas boiler',
    'heat200-500 industry solid biomass': 'biomass',
    'heat200-500 industry gas': 'direct firing gas',
    'Offshore Wind (AC)': 'Offshore Wind (AC)',
    'Offshore Wind (DC)': 'Offshore Wind (DC)',
    'Onshore Wind': 'Onshore Wind',
    'Run of River': 'Run-of-River',
    'Solar': 'Solar PV',
    'coal': 'Coal',
    'lignite': 'Lignite',
    'nuclear': 'Nuclear',
    'urban central solid biomass CHP': 'biomass CHP',
    'urban central solid biomass CHP CC': 'biomass CHP CC',
    'Reservoir & Dam': 'Hydro (res+dam)',
    'Combined-Cycle Gas': 'gas turbine',
    'urban central gas CHP': 'gas CHP',
    'rural biomass boiler': 'biomass boiler',
    'rural ground heat pump': 'electric heating',
    'rural oil boiler': 'oil boiler',
    'rural resistive heater': 'resistive heater',
    'rural gas boiler': 'gas boiler',
    'H2 Electrolysis': 'electrolysis',
    'Haber-Bosch': 'Haber-Bosch',
    'urban central air heat pump': 'air heat pump',
    'urban central resistive heater': 'resistive heater',
    'urban central gas boiler': 'gas boiler',
    'urban decentral air heat pump': 'air heat pump',
    'urban decentral biomass boiler': 'biomass boiler',
    'urban decentral oil boiler': 'oil boiler',
    'urban decentral resistive heater': 'resistive heater',
    'urban decentral gas boiler': 'gas boiler',
    'SMR': 'steam-methane reforming',
    'heat>500 industry hydrogen': 'hydrogen',
    'heat>500 industry gas': 'direct firing gas',
}


# ============================================================
# DATA PREPARATION
# ============================================================

consumption_threshold_color = '#2B1B07'

gb = bal.loc[:, :, 'gas'].mul(1e-6)
gb.index = gb.index.get_level_values(1)
gb.drop('gas', inplace=True)

# System cost vs gas consumption
cost_x = np.arange(len(costs)) + 0.5
cost_values = costs.values
cost_y_min = 840
cost_y_max = 1230

# Commodity prices vs gas consumption
commodity_x = np.arange(len(commodity_prices.columns)) + 0.5
commodity_y_min = 0
commodity_y_max = commodity_prices.max().max() * 1.1

# Sectoral gas allocation (stacked bar data)
gb = gb.groupby(carrier_mapping).sum()

sector_items = [
    '>500C industry heat and feedstock',
    'hydrogen',
    'building heat',
    'electricity',
    '<500C industry heat',
    'biomethane',
]

# Sectoral gas allocation panel colours: restore the original standalone palette
# (from plot_total_gas_use.ipynb) rather than the canonical PyPSA-Eur tech colours.
tech_colors['>500C industry heat and feedstock'] = '#4A9BA8'
tech_colors['<500C industry heat'] = '#6BBF8A'
tech_colors['biomethane'] = '#E8C547'
tech_colors['building heat'] = '#C2714F'
tech_colors['electricity'] = '#D94F5C'
tech_colors['hydrogen'] = '#8B6BBF'

area_data = []
bottoms_pos = pd.Series(0, index=gb.columns, dtype=float)
bottoms_neg = pd.Series(0, index=gb.columns, dtype=float)
running_pos = pd.Series(0, index=gb.columns, dtype=float)
running_neg = pd.Series(0, index=gb.columns, dtype=float)
envelope_data = []

for carrier in sector_items:
    row = gb.loc[carrier] * (-1)
    bottoms_arr = pd.Series(0.0, index=gb.columns)
    tops_arr = pd.Series(0.0, index=gb.columns)
    for con in row.index:
        value = row.loc[con]
        if value < 0:
            bottoms_arr.loc[con] = running_neg.loc[con] + value
            tops_arr.loc[con] = running_neg.loc[con]
            running_neg.loc[con] += value
            bottoms_neg.loc[con] += value
        else:
            bottoms_arr.loc[con] = running_pos.loc[con]
            tops_arr.loc[con] = running_pos.loc[con] + value
            running_pos.loc[con] += value
            bottoms_pos.loc[con] += value
    area_data.append((carrier, bottoms_arr.values, tops_arr.values, tech_colors[carrier]))
    envelope_data.append((running_pos.values.copy(), running_neg.values.copy()))

alloc_x = np.arange(len(gb.columns)) + 0.5
sector_xlim = (0, len(bottoms_pos) - 0.5)
sector_xticks = np.arange(len(bottoms_pos)) + 0.5
sector_xticklabels = [label if i % 4 == 0 else '' for i, label in enumerate(bottoms_pos.index)]

# Right column data: full technology share per bus carrier
gas_prices = prices.loc['gas']

price_thresholds = [20, 30, 50, 100]
x_at_price = {}
for pt in price_thresholds:
    x_at_price[pt] = np.interp(pt, gas_prices.values[::-1], gas_prices.index[::-1])

right_panel_data = {}

for bc, gas_techs in gas_techs_by_bus.items():
    ss = eb.loc[idx[:, :, bc], :]

    total_inflow = ss[ss > 0].sum(axis=0)

    all_techs = ss.index.get_level_values(1).unique()

    tech_shares = {}
    for tech in all_techs:
        try:
            contrib = ss.loc[idx[:, tech, :], :].sum(axis=0)
        except KeyError:
            continue
        share = (contrib / total_inflow * 100).fillna(0)
        if (share > 0.5).any():
            tech_shares[tech] = share

    shares_df = pd.DataFrame(tech_shares, index=prices.columns)

    gas_tech_names = [t for t in gas_techs if t in shares_df.columns]
    non_gas_tech_names = [t for t in shares_df.columns if t not in gas_techs]

    ordered_cols = gas_tech_names + non_gas_tech_names
    shares_df = shares_df[[c for c in ordered_cols if c in shares_df.columns]]

    row_sums = shares_df.sum(axis=1)
    shares_df = shares_df.div(row_sums, axis=0) * 100
    shares_df = shares_df.fillna(0)

    right_panel_data[bc] = {
        'shares': shares_df,
        'gas_techs': gas_tech_names,
        'non_gas_techs': non_gas_tech_names,
        'total': total_inflow,
    }

right_panel_data['AC']['shares']['Solar'] += right_panel_data['AC']['shares']['solar-hsat']
right_panel_data['AC']['shares'].drop(columns=['solar-hsat'], inplace=True)

tech_colors['electric heating'] = tech_colors['air heat pump']
tech_colors['Offshore Wind'] = tech_colors.get('Offshore Wind (AC)', '#6895dd')
nice_names['electric heating'] = 'electric heating'
nice_names['Offshore Wind'] = 'Offshore Wind'

# --- Split heat-pump output into electricity input vs ambient heat ---
# A heat pump delivers heat = electricity x COP. Only the electricity fraction
# (1/COP) is genuine "electric heating"; the (COP-1)/COP excess is drawn from
# the environment and is shown separately as 'ambient heat'. We split each heat
# pump's percentage share accordingly before it gets merged into the combined
# 'electric heating' band, so the total per panel still sums to 100%.
AMBIENT = 'ambient heat'
tech_colors[AMBIENT] = '#3FBFB3'   # turquoise
nice_names[AMBIENT] = 'ambient heat'

heat_pump_techs = [
    'urban central air heat pump',
    'urban decentral air heat pump',
    'rural air heat pump',
    'rural ground heat pump',
    'heat<100 industry industrial heat pump medium temperature',
    'heat100-200 industry industrial heat pump high temperature',
]


def _heat_pump_elec_fraction(tech):
    """Fraction of a heat pump's delivered heat supplied by electricity (1/COP),
    per gas-consumption column, read from the energy balance."""
    rows = eb.loc[idx[:, tech, :], :]
    heat_out = rows[rows > 0].sum()
    elec_in = -rows[rows < 0].sum()
    frac = (elec_in / heat_out).replace([np.inf, -np.inf], 0).fillna(0).clip(0, 1)
    return frac.reindex(prices.columns).fillna(0)


for _bc, _panel in right_panel_data.items():
    _shares = _panel['shares']
    _present_hps = [t for t in heat_pump_techs if t in _shares.columns]
    if not _present_hps:
        continue
    _ambient = pd.Series(0.0, index=_shares.index)
    for _hp in _present_hps:
        _frac = _heat_pump_elec_fraction(_hp).reindex(_shares.index).fillna(0)
        _ambient = _ambient + _shares[_hp] * (1 - _frac)
        _shares[_hp] = _shares[_hp] * _frac
    _shares[AMBIENT] = _ambient
    if AMBIENT not in _panel['non_gas_techs']:
        _panel['non_gas_techs'] = _panel['non_gas_techs'] + [AMBIENT]

heating_merges = {
    'urban central heat': ('urban central air heat pump', 'urban central resistive heater'),
    'urban decentral heat': ('urban decentral air heat pump', 'urban decentral resistive heater'),
    'rural heat': ('rural ground heat pump', 'rural resistive heater'),
}
for _bc, (_hp, _rh) in heating_merges.items():
    _panel = right_panel_data[_bc]
    _shares = _panel['shares']
    _present = [c for c in (_hp, _rh) if c in _shares.columns]
    if _present:
        _shares['electric heating'] = _shares[_present].sum(axis=1)
        _shares.drop(columns=_present, inplace=True)
        _panel['non_gas_techs'] = [t for t in _panel['non_gas_techs'] if t not in (_hp, _rh)] + ['electric heating']

# merge biomass CHP CC into biomass CHP for District Heating
_dh_panel = right_panel_data['urban central heat']
_dh_shares = _dh_panel['shares']
_chp_base, _chp_cc = 'urban central solid biomass CHP', 'urban central solid biomass CHP CC'
if _chp_cc in _dh_shares.columns:
    if _chp_base in _dh_shares.columns:
        _dh_shares[_chp_base] = _dh_shares[_chp_base] + _dh_shares[_chp_cc]
    else:
        _dh_shares[_chp_base] = _dh_shares[_chp_cc]
    _dh_shares.drop(columns=[_chp_cc], inplace=True)
    _dh_panel['non_gas_techs'] = [t for t in _dh_panel['non_gas_techs'] if t != _chp_cc]
    if _chp_base not in _dh_panel['non_gas_techs']:
        _dh_panel['non_gas_techs'].append(_chp_base)

# merge gas boiler + gas CHP into a single 'gas' band for District Heating
_dh_gas_techs = ['urban central gas CHP', 'urban central gas boiler']
_dh_present_gas = [c for c in _dh_gas_techs if c in _dh_shares.columns]
if _dh_present_gas:
    _dh_shares['gas'] = _dh_shares[_dh_present_gas].sum(axis=1)
    _dh_shares.drop(columns=_dh_present_gas, inplace=True)
    _dh_panel['gas_techs'] = ['gas']
    nice_names['gas'] = 'gas'

_ac_panel = right_panel_data['AC']
_ac_shares = _ac_panel['shares']
_ow_present = [c for c in ('Offshore Wind (AC)', 'Offshore Wind (DC)') if c in _ac_shares.columns]
if _ow_present:
    _ac_shares['Offshore Wind'] = _ac_shares[_ow_present].sum(axis=1)
    _ac_shares.drop(columns=_ow_present, inplace=True)
    _ac_panel['non_gas_techs'] = [t for t in _ac_panel['non_gas_techs'] if t not in ('Offshore Wind (AC)', 'Offshore Wind (DC)')] + ['Offshore Wind']

nice_names['heat<100 industry industrial heat pump medium temperature'] = 'electric heating'
nice_names['heat100-200 industry industrial heat pump high temperature'] = 'electric heating'
nice_demand_names['urban decentral heat'] = 'Urban Individual Heating'

# --- Merge Rural + Urban Individual heating into one 'Individual Heating' panel.
# The per-bus percentage shares are combined as an energy-weighted average using
# each bus's total heat supply as the weight (i.e. percentages first, then pooled
# by energy totals). Carrier names that differ only by prefix are unified, and
# the merged panel replaces the two originals directly under District Heating. ---
_indiv_src = ['rural heat', 'urban decentral heat']
_indiv_canon = {
    'rural gas boiler': 'gas',
    'urban decentral gas boiler': 'gas',
    'rural biomass boiler': 'biomass boiler',
    'urban decentral biomass boiler': 'biomass boiler',
    'rural oil boiler': 'oil boiler',
    'urban decentral oil boiler': 'oil boiler',
}
tech_colors['biomass boiler'] = tech_colors['rural biomass boiler']
nice_names['gas'] = 'gas'
nice_names['biomass boiler'] = 'biomass boiler'
nice_names['oil boiler'] = 'oil boiler'

_indiv_weights = {b: right_panel_data[b]['total'].reindex(prices.columns) for b in _indiv_src}
_indiv_wsum = sum(_indiv_weights.values())

_indiv_order = ['gas', 'ambient heat', 'electric heating', 'biomass boiler', 'oil boiler']
_indiv_shares = pd.DataFrame(0.0, index=prices.columns, columns=list(_indiv_order))
for _b in _indiv_src:
    _sdf = right_panel_data[_b]['shares']
    _w = _indiv_weights[_b]
    for _col in _sdf.columns:
        _canon = _indiv_canon.get(_col, _col)
        if _canon not in _indiv_shares.columns:
            _indiv_shares[_canon] = 0.0
        _indiv_shares[_canon] = (
            _indiv_shares[_canon]
            + _sdf[_col].reindex(prices.columns).fillna(0) * _w / _indiv_wsum
        )
_indiv_shares = _indiv_shares.loc[:, (_indiv_shares.abs() > 1e-9).any()]

right_panel_data['individual heat'] = {
    'shares': _indiv_shares,
    'gas_techs': [c for c in ['gas'] if c in _indiv_shares.columns],
    'non_gas_techs': [c for c in _indiv_order if c != 'gas' and c in _indiv_shares.columns],
    'total': _indiv_wsum,
}
nice_demand_names['individual heat'] = 'Individual Heating'

# rebuild in display order: Individual Heating directly under District Heating,
# with the two source panels removed.
_new_order = []
for _k in list(right_panel_data.keys()):
    if _k in _indiv_src or _k == 'individual heat':
        continue
    _new_order.append(_k)
    if _k == 'urban central heat':
        _new_order.append('individual heat')
right_panel_data = {_k: right_panel_data[_k] for _k in _new_order}


# ============================================================
# PLOTTING
# ============================================================

def compute_area_com(y_values, x_values, base_bottom):
    """Area-weighted center of mass of a stacked segment."""
    dxs = np.diff(x_values, prepend=x_values[0])
    segment_areas = y_values * dxs
    total_area = segment_areas.sum()
    if total_area == 0:
        return x_values[len(x_values) // 2], base_bottom[len(x_values) // 2] + y_values[len(x_values) // 2] / 2
    x_com = np.sum(x_values * segment_areas) / total_area
    y_centers = base_bottom + y_values / 2
    y_com = np.sum(y_centers * segment_areas) / total_area
    return x_com, y_com


line_kwargs = {
    'color': 'k',
    'linewidth': 0.5,
    'alpha': 0.8,
}

width = 1
n_right = len(right_panel_data)

fig = plt.figure(figsize=(14, 8.5))
outer = fig.add_gridspec(
    1, 2,
    width_ratios=[1, 1],
    wspace=0.1,
)

# Panel a (cost) extended by 1.4x (1.5 -> 2.1); panel c (allocation) keeps ratio
# 3. Figure height reduced 10 -> 8.5 so the ratio-sum drop (6 -> 5.1) leaves the
# per-unit height unchanged, i.e. panel c keeps its absolute height.
gs_left = outer[0, 0].subgridspec(2, 1, height_ratios=[2.1, 3], hspace=0.12)
ax_cost = fig.add_subplot(gs_left[0])
ax_alloc = fig.add_subplot(gs_left[1])

gs_right = outer[0, 1].subgridspec(n_right, 1, hspace=0.08)
ax_right = [fig.add_subplot(gs_right[i]) for i in range(n_right)]

# --- Left: System Cost ---
cost_color = 'forestgreen'
carbon_color = 'royalblue'

cost_values_with_carbon = cost_values + ets_costs.values

ax_cost.fill_between(cost_x, cost_values, cost_values_with_carbon,
                     color='gold', alpha=0.25, linewidth=0, zorder=-1)
_line_cost, = ax_cost.plot(cost_x, cost_values, color=cost_color, linewidth=4, zorder=0,
                            label='System cost excl. ETS payments')
_line_cost_carbon, = ax_cost.plot(cost_x, cost_values_with_carbon, color=carbon_color, linewidth=4, zorder=0,
                                   label='System cost incl. ETS payments')
ax_cost.set_ylim(cost_y_min, cost_y_max)
ax_cost.set_ylabel('bn€')
_cost_legend = ax_cost.legend(loc='lower left', frameon=True, fontsize=8)
_cost_legend.set_zorder(30)

for i in range(0, len(cost_x) - 2, 2):
    ax_cost.annotate(
        f"{int(cost_values[i])}",
        xy=(cost_x[i], cost_values[i]),
        xytext=(6, 5),
        textcoords="offset points",
        ha='left', va='bottom',
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=cost_color, lw=1, alpha=0.8),
        zorder=20,
    )
    ax_cost.annotate(
        f"{int(cost_values_with_carbon[i])}",
        xy=(cost_x[i], cost_values_with_carbon[i]),
        xytext=(6, 5),
        textcoords="offset points",
        ha='left', va='bottom',
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=carbon_color, lw=1, alpha=0.8),
        zorder=20,
    )
    ax_cost.plot(cost_x[i], cost_values[i], marker='o', color=cost_color,
                 markersize=10, markerfacecolor='white',
                 markeredgecolor=cost_color, markeredgewidth=2, zorder=15)
    ax_cost.plot(cost_x[i], cost_values_with_carbon[i], marker='o', color=carbon_color,
                 markersize=10, markerfacecolor='white',
                 markeredgecolor=carbon_color, markeredgewidth=2, zorder=15)

ax_cost.text(-0.04, 1.13, "a", transform=ax_cost.transAxes, fontsize=16, fontweight="bold", va="top", ha="left")

# --- Left: Sectoral Gas Allocation ---
ax_alloc.axhline(0, **line_kwargs)

for carrier, bottoms_arr, tops_arr, color in area_data:
    ax_alloc.fill_between(alloc_x, bottoms_arr, tops_arr, color=color, alpha=0.8, linewidth=0)

for env_pos, env_neg in envelope_data:
    ax_alloc.plot(alloc_x, env_pos, **line_kwargs)
    ax_alloc.plot(alloc_x, env_neg, **line_kwargs)

ax_alloc.set_xlim(*sector_xlim)
ax_alloc.set_xticks(sector_xticks)
ax_alloc.set_xticklabels(sector_xticklabels, rotation=0)
ax_alloc.set_ylabel('Sectoral Gas Allocation [TWh/a]')

ax_alloc.set_ylim(-500, 4700)
_existing_yticks = [t for t in ax_alloc.get_yticks() if -500 <= t <= 4700]
if -250 not in _existing_yticks:
    _existing_yticks = sorted(_existing_yticks + [-250])
ax_alloc.set_yticks(_existing_yticks)
ax_alloc.set_ylim(-500, 4700)

# Consumption legend (positive sectors); biomethane is the only supply (negative)
# sector and gets its own 'Supply' legend right below.
cons_order = [c for c in sector_items[::-1] if c != 'biomethane']
cons_handles, cons_labels = [], []
for carrier in cons_order:
    if carrier in tech_colors:
        cons_handles.append(plt.Rectangle((0, 0), 1, 1, fc=tech_colors[carrier], alpha=0.8))
        cons_labels.append(carrier)
_LEG_X = 0.015   # shared left anchor (axes fraction) so both legends line up
_leg_cons = ax_alloc.legend(cons_handles, cons_labels, loc='upper left',
                            bbox_to_anchor=(_LEG_X, 0.98), frameon=False, ncol=1,
                            title='Consumption', fontsize=8, title_fontsize=8,
                            alignment='left')
ax_alloc.add_artist(_leg_cons)

# anchor the supply legend just below the consumption legend, same left edge
fig.canvas.draw()
_bb = _leg_cons.get_window_extent().transformed(ax_alloc.transAxes.inverted())
_sup_handle = plt.Rectangle((0, 0), 1, 1, fc=tech_colors['biomethane'], alpha=0.8)
ax_alloc.legend([_sup_handle], ['biomethane'], loc='upper left',
                bbox_to_anchor=(_LEG_X, _bb.y0 - 0.01), frameon=False, ncol=1,
                title='Supply', fontsize=8, title_fontsize=8, alignment='left')

ax_alloc.text(-0.04, 1.02, "b", transform=ax_alloc.transAxes, fontsize=16, fontweight="bold", va="top", ha="left")

for ax in [ax_cost, ax_alloc]:
    ax.set_xlim(*sector_xlim)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(linestyle='--', alpha=0.3, linewidth=0.5, axis='y')
    ax.set_axisbelow(True)

ax_cost.set_xticks(sector_xticks)
ax_cost.tick_params(labelbottom=False)
ax_alloc.set_xlabel('Total fossil gas supply [TWh/a]')

# --- Right column ---
middle_ax_index = n_right // 2

for i, (bc, panel) in enumerate(right_panel_data.items()):
    ax = ax_right[i]
    shares = panel['shares']
    gas_techs = panel['gas_techs']
    non_gas_techs = panel['non_gas_techs']
    x_values = shares.index.values

    cumulative = np.zeros(len(x_values))

    for tech in gas_techs:
        if tech not in shares.columns:
            continue
        y = shares[tech].values
        ax.fill_between(x_values, cumulative, cumulative + y, color=GAS_COLOR, alpha=0.85)
        if y.max() > 8:
            x_com, y_com = compute_area_com(y, x_values, cumulative)

            ax.text(
                x_com,
                y_com,
                nice_names[tech],
                fontsize=8,
                ha='center',
                va='center',
                color='white',
                fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.18', fc=GAS_COLOR, alpha=0.7, ec='none')
            )
        cumulative += y

    for tech in non_gas_techs:
        if tech not in shares.columns:
            continue
        y = shares[tech].values
        color = tech_colors.get(tech, '#999999')
        if color == '#999999':
            print('warning: color for {} is grey'.format(tech))

        ax.fill_between(x_values, cumulative, cumulative + y, color=color, alpha=0.5)
        if y.max() > 8:
            x_com, y_com = compute_area_com(y, x_values, cumulative)

            if tech == 'heat>500 industry hydrogen':
                x_com, y_com = x_values[1], 90

            rgb = mcolors.to_rgb(color)
            brightness = (rgb[0] * 299 + rgb[1] * 587 + rgb[2] * 114) / 1000
            text_color = 'black'
            ax.text(
                x_com,
                y_com,
                nice_names[tech],
                fontsize=8,
                ha='center',
                va='center',
                color=text_color,
                bbox=dict(boxstyle='round,pad=0.18', fc=color, alpha=0.7, ec='none')
            )
        cumulative += y

    if bc == 'heat>500 industry':
        ypos = 0.2
    else:
        ypos = 0.98
    ax.text(-0.01, ypos, nice_demand_names[bc], transform=ax.transAxes, fontsize=9,
            va='top', ha='left',
            bbox=dict(fc='white', alpha=1, ec='grey', lw=0.5))

    if i == 0:
        ax.text(-0.04, 1.3, "c", transform=ax.transAxes, fontsize=16, fontweight="bold", va="top", ha="left")
        ax.text(2000, 103, "Autarky\n(2000 TWh)", color=consumption_threshold_color, ha='center', va='bottom', fontsize=8, fontweight='bold')
        ax.text(2750, 103, "No LNG\n(2750 TWh)", color=consumption_threshold_color, ha='center', va='bottom', fontsize=8, fontweight='bold')
        ax.text(4000, 103, "Today's\nDemand", color='dimgray', ha='center', va='bottom', fontsize=8, fontweight='bold')

    if i == middle_ax_index:
        ax.set_ylabel('Supply Share')

    ax.axvline(2000, color=consumption_threshold_color, linestyle='--', linewidth=1.5, alpha=0.7)
    ax.axvline(2750, color=consumption_threshold_color, linestyle='--', linewidth=1.5, alpha=0.7)
    ax.axvspan(3500, 4500, color='grey', alpha=0.4, zorder=-1)

    ax.set_ylim(0, 100)
    ax.set_xlim(0, prices.columns[-1])
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels(['', '', '', '', ''])

    for spine in ax.spines.values():
        spine.set_edgecolor('grey')
        spine.set_linewidth(0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(axis='y', which='both', left=True, right=True,
                   length=2, width=0.5, color='grey')

    ax.set_xticks(range(0, 4750, 250))
    if i < n_right - 1:
        ax.tick_params(labelbottom=False)

for i, ax in enumerate([ax_cost, ax_alloc]):
    ax.axvline(8.5, color=consumption_threshold_color, linestyle='--', linewidth=1.5, alpha=0.7)
    ax.axvline(11.5, color=consumption_threshold_color, linestyle='--', linewidth=1.5, alpha=0.7)
    ax.axvspan(14.5, 18.5, color='grey', alpha=0.4, zorder=-1)

    if i == 0:
        ylim = ax.get_ylim()
        y_text = ylim[1] + (ylim[1] - ylim[0]) * 0.03

        ax.text(8, y_text, "Autarky\n(2000 TWh)", color=consumption_threshold_color, ha='center', va='bottom', fontsize=8, fontweight='bold')
        ax.text(11, y_text, "No LNG\n(2750 TWh)", color=consumption_threshold_color, ha='center', va='bottom', fontsize=8, fontweight='bold')
        ax.text(16.5, y_text, "Today's\nDemand", color='dimgray', ha='center', va='bottom', fontsize=8, fontweight='bold')


ax_right[-1].set_xticks(range(0, 4750, 250))
ax_right[-1].set_xticklabels(sector_xticklabels)
ax_right[-1].set_xlabel('Total fossil gas supply [TWh/a]')
ax_right[-1].spines['bottom'].set_visible(True)

plt.tight_layout()
out = HERE / "main_results.pdf"
plt.savefig(out, bbox_inches='tight')
SIBLING_IMGS.mkdir(parents=True, exist_ok=True)
shutil.copy(out, SIBLING_IMGS / "main_results.pdf")
print(f"Wrote {out}")

_gas_levels = [int(c) for c in bottoms_pos.index]

_left_payload = {
    "system_cost": {
        "x_gas_TWh": _gas_levels,
        "value_BEUR": [float(v) for v in cost_values],
        "value_incl_ets_BEUR": [float(v) for v in cost_values_with_carbon],
        "ets_payment_BEUR": [float(v) for v in ets_costs.values],
        "y_label": "System Cost [B€]",
    },
    "commodity_prices": {
        "x_gas_TWh": [int(c) for c in commodity_prices.columns],
        "carriers": list(commodity_prices.index),
        "nice_names": commidity_nice_names,
        "colors": commidity_colors,
        "prices_EUR_per_MWh": {
            str(_c): commodity_prices.loc[_c].astype(float).tolist()
            for _c in commodity_prices.index
        },
        "x_at_price_threshold_TWh": {
            str(_pt): float(np.interp(_pt, gas_prices.values[::-1], gas_prices.index[::-1]))
            for _pt in price_thresholds
        },
    },
    "sectoral_gas_allocation": {
        "x_gas_TWh": _gas_levels,
        "sectors": sector_items,
        "values_TWh_negated": {
            _s: gb.loc[_s].astype(float).mul(-1).tolist() for _s in sector_items
        },
        "colors": {_s: tech_colors[_s] for _s in sector_items},
    },
}

_right_payload = {}
for _bc, _data in right_panel_data.items():
    _shares_df = _data["shares"]
    _right_payload[_bc] = {
        "nice_name": nice_demand_names.get(_bc, _bc),
        "x_gas_TWh": [int(c) for c in _shares_df.index],
        "technologies": list(_shares_df.columns),
        "tech_nice_names": {
            t: nice_names.get(t, t) for t in _shares_df.columns
        },
        "shares_pct": {
            t: _shares_df[t].astype(float).tolist() for t in _shares_df.columns
        },
        "colors": {
            t: tech_colors.get(t, "#999999") for t in _shares_df.columns
        },
        "gas_techs": _data["gas_techs"],
    }

export_frontend_data("main_results", {
    "x_label": "Total Net Gas Consumption [TWh/a]",
    "thresholds": {
        "autarky_TWh": 2000,
        "no_lng_TWh": 2750,
    },
    "left_panels": _left_payload,
    "right_panels": _right_payload,
})
