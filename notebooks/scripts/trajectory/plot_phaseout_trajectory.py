import sys
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as mcolors
from matplotlib.colorbar import ColorbarBase
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pypsa
import yaml
import pycountry


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "notebooks" / "scripts"))
from frontend_export import export_frontend_data  # noqa: E402

with open(REPO_ROOT / 'config.basicrun.yaml') as f:
    config = yaml.safe_load(f)
model_countries = config['countries']
tech_colors = config['plotting']['tech_colors']

countries = [
    'DE', 'FR', 'GB', 'IT', 'ES', 'PL', 'SE', 'NO', 'NL', 'FI', 'RO', 'GR',
    'PT', 'CZ', 'BE', 'AT', 'BG', 'DK', 'HU', 'IE'
]

df = pd.read_csv(
    REPO_ROOT / 'data' / 'ELEC-C_20260226-103428.csv',
    encoding='latin-1'
)

# Rename UK to GB in level 0 (country codes)
level_0_values = df.index.get_level_values(0)
renamed_values = level_0_values.map(lambda x: 'United Kingdom' if x == 'United Kingdom of Great Britain and Northern Ireland (the)' else x)

new_index_arrays = [renamed_values]
for i in range(1, df.index.nlevels):
    level_values = df.index.get_level_values(i)
    new_index_arrays.append(level_values)

df.index = pd.MultiIndex.from_arrays(
    new_index_arrays,
    names=df.index.names
)

_country_code_cache = {}

def get_country_code(country_name):
    """Convert country name to 2-letter ISO code using pycountry with caching."""
    if country_name in _country_code_cache:
        return _country_code_cache[country_name]

    if country_name == 'Czechia' or country_name == 'Czech Republic':
        _country_code_cache[country_name] = 'CZ'
        return 'CZ'

    try:
        country = pycountry.countries.search_fuzzy(country_name)[0]
        code = country.alpha_2
        _country_code_cache[country_name] = code
        return code
    except (LookupError, AttributeError):
        _country_code_cache[country_name] = None
        return None

level_0_values = df.index.get_level_values(0)
unique_countries = level_0_values.unique()
country_mapping = {country: get_country_code(country) for country in unique_countries}
mapped_values = level_0_values.map(country_mapping)

mask = mapped_values.notna()
df = df[mask]

new_index_arrays = [mapped_values[mask]]
for i in range(1, df.index.nlevels):
    level_values = df.index.get_level_values(i)
    new_index_arrays.append(level_values)

df.index = pd.MultiIndex.from_arrays(
    new_index_arrays,
    names=df.index.names
)

# Rename UK to GB in level 0 (country codes)
level_0_values = df.index.get_level_values(0)
renamed_values = level_0_values.map(lambda x: 'GB' if x == 'UK' else x)

new_index_arrays = [renamed_values]
for i in range(1, df.index.nlevels):
    level_values = df.index.get_level_values(i)
    new_index_arrays.append(level_values)

df.index = pd.MultiIndex.from_arrays(
    new_index_arrays,
    names=df.index.names
)

idx = pd.IndexSlice

inter = pd.Index(model_countries).intersection(df.index.unique(0))

diff = pd.Index(model_countries).difference(df.index.unique(0))
assert len(diff) <= 1

df = df.loc[df.iloc[:, 0] != '-'].astype(float)

irena_carriers = {
    'solar': 'Solar photovoltaic',
    'onwind': 'Onshore wind energy',
    'offwind': 'Offshore wind energy',
}

pypsa_carriers = {
    'solar': ['solar', 'solar-hsat'],
    'onwind': ['onwind'],
    'offwind': ['offwind-ac', 'offwind-dc', 'offwind-float'],
}

path = REPO_ROOT / 'results' / 'networks'

projected_2030_model_fn = 'base_s_50_lv1.25_3H-T-H-B-I-A-dist1_2030_free_2500.nc'
projected_2035_model_fn = 'base_s_50_lv1.25_3H-T-H-B-I-A-dist1_2030_free_2000.nc'

n_2030 = pypsa.Network(path / projected_2030_model_fn)
n_2035 = pypsa.Network(path / projected_2035_model_fn)

bat = pd.read_csv(
    REPO_ROOT / 'data' / 'eurostat_batteries.csv',
)

eurostat_countries = [
    'AL', 'AT', 'BA', 'BE', 'BG', 'CY', 'CZ', 'DE', 'DK',
    'EE', 'EL', 'ES', 'FI', 'FR', 'GE', 'HR',
    'HU', 'IE', 'IT', 'LI', 'LT', 'LU', 'LV', 'MD', 'ME', 'MK', 'MT',
    'NL', 'NO', 'PL', 'PT', 'RO', 'RS', 'SE', 'SI', 'SK', 'TR', 'XK']

bat = bat.set_index(['TIME_PERIOD', 'geo', 'Unit of measure'])['OBS_VALUE']
bat = bat.loc[idx[:, eurostat_countries, 'Megawatt']].groupby(level=0).sum().mul(1e-3)

# Main sources:
# 1) UK Government – Renewable Energy Planning Database (REPD) monthly extract
# 2) RenewableUK – EnergyPulse blog: "Stacking up the storage: where the UK battery market stands in 2025"
# 3) UK House of Commons Library Research Briefing – "Battery energy storage systems" (CBP-7621)
uk_bat = pd.Series(
    {
        2015: 0.0088,
        2016: 0.0342,
        2017: 0.1572,
        2018: 0.5444,
        2019: 0.7036,
        2020: 1.128,
        2021: 1.591,
        2022: 2.592,
        2023: 4.148,
        2024: 5.492,
    },
    name="uk_operational_battery_capacity_mw",
).sort_index()

bat = bat + uk_bat
bat.index = bat.index.astype(str)

# from https://www.ehpa.org/wp-content/uploads/2025/07/EHPA-Market-Report-2025-executive-summary.pdf, Fig 3
heat_pumps_europe = pd.Series({
    2012: 6.3,
    2013: 7.1,
    2014: 7.9,
    2015: 8.8,
    2016: 9.9,
    2017: 11.0,
    2018: 12.3,
    2019: 13.8,
    2020: 15.5,
    2021: 17.6,
    2022: 20.5,
    2023: 23.4,
    2024: 25.5,
}, name='heat_pumps_europe')
heat_pumps_europe.index = heat_pumps_europe.index.astype(str)

num_households = pd.Series({  # in millions
    'DE': 41.3,
    'FR': 31.7,
    'GB': 28.6,
    'IT': 26.1,
    'ES': 19.4,
    'PL': 14.7,
    'SE': 4.9,
    'NO': 2.65,
    'NL': 8.4,
    'FI': 2.7,
    'RO': 7.1,
    'GR': 4.6,
    'PT': 4.1,
    'CZ': 4.6,
    'BE': 5.2,
    'AT': 4.2,
    'BG': 2.9,
    'DK': 2.9,
    'HU': 4.4,
    'IE': 1.8,
})

# share of heat pumps supply times num households is the total number of heat pumps

def get_hp_shares(n, countries):
    heating_demands = ['rural heat', 'urban decentral heat', 'urban central heat']

    ss = n.statistics.energy_balance(groupby=['carrier', 'bus_carrier', 'bus'])
    ss = ss.loc[ss > 0.]

    hp_shares = pd.Series(np.nan, index=countries)

    for c in countries:
        sss = ss.loc[ss.index.get_level_values(3).str.startswith(c)]
        sss = sss.loc[sss.index.get_level_values(2).isin(heating_demands)]

        total_supply = sss.sum()
        hp_supply = sss.loc[
            sss.index.get_level_values(1).str.contains('heat pump') |
            sss.index.get_level_values(1).str.contains('resistive heater')
        ].sum()

        hp_share = hp_supply / total_supply
        hp_shares.loc[c] = hp_share

    return hp_shares


def get_industry_hp_supplied(n):
    eb = n.statistics.energy_balance(groupby=['carrier', 'bus_carrier'])

    return eb.loc[idx[:,
        [
            'heat<100 industry industrial heat pump medium temperature',
            'heat100-200 industry industrial heat pump high temperature',
        ],
        [
            'heat<100 industry',
            'heat100-200 industry',
        ]
    ]].sum() * 1e-6


def get_industry_solid_biomass_supplied(n):
    eb = n.statistics.energy_balance(groupby=['carrier', 'bus_carrier'])

    return eb.loc[idx[:,
        [
            'heat<100 industry solid biomass',
            'heat100-200 industry solid biomass',
            'heat200-500 industry solid biomass',
        ],
        [
            'heat<100 industry',
            'heat100-200 industry',
            'heat200-500 industry',
        ]
    ]].sum() * 1e-6


irena_ylabels = {
    'solar': 'Solar PV [GW]',
    'onwind': 'Onshore wind [GW]',
    'offwind': 'Offshore wind [GW]',
}

# total system one

years = pd.Series(range(2018, 2025)).astype(str)

# --- Gather all relevant model values for 2030 and 2035 at the top ---

# 1. For each technology/carrier in irena_carriers
carrier_model = {}
for carrier in irena_carriers.keys():
    model_2030 = n_2030.generators.loc[
        n_2030.generators.carrier.isin(pypsa_carriers[carrier]),
        'p_nom_opt'
    ].mul(1e-3).sum()
    model_2035 = n_2035.generators.loc[
        n_2035.generators.carrier.isin(pypsa_carriers[carrier]),
        'p_nom_opt'
    ].mul(1e-3).sum()
    carrier_model[carrier] = {
        2030: model_2030,
        2035: model_2035
    }

# 2. Battery
battery_model = {
    2030: n_2030.links.loc[n_2030.links.carrier.isin(['battery discharger', 'home battery discharger']), 'p_nom_opt'].mul(1e-3).sum(),
    2035: n_2035.links.loc[n_2035.links.carrier.isin(['battery discharger', 'home battery discharger']), 'p_nom_opt'].mul(1e-3).sum()
}

# 3. Heat pumps (buildings)
hp_shares_2030 = get_hp_shares(n_2030, num_households.keys())
hp_shares_2035 = get_hp_shares(n_2035, num_households.keys())
heat_pumps_model = {
    2030: hp_shares_2030.mul(num_households).sum(),
    2035: hp_shares_2035.mul(num_households).sum()
}

# 4. Industry heat pumps
industry_hp_model = {
    2030: get_industry_hp_supplied(n_2030),
    2035: get_industry_hp_supplied(n_2035)
}

# 5. Industry solid biomass
industry_solid_biomass_model = {
    2030: get_industry_solid_biomass_supplied(n_2030),
    2035: get_industry_solid_biomass_supplied(n_2035)
}

rows = []
for tech, model_dict in carrier_model.items():
    for year, val in model_dict.items():
        rows.append({'category': 'carrier', 'name': tech, 'year': year, 'value': val})
for year, val in battery_model.items():
    rows.append({'category': 'battery', 'name': 'battery', 'year': year, 'value': val})
for year, val in heat_pumps_model.items():
    rows.append({'category': 'heat_pumps', 'name': 'heat_pump_buildings', 'year': year, 'value': val})
for year, val in industry_hp_model.items():
    rows.append({'category': 'industry_hp', 'name': 'heat_pump_industry', 'year': year, 'value': val})
for year, val in industry_solid_biomass_model.items():
    rows.append({'category': 'industry_solid_biomass', 'name': 'solid_biomass_industry', 'year': year, 'value': val})

model_projection_df = pd.DataFrame(rows).set_index(['category', 'name', 'year'])

appendage = model_projection_df.loc[idx['carrier', 'onwind', [2030, 2035]]].values + model_projection_df.loc[idx['carrier', 'offwind', [2030, 2035]]].values
appendage = pd.DataFrame(appendage, index=pd.MultiIndex.from_product([['carrier'], ['wind'], [2030, 2035]]), columns=['value'])
model_projection_df = pd.concat([model_projection_df, appendage])
model_projection_df = model_projection_df.loc[~model_projection_df.index.get_level_values(1).isin(['onwind', 'offwind'])]
model_projection_df = pd.concat([
    model_projection_df.loc[idx[:, 'solar', :], :],
    model_projection_df.loc[idx[:, 'wind', :], :],
    model_projection_df.loc[idx[:, ['battery', 'heat_pump_buildings', 'heat_pump_industry', 'solid_biomass_industry'], :], :],
])

bat.index = bat.index.astype(int)
heat_pumps_europe.index = heat_pumps_europe.index.astype(int)

renewables = {}

for carrier in irena_carriers.keys():
    irena_carrier = irena_carriers[carrier]
    ss = df.loc[idx[inter, irena_carrier, :, :, years]].astype(float).groupby(level=4).sum().mul(1e-3)

    renewables[carrier] = ss.iloc[:, 0]

renewables['wind'] = renewables['onwind'] + renewables['offwind']
del renewables['onwind']
del renewables['offwind']


def add_gradient_fill(ax, val_2024, target_2030, target_2035,
                      x_2024=2024, x_deadline=2030, cmap_name='Greens'):
    """
    Fill a region with a colormap encoding the required deployment rate
    (gradient) from (2024, val_2024) to any point (x, y) in the region.
    """
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    x_left, x_right = xlim
    y_bot, y_top = ylim

    nx, ny = 600, 400
    x_arr = np.linspace(x_left, x_right, nx)
    y_arr = np.linspace(y_bot, y_top, ny)
    X, Y = np.meshgrid(x_arr, y_arr)

    dt = X - x_2024
    dy = Y - val_2024
    with np.errstate(divide='ignore', invalid='ignore'):
        G = np.where(dt > 0, dy / dt, np.nan)

    slope_diag = (target_2035 - val_2024) / (x_deadline - x_2024)
    y_diag = val_2024 + slope_diag * (X - x_2024)

    mask = (dt > 0.05) & (Y >= val_2024) & (Y <= y_diag) & (G >= 0)
    G_masked = np.where(mask, G, np.nan)

    grad_slow = 0
    grad_fast = slope_diag
    cmap = plt.get_cmap(cmap_name)
    norm = mcolors.Normalize(vmin=grad_slow, vmax=grad_fast)

    ax.imshow(
        G_masked,
        extent=[x_left, x_right, y_bot, y_top],
        origin='lower',
        aspect='auto',
        cmap=cmap,
        norm=norm,
        alpha=0.5,
        zorder=0,
        interpolation='bilinear',
    )

    x_diag_end = x_2024 + (y_top - val_2024) / slope_diag if slope_diag > 0 else x_right
    x_diag_end = min(x_diag_end, x_right)
    y_diag_end = min(val_2024 + slope_diag * (x_diag_end - x_2024), y_top)
    ax.plot([x_2024, x_diag_end], [val_2024, y_diag_end],
            color='black', lw=1.2, ls='-', alpha=0.6, zorder=1)

    ax.axhline(target_2030, color='magenta', lw=1.5, ls='--', alpha=0.85, zorder=1)
    ax.axhline(target_2035, color='firebrick', lw=1.5, ls='--', alpha=0.85, zorder=1)

    return norm, cmap


def add_colorbar_below(fig, ax, norm, cmap, unit='GW/yr'):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("bottom", size="5%", pad=0.32)
    cb = ColorbarBase(cax, cmap=cmap, norm=norm, orientation='horizontal', alpha=0.5)
    cb.set_label(f'rate of deployment [{unit}]', fontsize=7, labelpad=2)
    cb.ax.tick_params(labelsize=6)
    return cb


ylabel_themes = [
    "installed capacity [GW]",    # Solar PV
    "installed capacity [GW]",   # Onshore Wind
    "power capacity [GW]",           # Battery
    "number installed [million]",        # Heat Pumps
    "supply volume [TWh/a]",    # Industry HPs
    "supply volume [TWh/a]",    # Industry + DH solid biomass
]

fig, axs = plt.subplots(2, 3, figsize=(14, 6.5))

techs = ['Solar PV', 'Wind', 'Batteries', 'Building Electric Heating', 'Gas-Displacing\nIndustry Electric Heat', 'Gas-Displacing\nIndustry Solid Biomass']
units = ['GW/yr', 'GW/yr', 'GW/yr', 'M/yr', 'TWh/yr', 'TWh/yr']
colors_tech = ['#f9d71c', '#235ebc', '#ace37f', '#d35050', '#b8544f', 'k']
target_nolng = model_projection_df.loc[idx[:, :, 2030], 'value'].values.tolist()
target_autarky = model_projection_df.loc[idx[:, :, 2035], 'value'].values.tolist()

years_hist = np.arange(2018, 2025)
np.random.seed(42)
norms_cmaps = []
panel_meta = []

biomass_color = '#baa741'

year_marker_colors = {
    2030: '#dadaeb',
    2035: '#807dba',
    2040: '#3f007d',
    2045: '#0a0014',
}
year_marker_facecolors = {
    fy: mcolors.to_rgba(c, alpha=0.6) for fy, c in year_marker_colors.items()
}

for (i, ax), vals in zip(
    enumerate(axs.flatten()),
    [
        renewables['solar'],
        renewables['wind'],
        bat.loc[2018:],
        heat_pumps_europe.loc[2018:],
        pd.Series(22, index=[2024]),  # add source
        pd.Series(0, index=[2024]),  # add source
    ]
):
    vals.index = vals.index.astype(int)

    if i < 4:
        vals = vals.iloc[1:]

    # Compute 2025 reference value
    if i < 4:
        y_2025 = vals.loc[2024] + (vals.loc[2024] - vals.loc[2023])
    else:
        # Replacement: 2024 point moved one year right
        y_2025 = vals.loc[2024]
        vals = vals.drop(2024)

    y_max = max(target_nolng[i], target_autarky[i]) * 1.3
    ax.set_ylim(0, y_max)
    ax.set_xlim(2018.5, 2036.5)

    norm, cmap = add_gradient_fill(
        ax, y_2025, target_nolng[i], target_autarky[i],
        x_2024=2025, x_deadline=2030,
    )
    norms_cmaps.append((norm, cmap, units[i]))

    # marker at intersection of max-gradient line with upper axhline (year 2030)
    upper_target = max(target_nolng[i], target_autarky[i])
    slope_max = (target_autarky[i] - y_2025) / (2030 - 2025)
    if slope_max > 0:
        x_intersect = 2025 + (upper_target - y_2025) / slope_max
        ax.plot(
            x_intersect, upper_target,
            marker='H', markerfacecolor=year_marker_facecolors[2030],
            markeredgecolor='black', markersize=11, markeredgewidth=1.2,
            linestyle='None', zorder=10, clip_on=False,
        )

    # markers + thin connector lines for 2035, 2040, 2045
    for fy in [2035, 2040, 2045]:
        ax.plot(
            [2025, fy], [y_2025, upper_target],
            color=year_marker_colors[fy], lw=0.8, ls='-', alpha=0.9,
            zorder=2, clip_on=True,
        )
        if fy == 2035:
            ax.plot(
                fy, upper_target,
                marker='H', markerfacecolor=year_marker_facecolors[fy],
                markeredgecolor='black', markersize=11, markeredgewidth=1.2,
                linestyle='None', zorder=10, clip_on=False,
            )

    slope_trend = None
    if i < 5:
        if 2023 in vals.index and 2024 in vals.index:
            y_2023 = vals.loc[2023]
            y_2024 = vals.loc[2024]
            slope_trend = y_2024 - y_2023
            x_extra = np.array([2024, ax.get_xlim()[1]])
            y_extra = y_2024 + slope_trend * (x_extra - 2024)
            extrap_line, = ax.plot(x_extra, y_extra, color='dodgerblue', ls='--', lw=2.2, zorder=3)
            if i == 0:
                ax.text(
                    2027.8, y_2024 + slope_trend * (2027.8 - 2024) + 0.03 * y_max,
                    'current trend',
                    fontsize=9, fontweight='bold', color='dodgerblue',
                    ha='left', va='bottom', rotation=6, zorder=4,
                    bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='dodgerblue', lw=1, alpha=0.75)
                )

    if i < 4:
        ax.plot(
            vals.index, vals,
            color=colors_tech[i], lw=3, marker='o',
            markeredgecolor='black', markerfacecolor=colors_tech[i],
            markersize=8, markeredgewidth=1, zorder=5, clip_on=False
        )
        # connect 2024 to projected 2025 dot
        ax.plot(
            [2024, 2025], [vals.loc[2024], y_2025],
            color=colors_tech[i], lw=3, zorder=5, clip_on=False,
        )

    fc_2025 = biomass_color if i == 5 else colors_tech[i]
    sc_2025 = ax.scatter(
        [2025], [y_2025],
        s=8**2, facecolor=fc_2025, edgecolor='black',
        linewidth=1.2, zorder=6, clip_on=False,
    )
    sc_2025.set_linestyle('--')

    ax.text(
        0.01, 0.98, techs[i],
        transform=ax.transAxes,
        fontsize=10,
        fontweight='bold',
        va='top',
        ha='left',
        bbox=dict(facecolor='white', edgecolor='gray', alpha=0.87)
    )
    ax.set_ylabel(ylabel_themes[i], fontsize=9)

    panel_meta.append({
        'y_2025': y_2025,
        'upper_target': upper_target,
        'slope_max': slope_max,
        'slope_trend': slope_trend,
    })

for ax in axs.flatten():
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.fill_between([2018.5, 2025.5], *ax.get_ylim(),
                    color='lightgrey', alpha=0.25, zorder=0)
    ax.grid(True, axis='y', linestyle='--', alpha=0.4)
    ax.set_axisbelow(True)
    ax.set_xticks([2020, 2025, 2030, 2035])

plt.tight_layout(h_pad=1.8)

for i, ax in enumerate(axs.flatten()):
    norm, cmap, unit = norms_cmaps[i]
    cb = add_colorbar_below(fig, ax, norm, cmap, unit)
    meta = panel_meta[i]
    slope_diag = norm.vmax

    cb.ax.plot(
        1.0, 0.5,
        marker='H', markerfacecolor=year_marker_facecolors[2030],
        markeredgecolor='black', markersize=11, markeredgewidth=1.2,
        linestyle='None',
        transform=cb.ax.transAxes, zorder=10, clip_on=False,
    )

    if slope_diag > 0:
        for fy in [2035, 2040, 2045]:
            g = (meta['upper_target'] - meta['y_2025']) / (fy - 2025)
            x_pos = g / slope_diag
            if 0 <= x_pos <= 1:
                cb.ax.plot(
                    x_pos, 0.5,
                    marker='H', markerfacecolor=year_marker_facecolors[fy],
                    markeredgecolor='black', markersize=11, markeredgewidth=1.2,
                    linestyle='None',
                    transform=cb.ax.transAxes, zorder=10, clip_on=False,
                )

    if meta['slope_trend'] is not None and slope_diag > 0:
        if 0 <= meta['slope_trend'] <= slope_diag:
            cb.ax.plot(
                [meta['slope_trend'], meta['slope_trend']], [-0.5, 1.5],
                color='dodgerblue', ls='--', lw=2.5,
                transform=cb.ax.get_xaxis_transform(),
                zorder=11, clip_on=False,
            )

handles = [
    plt.Line2D([0], [0], color='magenta', lw=1.5, ls='--', label='No LNG (275 bcm)'),
    plt.Line2D([0], [0], color='firebrick', lw=1.5, ls='--', label='Autarky (200 bcm)'),
]
fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 1.08),
           ncol=2, frameon=False, title='Cost Optimal Build-Out for...')

cross_handles = [
    plt.Line2D([0], [0], marker='H',
               markerfacecolor=year_marker_facecolors[fy],
               markeredgecolor='black',
               markersize=11, markeredgewidth=1.2,
               linestyle='None', label=str(fy))
    for fy in [2030, 2035, 2040, 2045]
]
axs[0, 0].legend(
    handles=cross_handles, loc='lower right',
    ncol=2, fontsize=8, frameon=True, framealpha=0.9,
    handletextpad=0.4, columnspacing=0.8,
    title='Autarky by...', title_fontsize=8,
)

necp_points = {(0, 0): 709, (0, 1): 514}
for (_r, _c), _y in necp_points.items():
    axs[_r, _c].plot(
        2030, _y,
        marker='D', color='#f4a261', alpha=0.8,
        markersize=10, markeredgecolor='black', markeredgewidth=0.5,
        linestyle='None', zorder=11, clip_on=False,
    )

extrap_handle = axs[0, 1].scatter(
    [], [], s=8**2, facecolor='lightgrey', edgecolor='black', linewidth=1.2,
)
extrap_handle.set_linestyle('--')
necp_handle = plt.Line2D(
    [0], [0], marker='D', color='#f4a261', alpha=0.8,
    markersize=7, markeredgecolor='black', markeredgewidth=0.5,
    linestyle='None', label='Aggregate 2030 NECPs',
)
axs[0, 1].legend(
    [extrap_handle, necp_handle], ['extrapolation', 'Aggregate 2030 NECPs'],
    loc='lower right',
    fontsize=8, frameon=True, framealpha=0.9,
)

fr_points = {(1, 1): 80, (1, 2): 40.5}
for (_r, _c), _y in fr_points.items():
    axs[_r, _c].plot(
        2030, _y,
        marker='D', color='#2a9d8f', alpha=0.8,
        markersize=10, markeredgecolor='black', markeredgewidth=0.5,
        linestyle='None', zorder=11, clip_on=False,
    )

_fr_label = 'NECP Increment for\n2030 FR, ES, LU, DK'
fr_handle = plt.Line2D(
    [0], [0], marker='D', color='#2a9d8f', alpha=0.8,
    markersize=7, markeredgecolor='black', markeredgewidth=0.5,
    linestyle='None', label=_fr_label,
)
for _ax in (axs[1, 1], axs[1, 2]):
    _ax.legend(
        [fr_handle], [_fr_label],
        loc='center left',
        fontsize=8, frameon=True, framealpha=0.9,
    )

for _ax, _lbl in zip(axs.flatten(), ['a', 'b', 'c', 'd', 'e', 'f']):
    _ax.text(
        1.05, 1.03, _lbl,
        transform=_ax.transAxes,
        fontsize=16, fontweight='bold',
        va='top', ha='right', zorder=15,
    )

plt.savefig(REPO_ROOT / 'phaseout_trajectory_script.pdf', bbox_inches='tight')
plt.savefig(REPO_ROOT.parent / 'gas_resilience' / 'imgs' / 'phaseout_trajectory_script.pdf', bbox_inches='tight')

# === Second version: 2x2, no batteries, industry electrification + biomass merged ===

# Panel order for v2: solar, wind, building heat, merged industry (electric + biomass).
# Batteries (old panel c) are dropped; old panels e (industry electric heat) and f
# (industry solid biomass) are merged into a single bottom-right panel by summing
# their targets, NECP increments and 2025 values.
# v2-only marker colours: widen the 2030-vs-2035 contrast (v1 is already saved above)
year_marker_colors = dict(year_marker_colors)
year_marker_colors[2035] = '#54278f'
year_marker_facecolors = {
    fy: mcolors.to_rgba(c, alpha=0.6) for fy, c in year_marker_colors.items()
}

v2_techs = [
    'Solar PV',
    'Wind',
    'Building Electric Heating',
    'Gas-Displacing\nIndustry Electrification\nand Solid Biomass',
]
v2_units = ['GW/yr', 'GW/yr', 'M/yr', 'TWh/yr']
v2_colors = ['#f9d71c', '#235ebc', '#d35050', '#b8544f']
v2_ylabels = [
    'installed capacity [GW]',
    'installed capacity [GW]',
    'households [million]',
    'supply volume [TWh/a]',
]
v2_vals = [
    renewables['solar'],
    renewables['wind'],
    heat_pumps_europe.loc[2018:],
    pd.Series(22, index=[2024]),  # industry electric heat (22) + solid biomass (0)
]
v2_target_nolng = [
    target_nolng[0], target_nolng[1], target_nolng[3],
    target_nolng[4] + target_nolng[5],
]
v2_target_autarky = [
    target_autarky[0], target_autarky[1], target_autarky[3],
    target_autarky[4] + target_autarky[5],
]
v2_has_history = [True, True, True, False]
v2_labels = ['a', 'b', 'c', 'd']
# place the 2030 rate label below its hexagon (default above); wind needs it below
# so it does not sit on top of the NECP marker at 514.
v2_label_2030_below = [False, True, False, False]


def _format_rate_label(rate, slope_trend, unit):
    if slope_trend is not None and slope_trend > 0:
        pct = (rate - slope_trend) / slope_trend * 100
        sign = '+' if pct >= 0 else ''
        return f'{rate:.1f} {unit}\n{sign}{pct:.0f}%'
    return f'{rate:.1f} {unit}'


fig2, axs2 = plt.subplots(2, 2, figsize=(10, 6.5))

for (i, ax), vals in zip(enumerate(axs2.flatten()), v2_vals):
    vals = vals.copy()
    vals.index = vals.index.astype(int)
    has_history = v2_has_history[i]

    if has_history:
        vals = vals.iloc[1:]
        y_2025 = vals.loc[2024] + (vals.loc[2024] - vals.loc[2023])
    else:
        y_2025 = vals.loc[2024]
        vals = vals.drop(2024)

    tn = v2_target_nolng[i]
    ta = v2_target_autarky[i]
    y_max = max(tn, ta) * 1.3
    ax.set_ylim(0, y_max)
    ax.set_xlim(2018.5, 2036.5)

    upper_target = max(tn, ta)
    slope_max = (upper_target - y_2025) / (2030 - 2025)

    x_left, x_right = ax.get_xlim()
    if slope_max > 0:
        x_diag_end = min(2025 + (y_max - y_2025) / slope_max, x_right)
        y_diag_end = min(y_2025 + slope_max * (x_diag_end - 2025), y_max)
        ax.plot([2025, x_diag_end], [y_2025, y_diag_end],
                color='black', lw=1.2, ls='-', alpha=0.6, zorder=1)

    ax.axhline(tn, color='magenta', lw=1.5, ls='--', alpha=0.85, zorder=1)
    ax.axhline(ta, color='firebrick', lw=1.5, ls='--', alpha=0.85, zorder=1)

    x_intersect_2030 = None
    if slope_max > 0:
        x_intersect_2030 = 2030
        ax.plot(
            x_intersect_2030, upper_target,
            marker='H', markerfacecolor=year_marker_facecolors[2030],
            markeredgecolor='black', markersize=11, markeredgewidth=1.2,
            linestyle='None', zorder=10, clip_on=False,
        )

    ax.plot(
        [2025, 2035], [y_2025, upper_target],
        color=year_marker_colors[2035], lw=0.8, ls='-', alpha=0.9,
        zorder=2, clip_on=True,
    )
    ax.plot(
        2035, upper_target,
        marker='H', markerfacecolor=year_marker_facecolors[2035],
        markeredgecolor='black', markersize=11, markeredgewidth=1.2,
        linestyle='None', zorder=10, clip_on=False,
    )

    slope_trend = None
    if has_history and 2023 in vals.index and 2024 in vals.index:
        y_2023 = vals.loc[2023]
        y_2024 = vals.loc[2024]
        slope_trend = y_2024 - y_2023
        x_extra = np.array([2024, x_right])
        y_extra = y_2024 + slope_trend * (x_extra - 2024)
        ax.plot(x_extra, y_extra, color='dodgerblue', ls='--', lw=2.2, zorder=3)

    unit = v2_units[i]

    if x_intersect_2030 is not None:
        label_2030 = _format_rate_label(slope_max, slope_trend, unit)
        _off_2030, _va_2030 = ((0, -11), 'top') if v2_label_2030_below[i] else ((0, 11), 'bottom')
        ax.annotate(
            label_2030,
            xy=(x_intersect_2030, upper_target),
            xytext=_off_2030, textcoords='offset points',
            fontsize=8, fontweight='bold',
            ha='center', va=_va_2030, multialignment='center',
            bbox=dict(boxstyle='round,pad=0.22', fc='white', ec=year_marker_colors[2030], lw=0.9, alpha=0.95),
            zorder=12,
        )

    rate_2035 = (upper_target - y_2025) / (2035 - 2025)
    label_2035 = _format_rate_label(rate_2035, slope_trend, unit)
    ax.annotate(
        label_2035,
        xy=(2035, upper_target),
        xytext=(0, 11), textcoords='offset points',
        fontsize=8, fontweight='bold',
        ha='center', va='bottom', multialignment='center',
        bbox=dict(boxstyle='round,pad=0.22', fc='white', ec=year_marker_colors[2035], lw=0.9, alpha=0.95),
        zorder=12,
    )

    if has_history:
        ax.plot(
            vals.index, vals,
            color=v2_colors[i], lw=3, marker='o',
            markeredgecolor='black', markerfacecolor=v2_colors[i],
            markersize=8, markeredgewidth=1, zorder=5, clip_on=False
        )
        ax.plot(
            [2024, 2025], [vals.loc[2024], y_2025],
            color=v2_colors[i], lw=3, zorder=5, clip_on=False,
        )

    sc_2025 = ax.scatter(
        [2025], [y_2025],
        s=8**2, facecolor=v2_colors[i], edgecolor='black',
        linewidth=1.2, zorder=6, clip_on=False,
    )
    sc_2025.set_linestyle('--')

    ax.text(
        0.01, 0.98, v2_techs[i],
        transform=ax.transAxes,
        fontsize=10,
        fontweight='bold',
        va='top',
        ha='left',
        bbox=dict(facecolor='white', edgecolor='gray', alpha=0.87)
    )
    ax.set_ylabel(v2_ylabels[i], fontsize=9)

for ax in axs2.flatten():
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.fill_between([2018.5, 2025.5], *ax.get_ylim(),
                    color='lightgrey', alpha=0.25, zorder=0)
    ax.grid(True, axis='y', linestyle='--', alpha=0.4)
    ax.set_axisbelow(True)
    ax.set_xticks([2020, 2025, 2030, 2035])

plt.tight_layout(h_pad=1.8)

_leg_build = fig2.legend(
    handles=handles, loc='lower center', bbox_to_anchor=(0.35, 1.0),
    ncol=1, frameon=False, title='Cost Optimal Build-Out for...',
    fontsize=8, title_fontsize=9,
)

cross_handles2 = [
    plt.Line2D([0], [0], marker='H',
               markerfacecolor=year_marker_facecolors[fy],
               markeredgecolor='black',
               markersize=11, markeredgewidth=1.2,
               linestyle='None', label=str(fy))
    for fy in [2030, 2035]
]
_leg_autarky = fig2.legend(
    handles=cross_handles2, loc='lower center', bbox_to_anchor=(0.50, 1.0),
    ncol=2, frameon=False, title='Autarky by...',
    fontsize=8, title_fontsize=9,
    handletextpad=0.4, columnspacing=0.8,
)

_solar_upper = max(target_nolng[0], target_autarky[0])
_ax_a = axs2[0, 0]
_trans_rate_anchor = mpl.transforms.offset_copy(
    _ax_a.transData, fig=fig2, x=0, y=18, units='points'
)
_trans_pct_anchor = mpl.transforms.offset_copy(
    _ax_a.transData, fig=fig2, x=0, y=15, units='points'
)
_ax_a.annotate(
    'Needed rate of\ndeployment',
    xy=(2028, _solar_upper), xycoords=_trans_rate_anchor,
    xytext=(0.19, 0.62), textcoords=_ax_a.transAxes,
    fontsize=8, fontweight='bold',
    ha='center', va='center',
    arrowprops=dict(arrowstyle='->', color='gray', lw=1,
                    connectionstyle='arc3,rad=-0.18'),
    zorder=13, annotation_clip=False,
)
_ax_a.annotate(
    'Percentage\nchange',
    xy=(2029, _solar_upper), xycoords=_trans_pct_anchor,
    xytext=(0.19, 0.32), textcoords=_ax_a.transAxes,
    fontsize=8, fontweight='bold',
    ha='center', va='center',
    arrowprops=dict(arrowstyle='->', color='gray', lw=1,
                    connectionstyle='arc3,rad=-0.18'),
    zorder=13, annotation_clip=False,
)

for (_r, _c), _y in necp_points.items():
    axs2[_r, _c].plot(
        2030, _y,
        marker='D', color='#f4a261', alpha=0.8,
        markersize=10, markeredgecolor='black', markeredgewidth=0.5,
        linestyle='None', zorder=11, clip_on=False,
    )

extrap_handle2 = axs2[0, 0].scatter(
    [], [], s=8**2, facecolor='lightgrey', edgecolor='black', linewidth=1.2,
)
extrap_handle2.set_linestyle('--')
current_trend_handle = plt.Line2D(
    [0], [0], color='dodgerblue', ls='--', lw=2.2, label='current trend',
)
_leg_trend = fig2.legend(
    [current_trend_handle, extrap_handle2, necp_handle],
    ['current trend', 'extrapolation', 'Aggregate 2030 NECPs'],
    loc='lower center', bbox_to_anchor=(0.66, 1.0),
    ncol=1, frameon=False, fontsize=8,
)

import matplotlib.transforms as _mtransforms
from matplotlib.patches import FancyBboxPatch as _FancyBboxPatch

fig2.canvas.draw()
_renderer = fig2.canvas.get_renderer()
_inv = fig2.transFigure.inverted()
_union = _mtransforms.Bbox.union([
    leg.get_window_extent(_renderer).transformed(_inv)
    for leg in (_leg_build, _leg_autarky, _leg_trend)
])
_pad_x, _pad_y = 0.012, 0.008
_frame = _FancyBboxPatch(
    (_union.x0 - _pad_x, _union.y0 - _pad_y),
    _union.width + 2 * _pad_x, _union.height + 2 * _pad_y,
    boxstyle='round,pad=0.002',
    transform=fig2.transFigure,
    fc='white', ec='0.7', lw=0.8, zorder=1,
)
fig2.add_artist(_frame)

# merged industry panel: NECP increment = electric heat (80) + solid biomass (40.5)
fr_points_v2 = {(1, 1): 80 + 40.5}
for (_r, _c), _y in fr_points_v2.items():
    axs2[_r, _c].plot(
        2030, _y,
        marker='D', color='#2a9d8f', alpha=0.8,
        markersize=10, markeredgecolor='black', markeredgewidth=0.5,
        linestyle='None', zorder=11, clip_on=False,
    )

axs2[1, 1].legend(
    [fr_handle], [_fr_label],
    loc='center left',
    fontsize=8, frameon=True, framealpha=0.9,
)

for _ax, _lbl in zip(axs2.flatten(), v2_labels):
    _ax.text(
        -0.05, -0.05, _lbl,
        transform=_ax.transAxes,
        fontsize=16, fontweight='bold',
        va='top', ha='right', zorder=15,
    )

plt.savefig(REPO_ROOT / 'phaseout_trajectory_script_v2.pdf', bbox_inches='tight')
plt.savefig(REPO_ROOT.parent / 'gas_resilience' / 'imgs' / 'phaseout_trajectory_script_v2.pdf', bbox_inches='tight')

_panel_series = [
    ("solar_pv",          renewables['solar']),
    ("wind",              renewables['wind']),
    ("batteries",         bat.loc[2018:]),
    ("building_heat",     heat_pumps_europe.loc[2018:]),
    ("industry_heat",     pd.Series(22.0, index=[2024])),
    ("industry_biomass",  pd.Series(0.0,  index=[2024])),
]

_panels_payload = {}
for _i, (_key, _vals) in enumerate(_panel_series):
    _vals = _vals.copy()
    _vals.index = _vals.index.astype(int)
    _panels_payload[_key] = {
        "label": techs[_i],
        "y_label": ylabel_themes[_i],
        "deployment_unit": units[_i],
        "color": colors_tech[_i],
        "history": {
            "year": [int(y) for y in _vals.index],
            "value": [float(v) for v in _vals.values],
        },
        "target_no_lng_2030": float(target_nolng[_i]),
        "target_autarky_2035": float(target_autarky[_i]),
    }

export_frontend_data("phaseout_trajectory", {
    "panels": _panels_payload,
    "scenario_targets": {
        "no_lng": {"year": 2030, "gas_TWh": 2750},
        "autarky": {"year": 2035, "gas_TWh": 2000},
    },
    "model_projection": (
        model_projection_df.reset_index()
        .rename(columns={0: "value"})
        .to_dict(orient="records")
    ),
})
