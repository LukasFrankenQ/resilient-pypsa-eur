"""Reproduce the capex / opex / LCOH-per-technology figure as a standalone script.

Converted from notebooks/plot_capex_opex_lcoe.ipynb.

For each demand group (electricity 'AC', the residential/commercial heat
categories and the industrial process-heat temperature bands) the figure shows,
in three stacked panels:

    1. Overnight investment cost   [EUR/kW]   (de-annuitised capital cost)
    2. VOM                         [EUR/MWh]  (marginal cost + fuel cost)
    3. Levelized cost of energy    [EUR/MWh]  (LCOH / LCOE)

The per-bus spread of each technology is drawn as a violin + density-jittered
scatter.

Output: heat_capex_opex_lcohs.pdf (written to this folder and to the paper imgs).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import pypsa
import yaml
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = Path(__file__).resolve().parent
NETWORK_DIR = REPO_ROOT / "results" / "networks"
COSTS_DIR = REPO_ROOT.parent / "technology-data" / "outputs"
SIBLING_IMGS = REPO_ROOT.parent / "gas_resilience" / "imgs"

NETWORK_FILE = (
    sys.argv[1] if len(sys.argv) > 1 else "base_s_50_lv1.25_3H-T-H-B-I-A-dist1_2030_free_4000.nc"
)

idx = pd.IndexSlice
np.random.seed(0)  # reproducible density jitter


# ── Tech colors ──────────────────────────────────────────────────────────────
with open(REPO_ROOT / "config" / "plotting.default.yaml") as f:
    tech_colors = yaml.safe_load(f)["plotting"]["tech_colors"]
tech_colors["rural ground heat pump"] = tech_colors["ground heat pump"]
tech_colors["rural biomass boiler"] = tech_colors["biomass boiler"]
tech_colors["urban central air heat pump"] = tech_colors["air heat pump"]
tech_colors["urban central gas boiler"] = tech_colors["gas boiler"]
tech_colors["urban central gas CHP"] = tech_colors["gas CHP"]
tech_colors["urban central solid biomass CHP"] = tech_colors["biomass CHP"]
tech_colors["urban decentral air heat pump"] = tech_colors["air heat pump"]
tech_colors["urban decentral gas boiler"] = tech_colors["gas boiler"]
tech_colors["urban decentral biomass boiler"] = tech_colors["biomass boiler"]


# ── Techno-economic cost assumptions ─────────────────────────────────────────
with open(REPO_ROOT / "config.basicrun.yaml") as f:
    cost_year = yaml.safe_load(f)["costs"]["year"]
costs = pd.read_csv(COSTS_DIR / f"costs_{cost_year}.csv", index_col=[0, 1])

lifetimes = costs.loc[idx[:, "lifetime"], "value"]
lifetimes.index = lifetimes.index.droplevel(1)

discount_rates = costs.loc[idx[:, "discount rate"], "value"]
discount_rates.index = discount_rates.index.droplevel(1)


def get_annuity_factor(lifetime, discount_rate):
    if isinstance(discount_rate, (int, float)) and discount_rate == 0:
        return 1 / lifetime
    return discount_rate / (1 - (1 + discount_rate) ** (-lifetime))


# Map our display carriers onto the names used in the technology-data costs file.
tech_costs_mapper = {
    "rural gas boiler": "decentral gas boiler",
    "urban decentral air heat pump": "decentral air-sourced heat pump",
    "urban decentral gas boiler": "decentral gas boiler",
    "urban decentral biomass boiler": "biomass boiler",
    "urban central solid biomass CHP": "central solid biomass CHP",
    "heat<100 industry gas": "gas boiler steam",
    "heat<100 industry electric boiler steam": "electric boiler steam",
    "heat<100 industry solid biomass": "solid biomass boiler steam",
    "heat100-200 industry solid biomass": "solid biomass boiler steam",
    "heat200-500 industry solid biomass": "solid biomass boiler steam",
    "heat200-500 industry gas": "gas boiler steam",
    "heat>500 industry gas": "gas boiler steam",
    "rural ground heat pump": "decentral ground-sourced heat pump",
    "Combined-Cycle Gas": "CCGT",
    "Open-Cycle Gas": "OCGT",
    "rural biomass boiler": "biomass boiler",
    "urban central biomass CHP": "biomass CHP",
    "urban central air heat pump": "central air-sourced heat pump",
    "urban central gas CHP": "central gas CHP",
    "urban central gas boiler": "central gas boiler",
    "heat<100 industry industrial heat pump medium temperature": "industrial heat pump medium temperature",
    "heat100-200 industry industrial heat pump high temperature": "industrial heat pump high temperature",
    "heat100-200 industry electric boiler steam": "electric boiler steam",
    "heat100-200 industry gas": "gas boiler steam",
    "heat100-200 industry solid biomass": "solid biomass boiler steam",
    "Onshore Wind": "onwind",
    "Offshore Wind (AC)": "offwind",
    "Solar": "solar",
    "Reservoir & Dam": "ror",
    "nuclear": "nuclear",
}
# Map display carriers onto the carrier strings actually present in the network.
carrier_network_names = {
    "Combined-Cycle Gas": "CCGT",
    "Open-Cycle Gas": "OCGT",
    "Onshore Wind": "onwind",
    "Offshore Wind (AC)": "offwind-ac",
    "Solar": "solar",
    "Reservoir & Dam": "ror",
    "nuclear": "nuclear",
}


# ── Demand groups and the technologies shown for each ─────────────────────────
shown_techs = {
    "rural heat": ["rural gas boiler", "rural ground heat pump", "rural biomass boiler"],
    "urban decentral heat": [
        "urban decentral gas boiler",
        "urban decentral air heat pump",
        "urban decentral biomass boiler",
    ],
    "urban central heat": [
        "urban central gas boiler",
        "urban central air heat pump",
        "urban central solid biomass CHP",
    ],
    "heat<100 industry": [
        "heat<100 industry solid biomass",
        "heat<100 industry gas",
        "heat<100 industry industrial heat pump medium temperature",
        "heat<100 industry electric boiler steam",
    ],
    "heat100-200 industry": [
        "heat100-200 industry solid biomass",
        "heat100-200 industry gas",
        "heat100-200 industry industrial heat pump high temperature",
        "heat100-200 industry electric boiler steam",
    ],
    "heat200-500 industry": [
        "heat200-500 industry solid biomass",
        "heat200-500 industry gas",
        "heat200-500 industry hydrogen",
    ],
    "heat>500 industry": [
        "heat>500 industry gas",
        "heat>500 industry hydrogen",
    ],
}
fuels = {
    "Combined-Cycle Gas": "gas",
    "Open-Cycle Gas": "gas",
    "rural gas boiler": "gas",
    "rural biomass boiler": "solid biomass",
    "rural ground heat pump": "AC",
    "urban decentral gas boiler": "gas",
    "urban decentral air heat pump": "AC",
    "urban decentral biomass boiler": "solid biomass",
    "urban central gas boiler": "gas",
    "urban central air heat pump": "AC",
    "urban central gas CHP": "gas",
    "urban central solid biomass CHP": "solid biomass",
    "heat<100 industry solid biomass": "solid biomass",
    "heat<100 industry gas": "gas",
    "heat<100 industry industrial heat pump medium temperature": "AC",
    "heat<100 industry electric boiler steam": "AC",
    "heat100-200 industry solid biomass": "solid biomass",
    "heat100-200 industry gas": "gas",
    "heat100-200 industry industrial heat pump high temperature": "AC",
    "heat100-200 industry electric boiler steam": "AC",
    "heat200-500 industry solid biomass": "solid biomass",
    "heat200-500 industry gas": "gas",
    "heat200-500 industry hydrogen": "H2",
    "heat>500 industry gas": "gas",
    "heat>500 industry hydrogen": "H2",
}
# Gas uses the scarcity/glut-adjusted gas-*bus* marginal price (model result),
# not the flat exogenous commodity cost. Biomass stays at its commodity price.
fixed_carriers = ["solid biomass"]
endogenous_carriers = ["AC", "H2", "gas"]

# Tech tick labels, stripped of the demand category (shown as a section header).
nice_names = {
    "rural gas boiler": "Gas Boiler",
    "rural ground heat pump": "Ground HP",
    "rural biomass boiler": "Biomass Boiler",
    "urban decentral gas boiler": "Gas Boiler",
    "urban decentral air heat pump": "Air HP",
    "urban decentral biomass boiler": "Biomass Boiler",
    "urban central gas boiler": "Gas Boiler",
    "urban central air heat pump": "Air HP",
    "urban central solid biomass CHP": "Biomass CHP",
    "heat<100 industry solid biomass": "Biomass Boiler",
    "heat<100 industry gas": "Gas Boiler",
    "heat<100 industry industrial heat pump medium temperature": "HP (med temp)",
    "heat<100 industry electric boiler steam": "Electric Boiler",
    "heat100-200 industry solid biomass": "Biomass Boiler",
    "heat100-200 industry gas": "Gas Boiler",
    "heat100-200 industry industrial heat pump high temperature": "HP (high temp)",
    "heat100-200 industry electric boiler steam": "Electric Boiler",
    "heat200-500 industry solid biomass": "Biomass Boiler",
    "heat200-500 industry gas": "Gas (direct firing)",
    "heat200-500 industry hydrogen": "H2 (direct firing)",
    "heat>500 industry gas": "Gas (direct firing)",
    "heat>500 industry hydrogen": "H2 (direct firing)",
}

# Centered section headers naming the heat demand each group of techs supplies.
group_names = {
    "rural heat": "Rural\nheat",
    "urban decentral heat": "Urban\nindividual heat",
    "urban central heat": "Urban\ncentral heat",
    "heat<100 industry": "Industry\n<100°C",
    "heat100-200 industry": "Industry\n100–200°C",
    "heat200-500 industry": "Industry\n200–500°C",
    "heat>500 industry": "Industry\n>500°C",
}


# ── Helpers ──────────────────────────────────────────────────────────────────
def get_buses(buses):
    buses = pd.Index(buses)
    return buses.str.split().str[:2].str.join(" ")


def get_suffix(buses):
    buses = pd.Index(buses)
    return buses.str.split().str[2:].str.join(" ")[0]


def get_marginal_cost(n, components):
    if not isinstance(components, pd.Index):
        components = pd.Index([components])
    if len(n.links.index.intersection(components)) > 0:
        return n.links.marginal_cost.loc[components]
    elif len(n.generators.index.intersection(components)) > 0:
        return n.generators.marginal_cost.loc[components]
    else:
        raise ValueError(f"No marginal cost found for {components}")


def get_fuel_cost(n, buses, carrier):
    if carrier in fixed_carriers:
        return n.generators.loc[n.generators.carrier == carrier, "marginal_cost"].mean()

    elif carrier in endogenous_carriers:
        suffix = get_suffix(n.buses.index[n.buses.carrier == carrier])

        load = n.statistics.withdrawal(
            groupby="bus",
            aggregate_time=False,
            bus_carrier=carrier,
            aggregate_across_components=True,
        ).T[buses + " " * bool(len(suffix)) + suffix]

        if not load.empty and load.sum().sum() > 0:
            price = n.buses_t.marginal_price.loc[:, load.columns]
            price = price.reindex(columns=load.columns, fill_value=1)

            weights = n.snapshot_weightings.generators
            a = weights @ (load * price)
            b = weights @ load
            fuel_cost = a / b
            fuel_cost.index = buses
            return fuel_cost


def get_density_based_jitter(values, max_jitter=0.35):
    """Jitter points by their local probability density (denser -> wider)."""
    if len(values) < 2:
        return np.zeros_like(values)
    kde = stats.gaussian_kde(values)
    densities = kde(values)
    densities_norm = (densities - densities.min()) / (densities.max() - densities.min() + 1e-10)
    jitter_ranges = densities_norm * max_jitter
    return np.array([np.random.uniform(-jr, jr) for jr in jitter_ranges])


# ── Load network and per-bus statistics ──────────────────────────────────────
n = pypsa.Network(NETWORK_DIR / NETWORK_FILE)

nodal_caps = n.statistics.optimal_capacity(groupby=["bus", "carrier"])
nodal_supply = n.statistics.supply(groupby=["bus", "carrier"])
nodal_withdrawal = n.statistics.withdrawal(groupby=["bus", "carrier"])

all_locations = n.buses.index[n.buses.carrier == "AC"]

# Hydrogen price source. In the main run H2 demand is met almost entirely by SMR
# (cheap fossil H2 ~44 EUR/MWh). To show *electrolytic* H2 in the hydrogen techs'
# VOM/LCOH, take the per-bus H2 marginal price from the wiggle-0 network, where
# electrolysis sets the H2 price (~140 EUR/MWh).
ELECTROLYTIC_H2_NETWORK = "base_s_50_lv1.25_3H-T-H-B-I-A-dist1_2030_free_0.nc"


def h2_price_by_location(net):
    """Dispatch-weighted H2 marginal price per location (EUR/MWh)."""
    load = net.statistics.withdrawal(
        groupby="bus", aggregate_time=False, bus_carrier="H2",
        aggregate_across_components=True,
    ).T
    wts = net.snapshot_weightings.generators
    price = net.buses_t.marginal_price[load.columns]
    s = (wts @ (load * price)) / (wts @ load)
    s.index = get_buses(s.index)
    return s


electrolytic_h2_price = h2_price_by_location(
    pypsa.Network(NETWORK_DIR / ELECTROLYTIC_H2_NETWORK)
)

# Carbon price applied to the gas techs as a lift on top of their VOM/LCOH.
CO2_PRICE = 100.0  # EUR/tCO2


def rated_heat_efficiency(net, carrier_n):
    """Static (rated) output/input efficiency to the heat bus of a link carrier."""
    links = net.links[net.links.carrier == carrier_n]
    if not len(links):
        return 1.0
    row = links.iloc[0]
    for i, ecol in [(1, "efficiency"), (2, "efficiency2"), (3, "efficiency3")]:
        b = row.get(f"bus{i}", "")
        if isinstance(b, str) and b in net.buses.index and "heat" in str(net.buses.carrier[b]):
            return links[ecol].mean()
    return links.efficiency.mean()


def gas_co2_per_input(net, carrier_n):
    """tCO2 emitted to the atmosphere per MWh of gas input, for a link carrier."""
    links = net.links[net.links.carrier == carrier_n]
    if not len(links):
        return 0.0
    row = links.iloc[0]
    for i in range(2, 6):
        bcol, ecol = f"bus{i}", f"efficiency{i}"
        if bcol in net.links.columns and ecol in net.links.columns:
            b = row[bcol]
            if isinstance(b, str) and b in net.buses.index and net.buses.carrier[b] == "co2":
                return links[ecol].mean()
    return 0.0


# ── Build the figure ─────────────────────────────────────────────────────────
scatter_kwargs = {"s": 30, "edgecolor": "black", "linewidth": 0.5, "alpha": 0.7}

fig, axs = plt.subplots(
    3, 1, figsize=(12, 11), gridspec_kw={"height_ratios": [2, 1, 0.7]}
)
axs[0].set_ylabel("Overnight Investment Cost\n[EUR/kW$_{th}$ output]")
axs[1].set_ylabel("VOM + fuel cost\n[EUR/MWh$_{th}$]")
axs[2].set_ylabel("Weighted-avg\nefficiency / COP [-]")

xticklocs, xticklabels = [], []
group_spans = []  # (display name, first xloc, last xloc) per demand section
inter_load_gap = 1
xloc_tracker = 0


def draw_violin_scatter(ax, values, xloc, color, violin=True, edgecolor=None):
    """Violin + density-jittered scatter for an iterable; plain scatter otherwise."""
    kw = dict(scatter_kwargs)
    if edgecolor is not None:
        kw["edgecolor"], kw["linewidth"] = edgecolor, 1.0
    if isinstance(values, Iterable):
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]
        if len(values) == 0:
            return
        if not violin or len(values) < 2 or np.ptp(values) == 0:
            # no violin requested, or zero-variance data (which breaks the KDE):
            # draw bare points instead.
            ax.scatter(np.full(len(values), xloc), values, color=color, **kw)
            return
        parts = ax.violinplot(
            [values], positions=[xloc], widths=0.9,
            showmeans=False, showmedians=False, showextrema=False,
        )
        for pc in parts["bodies"]:
            pc.set_facecolor(color)
            pc.set_alpha(0.7)
            pc.set_edgecolor("none")
        jitter = get_density_based_jitter(values)
        ax.scatter(np.ones_like(values) * xloc + jitter, values, color=color, **kw)
    else:
        ax.scatter(np.ones_like(values) * xloc, values, color=color, **kw)


def draw_carbon_lift(ax, lifted_values, base_values, xloc, color):
    """Mirror the panel-a grid lift: red dashed connector + red-edged lifted blob."""
    base_med = np.nanmedian(np.asarray(base_values, dtype=float))
    lift_med = np.nanmedian(np.asarray(lifted_values, dtype=float))
    ax.plot([xloc, xloc], [base_med, lift_med], color="red", linestyle="--",
            linewidth=1.2, zorder=1)
    draw_violin_scatter(ax, lifted_values, xloc, color, edgecolor="red")


for load, carriers in shown_techs.items():
    group_start = xloc_tracker
    for carrier in carriers:

        # ── Overnight investment cost ───────────────────────────────────────
        carrier_n = carrier_network_names.get(carrier, carrier)

        if carrier_n in n.links.carrier.unique():
            asset_type = "link"
            assets = n.links.index[n.links.carrier == carrier_n]
            capital_cost = n.links.loc[assets, "capital_cost"].mean()
        elif carrier_n in n.generators.carrier.unique():
            asset_type = "generator"
            assets = n.generators.index[n.generators.carrier == carrier_n]
            capital_cost = n.generators.loc[assets, "capital_cost"].mean()
        else:
            raise ValueError(f"No assets found for {carrier_n}")

        costs_carrier_name = tech_costs_mapper.get(carrier, carrier)
        lt = lifetimes.loc[costs_carrier_name] if costs_carrier_name in lifetimes.index else 25
        dr = (
            discount_rates.loc[costs_carrier_name]
            if costs_carrier_name in discount_rates.index
            else 0.07
        )
        annuity_factor = get_annuity_factor(lt, dr)
        overnight = capital_cost / annuity_factor / 1e3  # EUR/kW input (fallback only)

        # Overnight investment per kW of *heat output*, taken straight from the
        # techno-economic 'investment' row (already EUR/kW_th, no FOM/annuity).
        # The network capital_cost is NOT used here: it folds in FOM and is keyed
        # to the input (fuel) capacity, which inflates boilers ~3-4x and would
        # make heat pumps look cheaper than gas boilers.
        if (costs_carrier_name, "investment") in costs.index:
            inv = costs.loc[(costs_carrier_name, "investment"), "value"]
            unit = str(costs.loc[(costs_carrier_name, "investment"), "unit"])
            if "kW_e" in unit:  # CHP is quoted per electrical kW -> scale to heat kW
                links_c = n.links[n.links.carrier == carrier_n]
                cost_value = inv * links_c.efficiency.mean() / rated_heat_efficiency(n, carrier_n)
            else:
                cost_value = inv
        else:
            # no techno-economic entry (e.g. direct firing): fall back to the
            # network capital cost, expressed per kW heat output.
            cost_value = overnight / rated_heat_efficiency(n, carrier_n)
        rated_cop = overnight / cost_value  # input->output capacity ratio (grid lift)
        cap_kwargs = {"s": 45, "linewidth": 0.5, "alpha": 1.0}

        axs[0].scatter(
            [xloc_tracker], [cost_value],
            color=tech_colors[carrier], edgecolor="black", zorder=3, **cap_kwargs,
        )

        if "industrial heat pump" in carrier:
            # Industry heat pumps carry a 500 EUR/kW_el grid-connection cost on top
            # of the link capex; per kW_th it is 500 / rated COP. Show it as a red
            # lift to a red-edged point.
            grid_add = 500.0 / rated_cop
            axs[0].plot(
                [xloc_tracker, xloc_tracker], [cost_value, cost_value + grid_add],
                color="red", linestyle="--", linewidth=1.2, zorder=1,
            )
            axs[0].scatter(
                [xloc_tracker], [cost_value + grid_add],
                color=tech_colors[carrier], edgecolor="red", linewidth=1.2,
                s=cap_kwargs["s"], alpha=1.0, zorder=3,
            )
            va, y_offset = "top", cost_value - 55  # label below the base point
        elif "electric boiler steam" in carrier:
            va, y_offset = "top", cost_value - 55  # below the point
        else:
            va, y_offset = "bottom", cost_value + 55

        axs[0].text(
            xloc_tracker, y_offset, f"{cost_value:.0f}",
            rotation=90, va=va, ha="center", fontsize=8, color="black",
        )

        # ── VOM ──────────────────────────────────────────────────────────────
        # marginal_cost and the input-carrier price are both expressed per unit
        # of link *input* (p0). Dividing by the effective efficiency (delivered
        # output / consumed input) converts them to per unit of *output*, so the
        # COP of a heat pump correctly squeezes its VOM and LCOH.
        marginal_cost = get_marginal_cost(n, assets)
        avg_marginal_cost = marginal_cost.mean()
        locations = pd.Index(get_buses(marginal_cost.index).unique())

        supply = nodal_supply.loc[idx[:, :, carrier]]
        supply.index = get_buses(supply.index.droplevel(0))

        built = nodal_caps.loc[idx[:, :, carrier]]
        built.index = get_buses(built.index.droplevel(0))

        if "CHP" in carrier:
            # CHP supplies electricity and heat; keep the heat side only.
            supply = supply.iloc[1::2]

        # Keep only AC buses where this tech actually supplies and is built.
        locations = locations.intersection(all_locations)
        locations = locations.intersection(supply.index).intersection(built.index)
        supply = supply.loc[locations]
        built = built.loc[locations]

        # Effective efficiency = output delivered / input consumed (per bus).
        # This captures time-varying heat-pump COPs and boiler losses alike.
        if asset_type == "link":
            withdrawal = nodal_withdrawal.loc[idx[:, :, carrier]]
            withdrawal.index = get_buses(withdrawal.index.droplevel(0))
            withdrawal = withdrawal.groupby(level=0).sum().reindex(locations)
            efficiency = (supply / withdrawal).replace([np.inf, -np.inf], np.nan)
        else:
            efficiency = pd.Series(1.0, index=locations)

        if fuels.get(carrier) == "H2":
            # electrolytic H2 price (from wiggle-0 net), not the SMR price here
            fuel_cost = electrolytic_h2_price.reindex(locations)
        elif carrier in fuels:
            fuel_cost = get_fuel_cost(n, locations, fuels[carrier])
        else:
            fuel_cost = 0
        vom = (avg_marginal_cost + fuel_cost) / efficiency

        draw_violin_scatter(axs[1], vom, xloc_tracker, tech_colors[carrier])

        # ── Weighted-average efficiency / COP (output-weighted across buses) ──
        wavg_eff = (efficiency * supply).sum() / supply.sum()
        axs[2].scatter(
            [xloc_tracker], [wavg_eff], color=tech_colors[carrier],
            s=55, edgecolor="black", linewidth=0.6, zorder=3,
        )

        # ── Panel b carbon-price lift (gas techs only) ────────────────────────
        if fuels.get(carrier) == "gas":
            # Panel b: base VOM + carbon-price lift.
            carbon_out = (CO2_PRICE * gas_co2_per_input(n, carrier_n)) / efficiency
            draw_carbon_lift(axs[1], vom + carbon_out, vom, xloc_tracker, tech_colors[carrier])

        if carrier == "urban decentral air heat pump":
            air_hp_xloc = xloc_tracker
        if carrier == "rural ground heat pump":
            ground_hp_xloc = xloc_tracker
        if carrier == "heat<100 industry industrial heat pump medium temperature":
            industry_hp_xloc = xloc_tracker
        if carrier == "heat100-200 industry industrial heat pump high temperature":
            industry_hp_high_xloc = xloc_tracker

        xticklocs.append(xloc_tracker)
        xticklabels.append(nice_names.get(carrier, carrier))
        xloc_tracker += 1

    group_spans.append((group_names.get(load, load), group_start, xloc_tracker - 1))

    for ax in axs:
        ax.axvline(xloc_tracker + inter_load_gap / 2 - 0.5, color="black", linestyle="--", alpha=0.7)
    xloc_tracker += inter_load_gap


axs[-1].set_xticks(xticklocs)
axs[-1].set_xticklabels(xticklabels, rotation=90)
for ax in axs[:-1]:
    ax.set_xticks(xticklocs)
    ax.set_xticklabels([])

for ax in axs:
    ax.set_xlim(-0.5, xloc_tracker - 0.5 - inter_load_gap)
    ax.grid(True, axis="y", linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

axs[2].set_ylim(bottom=0)  # efficiency / COP panel starts at 0

# Centered demand-section headers above the top panel.
header_tr = transforms.blended_transform_factory(axs[0].transData, axs[0].transAxes)
for name, start, end in group_spans:
    axs[0].text(
        (start + end) / 2, 1.03, name,
        transform=header_tr, ha="center", va="bottom",
        fontsize=9, fontweight="bold", linespacing=0.95,
    )

# ── Literature heat-pump CAPEX (household + industrial) ──────────────────────
# Real installed costs are quoted per kW_thermal; each CSV converts them to the
# figure's per-kW_el basis with the model's own thermal->electrical factor
# (model overnight / techno-economic investment for that HP), so a system that
# costs 2x the model's EUR/kW_th also reads as 2x on this axis. Marker colour
# encodes the heat-pump thermal capacity (log scale): industrial CAPEX drops
# strongly with size, so small units sit high and large units low.
lit_specs = [
    ("air_heat_pump_literature_capex.csv", air_hp_xloc),
    ("ground_heat_pump_literature_capex.csv", ground_hp_xloc),
    ("industry_lowtemp_heat_pump_literature_capex.csv", industry_hp_xloc),
    ("industry_hightemp_heat_pump_literature_capex.csv", industry_hp_high_xloc),
]
lit_data = [(pd.read_csv(SCRIPT_DIR / f), x) for f, x in lit_specs]
all_caps = pd.concat([d for d, _ in lit_data])["thermal_capacity_kw"]
cap_norm = LogNorm(vmin=all_caps.min(), vmax=all_caps.max())
cap_cmap = plt.cm.viridis

for df, x in lit_data:
    lit_x = x + 0.30  # just right of the model's heat-pump point
    axs[0].scatter(
        np.full(len(df), lit_x), df["eur_per_kw_thermal"],
        c=df["thermal_capacity_kw"], norm=cap_norm, cmap=cap_cmap,
        marker="D", s=45, edgecolor="black", linewidth=0.8, zorder=4,
    )
    for _, row in df.iterrows():
        axs[0].text(
            lit_x + 0.24, row["eur_per_kw_thermal"], row["country_code"],
            fontsize=6, va="center", ha="left",
        )

# Shared legend for the literature markers, top-left of the whole top panel.
lit_handle = Line2D(
    [0], [0], marker="D", color="none", markerfacecolor="lightgrey",
    markeredgecolor="black", markersize=7,
    label="Heat pump literature capex",
)
lit_legend = axs[0].legend(
    handles=[lit_handle], loc="upper right", bbox_to_anchor=(0.995, 0.99),
    bbox_transform=axs[0].transAxes, fontsize=7, frameon=True,
    handlelength=1.2, borderpad=0.5,
)
axs[0].add_artist(lit_legend)

# Legend for the gas carbon-price lift in panels b and d.
co2_handles = [
    Line2D([0], [0], color="red", linestyle="--", linewidth=1.2,
           label="CO$_2$ price (100 EUR/t)"),
    Line2D([0], [0], marker="o", color="none", markerfacecolor="lightgrey",
           markeredgecolor="red", markeredgewidth=1.2, markersize=8,
           label="incl. CO$_2$ price"),
]
axs[1].legend(
    handles=co2_handles, loc="upper left", fontsize=7, frameon=True,
    handlelength=1.6, borderpad=0.5, labelspacing=0.3,
)

# Small legend explaining the grid-connection lift, in the empty space at the
# top-left of the Industry <100°C section.
hp_legend_x = next(start for name, start, end in group_spans if "<100" in name)
hp_handles = [
    Line2D([0], [0], color="red", linestyle="--", linewidth=1.2,
           label="grid connection (+500 EUR/kW$_{el}$)"),
    Line2D([0], [0], color="none", marker="o", markersize=8,
           markerfacecolor="lightgrey", markeredgecolor="red", markeredgewidth=1.2,
           label="capex incl. grid connection"),
]
axs[0].legend(
    handles=hp_handles, loc="upper left",
    bbox_to_anchor=(hp_legend_x - 0.45, 0.97),
    bbox_transform=transforms.blended_transform_factory(axs[0].transData, axs[0].transAxes),
    fontsize=7, frameon=True, handlelength=1.8, borderpad=0.5, labelspacing=0.4,
)

# Bold A/B/C panel labels at the top-left of each panel.
for ax, lab in zip(axs, "abc"):
    ax.text(
        -0.045, 1.06, lab, transform=ax.transAxes,
        fontsize=19, fontweight="bold", va="top", ha="right",
    )

plt.tight_layout()


# ── Save ─────────────────────────────────────────────────────────────────────
out_local = SCRIPT_DIR / "heat_capex_opex_lcohs.pdf"
fig.savefig(out_local, bbox_inches="tight")
print(f"wrote {out_local}")

if SIBLING_IMGS.exists():
    out_paper = SIBLING_IMGS / "heat_capex_opex_lcohs.pdf"
    fig.savefig(out_paper, bbox_inches="tight")
    print(f"wrote {out_paper}")
