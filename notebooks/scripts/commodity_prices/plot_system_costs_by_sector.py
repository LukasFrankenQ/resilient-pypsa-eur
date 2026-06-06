"""
System cost breakdown by sector for different wiggle values,
plus difference plots showing the impact of the gas price hike (+50 EUR/MWh).

Plot 1: Stacked bar chart of absolute system costs by sector (supply-side).
Plot 2: Supply-side cost sensitivity per EUR/MWh of hike (capex + opex).
Plot 3: Demand-side cost sensitivity per EUR/MWh of hike (load × marginal price).

Usage:
    python plot_system_costs_by_sector.py              # plot from cached data
    python plot_system_costs_by_sector.py --recompute   # recompute data from networks, then plot
"""

import argparse
import numpy as np
import pypsa
import pandas as pd
import matplotlib.pyplot as plt
import yaml
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
NETWORK_DIR = ROOT / "results" / "networks"
PREFIX = "base_s_50__3H-T-H-B-I-A-dist1_2030"
SAVE_DIR = ROOT.parent / "gas_resilience" / "imgs"
SAVE_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

WIGGLES = [500, 1250, 2000, 2750, 3500]
HIKE_VALUE = 50  # EUR/MWh

with open(ROOT / "config.basicrun.yaml") as f:
    config = yaml.safe_load(f)
tech_colors = config["plotting"]["tech_colors"]


def network_path(scenario="free", wiggle=0, hike=None):
    hike_str = f"_{hike}" if hike is not None else ""
    return NETWORK_DIR / f"{PREFIX}_{scenario}_{wiggle}{hike_str}.nc"


def style_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)


# --- Supply-side sector grouping ---
SECTOR_MAP = {
    # Renewable power
    "solar-hsat": "renewable power",
    "Solar": "renewable power",
    "solar": "renewable power",
    "Onshore Wind": "renewable power",
    "onwind": "renewable power",
    "Offshore Wind (AC)": "renewable power",
    "Offshore Wind (DC)": "renewable power",
    "Offshore Wind (Floating)": "renewable power",
    "offwind-ac": "renewable power",
    "offwind-dc": "renewable power",
    "offwind-float": "renewable power",
    "Reservoir & Dam": "renewable power",
    "Run of River": "renewable power",
    "Pumped Hydro Storage": "renewable power",
    "ror": "renewable power",
    "PHS": "renewable power",
    "hydro": "renewable power",
    # Fossil power
    "Combined-Cycle Gas": "fossil power",
    "Open-Cycle Gas": "fossil power",
    "CCGT": "fossil power",
    "OCGT": "fossil power",
    "OCGT methanol": "fossil power",
    "coal": "fossil power",
    "lignite": "fossil power",
    "nuclear": "fossil power",
    "oil": "fossil power",
    # Electricity grid & storage
    "electricity distribution grid": "electricity grid & storage",
    "AC": "electricity grid & storage",
    "DC": "electricity grid & storage",
    "Battery Storage": "electricity grid & storage",
    "battery charger": "electricity grid & storage",
    "battery discharger": "electricity grid & storage",
    "battery storage": "electricity grid & storage",
    "home battery": "electricity grid & storage",
    "home battery charger": "electricity grid & storage",
    "home battery discharger": "electricity grid & storage",
    "transmission lines": "electricity grid & storage",
    # Building heat
    "urban decentral gas boiler": "building heat",
    "urban decentral air heat pump": "building heat",
    "urban decentral biomass boiler": "building heat",
    "urban decentral oil boiler": "building heat",
    "urban decentral resistive heater": "building heat",
    "urban decentral solar thermal": "building heat",
    "rural gas boiler": "building heat",
    "rural ground heat pump": "building heat",
    "rural air heat pump": "building heat",
    "rural biomass boiler": "building heat",
    "rural oil boiler": "building heat",
    "rural resistive heater": "building heat",
    "rural solar thermal": "building heat",
    "urban central gas boiler": "building heat",
    "urban central air heat pump": "building heat",
    "urban central resistive heater": "building heat",
    "urban central solar thermal": "building heat",
    "urban central solid biomass CHP": "building heat",
    "urban central solid biomass CHP CC": "building heat",
    "urban central gas CHP": "building heat",
    "urban central gas CHP CC": "building heat",
    # Industry heat
    "heat<100 industry industrial heat pump medium temperature": "industry heat",
    "heat100-200 industry industrial heat pump high temperature": "industry heat",
    "heat200-500 industry solid biomass": "industry heat",
    "heat<100 industry solid biomass": "industry heat",
    "heat100-200 industry solid biomass": "industry heat",
    "heat100-200 industry gas": "industry heat",
    "heat<100 industry gas": "industry heat",
    "heat>500 industry gas": "industry heat",
    "heat200-500 industry gas": "industry heat",
    "heat<100 industry electric boiler steam": "industry heat",
    "heat100-200 industry electric boiler steam": "industry heat",
    "heat>500 industry hydrogen": "industry heat",
    "heat200-500 industry hydrogen": "industry heat",
    # Steel & heavy industry
    "BOF": "steel & heavy industry",
    "EAF": "steel & heavy industry",
    "H2 DRI": "steel & heavy industry",
    "gas DRI": "steel & heavy industry",
    "HVC to air": "steel & heavy industry",
    "non-sequestered HVC": "steel & heavy industry",
    "naphtha for industry": "steel & heavy industry",
    "coal for industry": "steel & heavy industry",
    "gas for industry": "steel & heavy industry",
    "gas for industry CC": "steel & heavy industry",
    "solid biomass for industry": "steel & heavy industry",
    "solid biomass for industry CC": "steel & heavy industry",
    "industry wood": "steel & heavy industry",
    # Hydrogen & synfuels
    "H2 Electrolysis": "hydrogen & synfuels",
    "H2 Store": "hydrogen & synfuels",
    "H2 Fuel Cell": "hydrogen & synfuels",
    "H2 pipeline": "hydrogen & synfuels",
    "H2 pipeline retrofitted": "hydrogen & synfuels",
    "SMR": "hydrogen & synfuels",
    "SMR CC": "hydrogen & synfuels",
    "Fischer-Tropsch": "hydrogen & synfuels",
    "methanolisation": "hydrogen & synfuels",
    "biomass-to-methanol": "hydrogen & synfuels",
    "biomass to liquid": "hydrogen & synfuels",
    "Sabatier": "hydrogen & synfuels",
    "methanol": "hydrogen & synfuels",
    "Haber-Bosch": "hydrogen & synfuels",
    "ammonia cracker": "hydrogen & synfuels",
    "ammonia store": "hydrogen & synfuels",
    # Gas & biogas infrastructure
    "biogas to gas": "gas & biogas",
    "gas pipeline": "gas & biogas",
    "gas pipeline new": "gas & biogas",
    "gas": "gas & biogas",
    # Transport
    "land transport oil": "transport",
    "BEV charger": "transport",
    "EV battery": "transport",
    "V2G": "transport",
    "kerosene for aviation": "transport",
    "shipping oil": "transport",
    "shipping methanol": "transport",
    # CO2 management
    "co2 stored": "CO2 management",
    "co2 sequestered": "CO2 management",
    "CO2 pipeline": "CO2 management",
    "co2 Store": "CO2 management",
}

SECTOR_COLORS = {
    "renewable power": "#235ebc",
    "fossil power": "#a85522",
    "electricity grid & storage": "#6c9459",
    "building heat": "#d15959",
    "industry heat": "#f073da",
    "steel & heavy industry": "#545454",
    "hydrogen & synfuels": "#bf13a0",
    "gas & biogas": "#e3d37d",
    "transport": "#baf238",
    "CO2 management": "#f29dae",
}

SECTOR_ORDER = [
    "renewable power",
    "fossil power",
    "electricity grid & storage",
    "building heat",
    "industry heat",
    "steel & heavy industry",
    "hydrogen & synfuels",
    "gas & biogas",
    "transport",
    "CO2 management",
]

# --- Demand-side sector grouping (by bus carrier) ---
BUS_CARRIER_SECTOR_MAP = {
    "AC": "electricity",
    "low voltage": "electricity",
    "urban central heat": "building heat",
    "urban decentral heat": "building heat",
    "residential urban decentral heat": "building heat",
    "services urban decentral heat": "building heat",
    "rural heat": "building heat",
    "residential rural heat": "building heat",
    "services rural heat": "building heat",
    "heat<100 industry": "industry heat",
    "heat100-200 industry": "industry heat",
    "heat200-500 industry": "industry heat",
    "heat>500 industry": "industry heat",
    "gas": "gas",
    "gas for industry": "gas",
    "H2": "hydrogen",
    "oil": "oil & transport",
    "oil primary": "oil & transport",
    "kerosene for aviation": "oil & transport",
    "shipping oil": "oil & transport",
    "shipping methanol": "oil & transport",
    "methanol": "synfuels",
    "NH3": "synfuels",
    "naphtha for industry": "synfuels",
    "solid biomass": "biomass",
    "solid biomass for industry": "biomass",
    "biogas": "biomass",
    "coal": "coal",
    "coal for industry": "coal",
    "lignite": "coal",
    "co2": "CO2",
    "co2 stored": "CO2",
    "co2 sequestered": "CO2",
    "uranium": "electricity",
    "steel": "industry process",
    "hbi": "industry process",
    "non-sequestered HVC": "industry process",
}

DEMAND_SECTOR_COLORS = {
    "electricity": "#235ebc",
    "building heat": "#d15959",
    "industry heat": "#f073da",
    "gas": "#e3d37d",
    "hydrogen": "#bf13a0",
    "oil & transport": "#afafaf",
    "synfuels": "#25c49a",
    "biomass": "#baa741",
    "coal": "#545454",
    "CO2": "#f29dae",
    "industry process": "#586357",
}

DEMAND_SECTOR_ORDER = [
    "electricity",
    "building heat",
    "industry heat",
    "gas",
    "hydrogen",
    "oil & transport",
    "synfuels",
    "biomass",
    "coal",
    "CO2",
    "industry process",
]


def map_to_sector(carrier):
    return SECTOR_MAP.get(carrier, "other")


def get_demand_expenditure(n):
    """
    Compute demand-side expenditure by bus carrier sector in bn EUR/a.

    Only considers Load components (n.loads). For each load, retrieves
    its time series from n.loads_t.p_set (time-varying) or n.loads.p_set
    (static), then folds over the marginal price at the load's bus.

    expenditure = Σ_t (load_p × marginal_price_at_bus × snapshot_weight)
    """
    weights = n.snapshot_weightings.generators
    expenditures = {}

    for load_name, load_row in n.loads.iterrows():
        bus = load_row.bus
        bus_carrier = n.buses.at[bus, "carrier"]

        # Get load time series
        if load_name in n.loads_t.p_set.columns:
            load_ts = n.loads_t.p_set[load_name]
        else:
            load_ts = pd.Series(load_row.p_set, index=n.snapshots)

        # Get marginal price at the load's bus
        if bus in n.buses_t.marginal_price.columns:
            price_ts = n.buses_t.marginal_price[bus]
        else:
            continue

        # Expenditure = Σ_t (load × price × weight)
        expenditure = (load_ts * price_ts * weights).sum() / 1e9  # bn EUR
        sector = BUS_CARRIER_SECTOR_MAP.get(bus_carrier, "other")
        expenditures[sector] = expenditures.get(sector, 0.0) + expenditure

    return pd.Series(expenditures)


def get_cost_breakdown(n):
    """Get system costs grouped by sector in bn EUR/a."""
    c = n.statistics.capex()
    o = n.statistics.opex()
    union = c.index.union(o.index)
    c = c.reindex(union).replace(np.nan, 0)
    o = o.reindex(union).replace(np.nan, 0)
    costs = (c + o).droplevel(0).groupby(level=0).sum() / 1e9
    sector_costs = costs.groupby(costs.index.map(map_to_sector)).sum()
    return sector_costs


def _plot_stacked_diverging(ax, df, colors, bar_width=0.5, threshold=0.001):
    """Plot a stacked bar chart allowing positive and negative values."""
    pos = df.clip(lower=0)
    neg = df.clip(upper=0)
    bottom_pos = pd.Series(0.0, index=df.columns)
    bottom_neg = pd.Series(0.0, index=df.columns)
    bar_handles = []
    bar_labels = []
    x = range(len(df.columns))

    for sector in df.index:
        color = colors.get(sector, "#999999")
        vals_pos = pos.loc[sector]
        vals_neg = neg.loc[sector]

        if vals_pos.abs().max() > threshold:
            bars = ax.bar(x, vals_pos, bottom=bottom_pos, color=color, width=bar_width)
            bottom_pos = bottom_pos + vals_pos
            if sector not in bar_labels:
                bar_handles.append(bars[0])
                bar_labels.append(sector)

        if vals_neg.abs().max() > threshold:
            bars = ax.bar(x, vals_neg, bottom=bottom_neg, color=color, width=bar_width)
            bottom_neg = bottom_neg + vals_neg
            if sector not in bar_labels:
                bar_handles.append(bars[0])
                bar_labels.append(sector)

    ax.axhline(0, color="black", linewidth=0.8)
    bar_handles.reverse()
    bar_labels.reverse()
    return bar_handles, bar_labels


# ── Data preparation ────────────────────────────────────────────────────────

def compute_data():
    """Load all networks and compute supply- and demand-side cost data."""
    print("Loading networks and computing costs...")
    regular_costs = {}
    hike_costs = {}
    regular_expenditure = {}
    hike_expenditure = {}
    for w in WIGGLES:
        print(f"  wiggle={w} (regular)")
        n = pypsa.Network(network_path(wiggle=w))
        regular_costs[w] = get_cost_breakdown(n)
        regular_expenditure[w] = get_demand_expenditure(n)
        del n
        print(f"  wiggle={w} (hike={HIKE_VALUE})")
        n = pypsa.Network(network_path(wiggle=w, hike=HIKE_VALUE))
        hike_costs[w] = get_cost_breakdown(n)
        hike_expenditure[w] = get_demand_expenditure(n)
        del n

    # Build and save DataFrames
    reg_df = pd.DataFrame(regular_costs).reindex(SECTOR_ORDER).fillna(0)
    hike_df = pd.DataFrame(hike_costs).reindex(SECTOR_ORDER).fillna(0)
    reg_exp_df = pd.DataFrame(regular_expenditure).reindex(DEMAND_SECTOR_ORDER).fillna(0)
    hike_exp_df = pd.DataFrame(hike_expenditure).reindex(DEMAND_SECTOR_ORDER).fillna(0)

    reg_df.to_csv(DATA_DIR / "regular_costs.csv")
    hike_df.to_csv(DATA_DIR / "hike_costs.csv")
    reg_exp_df.to_csv(DATA_DIR / "regular_expenditure.csv")
    hike_exp_df.to_csv(DATA_DIR / "hike_expenditure.csv")
    print(f"Data saved to {DATA_DIR}")
    return reg_df, hike_df, reg_exp_df, hike_exp_df


def load_data():
    """Load cached data from CSV files."""
    reg_df = pd.read_csv(DATA_DIR / "regular_costs.csv", index_col=0)
    hike_df = pd.read_csv(DATA_DIR / "hike_costs.csv", index_col=0)
    reg_exp_df = pd.read_csv(DATA_DIR / "regular_expenditure.csv", index_col=0)
    hike_exp_df = pd.read_csv(DATA_DIR / "hike_expenditure.csv", index_col=0)
    # Restore integer column names
    for df in [reg_df, hike_df, reg_exp_df, hike_exp_df]:
        df.columns = df.columns.astype(int)
    return reg_df, hike_df, reg_exp_df, hike_exp_df


# ── Plotting ────────────────────────────────────────────────────────────────

def plot(reg_df, hike_df, reg_exp_df, hike_exp_df):
    diff_df = hike_df - reg_df
    diff_exp_df = hike_exp_df - reg_exp_df

    # Drop sectors with negligible costs across all wiggles
    threshold = 0.5  # bn EUR
    mask = reg_df.abs().max(axis=1) > threshold
    reg_df = reg_df[mask]
    diff_df = diff_df.reindex(reg_df.index)

    mask_exp = (reg_exp_df.abs().max(axis=1) > threshold) | (diff_exp_df.abs().max(axis=1) > 0.01)
    diff_exp_df = diff_exp_df[mask_exp]

    # --- Combined 1x3 figure ---
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(22, 6))

    # Plot 1: Absolute system costs (supply-side)
    reg_df.T.plot(
        kind="bar",
        ax=ax1,
        stacked=True,
        color=[SECTOR_COLORS.get(s, "#999999") for s in reg_df.index],
        width=0.7,
    )
    handles, labels = ax1.get_legend_handles_labels()
    handles.reverse()
    labels.reverse()
    ax1.set_ylabel("System cost [bn EUR/a]")
    ax1.set_xlabel("Gas budget [TWh]")
    ax1.set_xticklabels([str(w) for w in WIGGLES], rotation=0)
    ax1.legend(
        handles, labels, ncol=1, loc="upper left", bbox_to_anchor=[1, 1], frameon=True
    )
    style_ax(ax1)

    # Plot 2: Supply-side cost sensitivity per EUR/MWh of hike
    sensitivity_df = diff_df / HIKE_VALUE
    bar_handles, bar_labels = _plot_stacked_diverging(ax2, sensitivity_df, SECTOR_COLORS)
    ax2.set_ylabel("Supply-side cost sensitivity\n[bn EUR/a per EUR/MWh]")
    ax2.set_xlabel("Gas budget [TWh]")
    ax2.set_xticks(list(range(len(WIGGLES))))
    ax2.set_xticklabels([str(w) for w in WIGGLES])
    ax2.legend(
        bar_handles, bar_labels, ncol=1, loc="upper left", bbox_to_anchor=[1, 1],
        frameon=True,
    )
    style_ax(ax2)

    # Plot 3: Demand-side cost sensitivity per EUR/MWh of hike
    demand_sensitivity_df = diff_exp_df / HIKE_VALUE
    bar_handles3, bar_labels3 = _plot_stacked_diverging(
        ax3, demand_sensitivity_df, DEMAND_SECTOR_COLORS
    )
    ax3.set_ylabel("Demand-side cost sensitivity\n[bn EUR/a per EUR/MWh]")
    ax3.set_xlabel("Gas budget [TWh]")
    ax3.set_xticks(list(range(len(WIGGLES))))
    ax3.set_xticklabels([str(w) for w in WIGGLES])
    ax3.legend(
        bar_handles3, bar_labels3, ncol=1, loc="upper left", bbox_to_anchor=[1, 1],
        frameon=True,
    )
    style_ax(ax3)

    fig.tight_layout()
    fig.savefig(SAVE_DIR / "system_costs_by_sector.png", dpi=200, bbox_inches="tight")
    fig.savefig(SAVE_DIR / "system_costs_by_sector.pdf", bbox_inches="tight")
    print(f"Saved to {SAVE_DIR / 'system_costs_by_sector.png'}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--recompute", action="store_true",
        help="Recompute data from networks (slow). Otherwise use cached CSVs.",
    )
    args = parser.parse_args()

    cache_exists = (DATA_DIR / "regular_costs.csv").exists()

    if args.recompute or not cache_exists:
        reg_df, hike_df, reg_exp_df, hike_exp_df = compute_data()
    else:
        print("Using cached data. Pass --recompute to refresh.")
        reg_df, hike_df, reg_exp_df, hike_exp_df = load_data()

    plot(reg_df, hike_df, reg_exp_df, hike_exp_df)
