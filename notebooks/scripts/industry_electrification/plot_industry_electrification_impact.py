"""
System-level impact of slower industry electrification compared to cost-optimal.

Compares three industry electrification speed scenarios (fast, medium, slow)
against the unconstrained 'free' baseline, all at wiggle=2000. Plots the delta
for: total system cost (excl. ammonia store), heat pumps in buildings,
and biomass heat in industry.

Usage:
    python plot_industry_electrification_impact.py              # plot from cached data
    python plot_industry_electrification_impact.py --recompute  # recompute from networks
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

WIGGLE = 2000
SCENARIOS = ["free+fast", "free+medium", "free+slow"]
SCENARIO_LABELS = ["fast", "medium", "slow"]

with open(ROOT / "config.basicrun.yaml") as f:
    config = yaml.safe_load(f)
tech_colors = config["plotting"]["tech_colors"]


def network_path(scenario="free", wiggle=WIGGLE):
    return NETWORK_DIR / f"{PREFIX}_{scenario}_{wiggle}.nc"


def style_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)


# --- Quantity functions ---

EXCLUDE_CARRIERS = ["ammonia store"]


def get_total_system_cost(n):
    """Total annualised system cost in bn EUR/a, excluding ammonia store."""
    c = n.statistics.capex()
    o = n.statistics.opex()
    union = c.index.union(o.index)
    c = c.reindex(union).fillna(0)
    o = o.reindex(union).fillna(0)
    total = c + o
    mask = ~total.index.get_level_values(-1).isin(EXCLUDE_CARRIERS)
    return total[mask].sum() / 1e9


num_households = pd.Series({  # in millions
    'DE': 41.3, 'FR': 31.7, 'GB': 28.6, 'IT': 26.1, 'ES': 19.4,
    'PL': 14.7, 'SE': 4.9, 'NO': 2.65, 'NL': 8.4, 'FI': 2.7,
    'RO': 7.1, 'GR': 4.6, 'PT': 4.1, 'CZ': 4.6, 'BE': 5.2,
    'AT': 4.2, 'BG': 2.9, 'DK': 2.9, 'HU': 4.4, 'IE': 1.8,
})


def get_hp_in_buildings(n):
    """Heat pumps in buildings — millions of households equivalent."""
    heating_demands = ['rural heat', 'urban decentral heat', 'urban central heat']
    countries = num_households.index

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

        hp_share = hp_supply / total_supply if total_supply > 0 else 0.
        hp_shares.loc[c] = hp_share

    return hp_shares.mul(num_households).sum()


def get_industry_biomass_heat(n):
    """Biomass heat supply in industry in TWh."""
    idx = pd.IndexSlice
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


QUANTITIES = {
    "total_system_cost": ("Total system cost", "bn EUR/a", get_total_system_cost),
    "hp_buildings": ("Heat pumps in buildings", "million households", get_hp_in_buildings),
    "biomass_industry": ("Biomass heat in industry", "TWh", get_industry_biomass_heat),
}


# ── Data preparation ────────────────────────────────────────────────────────

def compute_data():
    """Load networks and compute all quantities."""
    print("Loading baseline network (free)...")
    n_base = pypsa.Network(network_path("free"))
    baseline = {k: fn(n_base) for k, (_, _, fn) in QUANTITIES.items()}
    del n_base

    scenario_data = {}
    for scenario, label in zip(SCENARIOS, SCENARIO_LABELS):
        print(f"Loading network ({label})...")
        n = pypsa.Network(network_path(scenario))
        scenario_data[label] = {k: fn(n) for k, (_, _, fn) in QUANTITIES.items()}
        del n

    baseline_s = pd.Series(baseline)
    scenario_df = pd.DataFrame(scenario_data)

    baseline_s.to_csv(DATA_DIR / "baseline.csv")
    scenario_df.to_csv(DATA_DIR / "scenarios.csv")
    print(f"Data saved to {DATA_DIR}")
    return baseline_s, scenario_df


def load_data():
    """Load cached data."""
    baseline_s = pd.read_csv(DATA_DIR / "baseline.csv", index_col=0).squeeze()
    scenario_df = pd.read_csv(DATA_DIR / "scenarios.csv", index_col=0)
    return baseline_s, scenario_df


# ── Plotting ────────────────────────────────────────────────────────────────

def plot(baseline_s, scenario_df):
    delta_df = scenario_df.sub(baseline_s, axis=0)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    colors = ["#4c72b0", "#dd8452", "#c44e52"]

    for ax, (key, (title, unit, _)) in zip(axes, QUANTITIES.items()):
        vals = delta_df.loc[key]
        bars = ax.bar(SCENARIO_LABELS, vals, color=colors, width=0.5)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel(f"\u0394 {unit}")
        style_ax(ax)
        ax.set_xticklabels(['50%', '25%', '2%'])
        ax.set_xlabel('Share of optimal industry electrification achieved')

    fig.suptitle(
        "Impact of slower industry electrification vs. cost-optimal (2000 TWh total gas consumption)",
        fontsize=12, y=1.02,
    )
    fig.tight_layout()
    fig.savefig(SAVE_DIR / "industry_electrification_impact.pdf", bbox_inches="tight")
    print(f"Saved to {SAVE_DIR / 'industry_electrification_impact.pdf'}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--recompute", action="store_true",
        help="Recompute data from networks (slow). Otherwise use cached CSVs.",
    )
    args = parser.parse_args()

    cache_exists = (DATA_DIR / "baseline.csv").exists()

    if args.recompute or not cache_exists:
        baseline_s, scenario_df = compute_data()
    else:
        print("Using cached data. Pass --recompute to refresh.")
        baseline_s, scenario_df = load_data()

    plot(baseline_s, scenario_df)
