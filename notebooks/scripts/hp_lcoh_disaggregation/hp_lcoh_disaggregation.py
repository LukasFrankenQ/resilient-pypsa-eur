"""Disaggregate the levelized cost of heat (LCOH) of heat-pump supply.

For the ``3H, lv1.25, free`` networks at three gas-budget settings
(wiggle = 0, 2000, 4000) we look at the heat-pump technology serving four
heat demands:

    * industry heat<100        (industrial heat pump, medium temperature)
    * industry heat100-200     (industrial heat pump, high temperature)
    * rural heat               (air + ground heat pump)
    * urban decentral heat     (air heat pump)

For each demand the LCOH (EUR/MWh_heat) is split into four additive parts,
computed as a heat-delivered-weighted average across all 50 regions (i.e. the
total annual cost of each part divided by the total annual heat delivered):

    1. capex heat pump          capital_cost * p_nom_opt of the heat-pump link
    2. capex distribution       share of the 'electricity distribution grid'
                                 link capex, allocated by the electricity the
                                 heat pump draws from the low-voltage bus
    3. opex (marginal_cost)     marginal_cost * p0  (the asset O&M term)
    4. electricity cost         AC (wholesale) price * p0  consumed

Electricity is valued at the *AC* bus price and the distribution-grid capex is
added separately, so the two are not double-counted (the low-voltage LMP would
already embed the distribution-grid cost).

The figure shows, at each demand, three stacked bars (one per wiggle setting).

Output: hp_lcoh_disaggregation.{pdf,csv} in this folder and a PDF copy in the
sibling gas_resilience/imgs folder.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pypsa
from matplotlib.patches import Patch

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = Path(__file__).resolve().parent
NETWORK_DIR = REPO_ROOT / "results" / "networks"
SIBLING_IMGS = REPO_ROOT.parent / "gas_resilience" / "imgs"

PREFIX = "base_s_50_lv1.25_3H-T-H-B-I-A-dist1_2030_free"
WIGGLES = [0, 2000, 4000]  # gas-budget settings -> the three stacked bars

# ── Demand groups and the heat-pump carrier(s) serving each ──────────────────
HP_TECHS = {
    "industry heat<100": ["heat<100 industry industrial heat pump medium temperature"],
    "industry heat100-200": ["heat100-200 industry industrial heat pump high temperature"],
    "rural heat": ["rural ground heat pump"],
    "urban individual heat": ["urban decentral air heat pump"],
}

# Frontend display labels: demand + the precise heat-pump technology referred to.
DEMAND_LABELS = {
    "industry heat<100": "industry heat < 100C\nmedium-temperature heat pump",
    "industry heat100-200": "industry heat 100-200C\nhigh-temperature heat pump",
    "rural heat": "ground heat pump",
    "urban individual heat": "air heat pump",
}

COMPONENTS = [
    "capex heat pump",
    "capex distribution",
    "opex",
    "electricity cost",
]
COMPONENT_COLORS = {
    "capex heat pump": "#1f6f5c",   # deep teal
    "capex distribution": "#8ec9b4",  # light teal
    "opex": "#f2a541",              # amber
    "electricity cost": "#3b6ea5",  # blue
}


def style_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)


def disaggregate(n):
    """Return a DataFrame (demand x component) of LCOH parts in EUR/MWh_heat."""
    w = n.snapshot_weightings.generators  # hours represented by each snapshot

    # Distribution-grid cost per MWh of electricity delivered to each LV bus.
    # The 'electricity distribution grid' link connects AC (bus0) -> low voltage
    # (bus1); its annualised capex / electricity delivered gives a per-MWh_elec
    # cost we allocate to heat pumps by the electricity they draw from that bus.
    dg = n.links[n.links.carrier == "electricity distribution grid"]
    dg_capex = dg.capital_cost * dg.p_nom_opt
    dg_delivered = (-n.links_t.p1[dg.index]).multiply(w, axis=0).sum()
    dist_cost_per_mwh = pd.Series(
        dg_capex.values / dg_delivered.values, index=dg.bus1.values
    )

    records = {}
    for demand, carriers in HP_TECHS.items():
        hp = n.links[n.links.carrier.isin(carriers)]

        elec_mwh = n.links_t.p0[hp.index].multiply(w, axis=0).sum()     # MWh_elec/link
        heat_mwh = (-n.links_t.p1[hp.index]).multiply(w, axis=0).sum()  # MWh_heat/link
        total_heat = heat_mwh.sum()

        capex_hp = (hp.capital_cost * hp.p_nom_opt).sum()
        capex_dist = (
            elec_mwh * dist_cost_per_mwh.reindex(hp.bus0.values).values
        ).sum()
        opex = (hp.marginal_cost * elec_mwh).sum()

        ac_buses = hp.bus0.str.replace(" low voltage", "", regex=False)
        ac_prices = n.buses_t.marginal_price[ac_buses.values].copy()
        ac_prices.columns = hp.index
        elec_cost = (
            (ac_prices * n.links_t.p0[hp.index]).multiply(w, axis=0).sum().sum()
        )

        records[demand] = {
            "capex heat pump": capex_hp / total_heat,
            "capex distribution": capex_dist / total_heat,
            "opex": opex / total_heat,
            "electricity cost": elec_cost / total_heat,
        }
    return pd.DataFrame(records).T[COMPONENTS]


def endogenous_elec_price(n):
    """Load-weighted average AC (wholesale) electricity price [EUR/MWh]."""
    w = n.snapshot_weightings.generators
    wd = n.statistics.withdrawal(
        bus_carrier="AC", groupby="bus", groupby_time=False,
        aggregate_across_components=True,
    ).T  # snapshots x AC buses
    price = n.buses_t.marginal_price[wd.columns]
    num = (price * wd).multiply(w, axis=0).sum().sum()
    den = wd.multiply(w, axis=0).sum().sum()
    return num / den


# ── Compute per wiggle ────────────────────────────────────────────────────────
data = {}
elec_prices = {}
for wiggle in WIGGLES:
    n = pypsa.Network(NETWORK_DIR / f"{PREFIX}_{wiggle}.nc")
    assert n.is_solved, f"network wiggle={wiggle} is not solved"
    df = disaggregate(n)
    df["total LCOH"] = df.sum(axis=1)
    data[wiggle] = df
    elec_prices[wiggle] = endogenous_elec_price(n)
    print(f"\nwiggle={wiggle}  (elec price {elec_prices[wiggle]:.1f} EUR/MWh)"
          f"  LCOH disaggregation [EUR/MWh_heat]:")
    print(df.round(2).to_string())

# Tidy CSV: rows = (wiggle, demand)
out = pd.concat(data, names=["wiggle", "demand"])
out.to_csv(SCRIPT_DIR / "hp_lcoh_disaggregation.csv")

# ── Grouped + stacked bar plot ────────────────────────────────────────────────
demands = list(HP_TECHS)
x = np.arange(len(demands))
bar_w = 0.26
offsets = np.linspace(-bar_w, bar_w, len(WIGGLES))  # one offset per wiggle

fig, ax = plt.subplots(figsize=(10, 6))

ymax = max(data[wg]["total LCOH"].max() for wg in WIGGLES)
for off, wiggle in zip(offsets, WIGGLES):
    df = data[wiggle]
    bottom = np.zeros(len(demands))
    xpos = x + off
    for comp in COMPONENTS:
        ax.bar(xpos, df[comp].values, bottom=bottom, width=bar_w,
               color=COMPONENT_COLORS[comp], edgecolor="white", linewidth=0.5)
        bottom += df[comp].values
    # total LCOH on top of each bar; endogenous electricity price as the x-tick
    for xi, tot in zip(xpos, df["total LCOH"].values):
        ax.text(xi, tot + ymax * 0.012, f"{tot:.0f}", ha="center", va="bottom",
                fontsize=8, fontweight="bold")
        ax.text(xi, -ymax * 0.02, f"{elec_prices[wiggle]:.0f}", ha="center",
                va="top", fontsize=8, color="0.2")

# ── Sideways arrow annotating the electricity-price tick row ──────────────────
# (price rises as the gas budget tightens, i.e. towards the left bar)
y_arrow = -ymax * 0.075
ax.annotate(
    "", xy=(x[-1] + bar_w + 0.15, y_arrow), xytext=(x[0] - bar_w - 0.15, y_arrow),
    arrowprops=dict(arrowstyle="<->", color="0.2", lw=1.2),
    annotation_clip=False,
)
ax.text((x[0] + x[-1]) / 2, y_arrow - ymax * 0.025,
        "endogenous\nelectricity price [EUR/MWh$_{el}$]",
        ha="center", va="top", fontsize=9, color="0.2")

# Demand + precise-technology labels at the bottom of each group.
ax.set_xticks([])
for xi, demand in zip(x, demands):
    ax.text(xi, -ymax * 0.20, DEMAND_LABELS[demand], ha="center", va="top",
            fontsize=9.5)

ax.set_ylabel("Levelized cost of heat  [EUR/MWh$_{heat}$]")
ax.set_ylim(0, ymax * 1.15)

style_ax(ax)
legend_handles = [Patch(facecolor=COMPONENT_COLORS[c], label=c) for c in COMPONENTS]
ax.legend(handles=legend_handles, frameon=True, loc="upper left")
plt.tight_layout()

# ── Save ──────────────────────────────────────────────────────────────────────
out_local = SCRIPT_DIR / "hp_lcoh_disaggregation.pdf"
fig.savefig(out_local, bbox_inches="tight")
print(f"\nwrote {out_local}")

if SIBLING_IMGS.exists():
    out_paper = SIBLING_IMGS / "hp_lcoh_disaggregation.pdf"
    fig.savefig(out_paper, bbox_inches="tight")
    print(f"wrote {out_paper}")
