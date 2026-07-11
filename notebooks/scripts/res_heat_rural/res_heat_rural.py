"""Reproduce res-heat_rural_majority-supply-bar.pdf and
res-heat_rural_lcoh-technology-panels.pdf as a standalone script.

Converted from notebooks/spatial_correlation.ipynb (cells 0, 1, 2, 7, 33, 35, 38).
The third figure produced by the original notebook function (renewable capacity
factors) is dropped — the paper does not use it.

Heavy: loads every solved 3H 2030 'free' network across the gas-budget wiggle
sweep to build per-bus LCOH and majority-supply tables. Run once; the JSON
exports are then frozen.
"""
from __future__ import annotations

import random
import sys
from itertools import product
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pypsa
import yaml
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = Path(__file__).resolve().parent
NETWORK_DIR = REPO_ROOT / "results" / "networks"
SIBLING_IMGS = REPO_ROOT.parent / "gas_resilience" / "imgs"
SIBLING_IMGS.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(REPO_ROOT / "notebooks" / "scripts"))
from frontend_export import export_frontend_data  # noqa: E402

idx = pd.IndexSlice
MODE = "rural"


# ── Set up bus list and tech colors ─────────────────────────────────────────
n_init = pypsa.Network(
    NETWORK_DIR / "base_s_50_lv1.25_3H-T-H-B-I-A-dist1_2030_free_4000.nc"
)
buses = n_init.buses.loc[n_init.buses.carrier == "AC"].index.unique()

with open(REPO_ROOT / "config" / "plotting.default.yaml") as f:
    tech_colors = yaml.safe_load(f)["plotting"]["tech_colors"]


# ── Build LCOH and majority-supply tables across the wiggle sweep ──────────
carriers = [
    "urban decentral air heat pump",
    "rural ground heat pump",
    "urban central air heat pump",
    "rural biomass boiler",
    "urban decentral biomass boiler",
    "urban central solid biomass CHP",
    "rural gas boiler",
    "urban decentral gas boiler",
    "urban central gas CHP",
    "urban central gas boiler",
    "urban decentral resistive heater",
]
demand_carriers = [
    "urban decentral heat",
    "rural heat",
    "urban central heat",
    "rural heat",
    "urban decentral heat",
    "urban central heat",
    "rural heat",
    "urban decentral heat",
    "urban central heat",
    "urban central heat",
    "urban decentral heat",
]
fuel_carriers = [
    "low voltage", "low voltage", "low voltage",
    "solid biomass", "solid biomass", "solid biomass",
    "gas", "gas", "gas", "gas",
    "low voltage",
]

lcohs = pd.DataFrame(
    index=pd.MultiIndex.from_product([buses, carriers]),
    columns=[0, 2000, 4000],
)

gas_consumptions = list(range(0, 5250, 250))
bus_majority_supply = pd.DataFrame(
    index=pd.MultiIndex.from_product([buses, set(demand_carriers)]),
    columns=gas_consumptions,
)

n = n_init
for wiggle in tqdm(gas_consumptions):
    path = NETWORK_DIR / f"base_s_50_lv1.25_3H-T-H-B-I-A-dist1_2030_free_{wiggle}.nc"
    if not path.exists():
        print(f"missing: {path.name} — skipping")
        continue
    n = pypsa.Network(path)
    w = n.snapshot_weightings["generators"]

    capex = n.statistics.capex(groupby=["bus", "carrier", "bus_carrier"])
    opex = n.statistics.opex(groupby=["bus", "carrier", "bus_carrier"])
    eb = n.statistics.energy_balance(
        aggregate_time=False,
        groupby=["bus", "carrier", "bus_carrier"],
    )

    for bus, demand in bus_majority_supply.index:
        try:
            bus_balance = eb.loc[idx[:, bus + " " + demand, :, :]].sum(axis=1)
            bus_balance.index = bus_balance.index.get_level_values(1)
            bus_majority_supply.loc[(bus, demand), wiggle] = bus_balance.idxmax()
        except KeyError:
            continue

    if wiggle not in lcohs.columns:
        continue
    for bus, (tech_carrier, demand_carrier, fuel) in product(
        buses, zip(carriers, demand_carriers, fuel_carriers)
    ):
        try:
            withdrawal = eb.loc[
                idx["Link", bus + " " + fuel, tech_carrier, fuel]
            ].mul(w).mul(-1)
        except KeyError:
            continue

        fuel_cost = (
            n.buses_t.marginal_price[bus + " " + fuel].mul(withdrawal).sum()
        )
        try:
            delivered_heat = eb.loc[
                idx["Link", bus + " " + demand_carrier, tech_carrier, demand_carrier]
            ].mul(w, axis=0).sum()
            o = opex.loc[idx["Link", bus + " " + fuel, tech_carrier, fuel]]
            c = capex.loc[idx["Link", bus + " " + fuel, tech_carrier, fuel]]
        except KeyError:
            continue

        if not (isinstance(o, float) and isinstance(c, float)
                and isinstance(delivered_heat, float)):
            continue
        if delivered_heat == 0.0:
            lcoh = np.nan
        else:
            lcoh = (fuel_cost + c + o) / delivered_heat
        lcohs.loc[(bus, tech_carrier), wiggle] = lcoh


# ── Plotting (cell 38, trimmed) ─────────────────────────────────────────────
def make_residential_heat_plot(mode):
    ms = bus_majority_supply.loc[idx[:, mode + " heat"], :]
    ms.index = ms.index.get_level_values(0)
    ms = ms.copy()

    def first_gas_col(row):
        for i, val in enumerate(row):
            if isinstance(val, str) and "gas" in val:
                return i
        return len(row)

    ms = ms.loc[ms.apply(first_gas_col, axis=1).sort_values().index]

    RANDOM_BUS_SUBSET_SIZE = 20
    random.seed(0)

    if len(ms) > RANDOM_BUS_SUBSET_SIZE:
        sample = random.sample(list(ms.index), RANDOM_BUS_SUBSET_SIZE)
    else:
        sample = list(ms.index)
    sample = [b for b in ms.index if b in sample]
    ms_sorted = ms.loc[sample]

    # ── Bar plot ────────────────────────────────────────────────────────────
    fig_bar, ax_bar = plt.subplots(1, 1, figsize=(8, 6))
    segment_height = 1
    n_segments = len(ms.columns)
    for i, (busname, row) in enumerate(ms_sorted.iterrows()):
        bottom = 0
        for j, col in enumerate(ms.columns):
            color = tech_colors.get(row[col], "#999999")
            ax_bar.bar(
                i, segment_height, bottom=bottom,
                color=color, width=0.8, edgecolor="white", linewidth=0.5,
            )
            bottom += segment_height

    ax_bar.set_xticks(range(len(ms_sorted)))
    ax_bar.set_xticklabels(ms_sorted.index, rotation=90, fontsize=7)
    ax_bar.set_ylabel("System Gas Consumption [bcm]")
    ax_bar.set_yticks([j + 0.5 for j in range(n_segments)])
    # Display gas consumption levels in bcm (10 TWh per bcm).
    ax_bar.set_yticklabels([int(c) // 10 for c in ms.columns])

    legend_handles = [
        mpatches.Patch(color=tech_colors.get(t, "#999999"), label=t)
        for t in set(ms.values.flatten()) if isinstance(t, str)
    ]
    ax_bar.legend(
        handles=legend_handles,
        loc="upper left", bbox_to_anchor=(0, 1.1),
        ncol=max(1, len(legend_handles) // 2),
        frameon=False, fontsize=8, handleheight=1.5, borderaxespad=0,
        title="Majority Supply Technology",
    )
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)

    out_bar_local = SCRIPT_DIR / f"res-heat_{mode}_majority-supply-bar.pdf"
    out_bar_paper = SIBLING_IMGS / f"res-heat_{mode}_majority-supply-bar.pdf"
    fig_bar.savefig(out_bar_local, bbox_inches="tight")
    fig_bar.savefig(out_bar_paper, bbox_inches="tight")
    print(f"wrote {out_bar_paper}")

    export_frontend_data(f"res-heat_{mode}_majority-supply-bar", {
        "y_label": "System Gas Consumption [TWh]",
        "wiggles_TWh": [int(c) for c in ms.columns],
        "buses": list(ms_sorted.index),
        "majority_supply_tech": {
            str(_b): [
                (str(_v) if isinstance(_v, str) else None)
                for _v in ms_sorted.loc[_b].values
            ]
            for _b in ms_sorted.index
        },
        "tech_colors": {
            t: tech_colors.get(t, "#999999")
            for t in set(ms.values.flatten()) if isinstance(t, str)
        },
    })

    # ── LCOH technology panels ─────────────────────────────────────────────
    technologies = sorted({t for t in ms.values.flatten() if isinstance(t, str)})
    n_technologies = len(technologies)

    ss = lcohs.loc[lcohs.index.get_level_values(1).str.contains(mode)]
    ss.columns = ss.columns.astype(int)
    ss.sort_index(inplace=True, axis=1)
    ss_subset = ss.loc[ss.index.get_level_values(0).isin(ms_sorted.index)]

    edge_cmap = plt.cm.viridis
    norm = plt.Normalize(ss.columns.min(), ss.columns.max())

    fig_lcoh, axes_row = plt.subplots(1, n_technologies, figsize=(16, 5))
    if n_technologies == 1:
        axes_row = [axes_row]

    panels_payload = {}
    for k, tech in enumerate(technologies):
        ax = axes_row[k]
        y_matrix = []
        for busname in ms_sorted.index:
            try:
                sss = ss_subset.loc[busname]
                if isinstance(sss, pd.Series):
                    sss = pd.DataFrame(
                        [sss.values], columns=sss.index, index=[sss.name]
                    )
                if tech in sss.index:
                    lcoh_row = pd.Series(
                        sss.loc[tech].values, index=sss.columns
                    ).reindex(ss.columns).values
                else:
                    lcoh_row = np.full(len(ss.columns), np.nan)
            except KeyError:
                lcoh_row = np.full(len(ss.columns), np.nan)
            y_matrix.append(lcoh_row)
        y_matrix = np.array(y_matrix, dtype=float)

        for jcol, col in enumerate(ss.columns):
            yvals = y_matrix[:, jcol]
            valid = np.isfinite(yvals)
            if valid.any():
                ax.plot(
                    np.arange(len(ms_sorted))[valid], yvals[valid],
                    color=edge_cmap(norm(col)),
                    linewidth=0.8, zorder=2, alpha=0.4,
                )
        for i in range(len(ms_sorted)):
            for jcol, col in enumerate(ss.columns):
                if np.isfinite(y_matrix[i, jcol]):
                    ax.scatter(
                        i, y_matrix[i, jcol],
                        color=[edge_cmap(norm(col))],
                        edgecolor="k", s=40, marker="o",
                        linewidths=1, zorder=3,
                    )

        ax.set_xlim(-0.5, len(ms_sorted) - 0.5)
        ax.set_xticks(range(len(ms_sorted)))
        ax.set_xticklabels(ms_sorted.index, rotation=90, fontsize=7)
        ax.set_title(tech)
        ax.set_ylim(15, 100)
        ax.set_ylabel("LCOH [EUR/MWh]")
        ax.grid(True, axis="y", linestyle="--", alpha=0.7)

        panels_payload[tech] = {
            "color": tech_colors.get(tech),
            "lcoh_EUR_per_MWh": {
                str(ms_sorted.index[i]): {
                    int(c): (float(y_matrix[i, jcol])
                             if np.isfinite(y_matrix[i, jcol]) else None)
                    for jcol, c in enumerate(ss.columns)
                }
                for i in range(len(ms_sorted))
            },
        }

    sm = plt.cm.ScalarMappable(cmap=edge_cmap, norm=norm)
    sm.set_array([])
    divider = make_axes_locatable(axes_row[-1])
    cax = divider.append_axes("right", size="3%", pad=0.2)
    cbar = plt.colorbar(sm, cax=cax, label="System Gas Consumption [bcm]")
    _cticks = np.linspace(ss.columns.min(), ss.columns.max(), 5).astype(int)
    cbar.set_ticks(_cticks)
    # Display gas consumption in bcm (10 TWh per bcm).
    cbar.set_ticklabels([str(int(t) // 10) for t in _cticks])

    plt.tight_layout()
    out_lcoh_local = SCRIPT_DIR / f"res-heat_{mode}_lcoh-technology-panels.pdf"
    out_lcoh_paper = SIBLING_IMGS / f"res-heat_{mode}_lcoh-technology-panels.pdf"
    fig_lcoh.savefig(out_lcoh_local, bbox_inches="tight")
    fig_lcoh.savefig(out_lcoh_paper, bbox_inches="tight")
    print(f"wrote {out_lcoh_paper}")

    export_frontend_data(f"res-heat_{mode}_lcoh-technology-panels", {
        "y_label": "LCOH [EUR/MWh]",
        "wiggles_TWh": [int(c) for c in ss.columns],
        "buses": list(ms_sorted.index),
        "panels": panels_payload,
    })


make_residential_heat_plot(MODE)
