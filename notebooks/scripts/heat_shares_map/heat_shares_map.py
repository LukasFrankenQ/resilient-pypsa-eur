"""Map of the electrification of building heat as the gas budget tightens.

Across the gas-budget sweep (``wiggle`` = 4000 -> 3000 -> 2000 -> 1000 -> 0,
with 3H resolution, lv1.25 transmission, ``free`` scenario) the model shifts
heat supply from gas/oil boilers towards heat pumps and resistive heaters. This
script renders that shift spatially: a 1x5 row of simplified onshore-region
maps, one per wiggle value, each region colour-coded by the share of its total
heat supply that comes from ELECTRIC technologies (heat pumps + resistive
heaters), aggregated across the three heat-demand vectors the model resolves:

    * urban central     -> district heating
    * urban decentral   -> individual heating in urban areas
    * rural             -> individual heating in rural areas

For each region and each heat vector, ``n.statistics.energy_balance`` gives the
heat supplied per carrier; only positive flows (actual supply into the heat
bus) are kept. The electric share is

    electric supply / total supply

summed over the three vectors. See the heat-fuel-mapping memory for the
"boiler contains oil" classification gotcha.

Output: ``heat_shares_map.pdf`` next to this script and a copy in
``../gas_resilience/imgs`` per the repo plotting convention (PDF only).
"""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pypsa
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

# ── Paths ────────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = Path(__file__).resolve().parent
NETWORK_DIR = REPO_ROOT / "results" / "networks"
SIBLING_IMGS = REPO_ROOT.parent / "gas_resilience" / "imgs"
SIBLING_IMGS.mkdir(parents=True, exist_ok=True)
REGIONS = REPO_ROOT / "resources" / "regions_onshore_base_s_50.geojson"

PREFIX = "base_s_50_lv1.25_3H-T-H-B-I-A-dist1_2030"
WIGGLES = [4000, 3000, 2000, 1000, 0]
HEAT_CARRIERS = ["urban central heat", "urban decentral heat", "rural heat"]

# Map extent (EPSG:3035, metres) — continental Europe, trimming far islands.
EXTENT_3035 = (2.4e6, 6.0e6, 1.4e6, 5.4e6)
SIMPLIFY_TOL = 5000  # metres; "simplified" region outlines


def network_path(wiggle: int, scenario: str = "free") -> Path:
    return NETWORK_DIR / f"{PREFIX}_{scenario}_{wiggle}.nc"


def is_electric(carrier: str) -> bool:
    """Electric heat tech = heat pump or resistive heater (see heat-fuel memory)."""
    return "heat pump" in carrier or "resistive heater" in carrier


def electric_share_by_region(n: pypsa.Network) -> pd.Series:
    """Share of total heat supply met by electric tech, per onshore region.

    Sums positive heat supply across the three heat vectors; numerator is the
    electric (heat pump + resistive) part. Returns a Series indexed by node
    (e.g. ``AT0 0``), values in [0, 1].
    """
    elec = pd.Series(dtype=float)
    total = pd.Series(dtype=float)
    for hc in HEAT_CARRIERS:
        eb = n.statistics.energy_balance(
            bus_carrier=hc, groupby=["bus", "carrier"]
        ).reset_index()
        eb.columns = ["component", "bus", "carrier", "value"]
        eb = eb[eb.value > 0]  # supply only
        eb["node"] = eb.bus.str[: -len(hc)].str.strip()  # drop the heat suffix
        total = total.add(eb.groupby("node").value.sum(), fill_value=0.0)
        e = eb[eb.carrier.map(is_electric)]
        elec = elec.add(e.groupby("node").value.sum(), fill_value=0.0)
    elec = elec.reindex(total.index).fillna(0.0)
    return (elec / total).rename("electric_share")


def style_map(ax):
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])


def main() -> None:
    regions = gpd.read_file(REGIONS).set_index("name").to_crs(3035)
    regions["geometry"] = regions.geometry.simplify(SIMPLIFY_TOL)

    shares = {}
    for w in WIGGLES:
        path = network_path(w)
        if not path.exists():
            raise FileNotFoundError(f"network not found: {path}")
        print(f"loading {path.name}")
        n = pypsa.Network(path)
        shares[w] = electric_share_by_region(n)

    norm = Normalize(vmin=0.0, vmax=1.0)
    cmap = plt.get_cmap("YlGnBu")

    fig, axes = plt.subplots(1, len(WIGGLES), figsize=(4 * len(WIGGLES), 5))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.86, bottom=0.16, wspace=0.02)
    for ax, w in zip(axes, WIGGLES):
        gdf = regions.join(shares[w])
        gdf.plot(
            column="electric_share", cmap=cmap, norm=norm, ax=ax,
            edgecolor="white", linewidth=0.3,
            missing_kwds={"color": "lightgrey"},
        )
        ax.set_xlim(EXTENT_3035[0], EXTENT_3035[1])
        ax.set_ylim(EXTENT_3035[2], EXTENT_3035[3])
        ax.set_title(f"gas budget {w}", fontsize=12)
        mean_share = shares[w].mean()
        ax.text(0.5, -0.04, f"mean {mean_share:.0%}", transform=ax.transAxes,
                ha="center", va="top", fontsize=9, color="0.3")
        style_map(ax)

    fig.suptitle(
        "Electric share of building-heat supply tightens with the gas budget\n"
        "(heat pumps + resistive heaters, across urban central / urban decentral / rural)",
        fontsize=14, y=0.98,
    )

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cax = fig.add_axes([0.30, 0.08, 0.40, 0.025])  # [left, bottom, w, h]
    cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cbar.set_label("share of total heat supplied by electric options")
    cbar.set_ticks(np.linspace(0, 1, 6))
    cbar.ax.set_xticklabels([f"{t:.0%}" for t in np.linspace(0, 1, 6)])

    out = SCRIPT_DIR / "heat_shares_map.pdf"
    fig.savefig(out)
    fig.savefig(SIBLING_IMGS / "heat_shares_map.pdf")
    print(f"wrote {out}")
    print(f"wrote {SIBLING_IMGS / 'heat_shares_map.pdf'}")


if __name__ == "__main__":
    main()
