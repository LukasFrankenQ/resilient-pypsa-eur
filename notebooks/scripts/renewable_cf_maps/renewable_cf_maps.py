"""
Two-row figure for a 3H pre-/post-solve network.

Row 1: 1x3 choropleth of the time-weighted average capacity factor per region
       for solar, onwind, offwind-ac (solar white->gold, wind white->blue).
Row 2: 1x3 choropleth of the time-weighted average air-source heat-pump COP
       per region for urban decentral, urban central, and rural air HPs
       (shared white->red colormap, colorbar matched in width to the top row).

Projection: EPSG:3035 (LAEA Europe).
"""

import textwrap
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pypsa
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize

ROOT = Path(__file__).resolve().parents[3]
NETWORK_DIR = ROOT / "results" / "networks"
PREFIX = "base_s_50__3H-T-H-B-I-A-dist1_2030"
NETWORK_PATH = NETWORK_DIR / f"{PREFIX}_free_1000.nc"
REGIONS_PATH = ROOT / "resources" / "regions_onshore_base_s_50.geojson"
LOCAL_OUT = Path(__file__).parent / "renewable_cf_maps.pdf"
SHARED_OUT = ROOT.parent / "gas_resilience" / "imgs" / "renewable_cf_maps.pdf"

CMAP_WIND = LinearSegmentedColormap.from_list("white_blue", ["#ffffff", "#08306b"])
CMAP_SOLAR = LinearSegmentedColormap.from_list("white_gold", ["#ffffff", "#b8860b"])
CMAP_COP = LinearSegmentedColormap.from_list("white_red", ["#ffffff", "#67000d"])

CARRIERS = [
    ("solar",      "Solar",         CMAP_SOLAR),
    ("onwind",     "Onshore Wind",  CMAP_WIND),
    ("offwind-ac", "Offshore Wind", CMAP_WIND),
]
HEAT_PUMPS = [
    ("urban decentral air heat pump", "Urban Individual Air HP"),
    ("urban central air heat pump",   "District Heat Air HP"),
    ("rural air heat pump",           "Rural Air HP"),
]


def capacity_factor_by_region(n: pypsa.Network, carrier: str) -> pd.Series:
    gens = n.generators[n.generators.carrier == carrier]
    if gens.empty:
        return pd.Series(dtype=float)
    cols = [g for g in gens.index if g in n.generators_t.p_max_pu.columns]
    if not cols:
        return pd.Series(dtype=float)
    p_max_pu = n.generators_t.p_max_pu[cols]
    weights = n.snapshot_weightings.generators
    cf_per_gen = p_max_pu.multiply(weights, axis=0).sum() / weights.sum()
    buses = gens.loc[cols, "bus"]
    return cf_per_gen.groupby(buses).mean()


def cop_by_region(n: pypsa.Network, carrier: str) -> pd.Series:
    links = n.links[n.links.carrier == carrier]
    if links.empty:
        return pd.Series(dtype=float)
    cols = [l for l in links.index if l in n.links_t.efficiency.columns]
    if not cols:
        return pd.Series(dtype=float)
    eff = n.links_t.efficiency[cols]
    weights = n.snapshot_weightings.generators
    cop_per_link = eff.multiply(weights, axis=0).sum() / weights.sum()
    heat_buses = links.loc[cols, "bus1"]
    locations = n.buses.loc[heat_buses.values, "location"].values
    out = pd.Series(cop_per_link.values, index=locations)
    return out.groupby(out.index).mean()


def style_map_ax(ax):
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])


def plot_choropleth(ax, regions, outer_boundary, values, cmap, vmin, vmax, label):
    regions_plot = regions.copy()
    regions_plot["value"] = values
    regions_plot.plot(
        column="value", cmap=cmap, vmin=vmin, vmax=vmax, ax=ax,
        edgecolor="black", linewidth=0.15,
        missing_kwds={"color": "lightgrey", "edgecolor": "black", "linewidth": 0.15},
    )
    outer_boundary.plot(ax=ax, color="black", linewidth=0.6)
    style_map_ax(ax)
    wrapped = textwrap.fill(label, width=16)
    ax.text(
        0.03, 0.97, wrapped, transform=ax.transAxes,
        ha="left", va="top", fontsize=12, fontweight="bold", color="#222222",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                  edgecolor="#444444", linewidth=0.8, alpha=0.92),
    )


def main():
    print(f"Loading network: {NETWORK_PATH.name}")
    n = pypsa.Network(NETWORK_PATH)

    print(f"Loading regions: {REGIONS_PATH.name}")
    regions = gpd.read_file(REGIONS_PATH).set_index("name")
    regions = regions.to_crs(epsg=3035)
    regions["geometry"] = regions.geometry.simplify(10000, preserve_topology=True)
    outer_boundary = gpd.GeoSeries([regions.union_all()], crs=regions.crs).boundary

    cf_panels = []
    for carrier, label, cmap in CARRIERS:
        vals = capacity_factor_by_region(n, carrier).reindex(regions.index)
        print(f"  [CF {carrier}] mean={vals.mean():.3f}  range={vals.min():.3f}-{vals.max():.3f}")
        cf_panels.append((label, cmap, vals))

    cop_panels = []
    for carrier, label in HEAT_PUMPS:
        vals = cop_by_region(n, carrier).reindex(regions.index)
        print(f"  [COP {carrier}] mean={vals.mean():.3f}  range={vals.min():.3f}-{vals.max():.3f}")
        cop_panels.append((label, vals))

    cop_combined = pd.concat([v for _, v in cop_panels])
    cop_vmin = float(np.nanmin(cop_combined))
    cop_vmax = float(np.nanmax(cop_combined))

    fig = plt.figure(figsize=(18, 13), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.0])

    # === Row 1: CF maps ===
    for col, (label, cmap, values) in enumerate(cf_panels):
        ax = fig.add_subplot(gs[0, col])
        if values.dropna().empty:
            style_map_ax(ax)
            ax.text(
                0.03, 0.97, f"{textwrap.fill(label, width=16)}\n(no data)",
                transform=ax.transAxes, ha="left", va="top",
                fontsize=12, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                          edgecolor="#444444", linewidth=0.8, alpha=0.92),
            )
            continue
        vmin = float(np.nanmin(values)); vmax = float(np.nanmax(values))
        if vmin == vmax:
            vmin -= 0.01; vmax += 0.01
        plot_choropleth(ax, regions, outer_boundary, values, cmap, vmin, vmax, label)
        sm = ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax), cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, shrink=0.7, pad=0.02, orientation="horizontal")
        cbar.set_label("Capacity factor", fontsize=10)
        cbar.ax.tick_params(labelsize=9)

    # === Row 2: COP maps with a single colorbar matched to top-row width ===
    cop_axes = []
    for col, (label, values) in enumerate(cop_panels):
        ax = fig.add_subplot(gs[1, col])
        cop_axes.append(ax)
        if values.dropna().empty:
            style_map_ax(ax)
            ax.text(
                0.03, 0.97, f"{textwrap.fill(label, width=16)}\n(no data)",
                transform=ax.transAxes, ha="left", va="top",
                fontsize=12, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                          edgecolor="#444444", linewidth=0.8, alpha=0.92),
            )
            continue
        plot_choropleth(ax, regions, outer_boundary, values, CMAP_COP,
                        cop_vmin, cop_vmax, label)
    sm = ScalarMappable(norm=Normalize(vmin=cop_vmin, vmax=cop_vmax), cmap=CMAP_COP)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=cop_axes[1], shrink=0.7, pad=0.02,
                        orientation="horizontal")
    cbar.set_label("Air heat-pump COP (annual mean)", fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    LOCAL_OUT.parent.mkdir(parents=True, exist_ok=True)
    SHARED_OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(LOCAL_OUT, bbox_inches="tight")
    fig.savefig(SHARED_OUT, bbox_inches="tight")
    print(f"Saved: {LOCAL_OUT}")
    print(f"Saved: {SHARED_OUT}")


if __name__ == "__main__":
    main()
