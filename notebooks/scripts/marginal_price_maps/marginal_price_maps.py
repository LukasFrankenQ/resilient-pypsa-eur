"""
2x4 grid of choropleth maps showing the load-weighted average marginal price
per network region for eight commodities in the
base_s_50_lv1.25_3H-T-H-B-I-A-dist1_2030_free_2000 network.

Top row: electricity (own scale) + building heat group (urban individual /
rural / district) sharing a single magma colorbar.
Bottom row: solid biomass, biogas, industry heat <100, industry heat >500 —
each with its own cmap and vmin/vmax.

Projection: EPSG:3035 (ETRS89 / LAEA Europe), equal-area, so Scandinavia
looks the right size.
"""

from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pypsa
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

ROOT = Path(__file__).resolve().parents[3]
NETWORK_DIR = ROOT / "results" / "networks"
PREFIX = "base_s_50_lv1.25_3H-T-H-B-I-A-dist1_2030"
NETWORK_PATH = NETWORK_DIR / f"{PREFIX}_free_2000.nc"
REGIONS_PATH = ROOT / "resources" / "regions_onshore_base_s_50.geojson"
LOCAL_OUT = Path(__file__).parent / "marginal_price_maps.pdf"
SHARED_OUT = ROOT.parent / "gas_resilience" / "imgs" / "marginal_price_maps.pdf"

GROUP_CMAPS = {"building_heat": "magma"}
GROUP_LABELS = {"building_heat": "Building Heating"}

# (bus_carrier, frontend name, cmap, group_or_None) — row-major 2x4.
# group=None  -> standalone (own vmin/vmax, own colorbar)
# group=str   -> shared cmap + shared vmin/vmax + one colorbar for that group
COMMODITIES = [
    ("low voltage",          "Electricity (Low Voltage)", "viridis", None),
    ("urban decentral heat", "Urban Individual Heating",  "magma",   "building_heat"),
    ("rural heat",           "Rural Heating",             "magma",   "building_heat"),
    ("urban central heat",   "District Heating",          "magma",   "building_heat"),
    ("solid biomass",        "Solid Biomass",             "YlGn",    None),
    ("biogas",               "Biogas",                    "BuGn",    None),
    ("heat<100 industry",    "Industry Heat <100 °C",     "Oranges", None),
    ("heat>500 industry",    "Industry Heat >500 °C",     "Reds",    None),
]


def load_weighted_price_by_location(n: pypsa.Network, bus_carrier: str) -> pd.Series:
    """Per-location load-weighted average marginal price (EUR/MWh).

    Loads at the carrier buses are the primary weight. If no Load components
    exist for the carrier (typical for H2, biogas, solid biomass), falls back
    to link withdrawals (p0 at the bus).
    """
    carrier_buses = n.buses.index[n.buses.carrier == bus_carrier]
    if len(carrier_buses) == 0:
        return pd.Series(dtype=float)

    prices = n.buses_t.marginal_price[carrier_buses]
    weights_t = n.snapshot_weightings.generators

    load_ts = pd.DataFrame(0.0, index=n.snapshots, columns=carrier_buses)
    loads_at_buses = n.loads[n.loads.bus.isin(carrier_buses)]

    if not loads_at_buses.empty:
        for load_name, load_row in loads_at_buses.iterrows():
            bus = load_row.bus
            if load_name in n.loads_t.p_set.columns:
                load_ts[bus] += n.loads_t.p_set[load_name]
            elif load_name in n.loads_t.p.columns:
                load_ts[bus] += n.loads_t.p[load_name]
            else:
                load_ts[bus] += load_row.p_set
    else:
        consuming_links = n.links[n.links.bus0.isin(carrier_buses)]
        for link_name, link_row in consuming_links.iterrows():
            bus = link_row.bus0
            if link_name in n.links_t.p0.columns:
                load_ts[bus] += n.links_t.p0[link_name]

    # Per-bus numerator/denominator, then aggregate to location.
    weighted_price_bus = (prices * load_ts).multiply(weights_t, axis=0).sum()
    weighted_load_bus = load_ts.multiply(weights_t, axis=0).sum()

    locations = n.buses.loc[carrier_buses, "location"]
    num = weighted_price_bus.groupby(locations).sum()
    den = weighted_load_bus.groupby(locations).sum()

    price_per_loc = num / den.replace(0, np.nan)
    price_per_loc.index.name = "location"
    return price_per_loc.dropna()


def style_map_ax(ax):
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])


def values_for_carrier(n, bus_carrier, regions):
    """Resolve a per-region value series for one carrier, handling
    empty / single-EU-bus cases."""
    price = load_weighted_price_by_location(n, bus_carrier)
    if price.empty:
        return None
    if price.size == 1 and price.index[0] not in regions.index:
        scalar = float(price.iloc[0])
        print(
            f"  [{bus_carrier}] single global bus at "
            f"{price.index[0]} — broadcasting {scalar:.1f} EUR/MWh"
        )
        return pd.Series(scalar, index=regions.index, name=bus_carrier)
    return price.reindex(regions.index).rename(bus_carrier)


def _safe_lims(series_or_list):
    if isinstance(series_or_list, list):
        combined = pd.concat(series_or_list)
    else:
        combined = series_or_list
    vmin = float(combined.min(skipna=True))
    vmax = float(combined.max(skipna=True))
    if vmin == vmax:
        vmin -= 0.5
        vmax += 0.5
    return vmin, vmax


def main():
    print(f"Loading network: {NETWORK_PATH.name}")
    n = pypsa.Network(NETWORK_PATH)

    print(f"Loading regions: {REGIONS_PATH.name}")
    regions = gpd.read_file(REGIONS_PATH).set_index("name")
    # EPSG:3035 (LAEA Europe): equal-area, undistorts Scandinavia
    regions = regions.to_crs(epsg=3035)
    regions["geometry"] = regions.geometry.simplify(10000, preserve_topology=True)
    outer_boundary = gpd.GeoSeries([regions.union_all()], crs=regions.crs).boundary

    panels = []
    for bus_carrier, label, cmap, group in COMMODITIES:
        values = values_for_carrier(n, bus_carrier, regions)
        panels.append((bus_carrier, label, cmap, group, values))

    # Group vmin/vmax (only for grouped panels).
    group_lims = {}
    for group_name in GROUP_CMAPS:
        group_vals = [v for (_, _, _, g, v) in panels if g == group_name and v is not None]
        if group_vals:
            group_lims[group_name] = _safe_lims(group_vals)

    fig, axes = plt.subplots(2, 4, figsize=(20, 11), constrained_layout=True)
    group_axes: dict[str, list] = {g: [] for g in GROUP_CMAPS}

    for ax, (bus_carrier, label, cmap, group, values) in zip(axes.flat, panels):
        if group is not None:
            group_axes[group].append(ax)

        if values is None:
            print(f"  [{bus_carrier}] no buses found — blank panel")
            style_map_ax(ax)
            ax.set_title(f"{label}\n(no data)", fontsize=11)
            continue

        if group is None:
            vmin, vmax = _safe_lims(values)
        else:
            vmin, vmax = group_lims[group]

        regions_plot = regions.copy()
        regions_plot["value"] = values
        regions_plot.plot(
            column="value",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            ax=ax,
            edgecolor="black",
            linewidth=0.15,
            missing_kwds={"color": "lightgrey", "edgecolor": "black", "linewidth": 0.15},
        )
        outer_boundary.plot(ax=ax, color="black", linewidth=0.6)

        style_map_ax(ax)
        ax.set_title(label, fontsize=12)

        # Per-panel colorbar for standalone panels only; grouped panels share
        # one colorbar attached after the loop.
        if group is None:
            sm = ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax), cmap=cmap)
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=ax, shrink=0.85, pad=0.02)
            cbar.set_label("EUR/MWh", fontsize=9)
            cbar.ax.tick_params(labelsize=8)

    # Shared colorbar(s) for grouped panels.
    for group_name, ax_list in group_axes.items():
        if group_name not in group_lims or not ax_list:
            continue
        cmap_name = GROUP_CMAPS[group_name]
        vmin, vmax = group_lims[group_name]
        sm = ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax), cmap=cmap_name)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax_list, shrink=0.85, pad=0.02)
        cbar.set_label(f"{GROUP_LABELS[group_name]} [EUR/MWh]", fontsize=10)
        cbar.ax.tick_params(labelsize=8)

    fig.suptitle(
        "Regional Commodity Prices for 2000 TWh Gas Consumption",
        fontsize=15,
    )

    LOCAL_OUT.parent.mkdir(parents=True, exist_ok=True)
    SHARED_OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(LOCAL_OUT, bbox_inches="tight")
    fig.savefig(SHARED_OUT, bbox_inches="tight")
    print(f"Saved: {LOCAL_OUT}")
    print(f"Saved: {SHARED_OUT}")


if __name__ == "__main__":
    main()
