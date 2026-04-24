"""
Duration curves of AC marginal prices across four gas-budget scenarios,
overlaid on a single axis. For each scenario a rectangle marks the portion
of the curve that sits inside the CCGT marginal-cost band (gas price /
CCGT efficiency). A horizontal red line along the bottom edge of each
rectangle anchors the percentage-of-time-in-band label.
"""

import pypsa
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import to_rgba
from matplotlib.patches import Rectangle
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
NETWORK_DIR = ROOT / "results" / "networks"
PREFIX = "base_s_50__3H-T-H-B-I-A-dist1_2030"
PAPER_DIR = ROOT.parent / "gas_resilience" / "imgs"
SCRIPT_DIR = Path(__file__).resolve().parent

WIGGLES = [0, 1000, 2000, 3000]
Y_LIM = (0, 300)


def network_path(scenario="free", wiggle=0, hike=None):
    hike_str = f"_{hike}" if hike is not None else ""
    return NETWORK_DIR / f"{PREFIX}_{scenario}_{wiggle}{hike_str}.nc"


def style_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)


def gas_electricity_share(n):
    """Share of AC supply produced from gas-consuming links (CCGT, OCGT, gas CHP, ...).

    Numerator: sum over snapshots and (gas -> AC) links of -p1 (MWh).
    Denominator: total positive AC injections from generators and links.
    """
    ac_buses = n.buses.index[n.buses.carrier == "AC"]
    gas_buses = n.buses.index[n.buses.carrier == "gas"]
    weights = n.snapshot_weightings.generators

    gas_to_ac = n.links[
        n.links.bus0.isin(gas_buses) & n.links.bus1.isin(ac_buses)
    ]
    if gas_to_ac.empty:
        return 0.0
    gas_gen = -(
        n.links_t.p1[gas_to_ac.index]
        .multiply(weights, axis=0).sum().sum()
    )

    ac_gens = n.generators[n.generators.bus.isin(ac_buses)]
    gen_sup = (
        n.generators_t.p[ac_gens.index].clip(lower=0)
        .multiply(weights, axis=0).sum().sum()
    )
    ac_inject = n.links[n.links.bus1.isin(ac_buses)]
    link_sup = (
        (-n.links_t.p1[ac_inject.index]).clip(lower=0)
        .multiply(weights, axis=0).sum().sum()
    )
    total = gen_sup + link_sup
    return gas_gen / total if total > 0 else 0.0


def curve_band_crossings(sorted_prices, band_lo, band_hi):
    """For a descending-sorted duration curve, return (idx_upper, idx_lower):
    the first index where the curve drops into the band (crosses band_hi)
    and the first index where it leaves below (crosses band_lo)."""
    N = len(sorted_prices)
    upper_mask = sorted_prices <= band_hi
    idx_upper = int(np.argmax(upper_mask)) if upper_mask.any() else N
    lower_mask = sorted_prices < band_lo
    idx_lower = int(np.argmax(lower_mask)) if lower_mask.any() else N
    if idx_lower < idx_upper:
        idx_lower = idx_upper
    return idx_upper, idx_lower


viridis_colors = cm.viridis(np.linspace(0.15, 0.85, len(WIGGLES)))

fig, ax = plt.subplots(figsize=(10, 5.5))

for w, color in zip(WIGGLES, viridis_colors):
    path = network_path(wiggle=w)
    print(f"Loading {path.name}")
    n = pypsa.Network(path)

    ac_buses = n.buses.index[n.buses.carrier == "AC"]
    prices = n.buses_t.marginal_price[ac_buses].values.ravel()
    sorted_prices = np.sort(prices)[::-1]
    N = len(sorted_prices)

    ax.scatter(
        np.arange(N), sorted_prices,
        s=1, alpha=0.5, color=color, rasterized=True,
    )

    # CCGT marginal-cost band (gas price range / CCGT efficiency)
    gas_buses = n.buses.index[n.buses.carrier == "gas"]
    gas_bus_prices = n.buses_t.marginal_price[gas_buses].values.ravel()
    ccgt_eff = n.links.loc[n.links.carrier == "CCGT", "efficiency"].mean()
    band_lo = gas_bus_prices.min() / ccgt_eff
    band_hi = gas_bus_prices.max() / ccgt_eff

    # Skip the rectangle when the CCGT band sits entirely outside the y-range
    if band_hi < Y_LIM[0] or band_lo > Y_LIM[1]:
        print(f"  band {band_lo:.1f}-{band_hi:.1f} outside y-range; skipping rectangle")
        continue

    # Rectangle: x-bounded by where the duration curve enters and leaves the band
    idx_upper, idx_lower = curve_band_crossings(sorted_prices, band_lo, band_hi)
    rect = Rectangle(
        (idx_upper, band_lo),
        idx_lower - idx_upper,
        band_hi - band_lo,
        facecolor=to_rgba(color, 0.22),
        edgecolor="black",
        linewidth=0.7,
        zorder=3,
    )
    ax.add_patch(rect)

    # Red marker: horizontal line along the bottom edge of the rectangle
    ax.hlines(
        band_lo, idx_upper, idx_lower,
        color="red", linewidth=2.0, zorder=5,
    )

    # Label placement: bottom-left of rectangle, except bottom-right for w == 0
    frac = (idx_lower - idx_upper) / N * 100
    gas_share = gas_electricity_share(n) * 100
    if w == 0:
        label_xy = (idx_lower, band_lo)
        label_offset = (6, -3)
        label_ha = "left"
    else:
        label_xy = (idx_upper, band_lo)
        label_offset = (-6, -3)
        label_ha = "right"
    ax.annotate(
        f"{w} TWh\n{frac:.1f}%\n{gas_share:.1f}%",
        xy=label_xy,
        xytext=label_offset,
        textcoords="offset points",
        fontsize=8, ha=label_ha, va="top",
        color="black",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                  edgecolor="red", lw=0.8),
        zorder=6,
    )

ax.set_xlabel("Duration")
ax.set_ylabel("AC marginal price [EUR/MWh]")
ax.set_xticks([])
ax.set_ylim(*Y_LIM)
style_ax(ax)

# Top-right key explaining the three rows inside each red-bordered label
ax.text(
    0.99, 0.98,
    "Gas Consumption\nShare of Time where Marginal Price could be set by Gas\nGas Share in Electricity Mix",
    transform=ax.transAxes,
    ha="right", va="top",
    fontsize=8,
    bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
              edgecolor="red", lw=0.8),
    zorder=6,
)

fig.tight_layout()

PAPER_DIR.mkdir(parents=True, exist_ok=True)
for out in (SCRIPT_DIR / "price_duration.pdf", PAPER_DIR / "price_duration.pdf"):
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved {out}")
