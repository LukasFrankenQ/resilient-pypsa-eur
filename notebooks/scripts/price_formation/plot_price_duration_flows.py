"""
Single-scenario (wiggle=500) AC price duration curve, shown three times
side-by-side with the scatter point size carrying a different signal per panel:

    1. AC demand-like withdrawal at the bus/snapshot
       (positive AC withdrawal minus anything that charges a Store/StorageUnit).
    2. Collective charging of Store/StorageUnit components whose bus is the AC
       bus itself or the co-located " low voltage" bus.
    3. Collective discharging of the same components.

The duration curve backdrop (CCGT gas band, price-setting reference lines,
bold wiggle label) mirrors plot_price_duration.py.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pypsa
import matplotlib.pyplot as plt
from matplotlib import cm

ROOT = Path(__file__).resolve().parents[3]
NETWORK_DIR = ROOT / "results" / "networks"
PREFIX = "base_s_50__3H-T-H-B-I-A-dist1_2030"
PAPER_DIR = ROOT.parent / "gas_resilience" / "imgs"
SCRIPT_DIR = Path(__file__).resolve().parent

WIGGLES = [500, 1500, 2500, 3500]
Y_LIM = (0, 200)


def network_path(scenario="free", wiggle=0, hike=None):
    hike_str = f"_{hike}" if hike is not None else ""
    return NETWORK_DIR / f"{PREFIX}_{scenario}_{wiggle}{hike_str}.nc"


def style_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)


WIGGLE = 500
MAX_SIZE = 90.0   # points^2 — upper bound of the scatter size
MIN_SIZE = 0.02   # points^2 — floor so nothing truly disappears
SIZE_CAP_QUANTILE = 0.9  # percentile of nonzero weights that saturates at MAX_SIZE
SIZE_EXPONENT = 1.4      # >1 amplifies contrast between small and large points


# --- Weight-matrix builders -------------------------------------------------

def _positive_link_withdrawal_at_ac(n, ac_buses):
    """Per (snapshot, AC bus) positive Link withdrawal (MW).

    Any Link with bus0 ∈ ac_buses contributes its clipped-positive p0 to its
    bus0 column.
    """
    consuming = n.links[n.links.bus0.isin(ac_buses)]
    present = consuming.index.intersection(n.links_t.p0.columns)
    out = pd.DataFrame(0.0, index=n.snapshots, columns=ac_buses)
    if len(present) == 0:
        return out
    p0 = n.links_t.p0[present].clip(lower=0)
    bus0 = consuming.loc[present, "bus0"]
    # groupby columns via bus0 mapping, then align to ac_buses
    grouped = p0.T.groupby(bus0).sum().T
    for col in grouped.columns.intersection(ac_buses):
        out[col] = out[col] + grouped[col]
    return out


def _direct_loads_at_ac(n, ac_buses):
    """Per (snapshot, AC bus) MW from direct Load components (usually 0)."""
    out = pd.DataFrame(0.0, index=n.snapshots, columns=ac_buses)
    loads = n.loads[n.loads.bus.isin(ac_buses)]
    for name, row in loads.iterrows():
        if name in n.loads_t.p_set.columns:
            series = n.loads_t.p_set[name]
        else:
            series = pd.Series(float(row.p_set), index=n.snapshots)
        out[row.bus] = out[row.bus] + series
    return out


def _storage_unit_charge_at_ac(n, ac_buses):
    """StorageUnit p_store at AC buses, bucketed into the ac_buses column."""
    out = pd.DataFrame(0.0, index=n.snapshots, columns=ac_buses)
    sus = n.storage_units[n.storage_units.bus.isin(ac_buses)]
    present = sus.index.intersection(n.storage_units_t.p_store.columns)
    if len(present) == 0:
        return out
    p_store = n.storage_units_t.p_store[present].clip(lower=0)
    bus = sus.loc[present, "bus"]
    grouped = p_store.T.groupby(bus).sum().T
    for col in grouped.columns.intersection(ac_buses):
        out[col] = out[col] + grouped[col]
    return out


def _store_charger_links_withdrawal(n, ac_buses):
    """Per (snapshot, AC bus) MW of Link withdrawal whose bus1 is a Store bus.

    A Link with bus0 ∈ ac_buses and bus1 ∈ n.stores.bus.values is, by
    construction, a component that 'loads' a Store from the AC side (the
    canonical example being the battery charger).
    """
    out = pd.DataFrame(0.0, index=n.snapshots, columns=ac_buses)
    store_buses = set(n.stores.bus.values)
    links = n.links[
        n.links.bus0.isin(ac_buses) & n.links.bus1.isin(store_buses)
    ]
    present = links.index.intersection(n.links_t.p0.columns)
    if len(present) == 0:
        return out
    p0 = n.links_t.p0[present].clip(lower=0)
    bus0 = links.loc[present, "bus0"]
    grouped = p0.T.groupby(bus0).sum().T
    for col in grouped.columns.intersection(ac_buses):
        out[col] = out[col] + grouped[col]
    return out


def build_w_demand(n, ac_buses):
    """AC demand-like withdrawal per (snapshot, AC bus) in MW.

    = (all positive Link withdrawal from bus0 ∈ AC) + (direct AC Loads)
      − (Link withdrawal that flows into a Store bus)
      − (StorageUnit p_store at AC).
    """
    link_w = _positive_link_withdrawal_at_ac(n, ac_buses)
    direct = _direct_loads_at_ac(n, ac_buses)
    charger_links = _store_charger_links_withdrawal(n, ac_buses)
    su_charge = _storage_unit_charge_at_ac(n, ac_buses)
    w = link_w + direct - charger_links - su_charge
    return w.clip(lower=0)


def _colocated_buses(ac_bus, all_buses):
    lv = f"{ac_bus} low voltage"
    if lv in all_buses:
        return [ac_bus, lv]
    return [ac_bus]


def _bucket_into_ac(series_by_bus, ac_bus_for_each_bus):
    """Sum a (snapshot, bus) DataFrame into AC-bus columns via a bus -> AC mapping."""
    grouped = series_by_bus.groupby(ac_bus_for_each_bus, axis=1).sum()
    return grouped


def build_w_charge_discharge(n, ac_buses):
    """Return (w_charge, w_discharge) DataFrames indexed (snapshots, ac_buses) in MW.

    For each AC bus `b`, sums over Store/StorageUnit components whose bus is
    `b` or `f"{b} low voltage"`.

    Sign convention:
        StorageUnit: p_store = charging (>=0), p_dispatch = discharging (>=0).
        Store: n.stores_t.p follows the 'p > 0 is out of the store' convention,
               so charging = -p clipped at 0, discharging = p clipped at 0.
    """
    all_buses = set(n.buses.index)
    # Map every relevant component-bus back to its AC location.
    bus_to_ac = {}
    for ac in ac_buses:
        for b in _colocated_buses(ac, all_buses):
            bus_to_ac[b] = ac

    # --- StorageUnits ---
    sus = n.storage_units[n.storage_units.bus.isin(bus_to_ac)]
    su_present = sus.index.intersection(n.storage_units_t.p_store.columns)
    su_bus_map = sus.loc[su_present, "bus"].map(bus_to_ac)

    if len(su_present):
        su_charge = n.storage_units_t.p_store[su_present].clip(lower=0)
        su_discharge = n.storage_units_t.p_dispatch[su_present].clip(lower=0)
        su_charge_ac = su_charge.T.groupby(su_bus_map).sum().T
        su_discharge_ac = su_discharge.T.groupby(su_bus_map).sum().T
    else:
        su_charge_ac = pd.DataFrame(0.0, index=n.snapshots, columns=[])
        su_discharge_ac = pd.DataFrame(0.0, index=n.snapshots, columns=[])

    # --- Stores ---
    stores = n.stores[n.stores.bus.isin(bus_to_ac)]
    st_present = stores.index.intersection(n.stores_t.p.columns)
    st_bus_map = stores.loc[st_present, "bus"].map(bus_to_ac)

    if len(st_present):
        # p > 0 is discharge (flow out of store toward bus), p < 0 is charge.
        st_p = n.stores_t.p[st_present]
        st_charge = (-st_p).clip(lower=0)
        st_discharge = st_p.clip(lower=0)
        st_charge_ac = st_charge.T.groupby(st_bus_map).sum().T
        st_discharge_ac = st_discharge.T.groupby(st_bus_map).sum().T
    else:
        st_charge_ac = pd.DataFrame(0.0, index=n.snapshots, columns=[])
        st_discharge_ac = pd.DataFrame(0.0, index=n.snapshots, columns=[])

    w_charge = pd.DataFrame(0.0, index=n.snapshots, columns=ac_buses)
    w_discharge = pd.DataFrame(0.0, index=n.snapshots, columns=ac_buses)
    for col in su_charge_ac.columns.intersection(ac_buses):
        w_charge[col] = w_charge[col] + su_charge_ac[col]
    for col in st_charge_ac.columns.intersection(ac_buses):
        w_charge[col] = w_charge[col] + st_charge_ac[col]
    for col in su_discharge_ac.columns.intersection(ac_buses):
        w_discharge[col] = w_discharge[col] + su_discharge_ac[col]
    for col in st_discharge_ac.columns.intersection(ac_buses):
        w_discharge[col] = w_discharge[col] + st_discharge_ac[col]

    return w_charge, w_discharge


# --- Size scaling -----------------------------------------------------------

def scale_sizes(w_flat, max_size=MAX_SIZE, min_size=MIN_SIZE):
    """Power-scale to the SIZE_CAP_QUANTILE of non-zero weights."""
    w = np.clip(w_flat.astype(float), 0, None)
    positive = w[w > 0]
    cap = np.quantile(positive, SIZE_CAP_QUANTILE) if positive.size else 1.0
    if cap <= 0:
        cap = 1.0
    s = np.power(w / cap, SIZE_EXPONENT) * max_size
    return np.clip(s, min_size, max_size), cap


# --- Main -------------------------------------------------------------------

def main():
    path = network_path(wiggle=WIGGLE)
    print(f"Loading {path.name}")
    n = pypsa.Network(path)

    ac_buses = n.buses.index[n.buses.carrier == "AC"]
    price_df = n.buses_t.marginal_price[ac_buses]

    print("Building weight matrices...")
    w_demand = build_w_demand(n, ac_buses)
    w_charge, w_discharge = build_w_charge_discharge(n, ac_buses)

    # Sanity diagnostics
    print(f"  w_demand    : mean={w_demand.values.mean():.1f} MW, "
          f"max={w_demand.values.max():.1f} MW")
    print(f"  w_charge    : mean={w_charge.values.mean():.1f} MW, "
          f"max={w_charge.values.max():.1f} MW")
    print(f"  w_discharge : mean={w_discharge.values.mean():.1f} MW, "
          f"max={w_discharge.values.max():.1f} MW")

    # Flatten in lock-step (C order: iterate snapshot-major, then buses).
    flat_price = price_df.values.ravel()
    order = np.argsort(-flat_price)
    x = np.arange(flat_price.size)
    y = flat_price[order]

    flat = {
        "demand": w_demand.values.ravel()[order],
        "charge": w_charge.values.ravel()[order],
        "discharge": w_discharge.values.ravel()[order],
    }

    # Match the viridis shade used in the original 2x2 plot when possible;
    # otherwise fall back to the darkest end (most-constrained end of the sweep).
    viridis_colors = cm.viridis(np.linspace(0.15, 0.85, len(WIGGLES)))
    if WIGGLE in WIGGLES:
        scatter_color = viridis_colors[WIGGLES.index(WIGGLE)]
    else:
        scatter_color = cm.viridis(0.15)

    # Figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5), sharex=True, sharey=True)

    panels = [
        ("demand", "Scaled by AC demand-like withdrawal"),
        ("charge", "Scaled by storage charging (AC + low voltage)"),
        ("discharge", "Scaled by storage discharging (AC + low voltage)"),
    ]

    # CCGT band (network-level, identical across panels)
    gas_buses = n.buses.index[n.buses.carrier == "gas"]
    gas_prices = n.buses_t.marginal_price[gas_buses].values.ravel()
    ccgt_eff = n.links.loc[n.links.carrier == "CCGT", "efficiency"].mean()
    band_lo = gas_prices.min() / ccgt_eff
    band_hi = gas_prices.max() / ccgt_eff

    weights = n.snapshot_weightings.generators
    in_band = ((price_df >= band_lo) & (price_df <= band_hi)).astype(float)
    frac = (in_band.multiply(weights, axis=0).sum().sum()
            / (weights.sum() * len(ac_buses)))

    for idx, (ax, (key, title)) in enumerate(zip(axes, panels)):
        sizes, _cap = scale_sizes(flat[key])
        ax.scatter(
            x, y,
            s=sizes,
            alpha=0.55,
            color=scatter_color,
            rasterized=True,
            edgecolors="none",
        )

        # CCGT band
        ax.axhspan(band_lo, band_hi, color=scatter_color, alpha=0.15, linewidth=0)
        if idx == 0:
            ax.text(
                0.5, (band_lo + band_hi) / 2,
                f"{frac * 100:.1f}% of time in CCGT band",
                transform=ax.get_yaxis_transform(),
                ha="center", va="center", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                          edgecolor="none", alpha=0.7),
            )

        # Wiggle label on leftmost panel only
        if idx == 0:
            ax.text(
                0.98, 0.04, f"{WIGGLE} TWh",
                transform=ax.transAxes,
                ha="right", va="bottom",
                fontsize=11, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                          edgecolor="black", lw=0.8),
            )

        ax.set_title(title, fontsize=10)
        ax.set_ylim(*Y_LIM)
        ax.set_xticks([])
        ax.set_xlabel("Duration")
        style_ax(ax)

    axes[0].set_ylabel("AC marginal price [EUR/MWh]")

    fig.tight_layout()

    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    fname = "price_duration_flows.pdf"
    for out in (SCRIPT_DIR / fname, PAPER_DIR / fname):
        fig.savefig(out, bbox_inches="tight")
        print(f"Saved {out}")


if __name__ == "__main__":
    main()
