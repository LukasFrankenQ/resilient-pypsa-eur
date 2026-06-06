"""
Per-bus maximum urban-decentral-heat marginal price across two solved
PyPSA-Eur networks (lv1.25, 3H, dist1, 2030, free_0 vs free_4000).

For each urban-decentral-heat bus we pick the snapshot at which its marginal
price is highest, then mark the transmission electricity price (AC bus
marginal price), low-voltage electricity price (LV bus marginal price), and
the gas-boiler-implied heat cost (gas / 0.98) at that same snapshot.

Three axvlines flag the capital-cost benchmarks:
  - distribution-grid Link capital_cost
  - mean transmission Line capital_cost
  - sum of the two
The unit on the x-axis is EUR/MWh whereas capital_cost is EUR/MW/yr — the
visual comparison still makes sense under the framing "to deliver one extra
MWh you need one extra MW of grid".

Output: urban_decentral_price_per_bus_decomposition.pdf
"""

import pypsa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
NETWORK_DIR = ROOT / "results" / "networks"
SCRIPT_DIR = Path(__file__).resolve().parent

NETWORKS = {
    "free_0":    NETWORK_DIR / "base_s_50_lv1.25_3H-T-H-B-I-A-dist1_2030_free_0.nc",
    "free_4000": NETWORK_DIR / "base_s_50_lv1.25_3H-T-H-B-I-A-dist1_2030_free_4000.nc",
}

SCENARIO_COLORS = {"free_0": "#cc4125", "free_4000": "#3d85c6"}
SCENARIO_LABELS = {
    "free_0":    "free, total gas consumption = 0 TWh (constrained)",
    "free_4000": "free, total gas consumption = 4000 TWh (relaxed)",
}

GAS_BOILER_EFF = 0.98
SYMLOG_LIN_THRESH = 200.0


def style_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)


def build_per_bus_max(n):
    """For each urban-decentral-heat bus, return the snapshot of max heat
    marginal price and the AC / LV / gas prices at that snapshot.

    Returns
    -------
    out : DataFrame indexed by `loc`, columns:
        snapshot, heat_price, ac_price, lv_price, gas_price
    cap_costs : dict with `dist_cap_cost`, `trans_cap_cost_mean`, `sum`.
    """
    heat_buses = n.buses.index[n.buses.carrier == "urban decentral heat"]
    locs = [b.replace(" urban decentral heat", "") for b in heat_buses]
    locs = [
        loc for loc in locs
        if loc in n.buses.index
        and f"{loc} low voltage" in n.buses.index
        and f"{loc} gas" in n.buses.index
    ]
    heat_cols = [f"{l} urban decentral heat" for l in locs]
    lv_cols = [f"{l} low voltage" for l in locs]
    gas_cols = [f"{l} gas" for l in locs]

    heat_p = n.buses_t.marginal_price[heat_cols].copy()
    heat_p.columns = locs

    # snapshot of per-bus maximum
    max_snap = heat_p.idxmax(axis=0)  # Series: loc -> snapshot

    rows = []
    for loc, sn in max_snap.items():
        rows.append({
            "loc": loc,
            "snapshot": sn,
            "heat_price": heat_p.at[sn, loc],
            "ac_price":   n.buses_t.marginal_price.at[sn, loc],
            "lv_price":   n.buses_t.marginal_price.at[sn, f"{loc} low voltage"],
            "gas_price":  n.buses_t.marginal_price.at[sn, f"{loc} gas"],
        })
    out = pd.DataFrame(rows).set_index("loc")

    # capital costs
    dg = n.links[
        (n.links.carrier == "electricity distribution grid")
        & n.links.bus1.str.endswith(" low voltage")
    ]
    dist_cap_cost = float(dg["capital_cost"].mean())  # uniform anyway

    trans_cap_cost_mean = float(n.lines["capital_cost"].mean())
    cap_costs = {
        "dist_cap_cost": dist_cap_cost,
        "trans_cap_cost_mean": trans_cap_cost_mean,
        "sum": dist_cap_cost + trans_cap_cost_mean,
    }
    return out, cap_costs


def plot_per_bus_decomposition(per_max, cap_costs, save):
    """Two-panel horizontal bar chart, one panel per scenario.  Each row is
    one urban-decentral-heat bus, sorted by per-bus max heat price.  Markers
    show transmission / low-voltage electricity price and gas-boiler-implied
    heat cost at the same snapshot.  Three axvlines mark capital costs."""
    fig, axes = plt.subplots(
        1, 2, figsize=(15, 0.20 * 50 + 2.5), sharey=False
    )
    for ax, scen in zip(axes, NETWORKS.keys()):
        sub = per_max[scen].sort_values("heat_price", ascending=False)
        sub = sub.assign(gas_implied=sub["gas_price"] / GAS_BOILER_EFF)
        y = np.arange(len(sub))[::-1]
        col = SCENARIO_COLORS[scen]

        ax.barh(y, sub["heat_price"], color=col, alpha=0.25, height=0.8,
                edgecolor="none", label="heat price (max)")

        ax.scatter(sub["ac_price"], y, marker="o", s=30, color="#4a4a4a",
                   label="transmission electricity price", zorder=3)
        ax.scatter(sub["lv_price"], y, marker="s", s=30, color="#888888",
                   label="low-voltage electricity price", zorder=3)
        ax.scatter(sub["gas_implied"], y, marker="X", s=35, color="#d95f02",
                   label="gas / 0.98 (gas-boiler-implied heat cost)", zorder=3)

        cc = cap_costs[scen]
        ax.axvline(cc["dist_cap_cost"], color="#1f77b4", lw=1.2, ls="--",
                   label=f"dist-grid capital_cost ({cc['dist_cap_cost']:,.0f})")
        ax.axvline(cc["trans_cap_cost_mean"], color="#9467bd", lw=1.2, ls="--",
                   label=f"transmission capital_cost (mean, "
                         f"{cc['trans_cap_cost_mean']:,.0f})")
        ax.axvline(cc["sum"], color="#000000", lw=1.4, ls="-",
                   label=f"sum of the two ({cc['sum']:,.0f})")

        labels = [
            f"{loc}  ({ts.strftime('%b-%d %H:%M')})"
            for loc, ts in zip(sub.index, sub["snapshot"])
        ]
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=7)
        ax.set_xlabel("price / cost  [EUR/MWh; vlines in EUR/MW/yr]   (symlog)")
        ax.set_xscale("symlog", linthresh=SYMLOG_LIN_THRESH)
        ax.set_xlim(left=0)
        ax.set_title(SCENARIO_LABELS[scen], color=col, loc="left", fontsize=10)
        style_ax(ax)
        ax.legend(frameon=True, fontsize=7, loc="lower right")

    fig.suptitle(
        "Per-bus max urban-decentral-heat price (one row per node).\n"
        "Bars = heat price at the bus's worst snapshot; markers = supply-option costs at that same snapshot; "
        "axvlines = grid capital_cost benchmarks (unit hack: 1 MWh dispatched ↔ 1 MW of grid).",
        fontsize=10, y=1.005,
    )
    fig.tight_layout()
    fig.savefig(save, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {save}")


def main():
    print("Loading networks...")
    per_max = {}
    cap_costs = {}
    for scen, path in NETWORKS.items():
        print(f"  {scen}: {path.name}")
        n = pypsa.Network(str(path))
        out, cc = build_per_bus_max(n)
        per_max[scen] = out
        cap_costs[scen] = cc
        print(f"    nodes={len(out)}  "
              f"min-of-max={out['heat_price'].min():.1f}  "
              f"median-of-max={out['heat_price'].median():.1f}  "
              f"max-of-max={out['heat_price'].max():.1f} EUR/MWh")
        print(f"    capital_cost benchmarks: "
              f"dist={cc['dist_cap_cost']:,.0f}  "
              f"trans-mean={cc['trans_cap_cost_mean']:,.0f}  "
              f"sum={cc['sum']:,.0f}  EUR/MW/yr")

    print("\nWriting plot...")
    plot_per_bus_decomposition(
        per_max, cap_costs,
        SCRIPT_DIR / "urban_decentral_price_per_bus_decomposition.pdf",
    )
    print("Done.")


if __name__ == "__main__":
    main()
