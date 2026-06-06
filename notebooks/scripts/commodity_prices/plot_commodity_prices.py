"""
Cleveland dot plot comparing commodity prices and system costs
between regular and gas-price-hike (+50 EUR/MWh) model runs
across three gas budget (wiggle) levels.
"""

import pypsa
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
NETWORK_DIR = ROOT / "results" / "networks"
PREFIX = "base_s_50__3H-T-H-B-I-A-dist1_2030"
SAVE_DIR = ROOT.parent / "gas_resilience" / "imgs"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

WIGGLES = [500, 2000, 3500]


def network_path(scenario="free", wiggle=0, hike=None):
    hike_str = f"_{hike}" if hike is not None else ""
    return NETWORK_DIR / f"{PREFIX}_{scenario}_{wiggle}{hike_str}.nc"


def carrier_price_weighted(n, bus_carrier):
    carrier_buses = n.buses.index[n.buses.carrier == bus_carrier]
    prices = n.buses_t.marginal_price[carrier_buses]
    weights_t = n.snapshot_weightings.generators

    loads_at_buses = n.loads[n.loads.bus.isin(carrier_buses)]
    if not loads_at_buses.empty:
        load_ts = pd.DataFrame(0.0, index=n.snapshots, columns=carrier_buses)
        for load_name, load_row in loads_at_buses.iterrows():
            bus = load_row.bus
            if load_name in n.loads_t.p_set.columns:
                load_ts[bus] += n.loads_t.p_set[load_name]
            elif load_name in n.loads_t.p.columns:
                load_ts[bus] += n.loads_t.p[load_name]
            else:
                load_ts[bus] += load_row.p_set
    else:
        load_ts = pd.DataFrame(0.0, index=n.snapshots, columns=carrier_buses)
        consuming_links = n.links[n.links.bus0.isin(carrier_buses)]
        for link_name, link_row in consuming_links.iterrows():
            bus = link_row.bus0
            if link_name in n.links_t.p0.columns:
                load_ts[bus] += n.links_t.p0[link_name]

    numerator = (prices * load_ts).multiply(weights_t, axis=0).sum().sum()
    denominator = load_ts.multiply(weights_t, axis=0).sum().sum()

    if denominator == 0:
        return float("nan")
    return numerator / denominator


def style_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)


def extract_metrics(n):
    total_cost = (n.statistics.capex().sum() + n.statistics.opex().sum()) / 1e9
    elec_price = carrier_price_weighted(n, "AC")
    urban_decentral = carrier_price_weighted(n, "urban decentral heat")
    heat_lt100 = carrier_price_weighted(n, "heat<100 industry")
    heat_gt500 = carrier_price_weighted(n, "heat>500 industry")
    return [total_cost, elec_price, urban_decentral, heat_lt100, heat_gt500]


# --- Load networks and extract data ---
print("Loading networks...")
regular = {}
hike = {}
for w in WIGGLES:
    print(f"  wiggle={w} (regular)")
    regular[w] = extract_metrics(pypsa.Network(network_path(wiggle=w)))
    print(f"  wiggle={w} (hike=50)")
    hike[w] = extract_metrics(pypsa.Network(network_path(wiggle=w, hike=50)))

# --- Plot ---
metric_labels = [
    "Total system cost\n[bn EUR/a]",
    "Avg. electricity price\n[EUR/MWh]",
    "Urban decentral\nheat price [EUR/MWh]",
    "Industry heat <100°C\nprice [EUR/MWh]",
    "Industry heat >500°C\nprice [EUR/MWh]",
]

fig, axes = plt.subplots(1, 5, figsize=(18, 4), sharey=False)

for i, (ax, label) in enumerate(zip(axes, metric_labels)):
    for w in WIGGLES:
        y_reg = regular[w][i]
        y_hike = hike[w][i]
        # connecting line
        ax.plot(
            [w, w], [y_reg, y_hike],
            color="grey", linewidth=1.5, zorder=1,
        )
    # scatter on top
    reg_vals = [regular[w][i] for w in WIGGLES]
    hike_vals = [hike[w][i] for w in WIGGLES]
    ax.scatter(
        WIGGLES, reg_vals,
        color="#2166ac", s=70, zorder=2, label="Regular",
        edgecolors="white", linewidths=0.5,
    )
    ax.scatter(
        WIGGLES, hike_vals,
        color="#b2182b", s=70, zorder=2, label="Hike (+50 EUR/MWh)",
        marker="D", edgecolors="white", linewidths=0.5,
    )

    ax.set_xlabel("Gas budget [TWh]")
    ax.set_ylabel(label)
    ax.set_xticks(WIGGLES)
    style_ax(ax)

axes[0].legend(frameon=True, fontsize=8)
fig.tight_layout()
fig.savefig(SAVE_DIR / "commodity_prices_cleveland.pdf", bbox_inches="tight")
print(f"Saved to {SAVE_DIR / 'commodity_prices_cleveland.png'}")
plt.show()
