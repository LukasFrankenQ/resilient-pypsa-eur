"""Consumer cost, production cost, and the infra-marginal rent gap between
them, as a function of the gas-budget wiggle, for 3H 2030 `free` networks.

- Consumer cost = Σ_b,t (price_{b,t} × consumption_{b,t} × weight_t) over all
  AC buses.
- Production cost = capex + opex of components attached to AC buses
  (n.statistics.{capex,opex}(bus_carrier="AC")).
- Infra-marginal rent = consumer - production.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pypsa
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "notebooks" / "scripts"))
from frontend_export import export_frontend_data  # noqa: E402
NET_DIR = ROOT / "results" / "networks"
SCRIPT_DIR = Path(__file__).resolve().parent
IMG_DIR = ROOT.parent / "gas_resilience" / "imgs"
IMG_DIR.mkdir(parents=True, exist_ok=True)

PREFIX = "base_s_50_lv1.25_3H-T-H-B-I-A-dist1_2030"
SCENARIO = "free"
WIGGLES = list(range(0, 4001, 250))

COLOR_CONSUMER = "#0b1f3a"
COLOR_PRODUCTION = "black"


def network_path(wiggle, scenario=SCENARIO, hike=None):
    hike_str = f"_{hike}" if hike is not None else ""
    return NET_DIR / f"{PREFIX}_{scenario}_{wiggle}{hike_str}.nc"


def consumer_cost_and_energy(n):
    """Return (consumer_cost_EUR, total_AC_consumption_MWh)."""
    ac_buses = n.buses.index[n.buses.carrier == "AC"]
    weights = n.snapshot_weightings.generators
    prices = n.buses_t.marginal_price[ac_buses]

    withdrawal = pd.DataFrame(0.0, index=n.snapshots, columns=ac_buses)

    loads_ac = n.loads[n.loads.bus.isin(ac_buses)]
    for name, row in loads_ac.iterrows():
        if name in n.loads_t.p_set.columns:
            withdrawal.loc[:, row.bus] += n.loads_t.p_set[name]
        else:
            withdrawal.loc[:, row.bus] += row.p_set

    links_ac = n.links[n.links.bus0.isin(ac_buses)]
    for name, row in links_ac.iterrows():
        if name in n.links_t.p0.columns:
            withdrawal.loc[:, row.bus0] += n.links_t.p0[name].clip(lower=0)

    su_ac = n.storage_units[n.storage_units.bus.isin(ac_buses)]
    for name, row in su_ac.iterrows():
        if name in n.storage_units_t.p_store.columns:
            withdrawal.loc[:, row.bus] += n.storage_units_t.p_store[name]

    cost = float((prices * withdrawal).multiply(weights, axis=0).sum().sum())
    energy = float(withdrawal.multiply(weights, axis=0).sum().sum())
    return cost, energy


def production_cost(n):
    # bus_carrier="AC" captures generators on AC buses and links outputting to
    # AC. Link opex here is the direct marginal_cost term only — upstream fuel
    # costs (e.g. gas) land in consumer_cost via the marginal price.
    capex = n.statistics.capex(bus_carrier="AC").sum()
    opex = n.statistics.opex(bus_carrier="AC").sum()
    return float(capex + opex)


records = []
for w in WIGGLES:
    p = network_path(w)
    if not p.exists():
        print(f"  missing: {p.name} — skipping")
        continue
    print(f"Loading wiggle={w}")
    n = pypsa.Network(p)
    cons_eur, energy_mwh = consumer_cost_and_energy(n)
    records.append({
        "wiggle": w,
        "consumer": cons_eur / 1e9,
        "production": production_cost(n) / 1e9,
        "energy_twh": energy_mwh / 1e6,
    })

df = pd.DataFrame(records)
df["rent"] = df["consumer"] - df["production"]
df["consumer_per_mwh"] = df["consumer"] * 1e9 / (df["energy_twh"] * 1e6)
df["production_per_mwh"] = df["production"] * 1e9 / (df["energy_twh"] * 1e6)
df["rent_per_mwh"] = df["consumer_per_mwh"] - df["production_per_mwh"]
print(df.to_string(index=False))

if (df["rent"] < 0).any():
    bad = df.loc[df["rent"] < 0, "wiggle"].tolist()
    print(f"WARNING: negative infra-marginal rent at wiggle={bad}")

fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.2))

CMAP = plt.get_cmap("rainbow")
BAR_WIDTH = 250  # gapless: step between adjacent wiggles


def draw_panel(ax, y_cons, y_prod, ylabel, cb_label, show_legend=True):
    x = df["wiggle"].to_numpy(float)
    yc = np.asarray(y_cons, float)
    yp = np.asarray(y_prod, float)
    rent = yc - yp

    # Each bar shows a vertical gradient: its bottom is at cmap(0), its top at
    # cmap(height / max_height). The tallest bar therefore reaches cmap(1.0).
    vmax = float(rent.max())
    norm = Normalize(vmin=0.0, vmax=vmax)

    for xi, yp_i, yc_i, rent_i in zip(x, yp, yc, rent):
        gradient = np.linspace(0.0, rent_i, 256).reshape(-1, 1)
        ax.imshow(
            gradient,
            extent=(
                xi - BAR_WIDTH / 2, xi + BAR_WIDTH / 2,
                yp_i, yc_i,
            ),
            origin="lower", aspect="auto",
            cmap=CMAP, norm=norm,
            zorder=1, interpolation="bilinear",
        )

    ax.plot(
        x, yc,
        color=COLOR_CONSUMER, lw=2.6, marker="o", markersize=7,
        markerfacecolor=COLOR_CONSUMER,
        markeredgecolor="white", markeredgewidth=1.0,
        label="Consumer cost", zorder=4,
    )
    ax.plot(
        x, yp,
        color=COLOR_PRODUCTION, lw=1.6, linestyle=(0, (4, 2)),
        marker="^", markersize=7,
        markerfacecolor="black",
        markeredgecolor="black", markeredgewidth=1.0,
        label="Production cost", zorder=3,
    )
    ax.set_xlabel("Total Gas Consumption [TWh]")
    ax.set_ylabel(ylabel)
    ax.set_xlim(-BAR_WIDTH / 2, 4000 + BAR_WIDTH / 2)
    ax.set_xticks(WIGGLES)
    ax.set_xticklabels([str(w) if w % 500 == 0 else "" for w in WIGGLES])

    # Slightly extend y-limits outwards
    y_lo, y_hi = float(yp.min()), float(yc.max())
    pad = 0.06 * (y_hi - y_lo)
    ax.set_ylim(y_lo - pad, y_hi + pad)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    if show_legend:
        ax.legend(frameon=True, loc="best")

    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=norm)
    sm.set_array([])
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("top", size="4%", pad=0.45)
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cax.xaxis.set_ticks_position("top")
    cax.xaxis.set_label_position("top")
    cb.set_label(cb_label)


draw_panel(
    axes[0], df["consumer"], df["production"],
    ylabel="Electricity Generation Cost [B€/a]",
    cb_label="Infra-marginal rent [B€/a]",
)
draw_panel(
    axes[1], df["consumer_per_mwh"], df["production_per_mwh"],
    ylabel="Electricity cost [€/MWh]",
    cb_label="Infra-marginal rent [€/MWh]",
    show_legend=False,
)

fig.tight_layout()

out_paper = IMG_DIR / "infra_marginal_rent.pdf"
out_script = SCRIPT_DIR / "infra_marginal_rent.pdf"
fig.savefig(out_paper, bbox_inches="tight")
fig.savefig(out_script, bbox_inches="tight")
print(f"wrote {out_paper}")
print(f"wrote {out_script}")

export_frontend_data("infra_marginal_rent", {
    "x_label": "Total Gas Consumption [TWh]",
    "wiggles_TWh": df["wiggle"].tolist(),
    "panels": {
        "absolute": {
            "y_label": "Electricity Generation Cost [B€/a]",
            "cb_label": "Infra-marginal rent [B€/a]",
            "consumer_BEUR_per_a": df["consumer"].tolist(),
            "production_BEUR_per_a": df["production"].tolist(),
            "rent_BEUR_per_a": df["rent"].tolist(),
        },
        "per_mwh": {
            "y_label": "Electricity cost [€/MWh]",
            "cb_label": "Infra-marginal rent [€/MWh]",
            "consumer_EUR_per_MWh": df["consumer_per_mwh"].tolist(),
            "production_EUR_per_MWh": df["production_per_mwh"].tolist(),
            "rent_EUR_per_MWh": df["rent_per_mwh"].tolist(),
        },
    },
    "energy_TWh": df["energy_twh"].tolist(),
    "colors": {"consumer": COLOR_CONSUMER, "production": COLOR_PRODUCTION},
})
