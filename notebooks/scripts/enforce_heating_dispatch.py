"""
Enforce realistic individual heating dispatch constraints.

In the model, individual heating buses aggregate millions of homes with
different technologies. Each home has exactly one tech, but the model sees
spare capacity and can cherry-pick the cheapest source each hour.

This script sets p_max_pu / p_min_pu so that each technology's heat output
is proportional to total demand at every timestep. The resulting heat share
per technology is constant over time and determined endogenously by the
optimizer through capacity decisions — no pre-computed shares are needed.

For a link with efficiency eff_i(t):
    p_pu_i(t) = demand(t) / (eff_i(t) * max_t[demand(t)/eff_i(t)])
guarantees heat_i(t) ∝ demand(t) regardless of capacity.

Gas boilers are left unconstrained as flex/balancing.

Usage:
    pixi run python notebooks/scripts/enforce_heating_dispatch.py
"""

import pypsa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yaml
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NETWORK_DIR = ROOT / "results" / "networks"
PREFIX = "base_s_50__168H-T-H-B-I-A-dist1_2030"
SAVE_DIR = ROOT.parent / "gas_resilience" / "imgs"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

with open(ROOT / "config.basicrun.yaml") as f:
    config = yaml.safe_load(f)
tech_colors = config["plotting"]["tech_colors"]


def network_path(scenario="free", wiggle=0, hike=None):
    hike_str = f"_{hike}" if hike is not None else ""
    return NETWORK_DIR / f"{PREFIX}_{scenario}_{wiggle}{hike_str}.nc"


def style_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(frameon=True)


def get_total_demand(n, bus_name):
    """Total heat demand time series at a bus."""
    loads_on_bus = n.loads[n.loads.bus == bus_name]
    total = pd.Series(0.0, index=n.snapshots)
    for load_name in loads_on_bus.index:
        if load_name in n.loads_t.p_set.columns:
            total += n.loads_t.p_set[load_name]
        else:
            total += loads_on_bus.loc[load_name, "p_set"]
    return total


def get_link_efficiency(n, link_name):
    """Get efficiency time series for a link (handles time-varying COP)."""
    if link_name in n.links_t.efficiency.columns:
        return n.links_t.efficiency[link_name]
    else:
        return pd.Series(
            n.links.loc[link_name, "efficiency"],
            index=n.snapshots,
        )


def enforce_individual_heating_dispatch(
    n,
    bus_name,
    capacity_field="p_nom_opt",
    flex_carriers=("gas boiler",),
    exclude_carriers=("heat vent",),
    tolerance=0.01,
    flex_reserve=0.05,
):
    """
    Constrain heating techs so each technology's heat output is proportional
    to total demand at every timestep (constant per-tech share of heat).

    Ideal per-tech per-unit profile:
        p_pu_i(t) = demand(t) / ( eff_i(t) · M_i ),    M_i = max_t[demand/eff_i]
    so heat_i(t) = p_pu_i · p_nom_i · eff_i(t) = demand(t) · (p_nom_i / M_i),
    giving a constant share s_i = p_nom_i / M_i.

    If Σ s_i > 1 − flex_reserve, all shares are scaled down by a common
    factor so the flex carrier (gas boiler) keeps at least `flex_reserve`
    of demand as headroom and the problem stays feasible. Scaling preserves
    the constant-share property and the between-tech ratios.

    Parameters
    ----------
    n : pypsa.Network
    bus_name : str
        Heat bus, e.g. "DE0 0 rural heat".
    capacity_field : str
        "p_nom_opt" (solved) or "p_nom" (brownfield).
    flex_carriers : tuple of str
        Carrier substrings left unconstrained (balancing role).
    exclude_carriers : tuple of str
        Carrier substrings to skip entirely.
    tolerance : float
        Fractional slack around the scaled profile.
    flex_reserve : float
        Minimum fraction of demand reserved for flex carriers (gas boiler).
    """
    total_demand = get_total_demand(n, bus_name)
    if total_demand.max() <= 0:
        return

    def _matches(carrier, patterns):
        return any(p in carrier for p in patterns)

    def _cap(row):
        v = row.get(capacity_field, 0.0)
        if v and not pd.isna(v):
            return v
        return row.get("p_nom", 0.0)

    # --- collect constrained techs with their nominal share s_i = p_nom_i / M_i ---
    demand_peak = total_demand.max()

    link_info = {}  # name -> (eff_ts, nominal_share)
    for name, link in n.links[n.links.bus1 == bus_name].iterrows():
        if _matches(link.carrier, exclude_carriers) or _matches(link.carrier, flex_carriers):
            continue
        p_nom = _cap(link)
        if p_nom < 1e-3:
            continue
        eff = get_link_efficiency(n, name)
        M = (total_demand / eff).max()
        if M <= 0:
            continue
        link_info[name] = (eff, p_nom / M)

    gen_info = {}  # name -> nominal_share
    for name, gen in n.generators[n.generators.bus == bus_name].iterrows():
        if _matches(gen.carrier, exclude_carriers) or _matches(gen.carrier, flex_carriers):
            continue
        p_nom = _cap(gen)
        if p_nom < 1e-3:
            continue
        gen_info[name] = p_nom / demand_peak

    total_share = sum(s for _, s in link_info.values()) + sum(gen_info.values())
    if total_share <= 0:
        return

    target = 1.0 - flex_reserve
    scale = min(1.0, target / (total_share * (1 + tolerance)))

    # --- apply constraints ---
    # At peak (demand/eff), p_pu = scale; dispatch = scale * p_nom * eff(peak);
    # heat share s_i_effective = scale * p_nom_i / M_i = scale * s_i_nominal.
    for name, (eff, _s) in link_info.items():
        normalised = (total_demand / eff) / (total_demand / eff).max()
        profile = normalised * scale

        p_max_pu = (profile * (1 + tolerance)).clip(0, 1)
        p_min_pu = (profile * (1 - tolerance)).clip(lower=0).clip(upper=p_max_pu)

        n.links_t.p_max_pu[name] = p_max_pu
        n.links_t.p_min_pu[name] = p_min_pu

    for name, _s in gen_info.items():
        profile = (total_demand / demand_peak) * scale  # eff = 1

        p_max_pu = (profile * (1 + tolerance)).clip(0, 1)
        p_min_pu = (profile * (1 - tolerance)).clip(lower=0).clip(upper=p_max_pu)

        # respect resource availability (e.g. solar thermal)
        if name in n.generators_t.p_max_pu.columns:
            resource = n.generators_t.p_max_pu[name]
            p_max_pu = p_max_pu.clip(upper=resource)
            p_min_pu = p_min_pu.clip(upper=p_max_pu)

        n.generators_t.p_max_pu[name] = p_max_pu
        n.generators_t.p_min_pu[name] = p_min_pu


def plot_dispatch_check(n, bus_name, save_path=None):
    """
    Stacked area plot of actual dispatch at a heat bus vs load, to verify
    the enforced constraints are working.
    """
    total_demand = get_total_demand(n, bus_name)

    # collect heat supply from links (bus1 side)
    supply = {}
    links = n.links[n.links.bus1 == bus_name]
    for name, link in links.iterrows():
        if name in n.links_t.p0.columns:
            eff = get_link_efficiency(n, name)
            heat = n.links_t.p0[name] * eff
            if heat.abs().sum() > 0:
                supply[link.carrier] = supply.get(link.carrier, 0) + heat

    gens = n.generators[n.generators.bus == bus_name]
    for name, gen in gens.iterrows():
        if name in n.generators_t.p.columns:
            p = n.generators_t.p[name]
            if p.abs().sum() > 0:
                supply[gen.carrier] = supply.get(gen.carrier, 0) + p

    if not supply:
        print("No dispatch data to plot.")
        return

    df = pd.DataFrame(supply)
    df = df[df.sum().sort_values(ascending=False).index]

    fig, ax = plt.subplots(figsize=(14, 5))
    rng = np.random.default_rng(seed=42)
    colors = [tech_colors.get(c, f"#{rng.integers(0, 0xFFFFFF):06x}") for c in df.columns]
    ax.stackplot(
        df.index,
        *[df[c].values for c in df.columns],
        labels=df.columns,
        colors=colors,
        alpha=0.85,
    )
    ax.plot(total_demand.index, total_demand.values, color="black", lw=1.2, label="load")
    ax.set_ylabel("MW$_{th}$")
    ax.set_title(f"Heating dispatch — {bus_name}")
    style_ax(ax)
    ax.legend(loc="upper right", fontsize=8, frameon=True)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        print(f"Saved to {save_path}")

    return fig, ax


def plot_pu_constraints(n, bus_name, save_path=None):
    """
    Plot p_max_pu and p_min_pu bands for each constrained carrier at a heat bus.
    """
    fig, ax = plt.subplots(figsize=(14, 5))
    rng = np.random.default_rng(seed=42)

    carriers_seen = set()

    # links
    links = n.links[n.links.bus1 == bus_name]
    for name, link in links.iterrows():
        carrier = link.carrier
        has_max = name in n.links_t.p_max_pu.columns
        has_min = name in n.links_t.p_min_pu.columns
        if not has_max and not has_min:
            continue
        color = tech_colors.get(carrier, f"#{rng.integers(0, 0xFFFFFF):06x}")
        label = carrier if carrier not in carriers_seen else None
        carriers_seen.add(carrier)
        if has_max:
            ax.plot(n.links_t.p_max_pu[name], color=color, lw=0.8, label=label)
        if has_min:
            ax.plot(n.links_t.p_min_pu[name], color=color, lw=0.8, ls="--")
        if has_max and has_min:
            ax.fill_between(
                n.snapshots,
                n.links_t.p_min_pu[name],
                n.links_t.p_max_pu[name],
                color=color, alpha=0.15,
            )

    # generators
    gens = n.generators[n.generators.bus == bus_name]
    for name, gen in gens.iterrows():
        carrier = gen.carrier
        has_max = name in n.generators_t.p_max_pu.columns
        has_min = name in n.generators_t.p_min_pu.columns
        if not has_max and not has_min:
            continue
        color = tech_colors.get(carrier, f"#{rng.integers(0, 0xFFFFFF):06x}")
        label = carrier if carrier not in carriers_seen else None
        carriers_seen.add(carrier)
        if has_max:
            ax.plot(n.generators_t.p_max_pu[name], color=color, lw=0.8, label=label)
        if has_min:
            ax.plot(n.generators_t.p_min_pu[name], color=color, lw=0.8, ls="--")
        if has_max and has_min:
            ax.fill_between(
                n.snapshots,
                n.generators_t.p_min_pu[name],
                n.generators_t.p_max_pu[name],
                color=color, alpha=0.15,
            )

    ax.set_ylabel("p.u.")
    ax.set_title(f"p_max_pu (solid) / p_min_pu (dashed) — {bus_name}")
    style_ax(ax)
    ax.legend(loc="upper right", fontsize=8, frameon=True)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        print(f"Saved to {save_path}")

    return fig, ax


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    path = network_path(scenario="free", wiggle=3500)
    print(f"Loading {path.name} ...")
    n = pypsa.Network(path)

    buses = list(n.buses.index[n.buses.carrier.isin(
        ["rural heat", "urban decentral heat"]
    )])
    print(f"Enforcing dispatch at {len(buses)} individual-heat buses ...")
    for b in buses:
        enforce_individual_heating_dispatch(n, b)

    # Capacities stay extendable: freezing the original mix is infeasible
    # because the pre-solve picked capacities under a different (cherry-picked)
    # dispatch. Re-investment finds a mix consistent with the constant-share
    # constraint.
    print("Re-optimising with constant-share constraints ...")
    n.optimize(solver_name="gurobi")

    demo_bus = buses[np.random.randint(len(buses))]
    print(f"\nDemo bus: {demo_bus}")
    plot_dispatch_check(n, demo_bus, save_path=SAVE_DIR / "heat_dispatch_enforced.png")
    plot_pu_constraints(n, demo_bus, save_path=SAVE_DIR / "heat_pu_constraints.png")
    plt.show()
