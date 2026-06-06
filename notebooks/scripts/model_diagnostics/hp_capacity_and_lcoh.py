"""
Heat-pump diagnostic — capacity binding, LCOH cost stack, carbon-price decomposition.

Loads base_s_50__3H-T-H-B-I-A-dist1_2030_free_2000.nc and produces:
  - hp_capacity_table.csv          per-link capacity status (p_nom, p_nom_min,
                                   p_nom_opt, headroom, capacity factor, flags)
  - hp_lcoh_stack.csv              per-(carrier, region) LCOH with components
                                   (capex / direct opex / fuel / co2)
  - hp_lcoh_stack.pdf              stacked-bar plot of the same
  - co2_price_summary.csv          ETS shadow price + EUR/MWh_gas markup
  - heat_energy_balance.csv        annual energy balance per heat bus carrier
  - heat_energy_balance.pdf        stacked-bar plot of the same

Outputs go in this script's directory (notebooks/scripts/model_diagnostics).
"""

import yaml
import numpy as np
import pandas as pd
import pypsa
import matplotlib.pyplot as plt
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[3]
NETWORK_DIR = ROOT / "results" / "networks"
SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_YAML = ROOT / "config.basicrun.yaml"
PREFIX = "base_s_50__3H-T-H-B-I-A-dist1_2030"

SCENARIO = "free"
WIGGLE = 2000
HIKE = None

HP_CARRIERS = [
    "rural air heat pump",
    "rural ground heat pump",
    "urban decentral air heat pump",
]
BOILER_CARRIERS = [
    "rural gas boiler",
    "rural oil boiler",
    "rural biomass boiler",
    "rural resistive heater",
    "urban decentral gas boiler",
    "urban decentral oil boiler",
    "urban decentral biomass boiler",
    "urban decentral resistive heater",
]
ALL_HEAT_CARRIERS = HP_CARRIERS + BOILER_CARRIERS

HEAT_BUS_CARRIERS = ["rural heat", "urban decentral heat"]


# ── Helpers ──────────────────────────────────────────────────────────────────
def network_path(scenario, wiggle, hike=None):
    hike_str = f"_{hike}" if hike is not None else ""
    return NETWORK_DIR / f"{PREFIX}_{scenario}_{wiggle}{hike_str}.nc"


def style_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)


def load_tech_colors():
    with open(CONFIG_YAML) as f:
        cfg = yaml.safe_load(f)
    return cfg["plotting"]["tech_colors"]


def region_of(carrier):
    if carrier.startswith("rural"):
        return "rural"
    if carrier.startswith("urban decentral"):
        return "urban decentral"
    return "other"


def time_eff(n, links):
    """Return (T, K) DataFrame of efficiency, broadcasting static when missing."""
    static = n.links.loc[links, "efficiency"].astype(float)
    out = pd.DataFrame(
        np.broadcast_to(static.values, (len(n.snapshots), len(links))),
        index=n.snapshots, columns=links, dtype=float,
    ).copy()
    tv = n.links_t.efficiency
    if not tv.empty:
        cols = [c for c in links if c in tv.columns]
        if cols:
            out.loc[:, cols] = tv[cols].astype(float)
    return out


def co2_bus_for(bus0):
    return "GB-co2 atmosphere" if bus0.startswith("GB") else "co2 atmosphere"


# ── Diagnostics ──────────────────────────────────────────────────────────────
def capacity_table(n):
    """Per-link capacity status for all rural/urban-decentral heat carriers."""
    weights = n.snapshot_weightings.generators
    total_h = weights.sum()

    rows = []
    for c in ALL_HEAT_CARRIERS:
        idx = n.links.index[n.links.carrier == c]
        if len(idx) == 0:
            continue
        eff_t = time_eff(n, list(idx))
        eff_mean = eff_t.multiply(weights, axis=0).sum() / total_h

        p1 = n.links_t.p1.reindex(columns=idx, fill_value=0.0)
        output_mwh = (-p1).multiply(weights, axis=0).sum()  # MWh_heat
        for link in idx:
            row = n.links.loc[link]
            p_nom_opt = float(row.p_nom_opt)
            p_nom_min = float(row.p_nom_min)
            p_nom = float(row.p_nom)
            headroom = p_nom_opt - p_nom_min
            cap_bind_up = (p_nom_opt > 0) and (headroom / max(p_nom_opt, 1e-9) < 1e-3)
            floored = (p_nom_min > 0) and ((p_nom_opt - p_nom_min) / max(p_nom_min, 1e-9) < 1e-3)
            full_load_h_eq = p_nom_opt * eff_mean[link] * total_h
            cf = output_mwh[link] / full_load_h_eq if full_load_h_eq > 0 else np.nan
            rows.append({
                "link": link,
                "carrier": c,
                "region": region_of(c),
                "country_node": " ".join(link.split(" ")[:2]),
                "p_nom": p_nom,
                "p_nom_min": p_nom_min,
                "p_nom_opt": p_nom_opt,
                "p_nom_extendable": bool(row.p_nom_extendable),
                "headroom": headroom,
                "headroom_share": headroom / p_nom_opt if p_nom_opt > 0 else np.nan,
                "eff_mean": float(eff_mean[link]),
                "output_mwh": float(output_mwh[link]),
                "cf": float(cf),
                "cap_binding_upward": bool(cap_bind_up),
                "floored_at_p_nom_min": bool(floored),
            })
    return pd.DataFrame(rows)


def lcoh_components(n, carrier):
    """Per-link LCOH decomposition.

    Returns a DataFrame with one row per link of this carrier with columns:
    [link, region, country_node, capex, opex_direct, fuel, co2, output_mwh,
     lcoh].

    capex      = capital_cost · p_nom_opt                          [EUR/a]
    opex_direct= Σ marginal_cost · p0 · w (link's own MC term)     [EUR/a]
    fuel       = Σ λ(bus0, t) · p0 · w                             [EUR/a]
    co2        = Σ λ(bus2, t) · efficiency2 · p0 · w  (≤ 0 for emitters,
                  but reported with sign flipped so positive = cost)
                                                                    [EUR/a]
    output_mwh = Σ (-p1) · w                                       [MWh_heat/a]
    lcoh       = (capex + opex_direct + fuel + co2) / output_mwh   [EUR/MWh]
    """
    idx = n.links.index[n.links.carrier == carrier]
    if len(idx) == 0:
        return pd.DataFrame()

    weights = n.snapshot_weightings.generators
    links_df = n.links.loc[idx]

    # Capex
    capex = links_df.capital_cost * links_df.p_nom_opt

    p0 = n.links_t.p0.reindex(columns=idx, fill_value=0.0)
    p1 = n.links_t.p1.reindex(columns=idx, fill_value=0.0)
    output_mwh = (-p1).multiply(weights, axis=0).sum()

    # Direct opex (own marginal_cost × p0)
    mc = links_df.marginal_cost.astype(float)
    opex_direct = (p0.mul(mc, axis=1)).multiply(weights, axis=0).sum()

    # Fuel cost = Σ λ(bus0,t) · p0 · w
    bus0 = links_df.bus0
    valid = bus0[bus0.isin(n.buses_t.marginal_price.columns)]
    if len(valid) > 0:
        lam0 = n.buses_t.marginal_price[valid.values]
        lam0.columns = valid.index
        fuel = (lam0 * p0[valid.index]).multiply(weights, axis=0).sum()
    else:
        fuel = pd.Series(0.0, index=idx)
    fuel = fuel.reindex(idx).fillna(0.0)

    # CO2 cost. For emitters, λ_co2 < 0 and efficiency2 > 0; raw contribution
    # is negative. We flip the sign so positive number = cost added to system
    # (to match how the LP duality gives an effective markup of -λ_co2 · η_co2
    # to the link's effective MC at the heat bus).
    co2 = pd.Series(0.0, index=idx)
    if "bus2" in links_df.columns:
        for link in idx:
            b2 = links_df.at[link, "bus2"]
            if not isinstance(b2, str) or b2 == "":
                continue
            if b2 not in n.buses_t.marginal_price.columns:
                continue
            eff2 = float(links_df.at[link, "efficiency2"])
            lam2 = n.buses_t.marginal_price[b2]
            contrib = (lam2 * p0[link] * eff2 * weights).sum()
            co2.at[link] = -contrib  # flip sign: positive = cost

    rows = []
    for link in idx:
        out = float(output_mwh[link])
        total = float(capex[link] + opex_direct[link] + fuel[link] + co2[link])
        lcoh = total / out if out > 0 else np.nan
        rows.append({
            "link": link,
            "carrier": carrier,
            "region": region_of(carrier),
            "country_node": " ".join(link.split(" ")[:2]),
            "capex": float(capex[link]),
            "opex_direct": float(opex_direct[link]),
            "fuel": float(fuel[link]),
            "co2": float(co2[link]),
            "output_mwh": out,
            "lcoh": float(lcoh),
        })
    return pd.DataFrame(rows)


def carrier_lcoh_aggregate(comp_df):
    """Carrier-level LCOH = total system EUR / total system MWh_heat."""
    if comp_df.empty:
        return pd.Series(dtype=float)
    grp = comp_df.groupby("carrier")
    out = pd.DataFrame({
        "capex": grp.capex.sum(),
        "opex_direct": grp.opex_direct.sum(),
        "fuel": grp.fuel.sum(),
        "co2": grp.co2.sum(),
        "output_mwh": grp.output_mwh.sum(),
    })
    total_eur = out[["capex", "opex_direct", "fuel", "co2"]].sum(axis=1)
    out["lcoh_eur_per_mwh"] = total_eur / out["output_mwh"]
    return out


def co2_price_decomposition(n):
    weights = n.snapshot_weightings.generators
    rows = []
    for b in ["co2 atmosphere", "GB-co2 atmosphere"]:
        if b not in n.buses_t.marginal_price.columns:
            continue
        lam = n.buses_t.marginal_price[b]
        weighted_mean = (lam * weights).sum() / weights.sum()
        # gas boiler effective markup: -λ_co2 · η_co2
        gas_eff2 = float(
            n.links.loc[
                n.links.carrier == "rural gas boiler", "efficiency2"
            ].iloc[0]
        )
        rows.append({
            "co2_bus": b,
            "lambda_co2_weighted_mean_eur_per_t": float(weighted_mean),
            "co2_price_magnitude_eur_per_t": float(-weighted_mean),
            "gas_co2_intensity_t_per_mwh": gas_eff2,
            "gas_boiler_markup_eur_per_mwh_gas": float(-weighted_mean) * gas_eff2,
        })
    return pd.DataFrame(rows)


def hp_envelope_check(n):
    """For each HP link, compare actual electric input p0(t) and heat output
    -p1(t) to the constraint envelope set by enforce_individual_heating_dispatch.

    The envelope is:
        p0_max(t) = p_nom_opt · p_max_pu(t)
        heat_max(t) = p_nom_opt · p_max_pu(t) · COP(t)

    A snapshot is "clipped at peak" when p_max_pu(t) ≥ 0.999. A link "rides the
    envelope" when its dispatch share at the clipping snapshots is itself ≥ 0.99
    of the cap — in that regime, the only way to deliver more heat is to grow
    p_nom_opt, so the constraint is functionally a sizing requirement.
    """
    weights = n.snapshot_weightings.generators
    rows = []
    for c in HP_CARRIERS:
        idx = n.links.index[n.links.carrier == c]
        if len(idx) == 0:
            continue
        eff_t = time_eff(n, list(idx))
        p0 = n.links_t.p0.reindex(columns=idx, fill_value=0.0)
        p1 = n.links_t.p1.reindex(columns=idx, fill_value=0.0)

        # p_max_pu time series, broadcasting static when missing
        pmax_static = n.links.loc[idx, "p_max_pu"].astype(float)
        pmax_t = pd.DataFrame(
            np.broadcast_to(pmax_static.values, (len(n.snapshots), len(idx))),
            index=n.snapshots, columns=idx, dtype=float,
        ).copy()
        if not n.links_t.p_max_pu.empty:
            cols = [k for k in idx if k in n.links_t.p_max_pu.columns]
            if cols:
                pmax_t.loc[:, cols] = n.links_t.p_max_pu[cols].astype(float)

        for link in idx:
            p_nom_opt = float(n.links.at[link, "p_nom_opt"])
            if p_nom_opt <= 0:
                continue
            pmax_link = pmax_t[link]
            cop_link = eff_t[link]
            actual_pu = p0[link] / p_nom_opt
            envelope_pu = pmax_link
            heat_actual = -p1[link]
            heat_envelope = pmax_link * p_nom_opt * cop_link

            clip_mask = pmax_link >= 0.999
            clip_hours_w = float((weights * clip_mask).sum())

            # snapshots where actual ≥ 99% of cap — link "rides envelope"
            ride_mask = actual_pu >= 0.99 * envelope_pu.replace(0, np.nan)
            ride_hours_w = float((weights * ride_mask.fillna(False)).sum())

            # snapshots simultaneously clipped AND riding (binding at peak)
            bind_peak_w = float((weights * (clip_mask & ride_mask.fillna(False))).sum())

            rows.append({
                "link": link,
                "carrier": c,
                "country_node": " ".join(link.split(" ")[:2]),
                "p_nom_opt": p_nom_opt,
                "cop_mean": float((cop_link * weights).sum() / weights.sum()),
                "max_p_max_pu": float(pmax_link.max()),
                "max_actual_pu": float(actual_pu.max()),
                "max_heat_actual_mw": float(heat_actual.max()),
                "max_heat_envelope_mw": float(heat_envelope.max()),
                "annual_heat_mwh": float((heat_actual * weights).sum()),
                "annual_envelope_mwh": float((heat_envelope * weights).sum()),
                "envelope_use_share": float(
                    (heat_actual * weights).sum() / max(
                        (heat_envelope * weights).sum(), 1e-9
                    )
                ),
                "clipped_hours_w": clip_hours_w,
                "envelope_riding_hours_w": ride_hours_w,
                "binding_at_peak_hours_w": bind_peak_w,
            })
    return pd.DataFrame(rows)


def distribution_grid_headroom(n):
    """Per-region headroom on the AC → low voltage distribution links.

    Reports p_nom_opt, peak directional flow (AC → low voltage), the share of
    weighted snapshots within 1% of p_nom_opt, and the minimum dual price (a
    proxy for whether the LP is paying a shadow price to enlarge the link).
    """
    weights = n.snapshot_weightings.generators
    idx = n.links.index[n.links.carrier == "electricity distribution grid"]
    if len(idx) == 0:
        return pd.DataFrame()

    p0 = n.links_t.p0.reindex(columns=idx, fill_value=0.0)
    rows = []
    for link in idx:
        row = n.links.loc[link]
        p_nom_opt = float(row.p_nom_opt)
        flow = p0[link]
        peak_flow = float(flow.max())
        binding_share = float(
            (weights * (flow >= 0.99 * p_nom_opt)).sum() / weights.sum()
        ) if p_nom_opt > 0 else np.nan
        headroom = p_nom_opt - peak_flow
        rows.append({
            "link": link,
            "country_node": " ".join(link.split(" ")[:2]),
            "p_nom": float(row.p_nom),
            "p_nom_min": float(row.p_nom_min),
            "p_nom_opt": p_nom_opt,
            "p_nom_extendable": bool(row.p_nom_extendable),
            "peak_flow_mw": peak_flow,
            "headroom_mw": headroom,
            "headroom_share": (
                headroom / p_nom_opt if p_nom_opt > 0 else np.nan
            ),
            "binding_share_w": binding_share,
        })
    return pd.DataFrame(rows)


def heat_energy_balance(n):
    pieces = []
    for bc in HEAT_BUS_CARRIERS:
        eb = n.statistics.energy_balance(bus_carrier=bc)
        # eb has MultiIndex; reduce to (component, carrier) → MWh
        s = eb.groupby(["component", "carrier"]).sum()
        s.name = "MWh"
        df = s.reset_index()
        df["bus_carrier"] = bc
        pieces.append(df)
    return pd.concat(pieces, ignore_index=True)


# ── Plots ────────────────────────────────────────────────────────────────────
def plot_lcoh_stack(carrier_lcoh, tech_colors, out_path):
    components = ["capex", "opex_direct", "fuel", "co2"]
    comp_colors = {
        "capex": "#5b8def",
        "opex_direct": "#a5b1c2",
        "fuel": "#e0b25b",
        "co2": "#9e2a2b",
    }

    df = carrier_lcoh.copy()
    df = df.loc[[c for c in ALL_HEAT_CARRIERS if c in df.index]]
    # Express each component on a per-MWh_heat basis for the stack
    for c in components:
        df[c + "_per_mwh"] = df[c] / df["output_mwh"]

    fig, ax = plt.subplots(figsize=(11, 5))
    bottom = np.zeros(len(df))
    x = np.arange(len(df))
    for c in components:
        vals = df[c + "_per_mwh"].fillna(0).values
        ax.bar(x, vals, bottom=bottom, label=c.replace("_", " "),
               color=comp_colors[c], edgecolor="white", linewidth=0.5)
        bottom = bottom + vals

    # Overlay total LCOH as a thick tick on top
    ax.plot(x, df["lcoh_eur_per_mwh"].values, "k_", markersize=22, mew=2,
            label="total LCOH")

    ax.set_xticks(x)
    ax.set_xticklabels(df.index, rotation=30, ha="right")
    ax.set_ylabel("EUR / MWh_heat")
    ax.set_title("LCOH stack — rural & urban decentral heating "
                 f"({SCENARIO}, wiggle={WIGGLE})")
    style_ax(ax)
    ax.legend(frameon=True, loc="upper left", ncol=2)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_heat_energy_balance(eb_df, tech_colors, out_path):
    # Aggregate by (bus_carrier, carrier) summing across components
    pivot = (
        eb_df.groupby(["bus_carrier", "carrier"]).MWh.sum().unstack("carrier")
        .fillna(0.0)
    )
    # Convert MWh → TWh for readability
    pivot = pivot / 1e6

    fig, ax = plt.subplots(figsize=(11, 5))
    # Use config tech_colors when available, fallback grey
    sorted_carriers = pivot.columns.tolist()
    colors = [tech_colors.get(c, "#888888") for c in sorted_carriers]
    # Stacked bar
    pivot.plot.bar(stacked=True, ax=ax, color=colors, edgecolor="white",
                   linewidth=0.3, width=0.7)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_ylabel("TWh / a (positive = supply, negative = withdrawal)")
    ax.set_title(f"Energy balance at heat buses ({SCENARIO}, wiggle={WIGGLE})")
    ax.set_xticklabels(pivot.index, rotation=0)
    style_ax(ax)
    ax.legend(frameon=True, bbox_to_anchor=(1.01, 1.0), loc="upper left",
              fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    npath = network_path(SCENARIO, WIGGLE, HIKE)
    print(f"loading {npath}")
    n = pypsa.Network(str(npath))
    assert getattr(n, "is_solved", True), "network not solved"
    print(f"  snapshots={len(n.snapshots)}  weight={n.snapshot_weightings.generators.iloc[0]}")
    print(f"  objective={n.objective:.3e}")

    tech_colors = load_tech_colors()

    # 1. Capacity status
    cap = capacity_table(n)
    cap_path = SCRIPT_DIR / "hp_capacity_table.csv"
    cap.to_csv(cap_path, index=False)
    print(f"\n[1/5] capacity table → {cap_path.name}  rows={len(cap)}")
    summary = (
        cap.groupby("carrier")
        .agg(p_nom_opt=("p_nom_opt", "sum"),
             p_nom_min=("p_nom_min", "sum"),
             output_TWh=("output_mwh", lambda s: s.sum() / 1e6),
             cf=("cf", lambda s: s.where(s.notna()).mean()),
             cap_bound_share=("cap_binding_upward", "mean"),
             floored_share=("floored_at_p_nom_min", "mean"))
    )
    print(summary.round(3).to_string())

    # 2. LCOH per carrier
    comp_pieces = []
    for c in ALL_HEAT_CARRIERS:
        df = lcoh_components(n, c)
        if not df.empty:
            comp_pieces.append(df)
    comp_df = pd.concat(comp_pieces, ignore_index=True) if comp_pieces else pd.DataFrame()
    comp_path = SCRIPT_DIR / "hp_lcoh_per_link.csv"
    comp_df.to_csv(comp_path, index=False)

    car_lcoh = carrier_lcoh_aggregate(comp_df)
    car_path = SCRIPT_DIR / "hp_lcoh_stack.csv"
    car_lcoh.to_csv(car_path)
    print(f"\n[2/5] LCOH per carrier → {car_path.name}")
    print(car_lcoh.round(2).to_string())

    # Sanity: empirical fuel + co2 vs analytic for gas boilers
    print("\n  sanity check (gas boilers): empirical (fuel + co2) vs analytic")
    for gb in ["rural gas boiler", "urban decentral gas boiler"]:
        if gb not in car_lcoh.index:
            continue
        emp = car_lcoh.loc[gb, "fuel"] + car_lcoh.loc[gb, "co2"]
        analytic = compute_analytic_gas_cost(n, gb)
        rel_err = abs(emp - analytic) / max(abs(emp), 1.0)
        print(f"    {gb}: emp={emp:.3e}  analytic={analytic:.3e}  rel_err={rel_err:.2%}")

    # 3. CO2 price decomposition
    co2_df = co2_price_decomposition(n)
    co2_path = SCRIPT_DIR / "co2_price_summary.csv"
    co2_df.to_csv(co2_path, index=False)
    print(f"\n[3/5] CO2 price summary → {co2_path.name}")
    print(co2_df.round(3).to_string(index=False))

    # 4. Heat energy balance
    eb_df = heat_energy_balance(n)
    eb_path = SCRIPT_DIR / "heat_energy_balance.csv"
    eb_df.to_csv(eb_path, index=False)
    print(f"\n[4/7] heat energy balance → {eb_path.name}")
    print(eb_df.groupby(["bus_carrier", "carrier"]).MWh.sum()
          .unstack("carrier").fillna(0).div(1e6).round(2).to_string())

    # 5. HP envelope vs actual dispatch
    env_df = hp_envelope_check(n)
    env_path = SCRIPT_DIR / "hp_envelope_check.csv"
    env_df.to_csv(env_path, index=False)
    print(f"\n[5/7] HP envelope check → {env_path.name}  rows={len(env_df)}")
    if not env_df.empty:
        env_summary = (
            env_df.groupby("carrier").agg(
                p_nom_opt_GW=("p_nom_opt", lambda s: s.sum() / 1e3),
                cop_mean=("cop_mean", "mean"),
                envelope_use_share=("envelope_use_share", "mean"),
                clipped_hours_w=("clipped_hours_w", "mean"),
                binding_at_peak_hours_w=("binding_at_peak_hours_w", "mean"),
            ).round(3)
        )
        print(env_summary.to_string())

    # 6. Distribution grid headroom
    dg_df = distribution_grid_headroom(n)
    dg_path = SCRIPT_DIR / "distribution_grid_headroom.csv"
    dg_df.to_csv(dg_path, index=False)
    print(f"\n[6/7] distribution grid headroom → {dg_path.name}  rows={len(dg_df)}")
    if not dg_df.empty:
        binding_count = int((dg_df.binding_share_w > 0.01).sum())
        tight_count = int((dg_df.headroom_share < 0.05).sum())
        print(f"  links with >1% weighted hours at ≥99% capacity: {binding_count}/{len(dg_df)}")
        print(f"  links with <5% headroom on p_nom_opt:           {tight_count}/{len(dg_df)}")
        print(dg_df.sort_values("headroom_share").head(10)[
            ["link", "p_nom_opt", "peak_flow_mw", "headroom_share", "binding_share_w"]
        ].round(3).to_string(index=False))

    # 7. Plots
    lcoh_pdf = SCRIPT_DIR / "hp_lcoh_stack.pdf"
    plot_lcoh_stack(car_lcoh, tech_colors, lcoh_pdf)
    print(f"\n[7/7] plot → {lcoh_pdf.name}")

    eb_pdf = SCRIPT_DIR / "heat_energy_balance.pdf"
    plot_heat_energy_balance(eb_df, tech_colors, eb_pdf)
    print(f"        plot → {eb_pdf.name}")


def compute_analytic_gas_cost(n, carrier):
    """Σ_t (λ_gas + (-λ_co2) · η_co2) · p0 · w  for sanity-check."""
    idx = n.links.index[n.links.carrier == carrier]
    if len(idx) == 0:
        return 0.0
    weights = n.snapshot_weightings.generators
    p0 = n.links_t.p0.reindex(columns=idx, fill_value=0.0)
    bus0 = n.links.loc[idx, "bus0"]
    eff2 = n.links.loc[idx, "efficiency2"].astype(float)
    bus2 = n.links.loc[idx, "bus2"].astype(str)

    # gas commodity price per link
    valid_gas = bus0[bus0.isin(n.buses_t.marginal_price.columns)]
    lam_gas = n.buses_t.marginal_price[valid_gas.values]
    lam_gas.columns = valid_gas.index

    # CO2 price per link (depends on which atmosphere bus)
    co2_buses = bus2.where(bus2.isin(n.buses_t.marginal_price.columns), other="")
    contrib = pd.DataFrame(0.0, index=n.snapshots, columns=idx)
    for link in idx:
        b2 = co2_buses[link]
        if b2 == "":
            continue
        contrib[link] = -n.buses_t.marginal_price[b2] * eff2[link]
    eff_fuel = lam_gas.reindex(columns=idx, fill_value=0.0).add(contrib, fill_value=0.0)
    return float((eff_fuel * p0).multiply(weights, axis=0).sum().sum())


if __name__ == "__main__":
    main()
