"""
For one AC bus in a solved sector-coupled PyPSA-Eur network, classify
snapshot-by-snapshot which mechanism is setting the local marginal
electricity price, and write out an equation showing how.

A component is taken to be marginal at (bus, t) if its dispatch is strictly
between bounds AND its effective cost-at-AC matches the bus price within
tolerance. Candidates:

    supply Link (bus1=b):          c_eff = (c + lambda_{bus0}) / eta
    storage discharger:             same, with lambda of the storage bus
    Generator at b (e.g. renewables): c_eff = marginal_cost
    demand-flex Link (bus0=b):     c_eff = eta * lambda_{bus1} - c
    storage charger:                same
    LV-projected flex:              bus0 = "{b} low voltage"  (distribution-grid eta ~= 1)
    StorageUnit (hydro/PHS):        c_eff ~= lambda_b (water value) when interior
    transmission:                   a neighbour's price imported via a congested line
    load-shedding:                  slack generator at VoLL

Classification priority: supply > demand_flex > storage > storage_unit >
transmission > load_shedding > unresolved.
"""

import pypsa
import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.patches import Patch

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[3]
NETWORK_DIR = ROOT / "results" / "networks"
PREFIX = "base_s_50__3H-T-H-B-I-A-dist1_2030"
PAPER_DIR = ROOT.parent / "gas_resilience" / "imgs" / "price_formation"
SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_YAML = ROOT / "config.basicrun.yaml"

# ── Analysis target ──────────────────────────────────────────────────────────
SCENARIO = "free"
WIGGLE = 1000
HIKE = None
BUS = "DE0 0"

# ── Tolerances ───────────────────────────────────────────────────────────────
BOUND_TOL_REL = 1e-3
BOUND_TOL_ABS = 1e-2
PRICE_TOL_ABS = 0.5
PRICE_TOL_REL = 0.01
FLOW_CONGESTION_REL = 0.99
VOLL_THRESH = 3000.0

# ── Carriers to ignore (pure topology wires) ─────────────────────────────────
EXCLUDE_LINK_CARRIERS = {"DC", "electricity distribution grid"}

# ── Carriers used only for labelling output classes ──────────────────────────
DISCHARGER_CARRIERS = {"battery discharger", "home battery discharger", "V2G"}
CHARGER_CARRIERS = {"battery charger", "home battery charger", "BEV charger"}

# ── Outputs ──────────────────────────────────────────────────────────────────
BUS_TAG = BUS.replace(" ", "_")
RUN_TAG = f"{SCENARIO}_{WIGGLE}" + (f"_{HIKE}" if HIKE is not None else "")
OUT_TABLE = SCRIPT_DIR / f"price_setter_all_buses_{RUN_TAG}.parquet"
OUT_CSV = SCRIPT_DIR / f"price_setter_all_buses_{RUN_TAG}.csv"
OUT_EXPLAIN = SCRIPT_DIR / f"price_setter_all_buses_{RUN_TAG}.txt"
OUT_DETAIL_PLOT = f"price_setter_detail_{BUS_TAG}_{RUN_TAG}.pdf"
OUT_SHARES_PLOT = f"price_setter_shares_by_bus_{RUN_TAG}.pdf"


# ── Helpers ──────────────────────────────────────────────────────────────────
def network_path(scenario="free", wiggle=0, hike=None):
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


def _ts_matrix(time_df, names, snapshots, static_values):
    """Return (T, len(names)) DataFrame for a time-varying attribute,
    broadcasting static_values where no time series is stored."""
    out = pd.DataFrame(
        np.broadcast_to(static_values.values, (len(snapshots), len(names))),
        index=snapshots, columns=names, dtype=float,
    ).copy()
    if time_df is not None and not time_df.empty:
        tv_cols = [c for c in names if c in time_df.columns]
        if tv_cols:
            out.loc[:, tv_cols] = time_df[tv_cols].astype(float)
    return out


def _p_nom_matrix(df, names, snapshots):
    """Return p_nom_opt (fallback p_nom) broadcast across snapshots."""
    col = "p_nom_opt" if "p_nom_opt" in df.columns else "p_nom"
    vals = df.loc[names, col].astype(float)
    return pd.DataFrame(
        np.broadcast_to(vals.values, (len(snapshots), len(names))),
        index=snapshots, columns=names,
    ).copy()


def _bus_contribution(n, link_df, names, bus_col, eff_col, eff_t_attr, snapshots):
    """Return (T, K) DataFrame of η_i(t) · λ_{bus_i}(t) for each link in
    `names`. Zero where `bus_i` is empty / not a valid bus in the network.
    Used for the CHP-heat and CO₂-emission contributions at bus2/bus3.
    """
    contrib = pd.DataFrame(0.0, index=snapshots, columns=names)
    if bus_col not in link_df.columns:
        return contrib
    bus_vals = link_df.loc[names, bus_col].astype(str).fillna("").values
    price_cols = n.buses_t.marginal_price.columns
    valid = [(nm, b) for nm, b in zip(names, bus_vals) if b and b in price_cols]
    if not valid:
        return contrib
    valid_names = [nm for nm, _ in valid]
    valid_buses = [b for _, b in valid]
    lam = (n.buses_t.marginal_price
           .reindex(columns=valid_buses, index=snapshots)
           .set_axis(valid_names, axis=1)
           .astype(float))
    eff_t = getattr(n.links_t, eff_t_attr, None)
    eff = _ts_matrix(eff_t, valid_names, snapshots,
                     link_df.loc[valid_names, eff_col].astype(float))
    contrib.loc[:, valid_names] = (lam * eff).values
    return contrib


# ── Candidate discovery (topology, not allow-lists) ──────────────────────────
def build_candidates(n, bus):
    """Discover all components whose price could set λ at `bus`.

    Returns
    -------
    supply_gens : DataFrame of Generators at `bus`.
    supply_links : DataFrame of Links with bus1 == bus (excluding DC + distrib grid).
                   Column `role` is 'supply' or 'discharger'.
    demand_links : DataFrame of Links with bus0 in {bus, f"{bus} low voltage"}.
                   Column `on_lv` flags LV-side.  Column `role` is 'demand_flex'
                   or 'charger'.
    storage_units : DataFrame of StorageUnits at `bus`.
    """
    lv_bus = f"{bus} low voltage"

    gens = n.generators[n.generators.bus == bus].copy()

    links = n.links
    sup_mask = (links.bus1 == bus) & (~links.carrier.isin(EXCLUDE_LINK_CARRIERS))
    supply_links = links[sup_mask].copy()
    supply_links["role"] = np.where(
        supply_links.carrier.isin(DISCHARGER_CARRIERS), "discharger", "supply"
    )

    dem_mask = links.bus0.isin([bus, lv_bus]) & (~links.carrier.isin(EXCLUDE_LINK_CARRIERS))
    demand_links = links[dem_mask].copy()
    demand_links["on_lv"] = demand_links.bus0 == lv_bus
    demand_links["role"] = np.where(
        demand_links.carrier.isin(CHARGER_CARRIERS), "charger", "demand_flex"
    )

    storage_units = n.storage_units[n.storage_units.bus == bus].copy()

    return gens, supply_links, demand_links, storage_units


# ── Cost / dispatch matrices ─────────────────────────────────────────────────
def supply_matrices(n, gens, supply_links, snapshots):
    """Return c_eff, p_ac, p_nom, pmax_pu, pmin_pu as (T, K) frames,
    plus meta DataFrame with columns {name, carrier, ctype, fuel_bus, role}."""
    metas = []
    c_eff_parts, p_ac_parts = [], []
    pnom_parts, pmax_parts, pmin_parts = [], [], []

    if len(gens):
        g_names = gens.index.tolist()
        c_g = _ts_matrix(
            n.generators_t.marginal_cost, g_names, snapshots,
            gens["marginal_cost"].astype(float),
        )
        p_g = n.generators_t.p.reindex(columns=g_names, index=snapshots).fillna(0.0)
        pnom_g = _p_nom_matrix(gens, g_names, snapshots)
        pmax_g = _ts_matrix(n.generators_t.p_max_pu, g_names, snapshots,
                            gens["p_max_pu"].astype(float))
        pmin_g = _ts_matrix(n.generators_t.p_min_pu, g_names, snapshots,
                            gens["p_min_pu"].astype(float))
        c_eff_parts.append(c_g)
        p_ac_parts.append(p_g)
        pnom_parts.append(pnom_g)
        pmax_parts.append(pmax_g)
        pmin_parts.append(pmin_g)
        metas.append(pd.DataFrame({
            "name": g_names,
            "carrier": gens.loc[g_names, "carrier"].values,
            "ctype": "Generator",
            "fuel_bus": "",
            "role": "supply",
        }))

    if len(supply_links):
        l_names = supply_links.index.tolist()
        bus0s = supply_links.loc[l_names, "bus0"].values
        lam_fuel = (
            n.buses_t.marginal_price
            .reindex(columns=bus0s, index=snapshots)
            .set_axis(l_names, axis=1)
            .astype(float)
        )
        c_l = _ts_matrix(
            n.links_t.marginal_cost, l_names, snapshots,
            supply_links.loc[l_names, "marginal_cost"].astype(float),
        )
        eff_l = _ts_matrix(
            n.links_t.efficiency, l_names, snapshots,
            supply_links.loc[l_names, "efficiency"].astype(float),
        )
        eff_l = eff_l.replace(0.0, np.nan)
        # Co-outputs at bus2 and bus3 (CHP heat, CO₂ emissions, etc.):
        # interior LP condition η₁·λ_bus1 + η₂·λ_bus2 + η₃·λ_bus3 = λ_bus0 + c
        bus2_contrib = _bus_contribution(
            n, supply_links, l_names, "bus2", "efficiency2",
            "efficiency2", snapshots,
        )
        bus3_contrib = _bus_contribution(
            n, supply_links, l_names, "bus3", "efficiency3",
            "efficiency3", snapshots,
        )
        c_eff_l = (c_l + lam_fuel - bus2_contrib - bus3_contrib) / eff_l
        # dispatch to AC: -p1 (p1 is negative when link delivers to bus1)
        p_ac_l = -n.links_t.p1.reindex(columns=l_names, index=snapshots).fillna(0.0)
        # bounds on the AC-side: p_nom * eta * p_max_pu (for |-p1|)
        pnom0 = _p_nom_matrix(supply_links, l_names, snapshots)
        pmax0 = _ts_matrix(n.links_t.p_max_pu, l_names, snapshots,
                           supply_links.loc[l_names, "p_max_pu"].astype(float))
        pmin0 = _ts_matrix(n.links_t.p_min_pu, l_names, snapshots,
                           supply_links.loc[l_names, "p_min_pu"].astype(float))
        # Convert bounds to the bus1 (AC-side) by multiplying by eta
        pnom_l = pnom0 * eff_l
        pmax_l = pmax0   # pu is identical
        pmin_l = pmin0
        c_eff_parts.append(c_eff_l)
        p_ac_parts.append(p_ac_l)
        pnom_parts.append(pnom_l)
        pmax_parts.append(pmax_l)
        pmin_parts.append(pmin_l)
        metas.append(pd.DataFrame({
            "name": l_names,
            "carrier": supply_links.loc[l_names, "carrier"].values,
            "ctype": "Link",
            "fuel_bus": bus0s,
            "role": supply_links.loc[l_names, "role"].values,
        }))

    if not metas:
        empty = pd.DataFrame(index=snapshots)
        return empty.copy(), empty.copy(), empty.copy(), empty.copy(), empty.copy(), \
               pd.DataFrame(columns=["name", "carrier", "ctype", "fuel_bus", "role"])

    c_eff = pd.concat(c_eff_parts, axis=1)
    p_ac = pd.concat(p_ac_parts, axis=1)
    p_nom = pd.concat(pnom_parts, axis=1)
    pmax = pd.concat(pmax_parts, axis=1)
    pmin = pd.concat(pmin_parts, axis=1)
    meta = pd.concat(metas, ignore_index=True).set_index("name")
    return c_eff, p_ac, p_nom, pmax, pmin, meta


def demand_matrices(n, demand_links, snapshots):
    """Return c_eff (willingness-to-pay), p_with (withdrawal at AC/LV side),
    p_nom, pmax_pu, pmin_pu, meta."""
    if not len(demand_links):
        empty = pd.DataFrame(index=snapshots)
        return empty.copy(), empty.copy(), empty.copy(), empty.copy(), empty.copy(), \
               pd.DataFrame(columns=["name", "carrier", "ctype", "sink_bus",
                                     "role", "on_lv"])
    names = demand_links.index.tolist()
    bus1s = demand_links.loc[names, "bus1"].values
    lam_sink = (
        n.buses_t.marginal_price
        .reindex(columns=bus1s, index=snapshots)
        .set_axis(names, axis=1)
        .astype(float)
    )
    c_l = _ts_matrix(
        n.links_t.marginal_cost, names, snapshots,
        demand_links.loc[names, "marginal_cost"].astype(float),
    )
    eff_l = _ts_matrix(
        n.links_t.efficiency, names, snapshots,
        demand_links.loc[names, "efficiency"].astype(float),
    )
    # Co-outputs / co-inputs at bus2, bus3 (e.g. Haber-Bosch consumes H2 at
    # bus2 via negative efficiency2). Interior LP condition for a demand link
    # (bus0 = AC input): η₁·λ_bus1 + η₂·λ_bus2 + η₃·λ_bus3 = λ_AC + c
    bus2_contrib = _bus_contribution(
        n, demand_links, names, "bus2", "efficiency2",
        "efficiency2", snapshots,
    )
    bus3_contrib = _bus_contribution(
        n, demand_links, names, "bus3", "efficiency3",
        "efficiency3", snapshots,
    )
    c_eff = eff_l * lam_sink + bus2_contrib + bus3_contrib - c_l
    p_with = n.links_t.p0.reindex(columns=names, index=snapshots).fillna(0.0).clip(lower=0.0)
    p_nom = _p_nom_matrix(demand_links, names, snapshots)
    pmax = _ts_matrix(n.links_t.p_max_pu, names, snapshots,
                      demand_links.loc[names, "p_max_pu"].astype(float))
    pmin = _ts_matrix(n.links_t.p_min_pu, names, snapshots,
                      demand_links.loc[names, "p_min_pu"].astype(float))
    meta = pd.DataFrame({
        "name": names,
        "carrier": demand_links.loc[names, "carrier"].values,
        "ctype": "Link",
        "sink_bus": bus1s,
        "role": demand_links.loc[names, "role"].values,
        "on_lv": demand_links.loc[names, "on_lv"].values,
    }).set_index("name")
    return c_eff, p_with, p_nom, pmax, pmin, meta


def storage_unit_matrices(n, storage_units, snapshots):
    """Return p_dispatch (positive), p_nom, and meta for SUs at `bus`."""
    if not len(storage_units):
        empty = pd.DataFrame(index=snapshots)
        return empty.copy(), empty.copy(), pd.DataFrame(columns=["name", "carrier"])
    names = storage_units.index.tolist()
    p = n.storage_units_t.p.reindex(columns=names, index=snapshots).fillna(0.0)
    p_nom = _p_nom_matrix(storage_units, names, snapshots)
    meta = pd.DataFrame({
        "name": names,
        "carrier": storage_units.loc[names, "carrier"].values,
    }).set_index("name")
    return p, p_nom, meta


def transmission_analysis(n, bus, snapshots):
    """For each snapshot, report the incident line/DC-link most at capacity
    whose flow is *into* `bus` (i.e. `bus` is importing). When `bus` is the
    exporting side of a saturated line the price is set locally, not by
    transmission, so those lines are discarded.

    Returns a DataFrame indexed by snapshot with columns:
        name, kind, other_bus, flow_frac, other_price, direction
    """
    lines = n.lines[(n.lines.bus0 == bus) | (n.lines.bus1 == bus)].copy()
    dc_links = n.links[
        ((n.links.bus0 == bus) | (n.links.bus1 == bus)) & (n.links.carrier == "DC")
    ].copy()

    cols = list(lines.index) + list(dc_links.index)
    if not cols:
        return pd.DataFrame(
            index=snapshots,
            columns=["name", "kind", "other_bus", "flow_frac", "other_price", "direction"],
        )

    flow_frac = pd.DataFrame(0.0, index=snapshots, columns=cols)
    inbound = pd.DataFrame(False, index=snapshots, columns=cols)
    other_bus = pd.Series("", index=cols, dtype=object)
    kind = pd.Series("", index=cols, dtype=object)

    # A line's "p0" is the flow out of bus0. For bus==bus0, p0>0 means
    # `bus` exports; p0<0 means `bus` imports. For bus==bus1, the sign is
    # reversed.
    if len(lines):
        sn = lines.get("s_nom_opt", lines["s_nom"]).astype(float)
        p0 = n.lines_t.p0.reindex(columns=lines.index, index=snapshots).fillna(0.0)
        signed_flow_into_bus = p0.multiply(
            np.where(lines.bus0.values == bus, -1.0, 1.0), axis=1
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            frac = p0.abs().divide(sn.replace(0.0, np.nan), axis=1).fillna(0.0)
        flow_frac.loc[:, lines.index] = frac
        inbound.loc[:, lines.index] = signed_flow_into_bus > 0
        other_bus.loc[lines.index] = np.where(
            lines.bus0 == bus, lines.bus1.values, lines.bus0.values
        )
        kind.loc[lines.index] = "AC-line"

    if len(dc_links):
        pn = dc_links.get("p_nom_opt", dc_links["p_nom"]).astype(float)
        p0 = n.links_t.p0.reindex(columns=dc_links.index, index=snapshots).fillna(0.0)
        signed_flow_into_bus = p0.multiply(
            np.where(dc_links.bus0.values == bus, -1.0, 1.0), axis=1
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            frac = p0.abs().divide(pn.replace(0.0, np.nan), axis=1).fillna(0.0)
        flow_frac.loc[:, dc_links.index] = frac
        inbound.loc[:, dc_links.index] = signed_flow_into_bus > 0
        other_bus.loc[dc_links.index] = np.where(
            dc_links.bus0 == bus, dc_links.bus1.values, dc_links.bus0.values
        )
        kind.loc[dc_links.index] = "DC-link"

    # restrict to inbound lines only, then pick the one most at capacity
    eligible_frac = flow_frac.where(inbound, other=0.0)
    best_name = eligible_frac.idxmax(axis=1)
    best_frac = eligible_frac.max(axis=1)

    other_bus_col = best_name.map(other_bus)
    kind_col = best_name.map(kind)
    other_price = pd.Series(
        [n.buses_t.marginal_price.at[t, ob]
         if ob and ob in n.buses_t.marginal_price.columns else np.nan
         for t, ob in zip(snapshots, other_bus_col)],
        index=snapshots,
    )
    return pd.DataFrame({
        "name": best_name,
        "kind": kind_col,
        "other_bus": other_bus_col,
        "flow_frac": best_frac,
        "other_price": other_price,
        "direction": "import",
    })


# ── Masks ────────────────────────────────────────────────────────────────────
def interior_mask(p, p_nom, pmin_pu, pmax_pu):
    lo = p_nom * pmin_pu
    hi = p_nom * pmax_pu
    tol = np.maximum(BOUND_TOL_ABS, BOUND_TOL_REL * p_nom.abs())
    return (p > lo + tol) & (p < hi - tol)


def price_match(c_eff, lam):
    tol = np.maximum(PRICE_TOL_ABS, PRICE_TOL_REL * np.abs(lam))
    return (c_eff.sub(lam, axis=0).abs()).le(tol, axis=0)


# ── Classification cascade ───────────────────────────────────────────────────
def classify(n, bus):
    snapshots = n.snapshots
    lam = n.buses_t.marginal_price[bus].astype(float).reindex(snapshots)

    gens, supply_links, demand_links, su = build_candidates(n, bus)
    c_s, p_s, pn_s, pmax_s, pmin_s, meta_s = supply_matrices(n, gens, supply_links, snapshots)
    c_d, p_d, pn_d, pmax_d, pmin_d, meta_d = demand_matrices(n, demand_links, snapshots)
    p_su, pn_su, meta_su = storage_unit_matrices(n, su, snapshots)

    cong = transmission_analysis(n, bus, snapshots)

    # --- residuals and masks for supply & demand
    def resolve(c_eff, p, p_nom, pmin, pmax, meta):
        if c_eff.empty:
            return None
        interior = interior_mask(p, p_nom, pmin, pmax)
        match = price_match(c_eff, lam)
        residual = c_eff.sub(lam, axis=0).abs()
        residual_masked = residual.where(interior & match, other=np.inf)
        best_col = residual_masked.idxmin(axis=1)
        best_val = residual_masked.min(axis=1)
        has_match = np.isfinite(best_val)
        return dict(best=best_col, resid=best_val, has=has_match,
                    c_eff=c_eff, residual_all=residual, meta=meta)

    sup_res = resolve(c_s, p_s, pn_s, pmin_s, pmax_s, meta_s)
    dem_res = resolve(c_d, p_d, pn_d, pmin_d, pmax_d, meta_d)

    # --- storage unit interior?
    if not p_su.empty:
        su_dispatching = p_su.abs() > np.maximum(BOUND_TOL_ABS, BOUND_TOL_REL * pn_su.abs())
        su_interior = (p_su.abs() < pn_su.abs() - np.maximum(BOUND_TOL_ABS, BOUND_TOL_REL * pn_su.abs()))
        su_active = su_dispatching & su_interior
        best_su = pd.Series("", index=snapshots, dtype=object)
        for t in snapshots:
            hits = su_active.columns[su_active.loc[t].values]
            if len(hits):
                best_su.loc[t] = hits[0]
    else:
        best_su = pd.Series("", index=snapshots, dtype=object)

    rows = []
    for t in snapshots:
        pr = lam.loc[t]
        row = {
            "snapshot": t,
            "price_eur_per_mwh": pr,
            "classification": "unresolved",
            "price_setter_carrier": "",
            "price_setter_component": "",
            "fuel_bus": "",
            "fuel_price": np.nan,
            "efficiency": np.nan,
            "on_lv": False,
            "residual": np.nan,
            "explanation": "",
        }

        def pick_supply():
            if sup_res is None or not sup_res["has"].loc[t]:
                return False
            nm = sup_res["best"].loc[t]
            m = sup_res["meta"].loc[nm]
            row["classification"] = "storage_discharger" if m["role"] == "discharger" else "supply"
            row["price_setter_carrier"] = m["carrier"]
            row["price_setter_component"] = nm
            row["fuel_bus"] = m["fuel_bus"]
            row["residual"] = sup_res["resid"].loc[t]
            if m["ctype"] == "Generator":
                mc = c_s.at[t, nm]
                row["efficiency"] = 1.0
                row["fuel_price"] = np.nan
                row["explanation"] = (
                    f"lambda_AC({bus}, {t:%Y-%m-%d %H:%M}) = {pr:.2f} EUR/MWh "
                    f"= marginal_cost[{m['carrier']}] = {mc:.2f}"
                )
            else:
                eff = n.links_t.efficiency.at[t, nm] if nm in getattr(n.links_t.efficiency, "columns", []) \
                      else float(n.links.at[nm, "efficiency"])
                c = n.links_t.marginal_cost.at[t, nm] if nm in getattr(n.links_t.marginal_cost, "columns", []) \
                    else float(n.links.at[nm, "marginal_cost"])
                lf = n.buses_t.marginal_price.at[t, m["fuel_bus"]]
                row["efficiency"] = eff
                row["fuel_price"] = lf
                row["explanation"] = (
                    f"lambda_AC({bus}, {t:%Y-%m-%d %H:%M}) = {pr:.2f} EUR/MWh "
                    f"= (c_{m['carrier']} + lambda_{{{m['fuel_bus']}}})/eta "
                    f"= ({c:.2f} + {lf:.2f})/{eff:.3f} = {(c + lf) / eff:.2f}"
                )
            return True

        def pick_demand():
            if dem_res is None or not dem_res["has"].loc[t]:
                return False
            nm = dem_res["best"].loc[t]
            m = dem_res["meta"].loc[nm]
            row["classification"] = "storage_charger" if m["role"] == "charger" else "demand_flex"
            row["price_setter_carrier"] = m["carrier"]
            row["price_setter_component"] = nm
            row["fuel_bus"] = m["sink_bus"]
            row["on_lv"] = bool(m["on_lv"])
            row["residual"] = dem_res["resid"].loc[t]
            eff = n.links_t.efficiency.at[t, nm] if nm in getattr(n.links_t.efficiency, "columns", []) \
                  else float(n.links.at[nm, "efficiency"])
            c = n.links_t.marginal_cost.at[t, nm] if nm in getattr(n.links_t.marginal_cost, "columns", []) \
                else float(n.links.at[nm, "marginal_cost"])
            ls = n.buses_t.marginal_price.at[t, m["sink_bus"]]
            row["efficiency"] = eff
            row["fuel_price"] = ls
            lv_note = " [LV-projected]" if m["on_lv"] else ""
            row["explanation"] = (
                f"lambda_AC({bus}, {t:%Y-%m-%d %H:%M}) = {pr:.2f} EUR/MWh "
                f"= eta * lambda_{{{m['sink_bus']}}} - c_{m['carrier']} "
                f"= {eff:.3f} * {ls:.2f} - {c:.2f} = {eff * ls - c:.2f}{lv_note}"
            )
            return True

        def pick_storage_unit():
            nm = best_su.loc[t]
            if not nm:
                return False
            m = meta_su.loc[nm]
            row["classification"] = "storage_unit"
            row["price_setter_carrier"] = m["carrier"]
            row["price_setter_component"] = nm
            row["explanation"] = (
                f"lambda_AC({bus}, {t:%Y-%m-%d %H:%M}) = {pr:.2f} EUR/MWh "
                f"= water value of {m['carrier']} (interior storage unit)"
            )
            return True

        def pick_transmission():
            if cong.empty or pd.isna(cong.at[t, "flow_frac"]):
                return False
            frac = cong.at[t, "flow_frac"]
            if frac < FLOW_CONGESTION_REL:
                return False
            ob = cong.at[t, "other_bus"]
            op = cong.at[t, "other_price"]
            if pd.isna(op) or not ob:
                return False
            # Economic sanity: an import only sets our price if the neighbour
            # is strictly cheaper (so our price = lambda_other + mu, mu>=0).
            if op > pr + PRICE_TOL_ABS:
                return False
            mu = pr - op
            row["classification"] = "transmission"
            row["price_setter_carrier"] = cong.at[t, "kind"]
            row["price_setter_component"] = cong.at[t, "name"]
            row["fuel_bus"] = ob
            row["fuel_price"] = op
            row["residual"] = 0.0  # by construction lambda_here = lambda_other + mu
            row["explanation"] = (
                f"lambda_AC({bus}, {t:%Y-%m-%d %H:%M}) = {pr:.2f} EUR/MWh "
                f"= lambda_{{{ob}}} + mu_line = {op:.2f} + {mu:.2f}  "
                f"(import via {cong.at[t, 'kind']} {cong.at[t, 'name']} "
                f"at {frac*100:.1f}% of capacity)"
            )
            return True

        def pick_load_shedding():
            gs = n.generators[(n.generators.bus == bus)
                              & (n.generators.marginal_cost >= VOLL_THRESH)]
            if not len(gs):
                return False
            active = gs.index[
                n.generators_t.p.reindex(columns=gs.index, index=[t]).fillna(0.0).iloc[0] > 1e-3
            ]
            if not len(active):
                return False
            nm = active[0]
            row["classification"] = "load_shedding"
            row["price_setter_carrier"] = n.generators.at[nm, "carrier"]
            row["price_setter_component"] = nm
            row["explanation"] = (
                f"lambda_AC({bus}, {t:%Y-%m-%d %H:%M}) = {pr:.2f} EUR/MWh "
                f"= VoLL ({n.generators.at[nm, 'marginal_cost']:.0f}) — load shedding via {nm}"
            )
            return True

        def pick_unresolved_best():
            # diagnostic fallback: report nearest c_eff across all categories
            candidates = []
            if sup_res is not None:
                r = sup_res["residual_all"].loc[t]
                if r.size:
                    nm = r.idxmin()
                    candidates.append(("supply?", nm, r.loc[nm], sup_res["meta"].loc[nm, "carrier"]))
            if dem_res is not None:
                r = dem_res["residual_all"].loc[t]
                if r.size:
                    nm = r.idxmin()
                    candidates.append(("demand_flex?", nm, r.loc[nm], dem_res["meta"].loc[nm, "carrier"]))
            if not candidates:
                row["explanation"] = (
                    f"lambda_AC({bus}, {t:%Y-%m-%d %H:%M}) = {pr:.2f} EUR/MWh — no candidate found"
                )
                return
            candidates.sort(key=lambda x: x[2])
            kind_, nm, resid, car = candidates[0]
            row["price_setter_carrier"] = car
            row["price_setter_component"] = nm
            row["residual"] = resid
            row["explanation"] = (
                f"lambda_AC({bus}, {t:%Y-%m-%d %H:%M}) = {pr:.2f} EUR/MWh "
                f"— unresolved; nearest candidate {car} ({nm}) with residual {resid:.2f}"
            )

        if not (pick_supply() or pick_demand() or pick_storage_unit()
                or pick_transmission() or pick_load_shedding()):
            pick_unresolved_best()

        rows.append(row)

    return pd.DataFrame(rows).set_index("snapshot")


# ── Plot ─────────────────────────────────────────────────────────────────────
CLASS_FALLBACK_COLOR = {
    "transmission": "#555555",
    "unresolved": "#bbbbbb",
    "load_shedding": "#ff0040",
    "storage_unit": "#2b8cbe",
}


def _color_for_row(row, tech_colors):
    car = row["price_setter_carrier"]
    if row["classification"] in CLASS_FALLBACK_COLOR and (car == "" or car not in tech_colors):
        return CLASS_FALLBACK_COLOR[row["classification"]]
    return tech_colors.get(car, CLASS_FALLBACK_COLOR["unresolved"])


def make_timeseries_plot(table, bus, tech_colors):
    fig, (ax_price, ax_class) = plt.subplots(
        2, 1, sharex=True, figsize=(12, 6),
        gridspec_kw={"height_ratios": [3, 1]},
    )

    colors = [_color_for_row(r, tech_colors) for _, r in table.iterrows()]

    ax_price.plot(table.index, table["price_eur_per_mwh"],
                  color="lightgrey", lw=0.8, zorder=1)
    ax_price.scatter(table.index, table["price_eur_per_mwh"],
                     c=colors, s=32, edgecolor="black", linewidth=0.3,
                     zorder=3)
    ax_price.set_ylabel(f"λ_AC({bus}) [EUR/MWh]")
    style_ax(ax_price)

    # bottom: classification strip
    tnum = pd.Series(range(len(table)), index=table.index)
    widths = np.ones(len(table))
    ax_class.bar(tnum, height=np.ones(len(table)), width=1.0,
                 color=colors, edgecolor="none", align="edge")
    ax_class.set_yticks([])
    ax_class.set_xticks([])
    ax_class.set_ylabel("class")
    for spine in ["top", "right", "left", "bottom"]:
        ax_class.spines[spine].set_visible(False)

    # figure-level legend (unique classes present)
    seen = {}
    for (_, row), col in zip(table.iterrows(), colors):
        lbl = row["classification"]
        if row["price_setter_carrier"]:
            lbl = f"{row['classification']}: {row['price_setter_carrier']}"
        seen.setdefault(lbl, col)
    handles = [Patch(facecolor=c, edgecolor="black", label=l) for l, c in seen.items()]
    fig.legend(handles=handles, loc="lower center", ncol=min(4, len(handles)),
               frameon=True, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(f"Price-setter classification at {bus}  ({SCENARIO}, wiggle={WIGGLE})")
    fig.tight_layout(rect=[0, 0.04, 1, 0.97])

    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    for out in (SCRIPT_DIR / OUT_DETAIL_PLOT, PAPER_DIR / OUT_DETAIL_PLOT):
        fig.savefig(out, bbox_inches="tight")
        print(f"Saved {out}")
    plt.close(fig)


def make_shares_plot(combined_table, n, tech_colors):
    """Stacked bar per AC bus: snapshot-weighted share of each price-setter label."""
    w = n.snapshot_weightings.generators

    # Row label = "{classification}: {carrier}" to distinguish, e.g.,
    # 'supply: onwind' from 'supply: CCGT'.
    df = combined_table.copy()
    df["label"] = df.apply(
        lambda r: (f"{r['classification']}: {r['price_setter_carrier']}"
                   if r["price_setter_carrier"]
                   else r["classification"]),
        axis=1,
    )
    df["weight"] = df["snapshot"].map(w)

    # (bus, label) -> weight
    shares = (
        df.groupby(["bus", "label"])["weight"].sum()
          .unstack(fill_value=0.0)
    )
    shares = shares.div(shares.sum(axis=1), axis=0)

    # order buses by mean price (cheap → expensive)
    mean_price = df.groupby("bus")["price_eur_per_mwh"].mean()
    shares = shares.loc[mean_price.sort_values().index]

    # order labels by total share descending
    shares = shares[shares.sum().sort_values(ascending=False).index]

    def color_for_label(lbl):
        if ": " in lbl:
            cls, car = lbl.split(": ", 1)
        else:
            cls, car = lbl, ""
        if cls in CLASS_FALLBACK_COLOR and (car == "" or car not in tech_colors):
            return CLASS_FALLBACK_COLOR[cls]
        return tech_colors.get(car, CLASS_FALLBACK_COLOR["unresolved"])

    colors = [color_for_label(l) for l in shares.columns]

    fig, ax = plt.subplots(figsize=(16, 7))
    bottoms = np.zeros(len(shares))
    for col, c in zip(shares.columns, colors):
        vals = shares[col].values
        ax.bar(shares.index, vals, bottom=bottoms, color=c,
               edgecolor="none", width=0.9, label=col)
        bottoms += vals

    ax.set_ylabel("share of snapshot-weighted hours")
    ax.set_xlabel("AC bus (sorted by mean price, low → high)")
    ax.set_ylim(0, 1)
    ax.set_xticks(range(len(shares.index)))
    ax.set_xticklabels(shares.index, rotation=90, fontsize=7)
    ax.legend(loc="center left", bbox_to_anchor=(1.005, 0.5),
              frameon=True, fontsize=7, title="price setter")
    style_ax(ax)
    fig.suptitle(f"Price-setter share per AC bus  ({SCENARIO}, wiggle={WIGGLE})")
    fig.tight_layout()

    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    for out in (SCRIPT_DIR / OUT_SHARES_PLOT, PAPER_DIR / OUT_SHARES_PLOT):
        fig.savefig(out, bbox_inches="tight")
        print(f"Saved {out}")
    plt.close(fig)


# ── Verification ─────────────────────────────────────────────────────────────
def verify(n, table, bus):
    w = n.snapshot_weightings.generators.reindex(table.index)
    # load at bus from Loads component
    loads_at_bus = n.loads[n.loads.bus == bus].index
    if len(loads_at_bus):
        load_ts = pd.Series(0.0, index=table.index)
        for ln in loads_at_bus:
            if ln in n.loads_t.p_set.columns:
                load_ts += n.loads_t.p_set[ln].reindex(table.index).fillna(0.0)
            else:
                load_ts += float(n.loads.at[ln, "p_set"])
        num = (table["price_eur_per_mwh"] * load_ts * w).sum()
        den = (load_ts * w).sum()
        price_lw = num / den if den > 0 else np.nan
        print(f"[verify] load-weighted mean price at {bus}: {price_lw:.2f} EUR/MWh")

    unresolved_frac = (
        (table["classification"] == "unresolved")
        .multiply(w, axis=0).sum() / w.sum()
    )
    passed = unresolved_frac <= 0.05
    print(f"[verify] unresolved share: {unresolved_frac*100:.1f}% "
          f"({'PASS' if passed else 'FAIL'} threshold 5%)")

    # CCGT sanity: only meaningful on snapshots where CCGT is actually
    # dispatching strictly inside its bounds AND the formula matches λ.
    ccgt = n.links[(n.links.bus1 == bus) & (n.links.carrier == "CCGT")]
    if len(ccgt):
        nm = ccgt.index[0]
        bus0 = ccgt.at[nm, "bus0"]
        eff = (n.links_t.efficiency[nm] if nm in n.links_t.efficiency.columns
               else pd.Series(float(n.links.at[nm, "efficiency"]), index=table.index))
        mc = (n.links_t.marginal_cost[nm] if nm in n.links_t.marginal_cost.columns
              else pd.Series(float(n.links.at[nm, "marginal_cost"]), index=table.index))
        lam_gas = n.buses_t.marginal_price[bus0].reindex(table.index)
        implied = (mc + lam_gas) / eff.replace(0.0, np.nan)
        diff = (implied - table["price_eur_per_mwh"]).abs()
        p0 = n.links_t.p0[nm].reindex(table.index).fillna(0.0)
        p_nom = float(n.links.at[nm, "p_nom_opt"] if "p_nom_opt" in n.links.columns
                      else n.links.at[nm, "p_nom"])
        tol = max(BOUND_TOL_ABS, BOUND_TOL_REL * p_nom)
        interior = (p0 > tol) & (p0 < p_nom - tol)
        mask = interior & (diff <= 0.1)
        if mask.any():
            tagged = (table.loc[mask, "price_setter_carrier"] == "CCGT").mean()
            print(f"[verify] formula-CCGT snapshots (dispatching & matching): {mask.sum()}, "
                  f"tagged CCGT: {tagged*100:.1f}% "
                  f"({'PASS' if tagged >= 0.95 else 'FAIL'} threshold 95%)")
        else:
            print("[verify] CCGT never both dispatching and formula-matching — skipped")

    # transmission consistency: each transmission row must correspond to
    # a snapshot where an incident line actually reached FLOW_CONGESTION_REL
    # capacity (tautological by construction, but catches logic regressions).
    trans = table[table["classification"] == "transmission"]
    if len(trans):
        cong = transmission_analysis(n, bus, trans.index)
        saturated = (cong["flow_frac"] >= FLOW_CONGESTION_REL).sum()
        ok = saturated == len(trans)
        print(f"[verify] transmission rows: {len(trans)}, "
              f"all at >={FLOW_CONGESTION_REL*100:.0f}% capacity: "
              f"{'PASS' if ok else 'FAIL'}")


# ── Main ─────────────────────────────────────────────────────────────────────
def save_combined_table(combined):
    try:
        combined.to_parquet(OUT_TABLE, index=False)
        print(f"Saved {OUT_TABLE}")
    except Exception as e:
        print(f"parquet write failed ({e}); writing CSV instead")
        combined.to_csv(OUT_CSV, index=False)
        print(f"Saved {OUT_CSV}")


def print_aggregate_summary(combined):
    print("\n=== Aggregate share of (bus, snapshot) pairs per classification ===")
    print((combined["classification"].value_counts(normalize=True) * 100)
          .round(1).to_string())
    print("\n=== One example per observed classification ===")
    for cls, sub in combined.groupby("classification"):
        ex = sub.iloc[0]
        print(f"[{cls}] bus={ex['bus']}  {ex['explanation']}")


def write_combined_explanation_file(combined):
    with open(OUT_EXPLAIN, "w") as f:
        for _, row in combined.iterrows():
            t = row["snapshot"]
            f.write(
                f"{row['bus']:<8}  {t:%Y-%m-%d %H:%M}  "
                f"[{row['classification']:<20}]  {row['explanation']}\n"
            )
    print(f"Saved {OUT_EXPLAIN}")


if __name__ == "__main__":
    path = network_path(scenario=SCENARIO, wiggle=WIGGLE, hike=HIKE)
    print(f"Loading {path.name}")
    n = pypsa.Network(path)

    ac_buses = list(n.buses.index[n.buses.carrier == "AC"])
    if BUS not in ac_buses:
        raise SystemExit(f"Bus {BUS!r} not in network. Available AC buses: "
                         f"{ac_buses[:10]}…")

    tech_colors = load_tech_colors()

    print(f"Classifying price-setter at {len(ac_buses)} AC buses "
          f"over {len(n.snapshots)} snapshots…")

    per_bus_tables = {}
    pieces = []
    for i, b in enumerate(ac_buses, 1):
        t = classify(n, b)
        t["bus"] = b
        per_bus_tables[b] = t
        pieces.append(t.reset_index())
        print(f"  [{i}/{len(ac_buses)}] {b}: "
              f"{(t['classification'] == 'unresolved').mean()*100:.1f}% unresolved")

    combined = pd.concat(pieces, ignore_index=True)
    combined = combined[[
        "bus", "snapshot", "price_eur_per_mwh", "classification",
        "price_setter_carrier", "price_setter_component", "fuel_bus",
        "fuel_price", "efficiency", "on_lv", "residual", "explanation",
    ]]

    save_combined_table(combined)
    write_combined_explanation_file(combined)
    print_aggregate_summary(combined)

    make_shares_plot(combined, n, tech_colors)

    # detail plot for the designated BUS
    print(f"\nDetail plot for {BUS}:")
    make_timeseries_plot(per_bus_tables[BUS], BUS, tech_colors)

    # verify on the designated BUS (full per-bus verify would flood stdout)
    print(f"\nVerification on {BUS}:")
    verify(n, per_bus_tables[BUS], BUS)
