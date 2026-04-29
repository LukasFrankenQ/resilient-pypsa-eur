"""
Across a sweep of gas-budget wiggle values, run the full priority cascade
(supply > demand > storage_unit > transmission, with fuzzy fallback for
unresolved snapshots) at every AC bus of every solved network. Collect the
per-(bus, snapshot) classification and the AC marginal price. Plot one
symmetric horizontal violin per (wiggle × classification) at:

    y = network-wide load-weighted gas price
    x = AC marginal price distribution for that subset

Violins are colored by classification, width scaled by sample count
relative to the global maximum across the sweep. Transmission-classified
pairs are excluded from the figure per request.
"""

import pypsa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.stats import gaussian_kde
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from classify_price_setter import (  # noqa: E402
    network_path,
    build_candidates,
    supply_matrices,
    demand_matrices,
    storage_unit_matrices,
    transmission_analysis,
    interior_mask,
    price_match,
    style_ax,
    _p_nom_matrix,
    SCENARIO,
    HIKE,
    BOUND_TOL_ABS,
    BOUND_TOL_REL,
    FLOW_CONGESTION_REL,
    PRICE_TOL_ABS,
)

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[3]
PAPER_DIR = ROOT.parent / "gas_resilience" / "imgs" / "price_formation"
SCRIPT_DIR = Path(__file__).resolve().parent
CACHE_DIR = SCRIPT_DIR / "cache_storage_vs_gas_v7"
CACHE_DIR.mkdir(exist_ok=True)

# ── Sweep ────────────────────────────────────────────────────────────────────
WIGGLES = [0, 125, 250, 500, 1000, 1500, 2000, 2500, 3000, 4000]

# Four price-setter groups: StorageUnits (hydro / PHS) are split by whether
# they are charging or discharging at each snapshot and folded into the
# corresponding Store-+-Link groups.
PLOT_CLASSES = [
    "supply",
    "demand_flex",
    "storage_discharger",
    "storage_charger",
]

# Supply is further broken down into buckets by carrier.
BIOMASS_CHP_CARRIERS = {
    "urban central solid biomass CHP",
    "urban central solid biomass CHP CC",
}

SUPPLY_BUCKETS = [
    ("wind & solar", [
        "onwind", "offwind-ac", "offwind-dc", "offwind-float",
        "solar", "solar-hsat",
    ]),
    ("biomass CHP", [
        "urban central solid biomass CHP",
        "urban central solid biomass CHP CC",
    ]),
    ("gas / waste CHP", [
        "urban central gas CHP", "waste CHP",
    ]),
    ("CCGT / OCGT", ["CCGT", "OCGT", "OCGT methanol"]),
    ("coal · lignite · oil", ["coal", "lignite", "oil"]),
    ("run-of-river hydro", ["ror"]),
]
SUPPLY_BUCKET_ORDER = [b[0] for b in SUPPLY_BUCKETS]
CARRIER_TO_BUCKET = {car: bucket for bucket, carriers in SUPPLY_BUCKETS
                     for car in carriers}

SUPPLY_BUCKET_COLORS = {
    "wind & solar":        "#f9c846",
    "biomass CHP":         "#6a7f3a",
    "gas / waste CHP":     "#e07b33",
    "CCGT / OCGT":         "#7a1410",
    "coal · lignite · oil": "#2f2f2f",
    "run-of-river hydro":  "#4a90e2",
}

# Some buckets want a different alpha to stand out visually (CCGT/OCGT
# otherwise blends with the orange gas / waste CHP bucket).
SUPPLY_BUCKET_ALPHA = {
    "CCGT / OCGT": 0.88,
}
DEFAULT_SUPPLY_ALPHA = 0.55

# Biomass CHP link efficiencies (static PyPSA-Eur inputs, verified in
# base_s_50__3H-T-H-B-I-A-dist1_2030_free_1000.nc at DE0 0).
BIOMASS_CHP_ETA_ELEC = 0.2699    # bus1 (AC) efficiency
BIOMASS_CHP_ETA_HEAT = 0.8245    # bus2 (urban central heat) efficiency2

CLASS_COLORS = {
    "supply":             "#d62728",
    "demand_flex":        "#9467bd",
    "storage_discharger": "#2ca02c",
    "storage_charger":    "#1f77b4",
    "transmission":       "#7f7f7f",
    "unresolved":         "#bbbbbb",
}

CLASS_TITLES = {
    "supply":             "Supply",
    "demand_flex":        "Demand-side flexibility",
    "storage_discharger": "Storage discharging",
    "storage_charger":    "Storage charging",
}

# Front-end-friendly description of technologies per group.
CLASS_DESCRIPTIONS = {
    "supply": (
        "Generators and conversion links that\n"
        "deliver electricity to the AC bus:\n"
        "  • Biomass / gas / waste CHPs\n"
        "  • CCGT, OCGT, coal, lignite\n"
        "  • Nuclear, oil\n"
        "  • H₂ turbines, H₂ fuel cells\n"
        "  • On- and offshore wind\n"
        "  • Utility and tracked solar\n"
        "  • Run-of-river hydro"
    ),
    "demand_flex": (
        "Flexible electric loads whose marginal\n"
        "willingness-to-pay sets the price:\n"
        "  • H₂ electrolysers\n"
        "  • Industrial & residential heat pumps\n"
        "  • Resistive heaters\n"
        "  • Industrial electric boilers\n"
        "  • Haber–Bosch (ammonia synthesis)\n"
        "  • H₂ DRI & EAF (electric steelmaking)"
    ),
    "storage_discharger": (
        "Components supplying electricity\n"
        "out of a stored-energy reservoir:\n"
        "  • Grid-battery dischargers\n"
        "  • Home-battery dischargers\n"
        "  • V2G (EVs discharging)\n"
        "  • Pumped hydro when generating\n"
        "  • Reservoir hydro when generating"
    ),
    "storage_charger": (
        "Components absorbing electricity\n"
        "to replenish a stored-energy reservoir:\n"
        "  • Grid-battery chargers\n"
        "  • Home-battery chargers\n"
        "  • BEV charging\n"
        "  • Pumped hydro when pumping"
    ),
}

OUT_PLOT = "price_setting_marginal_price_distribution.pdf"
XLIM = (0.0, 225.0)
DENSITY_CUTOFF_FRAC = 0.10
MIN_SPREAD_PILL = 1.5            # EUR/MWh — below this width, use a pill
PILL_HALF_WIDTH = 1.2            # half-width of pill in EUR/MWh
CONN_COLOR = "#c0001f"


# ── Full vectorised priority cascade (all classes) ──────────────────────────
def classify_all_vectorised(n, bus):
    """Return (cls, carrier, lam) for bus. `cls` is a Series of classification
    strings (supply / demand_flex / storage_discharger / storage_charger /
    transmission / unresolved), `carrier` is the winning component's carrier
    (empty string if transmission / unresolved), and `lam` is the AC price.
    """
    snaps = n.snapshots
    lam = n.buses_t.marginal_price[bus].astype(float).reindex(snaps)

    gens, sup_links, dem_links, su = build_candidates(n, bus)

    # ── supply cascade ──
    strict_has_s = np.zeros(len(snaps), dtype=bool)
    sup_cls_str = np.full(len(snaps), "", dtype=object)
    sup_car_str = np.full(len(snaps), "", dtype=object)
    sup_nearest_val = pd.Series(np.inf, index=snaps)
    sup_nearest_cls_str = np.full(len(snaps), "", dtype=object)
    sup_nearest_car_str = np.full(len(snaps), "", dtype=object)

    # Drop solid biomass CHPs from the AC-side supply candidate pool.
    # Their LP-rearranged c_eff_AC = (c + λ_fuel − η₂λ_heat − η₃λ_co2)/η₁
    # is the KKT stationarity identity rearranged for λ_AC: at any
    # snapshot where the link's dispatch is interior, the residual
    # collapses to machine-precision zero and biomass CHP wins the
    # strict cascade by construction — even when it is really the
    # heat-bus marginal and λ_AC is set by some other component. We
    # therefore exclude biomass CHPs from the supply candidates here
    # so the supply violin reflects only single-port AC marginals.
    sup_links = sup_links[~sup_links.carrier.isin(BIOMASS_CHP_CARRIERS)]

    if not sup_links.empty or not gens.empty:
        c_s, p_s, pn_s, pmax_s, pmin_s, meta_s = supply_matrices(
            n, gens, sup_links, snaps
        )
        if not c_s.empty:
            inter_s = interior_mask(p_s, pn_s, pmin_s, pmax_s)
            match_s = price_match(c_s, lam)
            residual_s = c_s.sub(lam, axis=0).abs()
            masked = residual_s.where(inter_s & match_s, other=np.inf)
            best = masked.idxmin(axis=1)
            strict_has_s = np.isfinite(masked.min(axis=1).values)
            best_ok = best.where(strict_has_s, other="")
            role_map_s = meta_s["role"].to_dict()
            car_map_s = meta_s["carrier"].to_dict()
            sup_roles = best_ok.map(lambda nm: role_map_s.get(nm, ""))
            sup_cls_str = np.where(sup_roles == "discharger",
                                   "storage_discharger", "supply")
            sup_car_str = best_ok.map(lambda nm: car_map_s.get(nm, "")).values

            sup_nearest_val = residual_s.min(axis=1)
            sup_nearest_name = residual_s.idxmin(axis=1)
            sup_nearest_roles = sup_nearest_name.map(
                lambda nm: role_map_s.get(nm, "")
            )
            sup_nearest_cls_str = np.where(sup_nearest_roles == "discharger",
                                           "storage_discharger", "supply")
            sup_nearest_car_str = sup_nearest_name.map(
                lambda nm: car_map_s.get(nm, "")
            ).values

    # ── demand cascade ──
    strict_has_d = np.zeros(len(snaps), dtype=bool)
    dem_cls_str = np.full(len(snaps), "", dtype=object)
    dem_car_str = np.full(len(snaps), "", dtype=object)
    dem_nearest_val = pd.Series(np.inf, index=snaps)
    dem_nearest_cls_str = np.full(len(snaps), "", dtype=object)
    dem_nearest_car_str = np.full(len(snaps), "", dtype=object)

    if not dem_links.empty:
        c_d, p_d, pn_d, pmax_d, pmin_d, meta_d = demand_matrices(
            n, dem_links, snaps
        )
        inter_d = interior_mask(p_d, pn_d, pmin_d, pmax_d)
        match_d = price_match(c_d, lam)
        residual_d = c_d.sub(lam, axis=0).abs()
        masked = residual_d.where(inter_d & match_d, other=np.inf)
        best = masked.idxmin(axis=1)
        strict_has_d = np.isfinite(masked.min(axis=1).values)
        best_ok = best.where(strict_has_d, other="")
        role_map_d = meta_d["role"].to_dict()
        car_map_d = meta_d["carrier"].to_dict()
        dem_roles = best_ok.map(lambda nm: role_map_d.get(nm, ""))
        dem_cls_str = np.where(dem_roles == "charger",
                               "storage_charger", "demand_flex")
        dem_car_str = best_ok.map(lambda nm: car_map_d.get(nm, "")).values

        dem_nearest_val = residual_d.min(axis=1)
        dem_nearest_name = residual_d.idxmin(axis=1)
        dem_nearest_roles = dem_nearest_name.map(
            lambda nm: role_map_d.get(nm, "")
        )
        dem_nearest_cls_str = np.where(dem_nearest_roles == "charger",
                                       "storage_charger", "demand_flex")
        dem_nearest_car_str = dem_nearest_name.map(
            lambda nm: car_map_d.get(nm, "")
        ).values

    # ── storage unit (hydro / PHS) ──
    # Unlike Store+Link pairs, a StorageUnit encapsulates both charging and
    # discharging in one component. We detect mode per snapshot from
    # p_dispatch (≥0) and p_store (≥0) and assign to storage_discharger or
    # storage_charger accordingly.
    su_disp_car = np.full(len(snaps), "", dtype=object)
    su_stor_car = np.full(len(snaps), "", dtype=object)
    if not su.empty:
        names_su = su.index.tolist()
        pn_su = _p_nom_matrix(su, names_su, snaps).abs()
        tol_su = np.maximum(BOUND_TOL_ABS, BOUND_TOL_REL * pn_su)
        p_disp = n.storage_units_t.p_dispatch.reindex(
            columns=names_su, index=snaps
        ).fillna(0.0)
        p_stor = n.storage_units_t.p_store.reindex(
            columns=names_su, index=snaps
        ).fillna(0.0)
        disp_active = (p_disp > tol_su) & (p_disp < pn_su - tol_su)
        stor_active = (p_stor > tol_su) & (p_stor < pn_su - tol_su)
        su_discharging = disp_active.any(axis=1).values
        su_charging = stor_active.any(axis=1).values
        # carrier of the most-strongly-dispatching/storing unit at each snap
        su_carriers = su["carrier"].to_dict()
        disp_vals = p_disp.where(disp_active, other=0.0)
        if len(disp_vals.columns):
            disp_name = disp_vals.idxmax(axis=1)
            su_disp_car = disp_name.map(lambda nm: su_carriers.get(nm, "")).values
        stor_vals = p_stor.where(stor_active, other=0.0)
        if len(stor_vals.columns):
            stor_name = stor_vals.idxmax(axis=1)
            su_stor_car = stor_name.map(lambda nm: su_carriers.get(nm, "")).values
    else:
        su_discharging = np.zeros(len(snaps), dtype=bool)
        su_charging = np.zeros(len(snaps), dtype=bool)
    su_has = su_discharging | su_charging

    # ── transmission ──
    cong = transmission_analysis(n, bus, snaps)
    if not cong.empty:
        frac = cong["flow_frac"].values
        op = cong["other_price"].values
        tr_has = (frac >= FLOW_CONGESTION_REL) & np.isfinite(op) \
                 & (op <= lam.values + PRICE_TOL_ABS)
    else:
        tr_has = np.zeros(len(snaps), dtype=bool)

    # ── cascade ──
    cls = np.full(len(snaps), "unresolved", dtype=object)
    car = np.full(len(snaps), "", dtype=object)
    m1 = strict_has_s
    cls[m1] = sup_cls_str[m1]
    car[m1] = sup_car_str[m1]
    m2 = (~strict_has_s) & strict_has_d
    cls[m2] = dem_cls_str[m2]
    car[m2] = dem_car_str[m2]
    # StorageUnit: route to discharger / charger bucket based on mode.
    # If both modes are active in the same snapshot (different SUs), prefer
    # the discharging side (it's on the supply cascade side of the balance).
    m3_disch = (~strict_has_s) & (~strict_has_d) & su_discharging
    cls[m3_disch] = "storage_discharger"
    car[m3_disch] = su_disp_car[m3_disch]
    m3_chg = (~strict_has_s) & (~strict_has_d) & (~su_discharging) & su_charging
    cls[m3_chg] = "storage_charger"
    car[m3_chg] = su_stor_car[m3_chg]
    m4 = (~strict_has_s) & (~strict_has_d) & (~su_has) & tr_has
    cls[m4] = "transmission"

    # ── fuzzy fallback ──
    # Remaining unresolved snapshots are attributed to the nearest-residual
    # candidate (supply or demand, whichever has smaller residual).
    unresolved = cls == "unresolved"
    sup_wins = sup_nearest_val.values <= dem_nearest_val.values
    fuzzy_cls = np.where(sup_wins, sup_nearest_cls_str, dem_nearest_cls_str)
    fuzzy_car = np.where(sup_wins, sup_nearest_car_str, dem_nearest_car_str)
    fuzzy_cls = np.where(fuzzy_cls == "", "unresolved", fuzzy_cls)
    cls[unresolved] = fuzzy_cls[unresolved]
    car[unresolved] = fuzzy_car[unresolved]

    return pd.Series(cls, index=snaps), pd.Series(car, index=snaps), lam


# ── Withdrawal-weighted AC electricity price ────────────────────────────────
def weighted_elec_price(n):
    """Withdrawal-weighted mean AC price. Withdrawals from each AC bus
    include: direct Loads, p0 of any Link with bus0 = that AC bus (heat
    pumps, electrolysers, battery chargers, DC links, distribution-grid
    links — everything that pulls power out of AC) and StorageUnit
    charging (p_store)."""
    ac_buses = n.buses.index[n.buses.carrier == "AC"]
    if not len(ac_buses):
        return float("nan")
    prices = n.buses_t.marginal_price[ac_buses].astype(float)
    w = n.snapshot_weightings.generators

    wd = pd.DataFrame(0.0, index=n.snapshots, columns=ac_buses)

    # direct Loads
    loads = n.loads[n.loads.bus.isin(ac_buses)]
    for name, row in loads.iterrows():
        bus = row.bus
        if name in n.loads_t.p_set.columns:
            wd[bus] += n.loads_t.p_set[name].reindex(n.snapshots).fillna(0.0)
        else:
            wd[bus] += float(row.p_set)

    # Link withdrawals (p0 > 0 at AC-side bus0)
    consumers = n.links[n.links.bus0.isin(ac_buses)]
    if len(consumers):
        p0 = n.links_t.p0.reindex(columns=consumers.index,
                                  index=n.snapshots).fillna(0.0).clip(lower=0.0)
        for lnk in consumers.index:
            wd[n.links.at[lnk, "bus0"]] += p0[lnk]

    # StorageUnit charging
    sus = n.storage_units[n.storage_units.bus.isin(ac_buses)]
    if len(sus):
        p_stor = n.storage_units_t.p_store.reindex(
            columns=sus.index, index=n.snapshots
        ).fillna(0.0)
        for su in sus.index:
            wd[n.storage_units.at[su, "bus"]] += p_stor[su]

    total_wd = wd.multiply(w, axis=0).sum().sum()
    if total_wd <= 0:
        return float(prices.multiply(w, axis=0).sum().sum()
                     / (w.sum() * len(ac_buses)))
    num = (prices * wd).multiply(w, axis=0).sum().sum()
    return float(num / total_wd)


# ── Load-weighted gas price ─────────────────────────────────────────────────
def weighted_gas_price(n):
    gas_buses = n.buses.index[n.buses.carrier == "gas"]
    if not len(gas_buses):
        return float("nan")
    prices = n.buses_t.marginal_price[gas_buses].astype(float)
    w = n.snapshot_weightings.generators
    consumer_links = n.links[n.links.bus0.isin(gas_buses)]
    load_ts = pd.DataFrame(0.0, index=n.snapshots, columns=gas_buses)
    if len(consumer_links):
        p0 = n.links_t.p0.reindex(columns=consumer_links.index,
                                  index=n.snapshots).fillna(0.0).clip(lower=0.0)
        for lnk in consumer_links.index:
            load_ts[n.links.at[lnk, "bus0"]] += p0[lnk]
    total_load = load_ts.multiply(w, axis=0).sum().sum()
    if total_load <= 0:
        return float(prices.multiply(w, axis=0).sum().sum()
                     / (w.sum() * len(gas_buses)))
    num = (prices * load_ts).multiply(w, axis=0).sum().sum()
    return float(num / total_load)


# ── Per-wiggle collection with parquet cache ────────────────────────────────
ALL_CLASSES = PLOT_CLASSES + ["transmission", "unresolved"]


def _bucket_for_carrier(car):
    return CARRIER_TO_BUCKET.get(car, "other supply")


def _split_supply_by_bucket(df):
    """Return dict mapping supply-bucket name → array of elec_price."""
    supply = df[df["classification"] == "supply"]
    out = {b: np.array([]) for b in SUPPLY_BUCKET_ORDER}
    if supply.empty:
        return out
    buckets = supply["carrier"].map(_bucket_for_carrier)
    for b in SUPPLY_BUCKET_ORDER:
        out[b] = supply.loc[buckets == b, "elec_price"].values
    return out


def collect_for_wiggle(w):
    """Return dict(wiggle, gas_price, battery_breakeven, ccgt_eff,
    supply_buckets, demand_flex, storage_discharger, storage_charger).
    supply_buckets is dict bucket_name → array of elec_price."""
    cache = CACHE_DIR / f"ac_price_by_class_{SCENARIO}_{w}.parquet"
    n_loaded = None

    def _load_network():
        nonlocal n_loaded
        if n_loaded is None:
            path = network_path(scenario=SCENARIO, wiggle=w, hike=HIKE)
            if not path.exists():
                return None
            print(f"  wiggle={w}: loading {path.name}")
            n_loaded = pypsa.Network(path)
        return n_loaded

    if cache.exists():
        df = pd.read_parquet(cache)
        if {"classification", "carrier"}.issubset(df.columns):
            gas_price = (float(df["gas_price"].iloc[0])
                         if len(df) else float("nan"))
            # break-even / CCGT / gas share / withdrawal-weighted avg elec
            # price — newer caches. "elec_price_wd_avg" supersedes the older
            # load-weighted "elec_price_avg" column: its absence forces a
            # recompute with the wider (all-withdrawals) weighting.
            needed = {"battery_breakeven", "ccgt_eff", "gas_share",
                      "elec_price_wd_avg"}
            if needed.issubset(df.columns) and len(df):
                breakeven = float(df["battery_breakeven"].iloc[0])
                ccgt_eff = float(df["ccgt_eff"].iloc[0])
                gas_share = float(df["gas_share"].iloc[0])
                elec_price_avg = float(df["elec_price_wd_avg"].iloc[0])
            else:
                n = _load_network()
                if n is None:
                    return None
                breakeven, ccgt_eff = battery_breakeven_and_ccgt_eff(n)
                gas_share = gas_share_in_power_mix(n)
                elec_price_avg = weighted_elec_price(n)
                df["battery_breakeven"] = breakeven
                df["ccgt_eff"] = ccgt_eff
                df["gas_share"] = gas_share
                df["elec_price_wd_avg"] = elec_price_avg
                df.to_parquet(cache, index=False)
                print(f"  wiggle={w}: break-even={breakeven:.2f}, "
                      f"η_CCGT={ccgt_eff:.3f}, "
                      f"gas_share={gas_share*100:.1f}%, "
                      f"λ_AC_avg(withdrawal-weighted)="
                      f"{elec_price_avg:.2f} (cache updated)")
            supply_buckets = _split_supply_by_bucket(df)
            other = {
                c: df.loc[df["classification"] == c, "elec_price"].values
                for c in ["demand_flex", "storage_discharger", "storage_charger"]
            }
            tot_sup = sum(len(v) for v in supply_buckets.values())
            print(f"  wiggle={w}: cached — supply={tot_sup} "
                  f"(in {sum(len(v)>0 for v in supply_buckets.values())} buckets)  "
                  f"disch={len(other['storage_discharger'])}  "
                  f"chg={len(other['storage_charger'])}  "
                  f"flex={len(other['demand_flex'])}  gas={gas_price:.2f}  "
                  f"batt_be={breakeven:.2f}")
            return dict(wiggle=w, gas_price=gas_price,
                        battery_breakeven=breakeven, ccgt_eff=ccgt_eff,
                        gas_share=gas_share,
                        elec_price_avg=elec_price_avg,
                        supply_buckets=supply_buckets, **other)
        print(f"  wiggle={w}: cache schema mismatch, recomputing")

    n = _load_network()
    if n is None:
        print(f"  wiggle={w}: network missing, skipping")
        return None
    ac_buses = list(n.buses.index[n.buses.carrier == "AC"])

    rows = []
    for b in ac_buses:
        cls, car, lam = classify_all_vectorised(n, b)
        rows.append(pd.DataFrame({
            "bus": b,
            "classification": cls.values,
            "carrier": car.values,
            "elec_price": lam.values,
        }))
    df_all = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
        columns=["bus", "classification", "carrier", "elec_price"]
    )
    gas_price = weighted_gas_price(n)
    breakeven, ccgt_eff = battery_breakeven_and_ccgt_eff(n)
    gas_share = gas_share_in_power_mix(n)
    elec_price_avg = weighted_elec_price(n)

    cache_df = df_all[["elec_price", "classification", "carrier"]].copy()
    cache_df["gas_price"] = gas_price
    cache_df["battery_breakeven"] = breakeven
    cache_df["ccgt_eff"] = ccgt_eff
    cache_df["gas_share"] = gas_share
    cache_df["elec_price_wd_avg"] = elec_price_avg
    cache_df.to_parquet(cache, index=False)

    supply_buckets = _split_supply_by_bucket(df_all)
    other = {
        c: df_all.loc[df_all["classification"] == c, "elec_price"].values
        for c in ["demand_flex", "storage_discharger", "storage_charger"]
    }
    tot_sup = sum(len(v) for v in supply_buckets.values())

    print(f"  wiggle={w}: supply={tot_sup} "
          f"(in {sum(len(v)>0 for v in supply_buckets.values())} buckets)  "
          f"disch={len(other['storage_discharger'])}  "
          f"chg={len(other['storage_charger'])}  "
          f"flex={len(other['demand_flex'])}  gas={gas_price:.2f}  "
          f"batt_be={breakeven:.2f}  η_CCGT={ccgt_eff:.3f}  "
          f"gas_share={gas_share*100:.1f}%  "
          f"(cached → {cache.name})")
    return dict(wiggle=w, gas_price=gas_price,
                battery_breakeven=breakeven, ccgt_eff=ccgt_eff,
                gas_share=gas_share,
                elec_price_avg=elec_price_avg,
                supply_buckets=supply_buckets, **other)


# ── Violin helper ────────────────────────────────────────────────────────────
def _kde_violin_poly(data, xlim, half_width, cutoff_frac=DENSITY_CUTOFF_FRAC):
    data = data[(data >= xlim[0]) & (data <= xlim[1])]
    if len(data) < 5:
        return None, None
    kde = gaussian_kde(data)
    lo = max(xlim[0], float(np.percentile(data, 0.5)))
    hi = min(xlim[1], float(np.percentile(data, 99.5)))
    if hi <= lo:
        return None, None
    xs = np.linspace(lo, hi, 400)
    dens = kde(xs)
    peak = dens.max()
    if peak <= 0:
        return None, None
    keep = dens >= cutoff_frac * peak
    if keep.sum() < 3:
        return None, None
    return xs[keep], dens[keep] / peak * half_width


def _draw_shape(ax, data, gp, half_max, face, edgecolor="black", alpha=0.55,
                zorder=3, xlim=None):
    """Draw a KDE violin when the distribution is wide enough, or a short
    rectangular 'pill' when it is very narrow (e.g. near-zero-cost
    renewables whose prices cluster at x ≈ 0). Returns (x_left, x_right)
    of the drawn shape, or None if nothing was drawn.

    `xlim` overrides the module-level XLIM for clipping — needed when the
    left axis extends below zero to show near-zero VRE violins.
    """
    xlim = xlim if xlim is not None else XLIM
    data = data[(data >= xlim[0]) & (data <= xlim[1])]
    if len(data) < 5:
        return None
    lo = max(xlim[0], float(np.percentile(data, 0.5)))
    hi = min(xlim[1], float(np.percentile(data, 99.5)))
    if hi - lo < MIN_SPREAD_PILL:
        med = float(np.median(data))
        x0 = max(xlim[0], med - PILL_HALF_WIDTH)
        x1 = min(xlim[1], med + PILL_HALF_WIDTH)
        ax.fill_between([x0, x1], gp - half_max, gp + half_max,
                        facecolor=face, edgecolor=edgecolor,
                        linewidth=0.25, alpha=alpha, zorder=zorder)
        return (x0, x1)
    xs, half_y = _kde_violin_poly(data, xlim, half_max)
    if xs is None:
        return None
    ax.fill_between(xs, gp - half_y, gp + half_y,
                    facecolor=face, edgecolor=edgecolor,
                    linewidth=0.25, alpha=alpha, zorder=zorder)
    return (float(xs[0]), float(xs[-1]))


# ── Battery break-even + CCGT efficiency ────────────────────────────────────
def battery_breakeven_and_ccgt_eff(n):
    """Break-even arbitrage spread for grid batteries (EUR/MWh delivered) and
    mean CCGT efficiency.

    Break-even = (annualised capex of all battery components) / (annual MWh
    delivered by battery dischargers). The capex sum includes the Store
    (energy), the charger Link (power, input), and the discharger Link
    (power, output). Discharge is measured as |p1| of battery discharger
    links, weighted by snapshot hours.
    """
    total_capex = 0.0
    stores = n.stores[n.stores.carrier == "battery"]
    if len(stores):
        total_capex += float(
            (stores["capital_cost"] * stores["e_nom_opt"]).sum()
        )
    chargers = n.links[n.links.carrier == "battery charger"]
    if len(chargers):
        total_capex += float(
            (chargers["capital_cost"] * chargers["p_nom_opt"]).sum()
        )
    dischargers = n.links[n.links.carrier == "battery discharger"]
    total_discharged = 0.0
    if len(dischargers):
        total_capex += float(
            (dischargers["capital_cost"] * dischargers["p_nom_opt"]).sum()
        )
        w = n.snapshot_weightings.generators
        p1 = n.links_t.p1.reindex(columns=dischargers.index,
                                  index=n.snapshots).fillna(0.0).abs()
        total_discharged = float(p1.multiply(w, axis=0).sum().sum())
    breakeven = (total_capex / total_discharged
                 if total_discharged > 0 else float("nan"))

    ccgt = n.links[n.links.carrier == "CCGT"]
    ccgt_eff = float(ccgt["efficiency"].mean()) if len(ccgt) else float("nan")
    return breakeven, ccgt_eff


def biomass_chp_events(n):
    """Collect every (bus, snapshot) pair where the supply cascade picks a
    biomass CHP as the price-setter, returning AC electricity price and the
    paired urban-central-heat bus price at the same snapshot."""
    ac_buses = list(n.buses.index[n.buses.carrier == "AC"])
    heat_prices = n.buses_t.marginal_price
    pieces = []
    for bus in ac_buses:
        cls, car, lam = classify_all_vectorised(n, bus)
        mask = ((cls == "supply")
                & car.isin(BIOMASS_CHP_CARRIERS)).values
        if not mask.any():
            continue
        heat_bus = f"{bus} urban central heat"
        if heat_bus in heat_prices.columns:
            lam_heat = heat_prices[heat_bus].reindex(lam.index)
        else:
            lam_heat = pd.Series(np.nan, index=lam.index)
        ts = lam.index[mask]
        pieces.append(pd.DataFrame({
            "day_of_year": ts.dayofyear.astype(int),
            "elec_price": lam.values[mask],
            "heat_price": lam_heat.values[mask],
        }))
    if not pieces:
        return pd.DataFrame(columns=["day_of_year", "elec_price", "heat_price"])
    return pd.concat(pieces, ignore_index=True)


def gas_share_in_power_mix(n):
    """Fraction of AC electricity supply produced from gas over the year.

    Gas-sourced supply = Σ |p1| of every link whose bus0 is on a gas bus
    and whose bus1 is on an AC bus (covers CCGT, OCGT, urban central gas
    CHP, etc.). Total AC supply = this plus all generators dispatching on
    AC buses and all StorageUnits (hydro/PHS) when discharging.
    """
    w = n.snapshot_weightings.generators
    ac_buses = n.buses.index[n.buses.carrier == "AC"]
    gas_buses = n.buses.index[n.buses.carrier == "gas"]

    # total AC supply
    total = 0.0
    # generators on AC buses
    gens = n.generators[n.generators.bus.isin(ac_buses)]
    if len(gens):
        p = n.generators_t.p.reindex(columns=gens.index,
                                     index=n.snapshots).fillna(0.0).clip(lower=0.0)
        total += float(p.multiply(w, axis=0).sum().sum())
    # links delivering to AC (excluding DC + distribution)
    sup_links = n.links[
        n.links.bus1.isin(ac_buses)
        & ~n.links.carrier.isin({"DC", "electricity distribution grid"})
    ]
    if len(sup_links):
        p1 = n.links_t.p1.reindex(columns=sup_links.index,
                                  index=n.snapshots).fillna(0.0).abs()
        total += float(p1.multiply(w, axis=0).sum().sum())
    # SU discharging on AC buses
    sus = n.storage_units[n.storage_units.bus.isin(ac_buses)]
    if len(sus):
        p_disp = n.storage_units_t.p_dispatch.reindex(
            columns=sus.index, index=n.snapshots
        ).fillna(0.0)
        total += float(p_disp.multiply(w, axis=0).sum().sum())

    # gas-sourced: any link with bus0 in gas_buses and bus1 in ac_buses
    gas_elec = n.links[
        n.links.bus0.isin(gas_buses) & n.links.bus1.isin(ac_buses)
    ]
    gas_supply = 0.0
    if len(gas_elec):
        p1 = n.links_t.p1.reindex(columns=gas_elec.index,
                                  index=n.snapshots).fillna(0.0).abs()
        gas_supply = float(p1.multiply(w, axis=0).sum().sum())

    return gas_supply / total if total > 0 else float("nan")


# ── Plot ─────────────────────────────────────────────────────────────────────
def make_plot(results):
    from matplotlib import cm
    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.0],
                          wspace=0.16,
                          left=0.07, right=0.98, top=0.90, bottom=0.10)
    ax_sup = fig.add_subplot(gs[0, 0])
    ax_dis = fig.add_subplot(gs[0, 1], sharey=ax_sup)
    # Render ax_sup last so its clip_on=False header box overlays anything
    # drawn on ax_dis that happens to land in the gap between them.
    ax_sup.set_zorder(10)
    ax_sup.patch.set_alpha(0.0)   # keep background transparent so we see ax_dis

    gas_prices = np.array([r["gas_price"] for r in results], dtype=float)
    gas_span = np.nanmax(gas_prices) - np.nanmin(gas_prices)
    base_half = max(1.5, (gas_span / 15.0)
                    if np.isfinite(gas_span) and gas_span > 0 else 3.0)

    # Global N_max: treat each supply bucket and the single discharger class
    # on equal footing so violins are directly comparable.
    all_counts = [len(r["storage_discharger"]) for r in results]
    for r in results:
        for v in r["supply_buckets"].values():
            all_counts.append(len(v))
    N_max = max(all_counts) if all_counts else 1
    if N_max == 0:
        N_max = 1

    CONNECTOR_KW = dict(color=CONN_COLOR, linestyle=(0, (1.5, 1.8)),
                        linewidth=0.8, zorder=0.5)

    # ── Left: supply, split into buckets ──
    # Within a wiggle's row, draw each bucket's violin on top of the same y,
    # largest-count bucket in the back so smaller ones stay visible.
    rng = np.random.default_rng(42)
    VRE_DISPLAY_JITTER = 2.5      # EUR/MWh Gaussian jitter, visual only
    # Left axis extends below zero so that VRE violins, jittered around 0,
    # can show both sides of the near-zero cluster.
    SUP_XLIM = (-5.0, XLIM[1])
    # Vertical-extent multiplier applied only to the supply panel — its
    # violins are smaller than the storage discharger ones because
    # supply-bucket counts are split across 5 carriers, so we widen them
    # so each bucket's distribution is legible.
    SUP_HALF_MULT = 2.0
    for r in results:
        gp = r["gas_price"]
        if not np.isfinite(gp):
            continue
        buckets = r["supply_buckets"]
        order = sorted(SUPPLY_BUCKET_ORDER,
                       key=lambda b: -len(buckets[b]))
        left_edges, right_edges = [], []
        for b in order:
            data = buckets[b]
            if len(data) < 5:
                continue
            # Wind & solar have effectively-zero marginal cost, so all their
            # price-setting snapshots land at λ ≈ 0 and the resulting violin
            # would be invisible. Add Gaussian jitter purely for display so
            # the distribution appears as a visibly broad bump centred on 0.
            if b == "wind & solar":
                data = data + rng.normal(0, VRE_DISPLAY_JITTER,
                                         size=data.shape)
            half = SUP_HALF_MULT * base_half * (len(data) / N_max)
            shape = _draw_shape(
                ax_sup, data, gp, half,
                face=SUPPLY_BUCKET_COLORS[b],
                alpha=SUPPLY_BUCKET_ALPHA.get(b, DEFAULT_SUPPLY_ALPHA),
                xlim=SUP_XLIM,
            )
            if shape is not None:
                left_edges.append(shape[0])
                right_edges.append(shape[1])
        # connector: dotted red from rightmost violin/pill edge to right spine
        if right_edges:
            ax_sup.plot([max(right_edges), SUP_XLIM[1]], [gp, gp],
                        clip_on=False, **CONNECTOR_KW)
        # extend dotted red leftward to the low x-limit of the left ax
        if left_edges:
            ax_sup.plot([SUP_XLIM[0], min(left_edges)], [gp, gp],
                        clip_on=False, **CONNECTOR_KW)

    # Weighted-average AC electricity price per wiggle: red dot on each panel
    anchor_250 = None
    for r in results:
        ep = r.get("elec_price_avg", float("nan"))
        gp = r["gas_price"]
        if not (np.isfinite(ep) and np.isfinite(gp)):
            continue
        for ax in (ax_sup, ax_dis):
            ax.scatter([ep], [gp], s=55, marker="o",
                       facecolor="#c0001f", edgecolor="white",
                       linewidth=0.8, zorder=8)
        if r["wiggle"] == 250:
            anchor_250 = (ep, gp)

    # Highlight the min/max of the red dots across wiggles with subtle red
    # dotted axvlines on both panels.
    ep_values = [r.get("elec_price_avg", float("nan")) for r in results]
    ep_values = [v for v in ep_values if np.isfinite(v)]
    if len(ep_values) >= 2:
        ep_min, ep_max = min(ep_values), max(ep_values)
        for ax in (ax_sup, ax_dis):
            for xv in (ep_min, ep_max):
                ax.axvline(xv, color="#c0001f", linestyle=(0, (1.2, 2.5)),
                           linewidth=0.6, alpha=0.55, zorder=0.7)

    # Annotate the 250-wiggle dot on the left panel to explain what the dots
    # represent.
    if anchor_250 is not None:
        ep, gp = anchor_250
        ax_sup.annotate(
            "withdrawal-weighted\nmean AC price",
            xy=(ep, gp), xycoords="data",
            xytext=(30, 30), textcoords="offset points",
            fontsize=8, color="#c0001f", ha="left", va="bottom",
            arrowprops=dict(arrowstyle="-", color="#c0001f",
                            linewidth=0.7, shrinkA=2, shrinkB=3),
            bbox=dict(boxstyle="round,pad=0.4",
                      facecolor="white", edgecolor="#c0001f",
                      linewidth=0.6, alpha=0.95),
            zorder=10,
        )

    ax_sup.set_xlim(*SUP_XLIM)
    ax_sup.set_title("…Generation Price Setting Events", fontsize=13,
                     fontweight="bold", color=CLASS_COLORS["supply"])
    ax_sup.set_xlabel("Electricity Marginal Price [€/MWh]")
    ax_sup.set_ylabel("Network-wide Load-Weighted Gas Price [€/MWh]")
    style_ax(ax_sup)
    ax_sup.yaxis.grid(False)

    # ── Right: storage discharging, one aggregated violin per wiggle ──
    n_w = len(results)
    viridis_colors = [
        cm.viridis(0.15 + 0.7 * i / max(1, n_w - 1)) for i in range(n_w)
    ]
    violin_edges = {}   # wiggle → (left, right) x of the discharger violin
    for r, w_col in zip(results, viridis_colors):
        gp = r["gas_price"]
        if not np.isfinite(gp):
            continue
        data = r["storage_discharger"]
        if len(data) < 5:
            continue
        half = base_half * (len(data) / N_max)
        shape = _draw_shape(ax_dis, data, gp, half, face=w_col, alpha=0.72)
        if shape is None:
            continue
        violin_edges[r["wiggle"]] = shape

    ax_dis.set_xlim(*XLIM)
    # Consistent x-ticks across both panels.
    shared_ticks = [0, 25, 50, 75, 100, 125, 150, 175, 200, 225]
    ax_sup.set_xticks(shared_ticks)
    ax_dis.set_xticks(shared_ticks)
    ax_dis.set_title("…Storage Discharge Price Setting Events", fontsize=13,
                     fontweight="bold", color=CLASS_COLORS["storage_discharger"])
    ax_dis.set_xlabel("Electricity Marginal Price [€/MWh]")
    plt.setp(ax_dis.get_yticklabels(), visible=False)
    style_ax(ax_dis)
    ax_dis.yaxis.grid(False)
    # Remove left spine of the right plot per request
    ax_dis.spines["left"].set_visible(False)
    ax_dis.tick_params(axis="y", length=0)

    # ── CCGT-equivalence line (shifted +25 EUR/MWh on the x-axis) ──
    # λ_AC = λ_gas / η_CCGT + 25 — a constant offset to separate it visually
    # from other vertical markers on the plot. The slope 1/η_CCGT is the
    # quantity of interest: a 1 EUR/MWh gas price change translates into
    # 1/η_CCGT EUR/MWh of AC electricity price if a gas CCGT is at the margin.
    CCGT_X_OFFSET = 20.0
    valid_ccgt = [r["ccgt_eff"] for r in results
                  if np.isfinite(r.get("ccgt_eff", np.nan))]
    eta_ccgt = float(np.mean(valid_ccgt)) if valid_ccgt else float("nan")
    if np.isfinite(eta_ccgt) and eta_ccgt > 0:
        gs = np.array([r["gas_price"] for r in results
                       if np.isfinite(r["gas_price"])], dtype=float)
        y_line = np.linspace(gs.min(), gs.max(), 50)
        x_line = y_line / eta_ccgt + CCGT_X_OFFSET
        mask = (x_line >= XLIM[0]) & (x_line <= XLIM[1])
        ax_dis.plot(x_line[mask], y_line[mask],
                    color="#c0001f", linewidth=1.4,
                    linestyle="-", alpha=0.9, zorder=6)
        if mask.any():
            ax_dis.annotate(
                (f"if a gas CCGT\nset the price\n"
                 f"slope = η_CCGT ≈ {eta_ccgt:.2f}"),
                xy=(x_line[mask][-1], y_line[mask][-1]),
                xytext=(-6, 6), textcoords="offset points",
                fontsize=8, color="#c0001f", ha="right", va="bottom",
                bbox=dict(boxstyle="round,pad=0.55",
                          facecolor="white", edgecolor="#c0001f",
                          linewidth=0.6, alpha=0.95),
            )

    # ── Battery arbitrage break-even (vertical line on right plot) ──
    be_values = [r["battery_breakeven"] for r in results
                 if np.isfinite(r.get("battery_breakeven", np.nan))]
    if be_values:
        be_mean = float(np.median(be_values))    # median across wiggles
        ax_dis.axvline(be_mean, color="#1a1a1a", linewidth=1.4,
                       linestyle=(0, (4, 2)), alpha=0.85, zorder=6)
        ax_dis.annotate(
            (f"battery arbitrage\nbreak-even spread\n"
             f"≈ {be_mean:.1f} EUR/MWh"),
            xy=(be_mean, max(r["gas_price"] for r in results) - 30),
            xytext=(6, -6), textcoords="offset points",
            fontsize=8, ha="left", va="top", color="#1a1a1a",
            bbox=dict(boxstyle="round,pad=0.55",
                      facecolor="white", edgecolor="#1a1a1a",
                      linewidth=0.6, alpha=0.95),
        )

    # ── Wiggle labels (centered) in a single vertical column just outside
    # the right edge of the left axis, with a header explaining the numbers.
    from matplotlib.transforms import blended_transform_factory
    label_trans_sup = blended_transform_factory(
        ax_sup.transAxes, ax_sup.transData
    )
    label_trans_dis = blended_transform_factory(
        ax_dis.transAxes, ax_dis.transData
    )

    # header box: 3 lines — first two describe the wiggle value,
    # third describes the gas-share value below it. High zorder so nothing
    # renders on top of it.
    header_text = "Gas Consumption\n[TWh/y]\nGas Share in Electricity Mix [%]"
    ax_sup.text(1.035, 0.965, header_text,
                transform=ax_sup.transAxes,
                fontsize=8, fontweight="bold", color="#333333",
                va="bottom", ha="center",
                clip_on=False, zorder=30,
                bbox=dict(boxstyle="round,pad=0.4",
                          facecolor="white",
                          edgecolor=CONN_COLOR, linewidth=0.6))

    # connector lines through the gap and individual wiggle labels
    for r in results:
        gp = r["gas_price"]
        if not np.isfinite(gp):
            continue
        # gap-crossing dotted line starting right of the label
        edges = violin_edges.get(r["wiggle"])
        if edges is not None:
            ax_dis.plot([-0.10, 0.0], [gp, gp],
                        transform=blended_transform_factory(
                            ax_dis.transAxes, ax_dis.transData),
                        clip_on=False, **CONNECTOR_KW)
            ax_dis.plot([XLIM[0], edges[0]], [gp, gp],
                        **CONNECTOR_KW)
            # extend rightward to the upper x-limit of the right ax
            ax_dis.plot([edges[1], XLIM[1]], [gp, gp],
                        **CONNECTOR_KW)

        gas_sh = r.get("gas_share", float("nan"))
        share_line = (f"{gas_sh*100:.1f}%" if np.isfinite(gas_sh) else "—")
        ax_sup.text(1.035, gp, f"{r['wiggle']}\n{share_line}",
                    transform=label_trans_sup,
                    fontsize=8, fontweight="bold",
                    va="center", ha="center",
                    color="#333333",
                    clip_on=False, zorder=6,
                    linespacing=1.25,
                    bbox=dict(boxstyle="round,pad=0.4",
                              facecolor="white",
                              edgecolor=CONN_COLOR, linewidth=0.6))

    # ── In-axis legend for the supply buckets, bottom-right of ax_sup ──
    bucket_handles = [
        Patch(facecolor=SUPPLY_BUCKET_COLORS[b],
              edgecolor="black", alpha=0.7, label=b)
        for b in SUPPLY_BUCKET_ORDER
    ]
    ax_sup.legend(
        handles=bucket_handles, loc="lower right",
        bbox_to_anchor=(0.99, 0.02),
        frameon=True, fontsize=8, fancybox=True,
        title="Supply Types", title_fontsize=9,
        borderpad=0.6, handletextpad=0.5, labelspacing=0.5,
    ).set_zorder(15)

    fig.suptitle("Distribution of Electricity Marginal Prices During…",
                 fontsize=15, fontweight="bold", y=0.985)

    # subplot labels
    ax_sup.text(0.01, 0.99, "a", transform=ax_sup.transAxes,
                fontsize=16, fontweight="bold", va="top", ha="left",
                zorder=50)
    ax_dis.text(0.99, 0.99, "b", transform=ax_dis.transAxes,
                fontsize=16, fontweight="bold", va="top", ha="right",
                zorder=50)

    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    for out in (SCRIPT_DIR / OUT_PLOT, PAPER_DIR / OUT_PLOT):
        fig.savefig(out, bbox_inches="tight")
        print(f"saved {out}")
    plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    results = []
    for w in WIGGLES:
        r = collect_for_wiggle(w)
        if r is None:
            continue
        for c in ["demand_flex", "storage_discharger", "storage_charger"]:
            r[c] = np.asarray(r[c], dtype=float)
        r["supply_buckets"] = {
            k: np.asarray(v, dtype=float) for k, v in r["supply_buckets"].items()
        }
        results.append(r)

    if not results:
        raise SystemExit("No wiggles yielded data.")

    print(f"\nCollected {len(results)} wiggles. Plotting…")
    make_plot(results)
