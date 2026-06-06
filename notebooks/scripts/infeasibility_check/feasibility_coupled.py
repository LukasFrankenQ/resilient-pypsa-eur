"""Coupled UDH-heat + LV-electricity per-bus adequacy LP.

Question: at every snapshot, can the LV bus deliver enough electricity to
power the HP/resistive heating that the UDH heat balance demands, on top of
the bus's exogenous LV electricity load (baseline + industry + agriculture)?

Variables (per AC location):
  - For each UDH supply asset i: additive capacity P_i ≥ 0 (extendable
    only) and snapshot dispatch x_{i,t} ≥ 0 (input-side, MW).
  - For the AC→LV distribution link: additive capacity P_dist ≥ 0 and
    snapshot inflow x_dist(t) ≥ 0 (input-side, MW from AC bus).
  - For rooftop solar generator: additive capacity P_solar ≥ 0 and
    snapshot dispatch x_solar(t) ≥ 0 (output-side, MW heat-equivalent
    treated as electricity directly since η=1).

Constraints (per snapshot t):
  - UDH heat balance:
        Σ_i  η_i(t) · x_{i,t}  ≥  demand_UDH(t)
  - LV electricity balance:
        η_dist · x_dist(t)  +  x_solar(t)
            ≥  load_LV_baseline(t)  +  Σ_{i ∈ LV-coupled UDH supply}  x_{i,t}
  - Per-asset dispatch caps:
        x_{i,t}    ≤ p_max_pu_i(t)    · (p_nom_i + P_i)
        x_dist(t)  ≤ p_max_pu_dist(t) · (p_nom_dist + P_dist)
        x_solar(t) ≤ p_max_pu_solar(t)· (p_nom_solar + P_solar)

`LV-coupled UDH supply` = UDH links whose `bus0` is the LV bus, i.e. air HP
and resistive heater. Gas/oil/biomass/solar-thermal don't draw from LV.

Solved with `scipy.optimize.linprog` (HiGHS). Objective: minimise total
extra capex (Σ capital_cost_i · P_i). For non-extendable assets P_i is
forced to 0 by tight bounds.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import csr_matrix, lil_matrix

from data import LVAsset, SupplyAsset


@dataclass
class CoupledResult:
    bus_udh: str
    bus_lv: str
    scenario: str
    status: str
    objective: float
    p_nom_added: dict[str, float]    # name → MW
    peak_lv_uplift_mw: float          # max over t of HP+resistive electric draw
    peak_lv_uplift_snapshot: pd.Timestamp | None
    peak_lv_total_mw: float           # max over t of (baseline + uplift)
    peak_lv_total_snapshot: pd.Timestamp | None
    binding_heat_snapshot: pd.Timestamp | None
    lv_load_baseline_t: pd.Series
    lv_load_total_t: pd.Series        # baseline + electric heating draw at LP optimum
    udh_demand_t: pd.Series
    distribution_inflow_t: pd.Series
    rooftop_dispatch_t: pd.Series


def _maybe_pmax(asset, snapshots) -> np.ndarray:
    """For SupplyAsset use p_max_pu_t; for LVAsset use p_max_pu_t — both attribs exist."""
    if isinstance(asset, SupplyAsset):
        s = asset.p_max_pu_t if not asset.p_max_pu_t.empty else pd.Series(1.0, index=snapshots)
    else:
        s = asset.p_max_pu_t
    return s.reindex(snapshots).fillna(0.0).values


def bus_coupled_feasibility(
    bus_udh: str,
    bus_lv: str,
    scenario: str,
    udh_demand_t: pd.Series,
    udh_assets: list[SupplyAsset],
    lv_load_baseline_t: pd.Series,
    lv_assets: list[LVAsset],
) -> CoupledResult:
    snapshots = udh_demand_t.index
    T = len(snapshots)

    # Identify LV-coupled UDH supply assets (those whose bus0 is the LV bus).
    udh_coupled = [a.bus0 == bus_lv for a in udh_assets]

    # Distribution and rooftop must each appear exactly once (or zero).
    dist_assets = [a for a in lv_assets if a.kind == "distribution_link"]
    roof_assets = [a for a in lv_assets if a.kind == "rooftop"]
    assert len(dist_assets) <= 1 and len(roof_assets) <= 1, \
        f"unexpected LV asset count for {bus_lv}"
    dist = dist_assets[0] if dist_assets else None
    roof = roof_assets[0] if roof_assets else None

    n_udh = len(udh_assets)

    # Variable layout (column index in x):
    #   [0 .. n_udh-1]                 P_i for UDH assets
    #   [n_udh .. n_udh+T-1]           x_dist(t)
    #   [n_udh+T .. n_udh+2T-1]        x_solar(t)
    #   [n_udh+2T .. n_udh+2T+T*n_udh-1]   x_{i,t} flattened (asset-major: i then t)
    #   last 2 columns:                P_dist, P_solar
    n_vars = n_udh + 2 * T + T * n_udh + 2
    idx_P = lambda i: i
    idx_dist_t = lambda t: n_udh + t
    idx_solar_t = lambda t: n_udh + T + t
    idx_xit = lambda i, t: n_udh + 2 * T + i * T + t
    idx_P_dist = n_udh + 2 * T + T * n_udh
    idx_P_solar = n_udh + 2 * T + T * n_udh + 1

    # Objective: minimize Σ capital_cost · P over extendable assets.
    c = np.zeros(n_vars)
    for i, a in enumerate(udh_assets):
        if a.p_nom_extendable:
            c[idx_P(i)] = a.capital_cost if a.capital_cost > 0 else 1.0
    if dist is not None and dist.p_nom_extendable:
        c[idx_P_dist] = dist.capital_cost if dist.capital_cost > 0 else 1.0
    if roof is not None and roof.p_nom_extendable:
        c[idx_P_solar] = roof.capital_cost if roof.capital_cost > 0 else 1.0

    # Bounds.
    bounds = [(0.0, None)] * n_vars
    for i, a in enumerate(udh_assets):
        if not a.p_nom_extendable:
            bounds[idx_P(i)] = (0.0, 0.0)
        elif np.isfinite(a.p_nom_max):
            bounds[idx_P(i)] = (0.0, max(0.0, a.p_nom_max - a.p_nom))
    if dist is not None and not dist.p_nom_extendable:
        bounds[idx_P_dist] = (0.0, 0.0)
    if roof is not None and not roof.p_nom_extendable:
        bounds[idx_P_solar] = (0.0, 0.0)

    # Pre-extract availability profiles.
    eff_t = [a.effective_t.values for a in udh_assets]                     # η · p_max_pu (heat-out per MW input)
    pmax_t_udh = [_maybe_pmax(a, snapshots) for a in udh_assets]           # p_max_pu (input cap per MW capacity)
    pmax_t_dist = _maybe_pmax(dist, snapshots) if dist is not None else None
    pmax_t_solar = _maybe_pmax(roof, snapshots) if roof is not None else None
    eta_dist = float(dist.efficiency) if dist is not None else 1.0

    pnom_udh = np.array([a.p_nom for a in udh_assets])
    pnom_dist = float(dist.p_nom) if dist is not None else 0.0
    pnom_solar = float(roof.p_nom) if roof is not None else 0.0

    # Build A_ub x ≤ b_ub.
    # Rows:
    #   2*T balance rows (UDH heat, LV electricity)        ← inequality flipped: -lhs ≤ -rhs
    #   T * n_udh dispatch caps for UDH
    #   T cap rows for distribution
    #   T cap rows for solar
    n_rows = 2 * T + T * n_udh + (T if dist is not None else 0) + (T if roof is not None else 0)
    A = lil_matrix((n_rows, n_vars))
    b = np.zeros(n_rows)

    row = 0
    # UDH heat balance:  -Σ_i η_i(t) x_{i,t}  ≤  -demand(t)
    for t in range(T):
        for i in range(n_udh):
            A[row, idx_xit(i, t)] = -eff_t[i][t]
        b[row] = -udh_demand_t.values[t]
        row += 1

    # LV electricity balance:
    #   load_baseline(t) + Σ_{coupled i} x_{i,t}  −  η_dist x_dist(t)  −  x_solar(t)  ≤  0
    for t in range(T):
        for i in range(n_udh):
            if udh_coupled[i]:
                A[row, idx_xit(i, t)] = 1.0
        if dist is not None:
            A[row, idx_dist_t(t)] = -eta_dist
        if roof is not None:
            A[row, idx_solar_t(t)] = -1.0
        b[row] = -lv_load_baseline_t.values[t]
        row += 1

    # Dispatch caps for UDH:  x_{i,t} − p_max_pu(t,i) P_i  ≤  p_max_pu(t,i) p_nom_i
    for i in range(n_udh):
        for t in range(T):
            A[row, idx_xit(i, t)] = 1.0
            A[row, idx_P(i)] = -pmax_t_udh[i][t]
            b[row] = pmax_t_udh[i][t] * pnom_udh[i]
            row += 1

    # Dist cap:  x_dist(t) − p_max_pu_dist(t) P_dist  ≤  p_max_pu_dist(t) p_nom_dist
    if dist is not None:
        for t in range(T):
            A[row, idx_dist_t(t)] = 1.0
            A[row, idx_P_dist] = -pmax_t_dist[t]
            b[row] = pmax_t_dist[t] * pnom_dist
            row += 1

    # Solar cap:  x_solar(t) − p_max_pu_solar(t) P_solar  ≤  p_max_pu_solar(t) p_nom_solar
    if roof is not None:
        for t in range(T):
            A[row, idx_solar_t(t)] = 1.0
            A[row, idx_P_solar] = -pmax_t_solar[t]
            b[row] = pmax_t_solar[t] * pnom_solar
            row += 1

    A = csr_matrix(A)

    res = linprog(c=c, A_ub=A, b_ub=b, bounds=bounds, method="highs")

    if res.status not in (0, 2):
        return CoupledResult(
            bus_udh=bus_udh, bus_lv=bus_lv, scenario=scenario,
            status=f"solver_error_{res.status}",
            objective=float("nan"), p_nom_added={},
            peak_lv_uplift_mw=float("nan"), peak_lv_uplift_snapshot=None,
            peak_lv_total_mw=float("nan"), peak_lv_total_snapshot=None,
            binding_heat_snapshot=None,
            lv_load_baseline_t=lv_load_baseline_t,
            lv_load_total_t=pd.Series(np.nan, index=snapshots),
            udh_demand_t=udh_demand_t,
            distribution_inflow_t=pd.Series(np.nan, index=snapshots),
            rooftop_dispatch_t=pd.Series(np.nan, index=snapshots),
        )

    if res.status == 2:
        return CoupledResult(
            bus_udh=bus_udh, bus_lv=bus_lv, scenario=scenario,
            status="infeasible", objective=float("nan"), p_nom_added={},
            peak_lv_uplift_mw=float("nan"), peak_lv_uplift_snapshot=None,
            peak_lv_total_mw=float("nan"), peak_lv_total_snapshot=None,
            binding_heat_snapshot=None,
            lv_load_baseline_t=lv_load_baseline_t,
            lv_load_total_t=pd.Series(np.nan, index=snapshots),
            udh_demand_t=udh_demand_t,
            distribution_inflow_t=pd.Series(np.nan, index=snapshots),
            rooftop_dispatch_t=pd.Series(np.nan, index=snapshots),
        )

    x = res.x
    added = {}
    for i, a in enumerate(udh_assets):
        added[a.name] = float(x[idx_P(i)])
    if dist is not None:
        added[dist.name] = float(x[idx_P_dist])
    if roof is not None:
        added[roof.name] = float(x[idx_P_solar])

    # Reconstruct dispatch series.
    x_dist_t = pd.Series(x[n_udh:n_udh + T], index=snapshots) if dist is not None else pd.Series(0.0, index=snapshots)
    x_solar_t = pd.Series(x[n_udh + T:n_udh + 2 * T], index=snapshots) if roof is not None else pd.Series(0.0, index=snapshots)
    udh_dispatch = pd.DataFrame(
        {a.name: x[idx_xit(i, 0):idx_xit(i, T - 1) + 1]
         for i, a in enumerate(udh_assets)},
        index=snapshots,
    )

    # LV electric draw from heating = sum of coupled UDH input dispatch.
    coupled_names = [a.name for a, c in zip(udh_assets, udh_coupled) if c]
    lv_uplift_t = udh_dispatch[coupled_names].sum(axis=1) if coupled_names else pd.Series(0.0, index=snapshots)
    lv_total_t = lv_load_baseline_t + lv_uplift_t

    binding_heat_t = pd.Timestamp((udh_demand_t -
                                   sum(eff_t[i] * udh_dispatch[a.name].values for i, a in enumerate(udh_assets))
                                   ).idxmax()) if n_udh else None

    return CoupledResult(
        bus_udh=bus_udh, bus_lv=bus_lv, scenario=scenario,
        status="feasible",
        objective=float(res.fun),
        p_nom_added=added,
        peak_lv_uplift_mw=float(lv_uplift_t.max()),
        peak_lv_uplift_snapshot=pd.Timestamp(lv_uplift_t.idxmax()),
        peak_lv_total_mw=float(lv_total_t.max()),
        peak_lv_total_snapshot=pd.Timestamp(lv_total_t.idxmax()),
        binding_heat_snapshot=binding_heat_t,
        lv_load_baseline_t=lv_load_baseline_t,
        lv_load_total_t=lv_total_t,
        udh_demand_t=udh_demand_t,
        distribution_inflow_t=x_dist_t,
        rooftop_dispatch_t=x_solar_t,
    )
