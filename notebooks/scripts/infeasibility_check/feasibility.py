"""Per-bus UDH adequacy LP.

Question answered per bus: given the existing capacities of non-extendable
supply assets and the dispatch envelope (p_max_pu, efficiency, including
time-varying COP and proportional-cap p_max_pu) of every UDH supply asset,
does there exist a non-negative capacity addition to the extendable assets
such that the hourly heat balance can be met at every snapshot?

Because UDH has no thermal storage in this network and `p_min_pu = 0` for
every supply asset, the LP collapses to one inequality per snapshot:

    Σ_i  η_i(t) · p_max_pu(t,i) · (p_nom_existing_i + P_i)  ≥  demand(t)

with `P_i ≥ 0`, `P_i = 0` for non-extendable assets, and objective
`min Σ_i capital_cost_i · P_i`. Solved with `scipy.optimize.linprog` (HiGHS).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import linprog

from data import SupplyAsset


@dataclass
class BusResult:
    bus: str
    scenario: str
    status: str                       # 'feasible' | 'infeasible' | 'solver_error'
    objective: float                  # minimum extra capex (EUR/year)
    p_nom_added: dict[str, float]     # carrier → MW added
    fixed_supply_max_t: pd.Series     # max heat supply from fixed assets only
    extendable_supply_potential_t: pd.Series  # per-MW-of-extendable-capex contribution metric (info)
    demand_t: pd.Series
    binding_snapshot: pd.Timestamp | None
    fixed_slack_t: pd.Series          # fixed_supply_max_t − demand_t (negative ⇒ extendable must close gap)


def bus_feasibility(
    bus: str,
    scenario: str,
    demand_t: pd.Series,
    assets: list[SupplyAsset],
) -> BusResult:
    snapshots = demand_t.index
    T = len(snapshots)

    # Per-snapshot baseline supply: every asset's existing `p_nom` is
    # available regardless of extendability. For non-extendable assets
    # this is also the upper limit; for extendable ones the LP can add
    # capacity on top via P_i.
    fixed_supply = pd.Series(0.0, index=snapshots)
    for a in assets:
        fixed_supply = fixed_supply + a.effective_t * a.p_nom

    # Extendable assets — these are the LP variables (additive capacity).
    ext_assets = [a for a in assets if a.p_nom_extendable]
    n_ext = len(ext_assets)

    # Aggregate per-MW-of-extendable contribution at each snapshot (summed
    # across extendable assets weighted by 1) — purely informational.
    ext_potential = pd.Series(0.0, index=snapshots)
    for a in ext_assets:
        ext_potential = ext_potential + a.effective_t

    fixed_slack = fixed_supply - demand_t
    binding_snap = pd.Timestamp(fixed_slack.idxmin()) if (fixed_slack < 0).any() else None

    if n_ext == 0:
        status = "feasible" if (fixed_slack >= -1e-6).all() else "infeasible"
        return BusResult(
            bus=bus, scenario=scenario, status=status, objective=0.0,
            p_nom_added={}, fixed_supply_max_t=fixed_supply,
            extendable_supply_potential_t=ext_potential,
            demand_t=demand_t, binding_snapshot=binding_snap,
            fixed_slack_t=fixed_slack,
        )

    # Build LP:
    #   variables  P[i] ≥ 0  for i in ext_assets, with optional upper bound p_nom_max
    #   constraints (per snapshot t):
    #     Σ_i  η_i(t) · p_max_pu(t,i) · P[i]  ≥  demand(t) − fixed_supply(t)
    #   ⇒  −Σ_i  effective_t,i · P[i]  ≤  −(demand(t) − fixed_supply(t))
    A = np.zeros((T, n_ext))
    for i, a in enumerate(ext_assets):
        A[:, i] = -a.effective_t.values
    b = -(demand_t.values - fixed_supply.values)

    # Drop trivially-satisfied rows (RHS ≥ 0 with all-zero LHS coefficients ⇒ row is 0·P ≤ ≥0).
    # Also drop rows where demand == fixed_supply already met (RHS ≤ 0 means -slack ≤ 0,
    # i.e. -Σ p ≤ 0 which is satisfied at P=0). Keep them anyway — they're cheap.

    c = np.array([a.capital_cost if a.capital_cost > 0 else 1.0 for a in ext_assets])
    # Upper bound on additive capacity P_i = max(0, p_nom_max - p_nom).
    bounds = []
    for a in ext_assets:
        if np.isfinite(a.p_nom_max):
            bounds.append((0.0, max(0.0, a.p_nom_max - a.p_nom)))
        else:
            bounds.append((0.0, None))

    res = linprog(c=c, A_ub=A, b_ub=b, bounds=bounds, method="highs")

    if res.status == 0:
        added = {a.carrier: float(p) for a, p in zip(ext_assets, res.x)}
        return BusResult(
            bus=bus, scenario=scenario, status="feasible",
            objective=float(res.fun),
            p_nom_added=added,
            fixed_supply_max_t=fixed_supply,
            extendable_supply_potential_t=ext_potential,
            demand_t=demand_t, binding_snapshot=binding_snap,
            fixed_slack_t=fixed_slack,
        )
    elif res.status == 2:
        return BusResult(
            bus=bus, scenario=scenario, status="infeasible",
            objective=float("nan"),
            p_nom_added={a.carrier: float("nan") for a in ext_assets},
            fixed_supply_max_t=fixed_supply,
            extendable_supply_potential_t=ext_potential,
            demand_t=demand_t, binding_snapshot=binding_snap,
            fixed_slack_t=fixed_slack,
        )
    else:
        return BusResult(
            bus=bus, scenario=scenario, status=f"solver_error_{res.status}",
            objective=float("nan"), p_nom_added={},
            fixed_supply_max_t=fixed_supply,
            extendable_supply_potential_t=ext_potential,
            demand_t=demand_t, binding_snapshot=binding_snap,
            fixed_slack_t=fixed_slack,
        )
