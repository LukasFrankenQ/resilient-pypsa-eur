"""Network-side helpers for the UDH adequacy audit.

Loads the network, applies / undoes the recent UDH oil-boiler pin, and
enumerates every supply asset feeding each `urban decentral heat` bus
together with its per-snapshot effective output `η(t) · p_max_pu(t)`.

The downstream LP only needs those time series and the static `p_nom`,
`p_nom_extendable`, and `capital_cost` per asset.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import pypsa


ROOT = Path(__file__).resolve().parents[3]
NETWORK_DIR = ROOT / "resources" / "networks"
PREFIX = "base_s_50__3H-T-H-B-I-A-dist1_2030"

UDH_CARRIER = "urban decentral heat"
UDH_OIL = "urban decentral oil boiler"
LV_CARRIER = "low voltage"
DIST_CARRIER = "electricity distribution grid"
ROOFTOP_CARRIER = "solar rooftop"


def network_path(scenario: str = "free", wiggle: int = 1000, hike: int | None = None) -> Path:
    hike_str = f"_{hike}" if hike is not None else ""
    return NETWORK_DIR / f"{PREFIX}_{scenario}_{wiggle}{hike_str}.nc"


def load_network(path: Path) -> pypsa.Network:
    return pypsa.Network(str(path))


def apply_udh_oil_pin(n: pypsa.Network) -> None:
    """Mirror of `prepare_sector_network.py:7780-7783`. Idempotent."""
    mask = n.links.carrier == UDH_OIL
    n.links.loc[mask, "p_nom_extendable"] = False


def unpin_udh_oil(n: pypsa.Network) -> None:
    """Make every UDH oil boiler extendable (no upper cap)."""
    mask = n.links.carrier == UDH_OIL
    n.links.loc[mask, "p_nom_extendable"] = True
    n.links.loc[mask, "p_nom_max"] = np.inf


def udh_buses(n: pypsa.Network) -> list[str]:
    """AC locations whose `{loc} urban decentral heat` load has non-zero peak."""
    loads = n.loads[n.loads.bus.str.endswith(UDH_CARRIER, na=False)]
    keep = []
    for L in loads.index:
        if L in n.loads_t.p_set.columns:
            peak = float(n.loads_t.p_set[L].max())
        else:
            peak = float(loads.at[L, "p_set"])
        if peak > 0:
            keep.append(loads.at[L, "bus"])
    return keep


def udh_demand(n: pypsa.Network, bus: str) -> pd.Series:
    """Heat-demand series at this UDH bus (MW), aligned to `n.snapshots`."""
    L = n.loads[n.loads.bus == bus].index
    if len(L) != 1:
        raise ValueError(f"expected exactly one load on {bus}, got {len(L)}")
    name = L[0]
    if name in n.loads_t.p_set.columns:
        return n.loads_t.p_set[name].astype(float)
    return pd.Series(float(n.loads.at[name, "p_set"]), index=n.snapshots)


@dataclass
class SupplyAsset:
    name: str          # component index
    kind: str          # 'link' or 'generator'
    carrier: str
    p_nom: float       # existing
    p_nom_extendable: bool
    p_nom_max: float
    capital_cost: float
    effective_t: pd.Series  # η(t) · p_max_pu(t), shape (T,)
    bus0: str = ""        # input bus (empty for generators); used to flag LV-coupled techs
    p_max_pu_t: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    efficiency_t: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))


def _series_or_scalar(df_t: pd.DataFrame, name: str, scalar: float, snapshots: pd.Index) -> pd.Series:
    if name in df_t.columns:
        return df_t[name].astype(float).reindex(snapshots)
    return pd.Series(float(scalar), index=snapshots)


def ac_location(udh_bus: str) -> str:
    """`'FR0 0 urban decentral heat'` → `'FR0 0'`."""
    return udh_bus[: -len(" " + UDH_CARRIER)]


def lv_bus(udh_bus: str) -> str:
    return f"{ac_location(udh_bus)} {LV_CARRIER}"


def lv_baseline_load(n: pypsa.Network, lv_bus_id: str) -> pd.Series:
    """Sum of every load on the LV bus, aligned to `n.snapshots`.

    Includes baseline electricity, industry electricity, agriculture
    electricity, agriculture machinery electric — i.e. the LV load that
    is exogenous to the heating LP.
    """
    snapshots = n.snapshots
    loads = n.loads[n.loads.bus == lv_bus_id]
    out = pd.Series(0.0, index=snapshots)
    for L in loads.index:
        if L in n.loads_t.p_set.columns:
            out = out + n.loads_t.p_set[L].astype(float).reindex(snapshots).fillna(0.0)
        else:
            out = out + float(n.loads.at[L, "p_set"])
    return out


@dataclass
class LVAsset:
    """Distribution link AC→LV or rooftop solar generator."""
    name: str
    kind: str             # 'distribution_link' | 'rooftop'
    p_nom: float
    p_nom_extendable: bool
    p_nom_max: float
    capital_cost: float
    efficiency: float     # static (1.0 for solar, ~0.97 for distribution)
    p_max_pu_t: pd.Series # availability profile aligned to snapshots


def lv_supply(n: pypsa.Network, lv_bus_id: str) -> list[LVAsset]:
    """LV-side inflow assets: distribution link and rooftop solar."""
    snapshots = n.snapshots
    out: list[LVAsset] = []

    # Distribution link AC → LV (bus1 = LV, carrier = electricity distribution grid).
    dist = n.links[(n.links.bus1 == lv_bus_id) & (n.links.carrier == DIST_CARRIER)]
    for name, row in dist.iterrows():
        pmax = _series_or_scalar(n.links_t.p_max_pu, name, row.p_max_pu, snapshots)
        out.append(LVAsset(
            name=name, kind="distribution_link",
            p_nom=float(row.p_nom),
            p_nom_extendable=bool(row.p_nom_extendable),
            p_nom_max=float(row.p_nom_max),
            capital_cost=float(row.get("capital_cost", 0.0) or 0.0),
            efficiency=float(row.efficiency),
            p_max_pu_t=pmax,
        ))

    # Rooftop solar generator on LV.
    roof = n.generators[(n.generators.bus == lv_bus_id) &
                        (n.generators.carrier == ROOFTOP_CARRIER)]
    for name, row in roof.iterrows():
        pmax = _series_or_scalar(n.generators_t.p_max_pu, name, row.p_max_pu, snapshots)
        out.append(LVAsset(
            name=name, kind="rooftop",
            p_nom=float(row.p_nom),
            p_nom_extendable=bool(row.p_nom_extendable),
            p_nom_max=float(row.p_nom_max),
            capital_cost=float(row.get("capital_cost", 0.0) or 0.0),
            efficiency=1.0,
            p_max_pu_t=pmax,
        ))

    return out


def udh_supply_assets(n: pypsa.Network, bus: str) -> list[SupplyAsset]:
    """Every link-into-bus / generator-on-bus that supplies UDH at `bus`.

    Returns a list of SupplyAsset where `effective_t = η(t) · p_max_pu(t)` is
    the per-MW-of-capacity heat output ceiling at each snapshot. For
    generators (solar thermal) η is implicit (= 1 since they generate the
    heat carrier directly).
    """
    snapshots = n.snapshots
    out: list[SupplyAsset] = []

    # Links with bus1 = UDH bus
    links = n.links[n.links.bus1 == bus]
    for name, row in links.iterrows():
        # Skip resistive heater rows that have no installed cap and won't
        # be considered (extendable but never seeded) — keep them in the
        # LP as extendable so they CAN be invested in.
        eff = _series_or_scalar(n.links_t.efficiency, name, row.efficiency, snapshots)
        pmax = _series_or_scalar(n.links_t.p_max_pu, name, row.p_max_pu, snapshots)
        effective = eff * pmax
        out.append(SupplyAsset(
            name=name,
            kind="link",
            carrier=row.carrier,
            p_nom=float(row.p_nom),
            p_nom_extendable=bool(row.p_nom_extendable),
            p_nom_max=float(row.p_nom_max),
            capital_cost=float(row.get("capital_cost", 0.0) or 0.0),
            effective_t=effective,
            bus0=str(row.bus0),
            p_max_pu_t=pmax,
            efficiency_t=eff,
        ))

    # Generators on the UDH bus — solar thermal supplies, vent absorbs.
    gens = n.generators[n.generators.bus == bus]
    for name, row in gens.iterrows():
        if row.carrier == "urban decentral heat vent":
            # Vent is a sink — does not help meet demand; ignore.
            continue
        pmax = _series_or_scalar(n.generators_t.p_max_pu, name, row.p_max_pu, snapshots)
        out.append(SupplyAsset(
            name=name,
            kind="generator",
            carrier=row.carrier,
            p_nom=float(row.p_nom),
            p_nom_extendable=bool(row.p_nom_extendable),
            p_nom_max=float(row.p_nom_max),
            capital_cost=float(row.get("capital_cost", 0.0) or 0.0),
            effective_t=pmax.astype(float),
        ))

    return out
