"""Run the UDH adequacy audit for the 3H wiggle-1000 network.

Two scenarios per bus:
  - baseline: oil boilers pinned non-extendable (current code path)
  - oil_unpinned: oil boilers freed (counterfactual)

Writes a per-(bus, scenario) summary to `cache/audit_v1.parquet` and
pickles the per-bus time series we need for plotting.
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data import (
    apply_udh_oil_pin,
    load_network,
    network_path,
    udh_buses,
    udh_demand,
    udh_supply_assets,
    unpin_udh_oil,
)
from feasibility import bus_feasibility


SCRIPT_DIR = Path(__file__).resolve().parent
CACHE_DIR = SCRIPT_DIR / "cache"
CACHE_DIR.mkdir(exist_ok=True)


def run_scenario(net_path: Path, scenario: str) -> tuple[pd.DataFrame, dict]:
    n = load_network(net_path)
    if scenario == "baseline":
        apply_udh_oil_pin(n)
    elif scenario == "oil_unpinned":
        unpin_udh_oil(n)
    else:
        raise ValueError(scenario)

    rows = []
    series_pack: dict[str, dict] = {}
    for bus in udh_buses(n):
        demand = udh_demand(n, bus)
        assets = udh_supply_assets(n, bus)
        r = bus_feasibility(bus, scenario, demand, assets)
        rows.append({
            "bus": bus,
            "scenario": scenario,
            "status": r.status,
            "objective_eur_per_year": r.objective,
            "min_fixed_slack_mw": float(r.fixed_slack_t.min()),
            "demand_peak_mw": float(demand.max()),
            "binding_snapshot": r.binding_snapshot,
            **{f"add_{c}_mw": v for c, v in r.p_nom_added.items()},
        })
        series_pack[bus] = {
            "demand_t": r.demand_t,
            "fixed_supply_max_t": r.fixed_supply_max_t,
            "extendable_supply_potential_t": r.extendable_supply_potential_t,
            "fixed_slack_t": r.fixed_slack_t,
            "p_nom_added": r.p_nom_added,
            "status": r.status,
        }
        print(f"  [{scenario:13s}] {bus:25s}  {r.status:11s}  "
              f"min_fixed_slack={float(r.fixed_slack_t.min()):>10.1f} MW  "
              f"peak_dem={float(demand.max()):>8.1f} MW  "
              f"obj={r.objective:.2e} EUR/yr")

    return pd.DataFrame(rows), series_pack


def main():
    net_path = network_path(scenario="free", wiggle=1000)
    if not net_path.exists():
        raise SystemExit(f"network not found: {net_path}")
    print(f"Auditing {net_path.name}")

    all_rows = []
    all_series: dict[tuple[str, str], dict] = {}
    for sc in ("baseline", "oil_unpinned"):
        print(f"\n===== scenario: {sc} =====")
        df, pack = run_scenario(net_path, sc)
        all_rows.append(df)
        for bus, ts in pack.items():
            all_series[(sc, bus)] = ts

    summary = pd.concat(all_rows, ignore_index=True)
    summary.to_parquet(CACHE_DIR / "audit_v1.parquet", index=False)
    with open(CACHE_DIR / "audit_v1_series.pkl", "wb") as f:
        pickle.dump(all_series, f)

    print("\n===== headline =====")
    counts = summary.groupby("scenario").status.value_counts().unstack(fill_value=0)
    print(counts.to_string())
    print(f"\nWrote {CACHE_DIR / 'audit_v1.parquet'} and "
          f"{CACHE_DIR / 'audit_v1_series.pkl'}")


if __name__ == "__main__":
    main()
