"""Run the coupled UDH-heat + LV-electricity adequacy audit.

Per scenario:
  baseline      — UDH oil boilers pinned non-extendable (current code).
  oil_unpinned  — UDH oil boilers freed.

For each AC location with a non-zero UDH heat load, solves the coupled LP
in `feasibility_coupled.py` and records:
  - status / objective / per-asset capacity addition
  - peak LV electricity uplift from heating (HP + resistive draw)
  - snapshot where LV total load (baseline + uplift) is highest

Caches summary parquet + per-bus time-series pickle.
"""

from __future__ import annotations

import pickle
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data import (
    apply_udh_oil_pin,
    ac_location,
    load_network,
    lv_baseline_load,
    lv_bus,
    lv_supply,
    network_path,
    udh_buses,
    udh_demand,
    udh_supply_assets,
    unpin_udh_oil,
)
from feasibility_coupled import bus_coupled_feasibility


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
        lvb = lv_bus(bus)
        if lvb not in n.buses.index:
            print(f"  skip {bus}: no LV bus {lvb}")
            continue
        demand = udh_demand(n, bus)
        udh_assets = udh_supply_assets(n, bus)
        lv_assets = lv_supply(n, lvb)
        lv_load = lv_baseline_load(n, lvb)

        t0 = time.time()
        r = bus_coupled_feasibility(
            bus, lvb, scenario, demand, udh_assets, lv_load, lv_assets,
        )
        dt = time.time() - t0
        rows.append({
            "bus": bus,
            "ac_loc": ac_location(bus),
            "scenario": scenario,
            "status": r.status,
            "objective_eur_per_year": r.objective,
            "udh_demand_peak_mw": float(demand.max()),
            "lv_baseline_peak_mw": float(lv_load.max()),
            "lv_uplift_peak_mw": r.peak_lv_uplift_mw,
            "lv_total_peak_mw": r.peak_lv_total_mw,
            "lv_uplift_snap": r.peak_lv_uplift_snapshot,
            "lv_total_snap": r.peak_lv_total_snapshot,
            **{f"add_{name}_mw": v for name, v in r.p_nom_added.items()},
        })
        series_pack[bus] = {
            "lv_baseline_t": r.lv_load_baseline_t,
            "lv_total_t": r.lv_load_total_t,
            "udh_demand_t": r.udh_demand_t,
            "dist_in_t": r.distribution_inflow_t,
            "solar_t": r.rooftop_dispatch_t,
            "p_nom_added": r.p_nom_added,
            "status": r.status,
        }
        print(f"  [{scenario:13s}] {ac_location(bus):8s}  {r.status:11s}  "
              f"udh_dem_peak={float(demand.max()):>8.0f}  "
              f"lv_base_peak={float(lv_load.max()):>8.0f}  "
              f"lv_uplift_peak={r.peak_lv_uplift_mw:>8.0f}  "
              f"lv_total_peak={r.peak_lv_total_mw:>8.0f}  "
              f"obj={r.objective:.2e}  ({dt:.1f}s)")

    return pd.DataFrame(rows), series_pack


def main():
    net_path = network_path(scenario="free", wiggle=1000)
    if not net_path.exists():
        raise SystemExit(f"network not found: {net_path}")
    print(f"Coupled audit on {net_path.name}")

    all_rows = []
    all_series: dict[tuple[str, str], dict] = {}
    for sc in ("baseline", "oil_unpinned"):
        print(f"\n===== scenario: {sc} =====")
        df, pack = run_scenario(net_path, sc)
        all_rows.append(df)
        for bus, ts in pack.items():
            all_series[(sc, bus)] = ts

    summary = pd.concat(all_rows, ignore_index=True)
    summary.to_parquet(CACHE_DIR / "audit_coupled_v1.parquet", index=False)
    with open(CACHE_DIR / "audit_coupled_v1_series.pkl", "wb") as f:
        pickle.dump(all_series, f)

    print("\n===== headline =====")
    counts = summary.groupby("scenario").status.value_counts().unstack(fill_value=0)
    print(counts.to_string())
    top = summary[summary.scenario == "baseline"].nlargest(10, "lv_uplift_peak_mw")[
        ["ac_loc", "udh_demand_peak_mw", "lv_baseline_peak_mw",
         "lv_uplift_peak_mw", "lv_total_peak_mw"]
    ]
    print(f"\nTop-10 LV uplift (baseline scenario):\n{top.to_string(index=False)}")
    print(f"\nWrote {CACHE_DIR / 'audit_coupled_v1.parquet'}")


if __name__ == "__main__":
    main()
