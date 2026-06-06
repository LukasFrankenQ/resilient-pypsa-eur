"""
Diagnose why urban decentral (individual) heating prices are very high in France
compared to neighbours in the base_s_50_lv1.25_..._free_2000 network.
"""

from pathlib import Path
import pandas as pd
import pypsa

pd.set_option("display.max_rows", 100)
pd.set_option("display.width", 200)

ROOT = Path(__file__).resolve().parents[3]
NETWORK_PATH = (
    ROOT / "results" / "networks"
    / "base_s_50_lv1.25_3H-T-H-B-I-A-dist1_2030_free_2000.nc"
)

n = pypsa.Network(NETWORK_PATH)
weights = n.snapshot_weightings.generators

# 1) Identify urban decentral heat buses by country.
heat = n.buses[n.buses.carrier == "urban decentral heat"].copy()
heat["country"] = heat.location.str[:2]
print(f"\nUrban decentral heat buses: {len(heat)} total, "
      f"across {heat.country.nunique()} countries")
by_country_count = heat.groupby("country").size()
print(by_country_count.to_string())

# 2) Average price per country (time-weighted, simple mean across buses).
prices = n.buses_t.marginal_price[heat.index]
avg_p = (prices * weights.values[:, None]).sum() / weights.sum()
avg_p = avg_p.rename(heat.country).groupby(level=0).mean()
print("\nMean urban decentral heat price by country (EUR/MWh):")
print(avg_p.sort_values(ascending=False).head(10).to_string())
print("...")
print(avg_p.sort_values(ascending=False).tail(10).to_string())

# 3) Highest-price French bus + compare with a cheaper neighbour (DE).
FR_buses = heat.index[heat.country == "FR"]
DE_buses = heat.index[heat.country == "DE"]
print(f"\nFR buses: {list(FR_buses)}")
print(f"DE buses: {list(DE_buses)}")

# 4) For each FR bus, find supplying links (those feeding heat into the bus,
#    i.e. with bus1 or bus2 == bus_name and positive flow out).
def supply_breakdown(net: pypsa.Network, target_buses):
    """Energy supplied to a set of heat buses, by tech carrier."""
    records = []
    links = net.links
    for port in (1, 2, 3):
        bus_col = f"bus{port}"
        p_col = f"p{port}"
        if bus_col not in links.columns:
            continue
        if p_col not in net.links_t:
            continue
        mask = links[bus_col].isin(target_buses)
        if not mask.any():
            continue
        sub = links[mask]
        p = net.links_t[p_col][sub.index]
        # heat output at bus1/2/3 is negative under PyPSA sign convention
        supply = -p.clip(upper=0).multiply(weights, axis=0).sum()
        for ln in sub.index:
            if supply[ln] > 1e-3:
                records.append({
                    "link": ln,
                    "carrier": sub.at[ln, "carrier"],
                    "bus0": sub.at[ln, "bus0"],
                    "target_bus": sub.at[ln, bus_col],
                    "MWh_supplied": float(supply[ln]),
                })
    return pd.DataFrame(records)


fr_supply = supply_breakdown(n, FR_buses)
de_supply = supply_breakdown(n, DE_buses)

print("\nFR urban decentral heat — supply by carrier (TWh):")
print((fr_supply.groupby("carrier").MWh_supplied.sum() / 1e6)
      .sort_values(ascending=False).to_string())

print("\nDE urban decentral heat — supply by carrier (TWh):")
print((de_supply.groupby("carrier").MWh_supplied.sum() / 1e6)
      .sort_values(ascending=False).to_string())

# 5) For each FR bus, find the *marginal* technology setting the price.
#    Look at the price level and compare with the inputs' marginal cost / fuel cost.
print("\nFR per-bus annual-average heat price (EUR/MWh):")
print(avg_p[avg_p.index == "FR"])  # only one entry — FR mean

# Per-bus detail
fr_bus_prices = (prices[FR_buses] * weights.values[:, None]).sum() / weights.sum()
print(fr_bus_prices.sort_values(ascending=False).to_string())

# 6) Check loads & their volumes to make sure FR has comparable load weight.
loads_at_fr = n.loads[n.loads.bus.isin(FR_buses)]
print(f"\nLoads at FR urban decentral heat buses: {len(loads_at_fr)}")
print(loads_at_fr[["bus", "carrier", "p_set"]].to_string())

# Total annual demand
def annual_load(net, loads_df):
    ts = pd.DataFrame(0.0, index=net.snapshots, columns=loads_df.index)
    for ln in loads_df.index:
        if ln in net.loads_t.p_set.columns:
            ts[ln] = net.loads_t.p_set[ln]
        else:
            ts[ln] = loads_df.at[ln, "p_set"]
    return float((ts.multiply(weights, axis=0).sum().sum()))

fr_demand_mwh = annual_load(n, loads_at_fr)
loads_at_de = n.loads[n.loads.bus.isin(DE_buses)]
de_demand_mwh = annual_load(n, loads_at_de)
print(f"\nFR urban decentral heat annual demand: {fr_demand_mwh/1e6:.2f} TWh")
print(f"DE urban decentral heat annual demand: {de_demand_mwh/1e6:.2f} TWh")

# 7) Inspect cost-defining inputs: marginal cost at bus0 of links feeding FR
fr_inputs = fr_supply.groupby("carrier").agg(
    MWh=("MWh_supplied", "sum"),
    n_links=("link", "count"),
).sort_values("MWh", ascending=False)
print("\nFR supplying tech summary:\n", fr_inputs.to_string())

# 8) For top FR carriers, look at the input fuel price (price at bus0)
top_carriers = fr_inputs.head(5).index.tolist()
for car in top_carriers:
    rows = fr_supply[fr_supply.carrier == car]
    bus0s = rows.bus0.unique()
    if len(bus0s) == 0:
        continue
    bus0_prices = (
        n.buses_t.marginal_price[list(bus0s)]
        .multiply(weights, axis=0).sum()
        / weights.sum()
    )
    print(f"\nFR carrier='{car}' input bus0 prices (EUR/MWh):")
    print(bus0_prices.to_string())

# 9) Capacities of urban decentral heat techs in FR vs DE
print("\nUrban decentral tech capacities (MW):")
links_fr = n.links[n.links.bus1.isin(FR_buses) | n.links.bus2.isin(FR_buses)]
links_de = n.links[n.links.bus1.isin(DE_buses) | n.links.bus2.isin(DE_buses)]
cap_fr = links_fr.groupby("carrier").p_nom_opt.sum()
cap_de = links_de.groupby("carrier").p_nom_opt.sum()
cap = pd.concat({"FR": cap_fr, "DE": cap_de}, axis=1).fillna(0).sort_values("FR", ascending=False)
print(cap.to_string())
