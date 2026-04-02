# PyPSA-Eur Model — Claude Code Instructions (Post-Solve Outputs)

> **Companion to `ANALYSE_MODEL_INPUT_DATA.md`.** Everything below assumes the network `n` has been solved (`n.optimize()` or loaded from a solved `.nc` file). Check with `n.is_solved`.

## The `n.statistics` module

`n.statistics` is the primary interface for extracting aggregated results. All methods return pandas `Series` or `DataFrame` objects. They share a common set of keyword arguments:

### Common parameters (all methods)

| Parameter | Type | Default | Description |
|---|---|---|---|
| `components` | `str \| list[str] \| None` | `None` (all) | Component types to include, e.g. `["Generator", "Link"]`. |
| `groupby` | `str \| list[str] \| callable` | `"carrier"` | How to group results. Built-in keys: `"carrier"`, `"bus_carrier"`, `"bus"`, `"country"`, `"name"`. Pass a list for multi-index grouping, e.g. `["bus", "carrier"]`. |
| `groupby_method` | `str \| callable` | `"sum"` | Aggregation across groups: `"sum"`, `"mean"`, `"max"`, etc. |
| `groupby_time` | `str \| bool` | `"sum"` (most) | Time aggregation. `"sum"` and `"mean"` respect snapshot weightings. `False` returns the full time series. |
| `bus_carrier` | `str \| list[str] \| None` | `None` | Filter to components attached to buses of this carrier (e.g. `"AC"`, `"H2"`, `"rural heat"`). |
| `carrier` | `str \| list[str] \| None` | `None` | Filter to components with this technology carrier. |
| `nice_names` | `bool \| None` | `None` | Use `n.carriers.nice_name` for labelling. |
| `aggregate_across_components` | `bool` | `False` | If `True`, collapse the component-level index. |

> **Version note:** In PyPSA ≥ 1.0.4, the old parameter names (`comps`, `aggregate_groups`, `aggregate_time`) are deprecated in favour of `components`, `groupby_method`, `groupby_time`. Both still work but prefer the new names.

### Available methods

| Method | Returns | Unit | Notes |
|---|---|---|---|
| `n.statistics()` | Overview DataFrame with all metrics as columns | mixed | Quick summary table. |
| `n.statistics.optimal_capacity()` | Optimised capacity | MW | `p_nom_opt` / `s_nom_opt` / `e_nom_opt`. |
| `n.statistics.installed_capacity()` | Pre-existing (brownfield) capacity | MW | Based on `p_nom` / `s_nom`. |
| `n.statistics.expanded_capacity()` | New capacity added by optimiser | MW | `p_nom_opt - p_nom` for extendables. |
| `n.statistics.capex()` | Capital expenditure | EUR/a | `capital_cost × p_nom_opt`. Annualised. |
| `n.statistics.installed_capex()` | Capex of pre-existing capacity | EUR/a | `capital_cost × p_nom`. |
| `n.statistics.expanded_capex()` | Capex of new capacity only | EUR/a | `capital_cost × (p_nom_opt - p_nom)`. |
| `n.statistics.opex()` | Operational expenditure | EUR/a | `marginal_cost × dispatch × weighting`. Defaults to `groupby_time="sum"`. |
| `n.statistics.supply()` | Energy supplied (positive flows) | MWh | Positive dispatch at the component's bus. |
| `n.statistics.withdrawal()` | Energy withdrawn (negative flows) | MWh | Consumption / charging. |
| `n.statistics.curtailment()` | Curtailed energy | MWh | Only for components with `p_max_pu` time series. |
| `n.statistics.capacity_factor()` | Capacity factor / utilisation | — | Defaults to `groupby_time="mean"`. |
| `n.statistics.revenue()` | Revenue from marginal prices | EUR/a | `Σ_t (marginal_price_at_bus × dispatch × weighting)`. |
| `n.statistics.market_value()` | Average captured price | EUR/MWh | `revenue / dispatch`. Problematic for storage — see caveats. |
| `n.statistics.energy_balance()` | Net energy balance per bus carrier | MWh | Indexed by `(component, carrier, bus_carrier)`. The key method for energy flow analysis. |
| `n.statistics.transmission()` | Energy transmitted | MWh | Lines and links connecting buses of the same carrier. |

## Energy balances

The energy balance is the most important output for understanding carrier-level flows.

### System-wide energy balance by carrier, for a single bus carrier

```python
# Electricity balance aggregated across all regions
eb_elec = n.statistics.energy_balance(bus_carrier="AC")
# Returns Series indexed by (component, carrier) — positive = supply, negative = demand

# For heat
eb_heat = n.statistics.energy_balance(bus_carrier="urban central heat")
```

### Energy balance disaggregated by region

```python
# Group by bus AND carrier to get regional breakdown
eb_regional = n.statistics.energy_balance(
    bus_carrier="AC",
    groupby=["bus", "carrier"],
)
# Returns Series with MultiIndex (component, bus, carrier)
```

### Energy balance as a time series

```python
# Full hourly time series — no temporal aggregation
eb_ts = n.statistics.energy_balance(
    bus_carrier="AC",
    groupby_time=False,
)
# Returns DataFrame: columns = (component, carrier), index = snapshots
```

### Energy balance disaggregated by BOTH region and time

```python
eb_regional_ts = n.statistics.energy_balance(
    bus_carrier="AC",
    groupby=["bus", "carrier"],
    groupby_time=False,
)
# Returns DataFrame: columns = (component, bus, carrier), index = snapshots
```

### Plotting the energy balance

```python
import matplotlib.pyplot as plt

eb = n.statistics.energy_balance(bus_carrier="AC")

# Drop component level, keep carrier
eb_by_carrier = eb.groupby("carrier").sum()

fig, ax = plt.subplots()
eb_by_carrier.to_frame().T.plot.bar(stacked=True, ax=ax)
ax.set_ylabel("MWh")
ax.set_title("Electricity Energy Balance")
```

## Costs — capex, opex, and total system cost

### Total annualised system cost

```python
# The objective function value (does NOT include brownfield capex)
n.objective

# The brownfield constant (capex of pre-existing infrastructure)
n.objective_constant

# Total system cost including brownfield
total_system_cost = n.objective + n.objective_constant

# Cross-check via statistics (the trusted calculation)
total_check = n.statistics.capex().sum() + n.statistics.opex().sum()
# This should equal n.objective + n.objective_constant
```

Warning: Computing total costs from `n.statistics.capex() + n.statistics.opex()` will yield an error of 
factor two, because these series have different carriers. Reindex to the union first:

```
```
c = n.statistics.capex()
o = n.statistics.opex()

union = c.index.union(o.index)
c = c.reindex(union).replace(np.nan, 0)
o = o.reindex(union).replace(np.nan, 0)
```

then proceed with processing the series.


### Cost breakdown by technology

```python
costs = pd.concat({
    "capex": n.statistics.capex(),
    "opex": n.statistics.opex(),
}, axis=1).replace(np.nan, 0.)
costs["total"] = costs.sum(axis=1)
# Indexed by (component, carrier)
```


### Cost breakdown by technology AND region

```python
costs_regional = pd.concat({
    "capex": n.statistics.capex(groupby=["bus", "carrier"]),
    "opex": n.statistics.opex(groupby=["bus", "carrier"]),
}, axis=1)
```

## Capacities

```python
# Optimal capacity by carrier (system-wide)
n.statistics.optimal_capacity()

# By region
n.statistics.optimal_capacity(groupby=["bus", "carrier"])

# Only generators
n.statistics.optimal_capacity(components=["Generator"])

# Storage energy capacity (MWh) — use the Store component
n.statistics.optimal_capacity(components=["Store"])
```

## LCOE calculations

### Generators — straightforward via `n.statistics`

For generators, LCOE = (capex + opex) / supply.

```python
capex = n.statistics.capex(components=["Generator"])
opex = n.statistics.opex(components=["Generator"])
supply = n.statistics.supply(components=["Generator"])

lcoe_gen = (capex + opex) / supply  # EUR/MWh
```

This works because generator `opex` via `n.statistics` is simply `marginal_cost × dispatch`, and generators have no fuel input cost — their `marginal_cost` already encodes fuel cost (fuel price / efficiency) as set during network preparation.

### Links — requires manual fuel cost accounting

For links (conversion technologies like heat pumps, electrolysers, CHPs), `n.statistics.opex()` only captures the **direct** `marginal_cost` attribute (typically small O&M costs). It does **not** include the cost of the input commodity consumed at `bus0`. To get a true LCOE you must add the fuel cost.

```python
def lcoe_links(n, carrier, bus_port=1):
    """
    Compute LCOE for a link carrier, including input fuel costs.

    Parameters
    ----------
    n : pypsa.Network
        Solved network.
    carrier : str
        Link carrier name (e.g. "rural ground heat pump", "H2 Electrolysis").
    bus_port : int
        Output bus port for which to compute LCOE (default 1 = bus1).

    Returns
    -------
    float
        System-wide average LCOE in EUR/MWh of output at bus_port.
    """
    links = n.links[n.links.carrier == carrier]
    if links.empty:
        return float("nan")

    # --- Capital cost ---
    capex = (links.capital_cost * links.p_nom_opt).sum()  # EUR/a

    # --- Direct marginal cost (O&M component from marginal_cost attribute) ---
    # n.statistics.opex accounts for snapshot weightings
    opex_direct = n.statistics.opex(
        components=["Link"], carrier=carrier
    ).sum()

    # --- Input fuel cost ---
    # Get input dispatch at bus0 (always positive for links)
    p0 = n.links_t.p0[links.index]  # MW, shape (snapshots, n_links)
    weights = n.snapshot_weightings.generators  # hours per snapshot

    # Marginal price at each link's input bus
    bus0_prices = n.buses_t.marginal_price[links.bus0.values]
    bus0_prices.columns = links.index  # align columns

    # Fuel cost = Σ_t (price_at_bus0 × input_dispatch × snapshot_weight)
    fuel_cost = (bus0_prices * p0).multiply(weights, axis=0).sum().sum()

    # --- Output energy ---
    if bus_port == 1:
        # Output at bus1 (negative by PyPSA convention for links)
        output = -n.links_t.p1[links.index]
    elif bus_port == 2:
        output = -n.links_t.p2[links.index]
    else:
        output = -getattr(n.links_t, f"p{bus_port}")[links.index]

    total_output_mwh = output.multiply(weights, axis=0).sum().sum()

    if total_output_mwh == 0:
        return float("nan")

    return (capex + opex_direct + fuel_cost) / total_output_mwh
```

#### Example usage

```python
# LCOE of rural air heat pumps (EUR/MWh_heat)
lcoe_hp = lcoe_links(n, "rural air heat pump", bus_port=1)

# LCOE of electrolysis (EUR/MWh_H2)
lcoe_elec = lcoe_links(n, "H2 Electrolysis", bus_port=1)
```

#### LCOE for links — disaggregated by region

```python
def lcoe_links_by_region(n, carrier, bus_port=1):
    """
    Per-region LCOE for a link carrier.

    Returns
    -------
    pd.Series
        LCOE indexed by location (derived from bus0).
    """
    links = n.links[n.links.carrier == carrier]
    if links.empty:
        return pd.Series(dtype=float)

    locations = n.buses.index[n.buses.carrier == "AC"]
    weights = n.snapshot_weightings.generators

    records = {}
    for loc in locations:
        # Find links at this location (match bus0 prefix)
        mask = links.bus0.str.startswith(loc)
        loc_links = links[mask]
        if loc_links.empty:
            continue

        capex = (loc_links.capital_cost * loc_links.p_nom_opt).sum()

        p0 = n.links_t.p0[loc_links.index]
        bus0_prices = n.buses_t.marginal_price[loc_links.bus0.values]
        bus0_prices.columns = loc_links.index
        fuel_cost = (bus0_prices * p0).multiply(weights, axis=0).sum().sum()

        # Direct opex
        mc = loc_links.marginal_cost
        if loc_links.index.isin(n.links_t.marginal_cost.columns).any():
            mc_t = n.links_t.marginal_cost[
                loc_links.index.intersection(n.links_t.marginal_cost.columns)
            ]
            opex_direct = (mc_t * p0[mc_t.columns]).multiply(weights, axis=0).sum().sum()
            # Add static marginal cost for links not in time-varying
            static_links = loc_links.index.difference(mc_t.columns)
            if not static_links.empty:
                opex_direct += (
                    (mc[static_links] * p0[static_links])
                    .multiply(weights, axis=0).sum().sum()
                )
        else:
            opex_direct = (mc * p0).multiply(weights, axis=0).sum().sum()

        if bus_port == 1:
            output = -n.links_t.p1[loc_links.index]
        else:
            output = -getattr(n.links_t, f"p{bus_port}")[loc_links.index]
        total_output = output.multiply(weights, axis=0).sum().sum()

        if total_output > 0:
            records[loc] = (capex + opex_direct + fuel_cost) / total_output

    return pd.Series(records, name=f"LCOE_{carrier}")
```

#### LCOE caveats

- For **CHP links** with multiple valuable outputs (electricity at `bus1` AND heat at `bus2`), a single-output LCOE is misleading. Consider allocating costs across outputs proportionally, or compute a combined cost per unit of primary output and note the revenue offset from the co-product.
- For links with **time-varying efficiency** (e.g. heat pumps), the output at `bus1` already reflects the time-varying COP, so the above calculation is correct without any COP adjustment.
- For links that **consume** at `bus2` (negative `efficiency2`, e.g. co-electrolysis consuming CO₂), add the `bus2` input cost analogously to the `bus0` fuel cost.
- `n.statistics.opex()` for links uses the `marginal_cost` attribute only. It does **not** multiply by bus prices. This is the key reason a manual fuel cost calculation is needed.

## Commodity prices from `n.buses_t.marginal_price`

After solving, `n.buses_t.marginal_price` contains the shadow price (dual variable) at every bus for every snapshot. This is the locational marginal price (LMP) — the system cost of supplying one additional MWh at that bus and time.

### Raw access

```python
# Full matrix: shape (n_snapshots, n_buses)
n.buses_t.marginal_price

# Price at a specific bus
n.buses_t.marginal_price["DE0 0"]  # electricity price at node DE0 0

# Price at a carrier-specific bus
n.buses_t.marginal_price["DE0 0 H2"]  # hydrogen price
n.buses_t.marginal_price["DE0 0 rural heat"]  # rural heat price
```

### System-wide average price for a carrier (load-weighted)

A simple time-mean is misleading because demand varies across regions and hours. Weight by load to get an economically meaningful average.

```python
def carrier_price_weighted(n, bus_carrier, weight_by_load=True):
    """
    Load-weighted average commodity price for a bus carrier.

    Parameters
    ----------
    n : pypsa.Network
        Solved network.
    bus_carrier : str
        Bus carrier, e.g. "AC", "H2", "rural heat".
    weight_by_load : bool
        If True, weight by total load/withdrawal at each bus.
        If False, return simple time-mean across all buses.

    Returns
    -------
    float
        Weighted average price in EUR/MWh.
    """
    carrier_buses = n.buses.index[n.buses.carrier == bus_carrier]
    prices = n.buses_t.marginal_price[carrier_buses]
    weights_t = n.snapshot_weightings.generators  # hours per snapshot

    if not weight_by_load:
        # Simple weighted time-mean, then average across buses
        return (prices.multiply(weights_t, axis=0).sum() / weights_t.sum()).mean()

    # --- Compute loads at each bus ---
    # Loads component
    loads_at_buses = n.loads[n.loads.bus.isin(carrier_buses)]
    if not loads_at_buses.empty:
        # Build load time series per bus
        load_ts = pd.DataFrame(0.0, index=n.snapshots, columns=carrier_buses)
        for load_name, load_row in loads_at_buses.iterrows():
            bus = load_row.bus
            if load_name in n.loads_t.p_set.columns:
                load_ts[bus] += n.loads_t.p_set[load_name]
            elif load_name in n.loads_t.p.columns:
                load_ts[bus] += n.loads_t.p[load_name]
            else:
                load_ts[bus] += load_row.p_set
    else:
        # Fall back: use link withdrawals at these buses as weight
        load_ts = pd.DataFrame(0.0, index=n.snapshots, columns=carrier_buses)
        # Links consuming from these buses
        consuming_links = n.links[n.links.bus0.isin(carrier_buses)]
        for link_name, link_row in consuming_links.iterrows():
            bus = link_row.bus0
            if link_name in n.links_t.p0.columns:
                load_ts[bus] += n.links_t.p0[link_name]

    # Weighted price: Σ(price × load × weight) / Σ(load × weight)
    numerator = (prices * load_ts).multiply(weights_t, axis=0).sum().sum()
    denominator = load_ts.multiply(weights_t, axis=0).sum().sum()

    if denominator == 0:
        return float("nan")

    return numerator / denominator
```

#### Example usage

```python
# Average wholesale electricity price (load-weighted)
elec_price = carrier_price_weighted(n, "AC")

# Average hydrogen price
h2_price = carrier_price_weighted(n, "H2")

# Average rural heat price
heat_price = carrier_price_weighted(n, "rural heat")
```

### Prices disaggregated by region

```python
def carrier_prices_by_region(n, bus_carrier):
    """
    Time-averaged commodity price per region, weighted by snapshot hours.

    Returns
    -------
    pd.Series
        Average price indexed by location.
    """
    locations = n.buses.index[n.buses.carrier == "AC"]
    carrier_buses = n.buses.index[n.buses.carrier == bus_carrier]
    weights_t = n.snapshot_weightings.generators

    if bus_carrier == "AC":
        price_buses = locations
    else:
        price_buses = pd.Index([f"{loc} {bus_carrier}" for loc in locations])
        price_buses = price_buses.intersection(carrier_buses)

    prices = n.buses_t.marginal_price[price_buses]
    avg = prices.multiply(weights_t, axis=0).sum() / weights_t.sum()
    avg.index = avg.index.str.replace(f" {bus_carrier}", "") if bus_carrier != "AC" else avg.index
    avg.name = f"price_{bus_carrier}"
    return avg
```

### Prices disaggregated by time (system-wide, load-weighted per snapshot)

```python
def carrier_price_timeseries(n, bus_carrier):
    """
    Load-weighted system price as a time series.

    Returns
    -------
    pd.Series
        Hourly load-weighted price, indexed by snapshot.
    """
    carrier_buses = n.buses.index[n.buses.carrier == bus_carrier]
    prices = n.buses_t.marginal_price[carrier_buses]

    # Build load weights per bus
    loads_at_buses = n.loads[n.loads.bus.isin(carrier_buses)]
    load_ts = pd.DataFrame(0.0, index=n.snapshots, columns=carrier_buses)

    if not loads_at_buses.empty:
        for load_name, load_row in loads_at_buses.iterrows():
            bus = load_row.bus
            if load_name in n.loads_t.p_set.columns:
                load_ts[bus] += n.loads_t.p_set[load_name]
            elif load_name in n.loads_t.p.columns:
                load_ts[bus] += n.loads_t.p[load_name]
            else:
                load_ts[bus] += load_row.p_set
    else:
        # Fallback: link withdrawals
        consuming_links = n.links[n.links.bus0.isin(carrier_buses)]
        for link_name, link_row in consuming_links.iterrows():
            bus = link_row.bus0
            if link_name in n.links_t.p0.columns:
                load_ts[bus] += n.links_t.p0[link_name]

    # Per-snapshot weighted average price
    total_load = load_ts.sum(axis=1)
    weighted_price = (prices * load_ts).sum(axis=1) / total_load.replace(0, float("nan"))
    weighted_price.name = f"price_{bus_carrier}"
    return weighted_price
```

### Prices disaggregated by BOTH region and time

```python
def carrier_price_matrix(n, bus_carrier):
    """
    Full price matrix: time × region.

    Returns
    -------
    pd.DataFrame
        Columns = locations, index = snapshots.
    """
    locations = n.buses.index[n.buses.carrier == "AC"]

    if bus_carrier == "AC":
        buses = locations
    else:
        buses = pd.Index([f"{loc} {bus_carrier}" for loc in locations])
        buses = buses.intersection(n.buses.index)

    prices = n.buses_t.marginal_price[buses].copy()
    if bus_carrier != "AC":
        prices.columns = prices.columns.str.replace(f" {bus_carrier}", "")
    return prices
```

## Quick-reference via `n.statistics` (no custom code needed)

For many common queries, `n.statistics` suffices without custom functions:

```python
# Capacity factor by carrier
n.statistics.capacity_factor()

# Revenue by carrier
n.statistics.revenue()

# Revenue by carrier AND region
n.statistics.revenue(groupby=["bus", "carrier"])

# Curtailment
n.statistics.curtailment()

# Transmission flows
n.statistics.transmission()

# Opex time series (not aggregated over time)
n.statistics.opex(groupby_time=False)

# Supply time series for generators only
n.statistics.supply(components=["Generator"], groupby_time=False)
```

## Dispatch time series (raw access)

When `n.statistics` grouping does not give you what you need, access raw dispatch:

```python
# Generator dispatch (MW)
n.generators_t.p

# Link input (bus0) — always positive
n.links_t.p0

# Link output (bus1) — negative by convention
n.links_t.p1

# Store state of charge
n.stores_t.e

# StorageUnit state of charge
n.storage_units_t.state_of_charge

# StorageUnit dispatch and storage
n.storage_units_t.p_dispatch
n.storage_units_t.p_store
n.storage_units_t.p  # net = p_dispatch - p_store
```

## Snapshot weightings

**Always** account for snapshot weightings when aggregating time series manually. If the model uses subsampled snapshots (e.g. every 3rd hour), each snapshot represents multiple hours:

```python
weights = n.snapshot_weightings.generators  # Series indexed by snapshot, values = hours

# Correct energy sum
energy_mwh = (dispatch_mw * weights).sum()

# Correct time-weighted mean
mean_mw = (dispatch_mw * weights).sum() / weights.sum()
```

`n.statistics` handles this automatically — you only need to worry about it when computing things manually.

## Anti-patterns for output analysis — avoid these

1. **Don't sum dispatch without snapshot weightings.** If `n.snapshot_weightings` is not all 1.0, a bare `.sum()` gives wrong energy totals.
2. **Don't assume `n.statistics.opex()` includes fuel costs for links.** It only uses the `marginal_cost` attribute. For links, the commodity input cost must be computed separately from `n.buses_t.marginal_price`.
3. **Don't use `n.statistics.market_value()` for storage.** Net dispatch of storage is ≤ 0 (due to round-trip losses), making the ratio `revenue / dispatch` negative or undefined.
4. **Don't confuse `n.objective` with total system cost.** Add `n.objective_constant` for the full picture, or use `n.statistics.capex().sum() + n.statistics.opex().sum()`.
5. **Don't take simple time-means of `n.buses_t.marginal_price`.** Weight by load or demand at the bus to get economically meaningful average prices.
6. **Don't forget `bus_carrier` when calling `energy_balance`.** Without it, results are indexed by `(component, carrier, bus_carrier)` and mix all carriers together. Filter to the carrier of interest.
7. **Don't use old parameter names in new PyPSA versions.** Use `groupby_time` not `aggregate_time`, `groupby_method` not `aggregate_groups`, `components` not `comps`.


