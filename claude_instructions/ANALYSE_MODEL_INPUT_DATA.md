# PyPSA-Eur Model — Claude Code Instructions

## Conventions

- All costs are **annualized** (EUR/MW/a for capital, EUR/MWh for marginal).
- Capacities (`p_nom`, `p_nom_min`, `p_nom_max`) are in **MW**.
- The network object is always `n` (a `pypsa.Network`).
- Snapshots (`n.snapshots`) are the temporal resolution of the model (hourly or subsampled).
- Technologies are either `generators` (dispatch/produce) or `links` (convert between carriers).
- Links can connect up to 4 buses (`bus0`–`bus3`). `efficiency`, `efficiency2`, `efficiency3` correspond to the conversion ratios for `bus1`, `bus2`, `bus3` respectively.
- `p_nom_extendable = True` means the optimizer can invest in new capacity.
- `p_nom_min > 0` indicates existing brownfield capacity that cannot be removed.

## Spatial structure

The model uses 50 nodes representing European regions:

```python
locations = n.buses.index[n.buses.carrier == "AC"]
```

Never hardcode the location list — always derive it from the network.

## Bus and carrier system

Each location has multiple buses representing different energy carriers. Access them via:

```python
# Electricity (AC) buses — these ARE the locations index
n.buses.loc[locations]

# Carrier-specific buses at a location
n.buses.loc[locations + f" {carrier}"]

# Example: hydrogen buses
n.buses.loc[locations + " H2"]
```

Do not hardcode carrier names — derive them from:

```python
n.buses.carrier.unique()
```

### Carrier categories (mental model)

- **Electricity**: `AC`, `low voltage`
- **Heat**: `rural heat`, `urban decentral heat`, `urban central heat`, `heat<100 industry`, `heat100-200 industry`, `heat200-500 industry`, `heat>500 industry`
- **Fuels**: `gas`, `oil`, `H2`, `methanol`, `NH3`, `solid biomass`, `biogas`, `coal`, `lignite`, `uranium`
- **Transport**: `land transport oil`, `EV battery`, `kerosene for aviation`, `shipping methanol`, `shipping oil`, `agriculture machinery oil`
- **Industry**: `steel`, `hbi`, `naphtha for industry`, `gas for industry`, `industry methanol`, `non-sequestered HVC`, `process emissions`, `coal for industry`, `solid biomass for industry`
- **Storage**: `battery`, `home battery`, `EV battery`, `H2` (when linked to H2 stores)

## Generators

Generators sit on a single bus and produce a carrier. Access them via:

```python
# All generators and their key attributes
n.generators[["bus", "carrier", "p_nom", "p_nom_max", "capital_cost", "marginal_cost", "p_nom_extendable"]]

# Filter by carrier
n.generators[n.generators.carrier == "onwind"]

# Filter by location — use the bus column, not string matching on the index
n.generators[n.generators.bus.isin(locations)]  # all AC-bus generators
```

### Capacity factors (renewables)

Wind, solar, and hydro have time-varying capacity factors:

```python
# Get capacity factor time series for a specific generator
n.generators_t.p_max_pu["DE0 0 onwind"]

# Shape: (len(n.snapshots),) — values between 0 and 1
```

**Never assume a fixed capacity factor for renewables — always use the time series.**

## Links (conversion technologies)

Links convert energy between carriers. They connect `bus0` (input) to `bus1` (primary output), with optional `bus2` and `bus3` for co-products or emissions.

```python
# All links and their key attributes
n.links[["bus0", "bus1", "bus2", "bus3", "carrier", "efficiency", "efficiency2", "efficiency3", "p_nom", "capital_cost", "marginal_cost", "p_nom_extendable"]]

# Filter by carrier
n.links[n.links.carrier == "rural air heat pump"]

# All heating technologies at a location
n.links[n.links.bus1.str.startswith("DE0 0") & n.links.bus1.str.contains("heat")]
```

### Reading link efficiencies

- `efficiency` = output at `bus1` per unit input at `bus0`
- `efficiency2` = output at `bus2` per unit input at `bus0` (negative means consumption)
- `efficiency3` = output at `bus3` per unit input at `bus0`

Example: a gas CHP with `efficiency=0.41`, `efficiency2=0.41`, `efficiency3=0.198` produces 0.41 MW electricity + 0.41 MW heat per 1 MW gas input, while emitting 0.198 t CO2/MWh_gas.

### Time-varying efficiencies (heat pumps)

Heat pump COP is time-varying and weather-dependent. **Never use a scalar efficiency for heat pumps.**

```python
# Get COP time series for a heat pump link
n.links_t.efficiency["DE0 0 rural air heat pump"]
```

If `n.links_t.efficiency` is empty for a given link, the static value in `n.links.efficiency` applies.

## Anti-patterns — avoid these

1. **Don't filter by string matching on component indices.** Use the `carrier`, `bus`, `bus0`, `bus1` columns instead.
2. **Don't hardcode location or carrier lists.** Always derive from the network object.
3. **Don't assume scalar COP for heat pumps.** Always check `n.links_t.efficiency`.
4. **Don't confuse `p_nom` with dispatch.** `p_nom` is installed capacity; actual dispatch is in `n.generators_t.p` and `n.links_t.p0` / `n.links_t.p1` (after solving).
5. **Don't use `.loc` with carrier names on the index.** Generator/link indices are formatted as `"{location} {carrier}"` but this is not guaranteed to be unique. Always filter via columns.
6. **Don't forget that `efficiency2` can be negative** — this means `bus2` is an input (e.g., CO2 captured from atmosphere, or hydrogen consumed as co-input).

## Common tasks

### Get all technologies supplying a carrier at a location

```python
loc = "DE0 0"
carrier_bus = f"{loc} rural heat"

# Generators on this bus
gens = n.generators[n.generators.bus == carrier_bus]

# Links outputting to this bus (bus1)
links_out = n.links[n.links.bus1 == carrier_bus]

# Links with co-product to this bus (bus2, bus3)
links_bus2 = n.links[n.links.bus2 == carrier_bus]
links_bus3 = n.links[n.links.bus3 == carrier_bus]
```

### Compute LCOE for an extendable generator

```python
gen = n.generators.loc["DE0 0 onwind"]
cf = n.generators_t.p_max_pu["DE0 0 onwind"].mean()
lcoe = gen.capital_cost / (cf * 8760) + gen.marginal_cost  # EUR/MWh
```

Note: `capital_cost` is already annualized, so divide by full-load hours (CF × 8760).

### Get total system cost

```python
n.objective  # after solving — total annualized system cost in EUR
```

### Get optimal capacity of a technology

```python
n.links[n.links.carrier == "rural air heat pump"]["p_nom_opt"]  # after solving
n.generators[n.generators.carrier == "onwind"]["p_nom_opt"]
```

### Get dispatch time series (post-solve)

```python
# Generator dispatch
n.generators_t.p["DE0 0 onwind"]

# Link flows (p0 = input from bus0, p1 = output to bus1)
n.links_t.p0["DE0 0 rural air heat pump"]
n.links_t.p1["DE0 0 rural air heat pump"]
```

### Aggregate capacity by carrier across all locations

```python
n.generators.groupby("carrier").p_nom_opt.sum()
n.links.groupby("carrier").p_nom_opt.sum()
```

## Stores and storage units

For storage (batteries, H2, thermal), check:

```python
n.stores[["bus", "carrier", "e_nom", "e_nom_opt", "capital_cost"]]
n.storage_units[["bus", "carrier", "p_nom", "state_of_charge_initial"]]
```

Storage energy capacity is `e_nom` (MWh), power capacity is linked via the charger/discharger links.
