# Price-setting classification in a solved PyPSA-Eur network

This document describes, in full detail, how `classify_price_setter.py` and
its downstream consumer `plot_storage_price_vs_gas.py` decide which component
is setting the AC marginal electricity price at a given `(bus, snapshot)`
pair in a sector-coupled PyPSA-Eur network.

The script operates on a solved network (`n.is_solved == True`) — the LP
has been solved and dual variables (shadow prices) are available in
`n.buses_t.marginal_price`.

---

## 1. What “price-setting” means in an LP

The node price at bus `b` and snapshot `t` is the dual variable of the
energy balance at that bus:

```
λ_b(t)  =  ∂(objective) / ∂(demand at bus b at time t)     [EUR / MWh]
```

Under LP optimality, this equals the effective marginal cost of whichever
flexible component is *currently* used to cover an infinitesimal extra
MWh at `b`. We call that component the **price setter**. It must be
simultaneously:

1. **Interior**: dispatching strictly between its lower and upper bound
   (not at a capacity-binding corner).
2. **Matching**: its effective cost of delivering 1 MWh at `b` equals
   `λ_b(t)` exactly (modulo LP tolerance).

Every component (Generator, Link, StorageUnit) has a natural effective
cost of delivering (or withdrawing) 1 MWh at `b`; the goal is to find, for
every `(b, t)`, which one is both interior and matching.

---

## 2. Effective cost per candidate type

For each candidate we derive a formula for `c_eff_k(t)` — what it would
charge (supply) or pay (demand) to absorb 1 MWh of AC electricity at bus
`b` at snapshot `t`.

### 2.1 Generator at `b` (e.g. wind, solar, ror)

```
c_eff(t)  =  marginal_cost(t)        (EUR / MWh, typically ≈ 0 for renewables)
```

Read `n.generators_t.marginal_cost` if time-varying, else the scalar
`n.generators.marginal_cost`.

### 2.2 Supply Link (bus1 = b) — single-output, e.g. CCGT, OCGT, coal, H2 turbine

For a link with input at `bus0` (fuel), output at `bus1` (AC), and
efficiency `η`, the LP interior condition is

```
η · λ_AC(t)  =  λ_bus0(t)  +  c_link(t)
     →   c_eff_link(t)  =  (c_link(t) + λ_bus0(t)) / η(t)
```

`λ_bus0` is the shadow price of the fuel bus (e.g. gas, coal, H2). `η`
and `c_link` may be time-varying (`n.links_t.efficiency`,
`n.links_t.marginal_cost`).

### 2.3 CHP (and any link with co-outputs at bus2 / bus3)

PyPSA links can connect up to four buses. A CHP produces electricity at
`bus1` **and** heat at `bus2`, plus optionally CO₂ emissions to an
atmosphere bus at `bus3`. The full interior condition is

```
η₁·λ_bus1 + η₂·λ_bus2 + η₃·λ_bus3  =  λ_bus0 + c_link
     →   λ_AC  =  (c_link + λ_bus0 − η₂·λ_bus2 − η₃·λ_bus3) / η₁
```

Implemented in `supply_matrices()` via a `_bus_contribution(…)` helper that
pulls `η_i(t)·λ_{bus_i}(t)` for each co-output when `bus_i` is non-empty
and subtracts/adds it to the numerator. **Without** this correction
biomass and gas CHPs appear far more expensive than they truly are and
never get classified as marginal, even when they demonstrably set the
price.

### 2.4 Demand-flex Link (bus0 = b) — e.g. electrolyser, heat pump, resistive heater

Here the link **consumes** AC to produce something else (H₂, heat). Its
willingness-to-pay per MWh AC consumed is

```
     →   c_eff_demand(t)  =  η(t) · λ_bus1(t)  +  η₂·λ_bus2  +  η₃·λ_bus3  −  c_link(t)
```

An interior demand-flex link sets `λ_AC(t)` to its willingness-to-pay.

For the main use-case, LV-side flex (heat pumps, EV chargers on the
`"low voltage"` bus) are included, since distribution-grid efficiency
≈ 1 means LV ≈ AC in price. This is done transparently: any link with
`bus0 ∈ {bus, f"{bus} low voltage"}` is a candidate.

### 2.5 Storage charger / discharger Links

A battery is modelled in PyPSA-Eur as **three** components: a Store
(energy capacity), a charger Link (bus0=AC → bus1=battery), and a
discharger Link (bus0=battery → bus1=AC). Each Link is treated exactly
like a supply or demand Link above:

* `battery discharger`: supply cascade, `c_eff = (c + λ_battery) / η` —
  where `λ_battery` is the intertemporal shadow price of stored energy
  (the "water value" of the battery). When this link is interior, the
  AC price is set by the market value of stored electricity.
* `battery charger`: demand cascade, `c_eff = η · λ_battery − c`.
* `home battery discharger`, `home battery charger`, `BEV charger`,
  `V2G`: treated identically.

### 2.6 StorageUnit (hydro, PHS) — single component encapsulating both modes

A StorageUnit carries `p_dispatch(t) ≥ 0` (discharging) and
`p_store(t) ≥ 0` (charging) as two separate variables on one component.
At each snapshot we inspect both to decide mode:

* If `BOUND_TOL < p_dispatch < p_nom − BOUND_TOL`, the SU is interior on
  the discharge side → treat as a `storage_discharger`.
* If `BOUND_TOL < p_store < p_nom − BOUND_TOL`, the SU is interior on
  the charge side → treat as a `storage_charger`.
* If both are active at the same snapshot (typically across different
  SUs on the same bus), the discharger wins (priority matches the supply
  cascade which takes precedence over demand).

For SU snapshots the carrier passed downstream is `"hydro"` or `"PHS"`,
so they co-populate the same charger / discharger buckets that the
Store-+-Link pairs do.

---

## 3. Tolerance thresholds

All tests against bounds and prices use a small tolerance to absorb LP
numerical noise. From the script:

| Constant             | Value           | Meaning                                     |
|----------------------|-----------------|---------------------------------------------|
| `BOUND_TOL_REL`      | `1e-3`          | Relative slack on interior test             |
| `BOUND_TOL_ABS`      | `1e-2` MW       | Absolute floor on interior test             |
| `PRICE_TOL_ABS`      | `0.5` EUR/MWh   | Absolute slack on price match               |
| `PRICE_TOL_REL`      | `0.01`          | Relative (`·|λ|`) slack on price match      |
| `FLOW_CONGESTION_REL`| `0.99`          | Line-flow fraction considered "saturated"   |
| `MU_TOL`             | `1e-2` EUR/MWh  | Threshold on `μ` (unused without duals)     |
| `VOLL_THRESH`        | `3000` EUR/MWh  | Flag load-shedding slack generators         |

`interior = (p > p_nom·p_min_pu + tol) AND (p < p_nom·p_max_pu − tol)`  
`match = |c_eff − λ| ≤ max(PRICE_TOL_ABS, PRICE_TOL_REL·|λ|)`

---

## 4. The priority cascade

For every `(b, t)` the classifier asks, in order:

1. **Supply strict**: is any supply candidate both interior *and*
   matching? If yes, the winner with smallest `|c_eff − λ|` is the price
   setter. Class is `supply` if its role is "supply", `storage_discharger`
   if its role is "discharger".
2. **Demand-flex strict**: same test against demand candidates. Class is
   `demand_flex` or `storage_charger`.
3. **StorageUnit**: if no Link cascade matches, but an SU is interior on
   the discharge side → `storage_discharger`; on the charge side →
   `storage_charger`.
4. **Transmission** (excluded from the plot): if no local component is
   marginal *and* an AC line or DC link incident to `b` is ≥99 % loaded
   **in the direction flowing into `b`** *and* the other end's price is
   lower, price is set via import. This requires sign-direction awareness
   (`λ_neighbour + μ = λ_b`, with `μ ≥ 0`).
5. **Load shedding**: if a slack generator with `marginal_cost ≥
   VOLL_THRESH` is dispatching at `b` → `load_shedding`.
6. **Unresolved**: none of the above triggered.

### 4.1 Fuzzy fallback for unresolved snapshots

For `unresolved` rows we compute the best *non-interior / non-matching*
candidate and re-assign:

```
sup_nearest = argmin |c_eff_supply − λ|     (without requiring interior or match)
dem_nearest = argmin |c_eff_demand − λ|
winner      = supply-side if its nearest residual ≤ demand-side nearest
class       = class of the winner's cascade
```

In practice the LP's numerical noise often drives otherwise-valid
matches just outside the 0.5 EUR/MWh tolerance — the fuzzy step
recovers them. The fraction of still-unresolved snapshots is typically
< 5 %.

---

## 5. How supply is sub-bucketed for the plot

Once a `(b, t)` pair is classified as `supply`, the winning component's
`carrier` is mapped to a display bucket:

| Bucket                  | Carriers                                                         |
|-------------------------|------------------------------------------------------------------|
| wind & solar            | onwind, offwind-ac, offwind-dc, offwind-float, solar, solar-hsat |
| biomass CHP             | urban central solid biomass CHP (+ CC variant)                   |
| gas / waste CHP         | urban central gas CHP, waste CHP                                 |
| CCGT / OCGT             | CCGT, OCGT, OCGT methanol                                        |
| coal · lignite · oil    | coal, lignite, oil                                               |
| run-of-river hydro      | ror                                                              |
| other supply            | any carrier not matched above (e.g. nuclear, H2 turbine)         |

Unmapped carriers fall into `"other supply"`; `"nuclear"` and
`"H2 → electricity"` are deliberately not given their own bucket so
they don't visually clutter the legend at scenarios where they hardly
dispatch.

---

## 6. What this procedure does *not* capture

* **Network-wide AC sub-networks**: within a synchronous region whose
  inter-bus AC lines are uncongested, every AC bus sees the same λ.
  The price setter lives physically at *one* of them. The classifier
  only searches locally, so the other buses in that sub-network may
  report `unresolved` (with fuzzy fallback filling the gap most of the
  time). A more complete implementation would unify uncongested AC
  buses before classification.

* **Network duals not written**: the PyPSA run in this repo does not set
  `solver_options.assign_all_duals=true`, so `n.lines_t.mu_upper` and
  `n.links_t.mu_upper` are empty. Transmission congestion is therefore
  detected from line flow fraction, not from direct shadow prices.

* **Multi-way ties**: if two different candidates satisfy interior AND
  match equally well, the one with the smallest residual wins; ties are
  broken arbitrarily by `DataFrame.idxmin` behaviour.

* **Negative marginal prices**: the classifier works fine for negative
  λ; the interpretation (e.g. a renewable paying to dispatch rather
  than curtail) is preserved.

* **Non-AC buses**: the workflow is written around `carrier == "AC"`
  buses. Hydrogen, heat, methanol price-setting would need an analogous
  cascade on the relevant bus carrier.

---

## 7. File references

* Classifier module:
  `notebooks/scripts/price_formation/classify_price_setter.py`
* Vectorised sweep + plots:
  `notebooks/scripts/price_formation/plot_storage_price_vs_gas.py`
* Cache directory:
  `notebooks/scripts/price_formation/cache_storage_vs_gas_v5/`
* Network files consumed:
  `results/networks/base_s_50__3H-T-H-B-I-A-dist1_2030_free_{wiggle}.nc`
