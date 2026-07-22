# Building the paper figures

This file maps every figure of the gas-resilience paper (LaTeX repo:
`../gas_resilience`, figures included from `../gas_resilience/imgs/`) to the
script that generates it and the network files that must exist for the script
to run.

## General conventions

- Figure scripts live under `notebooks/scripts/<figure_name>/` (a few
  exceptions noted below). They are **standalone Python scripts** — no
  snakemake needed — and are run from the repo root or from their own
  directory:
  ```bash
  python notebooks/scripts/<name>/<script>.py
  ```
- Each script saves the PDF locally next to itself **and** copies it to
  `../gas_resilience/imgs/`. Many also export a JSON to `frontend_data/`.
- Tech colours come from `config.basicrun.yaml` or
  `config/plotting.default.yaml` depending on the script.

### Network file naming

Solved networks live in `results/networks/`. The main family is

```
results/networks/base_s_50_lv1.25_3H-T-H-B-I-A-dist1_2030_free_<wiggle>.nc
```

where `<wiggle>` is the permitted annual fossil-gas consumption in TWh,
swept in steps of 250 (0 … 4000, some scripts up to 4500/5000). Variants:

- **Gas-price multipliers**: `...-dist1-gas+Generator+m<f>_2030_free_<wiggle>.nc`
  with `f ∈ {1.25, 1.5, 1.75, 2.0}` (also m1.1–m1.9 for the tax notebook).
- **Gas-price hikes**: `..._2030_free_<wiggle>_<hike>u.nc` (hike in €/MWh, e.g. `1u`, `200u`).
- **Electrification-speed scenarios**: `..._2030_free+fast_<wiggle>.nc`
  (also `+medium`, `+slow`).
- **Tax/carbon-price sweep (168H)**:
  `base_s_50__168H-T-H-B-I-A-dist1<sector_opt><+carbonprice-XX>_2030_free_endo.nc`.

### Aggregated CSVs

Some figures do not read `.nc` files but the aggregated summary CSVs in
`results/csvs_3_latest/` (`energy_balance.csv`, `prices.csv`, `costs.csv`),
produced by the snakemake `make_summary`/postprocessing rules from the full
wiggle sweep.

## Figure → script → networks

### Main-results & cost figures

| Figure | Script | Networks / inputs |
|---|---|---|
| `main_results.pdf` | `notebooks/scripts/main_results/main_results.py` | `results/csvs_3_latest/{energy_balance,prices,costs}.csv` (no `.nc`) |
| `cost_vs_gas_consumption_annotated.pdf` | `notebooks/scripts/plot_cost_price_surfaces.py` | Wiggle sweep 0–4000 for base **and** the four `m1.25/m1.5/m1.75/m2.0` gas-price sweeps (only wiggles common to all five are used). Caches to `notebooks/scripts/.cache/cost_vs_gas_consumption.json` |
| `cost_vs_gas_consumption_v2.pdf` | `notebooks/scripts/plot_cost_price_surfaces_v2.py` | Base sweep only, wiggles 0–4500; price variants derived by re-pricing (shares v1's cache) |
| `optimal_installed_capacities_power_sector.pdf` | `notebooks/scripts/plot_capacities/plot_capacities.py` | Base sweep, wiggles 0–4000 step 250 |
| `battery_gas_tradeoff.pdf` | `notebooks/scripts/battery_gas_tradeoff/battery_gas_tradeoff.py` | Base sweep, wiggles 0–4000 step 250 |
| `abatement_curve*.pdf` (if used) | `notebooks/scripts/abatement_curve/abatement_curve.py` | Base sweep |

### Intro / trajectory / maps

| Figure | Script | Networks / inputs |
|---|---|---|
| `intro_plot.pdf` | `notebooks/scripts/intro_plot/intro_plot.py` | No networks. SciGRID gas GeoJSONs (`data/gas_network/scigrid-gas/...`), `data/Industrial_Database.csv`, Eurostat API (hard-coded fallback) |
| `phaseout_trajectory_script_v2.pdf` | `notebooks/scripts/trajectory/plot_phaseout_trajectory.py` | `..._free_2500.nc` (2030) and `..._free_2000.nc` (2035 autarky); IRENA CSV `data/ELEC-C_*.csv`, `data/eurostat_batteries.csv` |
| `marginal_price_maps.pdf` | `notebooks/scripts/marginal_price_maps/marginal_price_maps.py` | Single network `..._free_2000.nc`; `resources/regions_onshore_base_s_50.geojson` |
| `renewable_cf_maps.pdf` | `notebooks/scripts/renewable_cf_maps/renewable_cf_maps.py` | Single network `..._free_1000.nc`; onshore regions GeoJSON |

### Heat figures

| Figure | Script | Networks / inputs |
|---|---|---|
| `03_balance_timeseries_heat.pdf` | `notebooks/scripts/model_overview/model_overview.py` | One network, default `..._free_2000.nc` (network file can be passed as CLI arg). Writes into `imgs/model_overview/<network>/`, flat copy in `imgs/` |
| `existing_heating_capacities.pdf` | `notebooks/scripts/methods_heat_demand/plot_heat_demand.py` | `..._free_4000.nc`; `resources/existing_heating_distribution_base_s_50_2030.csv` |
| `heat_share_validation.pdf` | `notebooks/scripts/heat_validation/heat_share_validation.py` | `..._free_4000.nc`; existing-heating CSV (validates against EU fuel-share data) |
| `heat_capex_opex_lcohs.pdf` | `notebooks/scripts/heat_capex_opex_lcohs/heat_capex_opex_lcohs.py` | `..._free_4000.nc` (CLI-overridable); `../technology-data/outputs/costs_2030.csv` |
| `heat_lcohs_distribution.pdf` | `notebooks/scripts/heat_lcohs/plot_heat_lcohs.py` | Base sweep, wiggles 0–4000 step 250 |
| `heat_pump_vs_gas_boiler_lcoh.pdf` | `notebooks/scripts/heat_manifold/heat_pump_vs_gas_boiler_manifold.py` | `..._free_2000.nc` (price overlay only; LCOH analytic); `resources/costs_2030.csv`, `resources/*_literature_capex.csv` |
| `hp_lcoh_disaggregation.pdf` | `notebooks/scripts/hp_lcoh_disaggregation/hp_lcoh_disaggregation.py` | Wiggles {0, 2000, 4000} |
| `res-heat_rural_lcoh-technology-panels.pdf`, `res-heat_rural_majority-supply-bar.pdf` | `notebooks/scripts/res_heat_rural/res_heat_rural.py` (one run makes both) | Full base sweep, wiggles 0–5000 step 250 (heavy; exports frozen JSON for reuse) |

### Industry figures

| Figure | Script | Networks / inputs |
|---|---|---|
| `industry_heat_demand_by_country.pdf` | `plot_industry_heat_demand.py` (repo root) | Single network `..._free_1000.nc` |
| `industry_electrification_impact.pdf` | `notebooks/scripts/industry_electrification/plot_industry_electrification_impact.py` | `..._free_2000.nc` plus `free+fast/+medium/+slow` at wiggle 2000. Cached in `data/`; `--recompute` to rebuild from networks |
| `industry_gas_fills_hot_end.pdf` | `notebooks/scripts/industry_temp_bands_text/plot_gas_fills_hot_end.py` | No networks. Via `_compute.py`: `resources/industrial_production_per_country.csv`, `resources/industrial_energy_demand_per_country_today.csv`, Fleiter et al. Excel `data/ente202300981-sup-0001-suppdata-s1.xlsx` |
| `industry_gas_heat_by_band.pdf` | `notebooks/scripts/industry_temp_bands_text/plot_gas_heat_by_band.py` | Same inputs as above (no networks) |
| `industry_sectors_production.pdf` | `notebooks/scripts/industry_temp_bands_text/plot_industry_sectors_production.py` | `resources/industrial_production_per_country.csv` (no networks) |
| `industry_temperature_band_shares.pdf` | `notebooks/scripts/temperature_band_shares/plot_temperature_band_shares.py` | Fleiter et al. Excel only (no networks) |

### Price-formation & policy figures

| Figure | Script | Networks / inputs |
|---|---|---|
| `price_formation/price_setting_marginal_price_distribution_simple.pdf`, `price_formation/store_bus_marginal_price_distribution_simple.pdf` | `notebooks/scripts/price_formation/plot_storage_price_vs_gas_simple.py` (one run makes both) | Wiggles {1000, 1500, 1750, 2000, 2500, 3000, 4000}; caches per-wiggle parquet in `cache_low_voltage_bus_prices_v1/`; helpers `plot_storage_price_vs_gas.py`, `classify_price_setter.py` |
| `infra_marginal_rent.pdf` | `notebooks/scripts/infra_marginal_rent/plot_infra_marginal_rent.py` | Base sweep 0–4000 step 250 (skips missing wiggles gracefully) |
| `hike_cost_sensitivity.pdf` | `notebooks/scripts/hike_analysis/plot_hike_cost_sensitivity.py` | Hike networks `..._free_4000_{0,1,200}u.nc`; cached CSVs in `hike_analysis/data/`; `--recompute` to rebuild |
| `tax_policy_burden.pdf` | `notebooks/contourplot.ipynb` (**notebook**, the only one) | 168H sweep: `base_s_50__168H-T-H-B-I-A-dist1<m1.1…m2.0><+carbonprice-50…-120>_2030_free_endo.nc` (11 × 9 = 99 networks). Output lands in the notebook cwd; copy to `imgs/` manually |
| `cfd_shares_by_country_2030.pdf` | `notebooks/scripts/cfds/plot_cfd_shares_by_country.py` | No networks; hand-curated `notebooks/scripts/cfds/data/policy_shares_2030.csv` |

### Biomass figures

| Figure | Script | Networks / inputs |
|---|---|---|
| `biomass_supply_curve.pdf` | `notebooks/scripts/biomass_potential/biomass_supply_curve.py` | Single network `..._free_0.nc` |
| `total_biomass_balance_split.pdf` | `notebooks/scripts/total_biomass_balance_split/total_biomass_balance_split.py` | `results/csvs_3_latest/energy_balance.csv` (no `.nc`); shared `forest_palette.py` colours |

### Snakemake-produced figure

| Figure | Source | Notes |
|---|---|---|
| `balances-co2.pdf` | `scripts/plot_summary.py` (snakemake postprocessing, `plot_balances()`) | Reads the aggregated balances CSV (`results/csvs_*`) and writes `results/graphs_*/balances-<bus_carrier>.svg`; the paper PDF was converted from `balances-co2.svg` and copied to `imgs/` manually |

## Regenerating everything

1. Make sure the base wiggle sweep exists in `results/networks/`
   (`..._free_0.nc` … `..._free_4000.nc` in steps of 250; up to 4500/5000 for
   `plot_cost_price_surfaces_v2.py` and `res_heat_rural.py`), plus the variant
   families needed by the specific figure (gas-price multipliers, hikes,
   `free+fast/medium/slow`, 168H `endo` sweep).
2. Refresh the aggregated CSVs (`results/csvs_3_latest/`) via the snakemake
   summary rules if `main_results` or `total_biomass_balance_split` need
   updating.
3. Run the relevant script(s); each copies its PDF into
   `../gas_resilience/imgs/` itself.
4. Rebuild the paper in `../gas_resilience` (pdflatex + bibtex +
   makeglossaries + 2× pdflatex on `nice_paper.tex`).

Scripts with caches (`plot_cost_price_surfaces*`, `hike_analysis`,
`industry_electrification`, `price_formation`) are fast on re-run; pass
`--recompute` (where supported) or delete the cache when the underlying
networks changed.
