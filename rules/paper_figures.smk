# Rules to (re)generate every figure that ends up in
# `../gas_resilience/nice_paper.tex` (and its `sections/*.tex`).
#
# Each rule:
#   1. runs the producing Python script (which writes the figure(s) into
#      `../gas_resilience/imgs/` and a JSON payload into `frontend_data/`),
#   2. copies the figure(s) into `results/paper_figures/` so there is a
#      run-tracked, repo-local copy as well.
#
# The scripts are not snakemake-aware (they hard-code their save paths and
# just need CWD = repo root, which snakemake provides). For that reason we
# do NOT declare CSV/network inputs — these rules are intended to be invoked
# manually with `-F paper_figures` (or a single `plot_paper_<key>`) after a
# new network run.

PAPER_FIGS_DIR = "results/paper_figures"
GAS_RES_IMGS = "../gas_resilience/imgs"

# key -> (script path relative to repo root,
#         [(source filename in gas_resilience/imgs, dest filename in results/paper_figures)],
#         frontend_data json stem or None)
PAPER_FIGURES = {
    "intro_plot": (
        "notebooks/scripts/intro_plot/intro_plot.py",
        [("intro_plot.pdf", "intro_plot.pdf")],
        "intro_plot",
    ),
    "main_results": (
        "notebooks/scripts/main_results/main_results.py",
        [("main_results.pdf", "main_results.pdf")],
        "main_results",
    ),
    "phaseout_trajectory": (
        "notebooks/scripts/trajectory/plot_phaseout_trajectory.py",
        # script writes phaseout_trajectory_script.pdf; nice_paper.tex includes phaseout_trajectory.pdf
        [("phaseout_trajectory_script.pdf", "phaseout_trajectory.pdf")],
        "phaseout_trajectory",
    ),
    "industry_electrification_impact": (
        "notebooks/scripts/industry_electrification/plot_industry_electrification_impact.py",
        [("industry_electrification_impact.pdf", "industry_electrification_impact.pdf")],
        "industry_electrification_impact",
    ),
    "industry_heat_demand_by_country": (
        # this script lives at the repo root, not under notebooks/scripts/
        "plot_industry_heat_demand.py",
        [("industry_heat_demand_by_country.pdf", "industry_heat_demand_by_country.pdf")],
        "industry_heat_demand_by_country",
    ),
    "battery_gas_tradeoff": (
        "notebooks/scripts/battery_gas_tradeoff/battery_gas_tradeoff.py",
        [("battery_gas_tradeoff.pdf", "battery_gas_tradeoff.pdf")],
        "battery_gas_tradeoff",
    ),
    "total_biomass_balance_split": (
        "notebooks/scripts/total_biomass_balance_split/total_biomass_balance_split.py",
        [("total_biomass_balance_split.pdf", "total_biomass_balance_split.pdf")],
        "total_biomass_balance_split",
    ),
    "biomass_supply_curve": (
        "notebooks/scripts/biomass_potential/biomass_supply_curve.py",
        [("biomass_supply_curve.pdf", "biomass_supply_curve.pdf")],
        "biomass_supply_curve",
    ),
    "price_classification_explainer": (
        "notebooks/scripts/price_formation/plot_price_classification_explainer.py",
        [("price_formation/price_classification_explainer_free.pdf",
          "price_classification_explainer_free.pdf")],
        "price_classification_explainer_free",
    ),
    "infra_marginal_rent": (
        "notebooks/scripts/infra_marginal_rent/plot_infra_marginal_rent.py",
        [("infra_marginal_rent.pdf", "infra_marginal_rent.pdf")],
        "infra_marginal_rent",
    ),
    "cfd_shares_by_country": (
        "notebooks/scripts/cfds/plot_cfd_shares_by_country.py",
        [("cfd_shares_by_country_2030.pdf", "cfd_shares_by_country_2030.pdf")],
        "cfd_shares_by_country_2030",
    ),
    "res_heat_rural": (
        "notebooks/scripts/res_heat_rural/res_heat_rural.py",
        [("res-heat_rural_majority-supply-bar.pdf", "res-heat_rural_majority-supply-bar.pdf"),
         ("res-heat_rural_lcoh-technology-panels.pdf", "res-heat_rural_lcoh-technology-panels.pdf")],
        # script writes one JSON per panel; track the bar variant as the canonical one
        "res-heat_rural_majority-supply-bar",
    ),
    "cost_vs_gas_consumption": (
        "notebooks/scripts/plot_cost_price_surfaces.py",
        [("cost_vs_gas_consumption.pdf", "cost_vs_gas_consumption.pdf")],
        "cost_vs_gas_consumption",
    ),
    "price_setting_marginal_dist": (
        "notebooks/scripts/price_formation/plot_storage_price_vs_gas.py",
        [("price_formation/price_setting_marginal_price_distribution.pdf",
          "price_setting_marginal_price_distribution.pdf")],
        "price_setting_marginal_price_distribution",
    ),
    "balance_timeseries_heat": (
        # model_overview writes 03_balance_timeseries_heat.pdf into a per-network
        # subdir (gas_resilience/imgs/model_overview/<stem>/...); the shell picks
        # the most recently modified one.
        "notebooks/scripts/model_overview/model_overview.py",
        [("model_overview/*/03_balance_timeseries_heat.pdf", "03_balance_timeseries_heat.pdf")],
        "03_balance_timeseries_heat",
    ),
    "existing_heating_capacities": (
        "notebooks/scripts/methods_heat_demand/plot_heat_demand.py",
        [
            ("existing_heating_capacities.pdf", "existing_heating_capacities.pdf"),
            ("existing_heating_capacities_vs_model.pdf",
             "existing_heating_capacities_vs_model.pdf"),
        ],
        "existing_heating_capacities",
    ),
    "energy_prices_wholesale": (
        "notebooks/scripts/gas_electricity_price_ratio.py",
        [("energy_prices_wholesale.pdf", "energy_prices_wholesale.pdf")],
        "energy_prices_wholesale",
    ),
    # NOTE: tax_policy_burden.pdf (used by sections/gas_tax.tex) has no .py
    # producer in notebooks/scripts/ — only a notebook output. Wire it up here
    # once a script exists.
}


def _shell_for(script, figs):
    lines = [f"python {script}", f"mkdir -p {PAPER_FIGS_DIR}"]
    for src, dst in figs:
        src_path = f"{GAS_RES_IMGS}/{src}"
        dst_path = f"{PAPER_FIGS_DIR}/{dst}"
        # Also drop a renamed copy at gas_resilience/imgs/<dst> so nice_paper.tex
        # picks up the fresh version (its \includegraphics uses the dst name).
        paper_path = f"{GAS_RES_IMGS}/{dst}"
        if "*" in src:
            # glob-style: pick the most recently modified match
            pick = f"$(ls -t {src_path} | head -n1)"
            lines.append(f"cp \"{pick}\" {dst_path}")
            if src != dst:  # file lives in a subdir; also drop a flat copy
                lines.append(f"cp \"{pick}\" {paper_path}")
        else:
            lines.append(f"cp {src_path} {dst_path}")
            if src != dst:
                lines.append(f"cp {src_path} {paper_path}")
    return "\n".join(lines)


def _make_paper_rule(key, script, figs, json_stem):
    # Bind values via function arguments so each rule's `shell` resolves to
    # its OWN script — Snakemake defers evaluation of the rule body, so a
    # bare `for` loop would leak the last loop variable into all rules.
    outputs = {f"pdf{i}": f"{PAPER_FIGS_DIR}/{dst}" for i, (_, dst) in enumerate(figs)}
    if json_stem is not None:
        outputs["json"] = f"frontend_data/{json_stem}.json"
    cmd = _shell_for(script, figs)

    rule:
        name:
            f"plot_paper_{key}"
        output:
            **outputs,
        shell:
            cmd


for _key, (_script, _figs, _json_stem) in PAPER_FIGURES.items():
    _make_paper_rule(_key, _script, _figs, _json_stem)


rule paper_figures:
    input:
        [f"{PAPER_FIGS_DIR}/{dst}"
         for (_script, _figs, _json) in PAPER_FIGURES.values()
         for (_src, dst) in _figs],


localrules:
    paper_figures,
    plot_paper_intro_plot,
    plot_paper_main_results,
    plot_paper_phaseout_trajectory,
    plot_paper_industry_electrification_impact,
    plot_paper_industry_heat_demand_by_country,
    plot_paper_battery_gas_tradeoff,
    plot_paper_total_biomass_balance_split,
    plot_paper_biomass_supply_curve,
    plot_paper_price_classification_explainer,
    plot_paper_infra_marginal_rent,
    plot_paper_cfd_shares_by_country,
    plot_paper_res_heat_rural,
    plot_paper_cost_vs_gas_consumption,
    plot_paper_price_setting_marginal_dist,
    plot_paper_balance_timeseries_heat,
    plot_paper_existing_heating_capacities,
    plot_paper_energy_prices_wholesale,
