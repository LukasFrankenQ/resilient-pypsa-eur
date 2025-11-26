rule plot_gas_resilience:
    input:
        expand(
            RESULTS
            + "networks/base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}_{tyndp_scenario}_{phaseout}.nc",
            **config["scenario"],
            run=config["run"]["name"],
        ),
    output:
        RESULTS + "resilient/gas_resilience.pdf",
    script:
        "../scripts/plot_gas_resilience.py"