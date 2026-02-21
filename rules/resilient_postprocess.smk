'''
rule plot_gas_resilience:
    input:
        expand(
            RESULTS
            + "networks/base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}_{tyndp_scenario}_{wiggle}.nc",
            **config["scenario"],
            run=config["run"]["name"],
        ) + expand(
            RESULTS
            + "networks/base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}_{tyndp_scenario}_{wiggle}_50.nc",
            **config["scenario"],
            run=config["run"]["name"],
        ),
    output:
        RESULTS + "resilient/gas_resilience.pdf",
    script:
        "../scripts/plot_gas_resilience.py"


# rule plot_validation_power_sector:
#     pass
    

'''