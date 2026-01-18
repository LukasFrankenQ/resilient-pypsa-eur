# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT


rule solve_sector_network:
    params:
        solving=config_provider("solving"),
        foresight=config_provider("foresight"),
        co2_sequestration_potential=config_provider(
            "sector", "co2_sequestration_potential", default=200
        ),
        custom_extra_functionality=input_custom_extra_functionality,
    input:
        network=resources(
            "networks/base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}_{tyndp_scenario}_{wiggle}.nc"
        ),
        gas_heating_progress_factors=resources(
            "gas_heating_progress_factors_base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}_{tyndp_scenario}_{wiggle}.csv"
        )
    output:
        network=RESULTS
        + "networks/base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}_{tyndp_scenario}_{wiggle}.nc",
        config=RESULTS
        + "configs/config.base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}_{tyndp_scenario}_{wiggle}.yaml",
    shadow:
        shadow_config
    log:
        solver=RESULTS
        + "logs/base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}_{tyndp_scenario}_{wiggle}_solver.log",
        memory=RESULTS
        + "logs/base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}_{tyndp_scenario}_{wiggle}_memory.log",
        python=RESULTS
        + "logs/base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}_{tyndp_scenario}_{wiggle}_python.log",
    threads: solver_threads
    resources:
        mem_mb=config_provider("solving", "mem_mb"),
        runtime=config_provider("solving", "runtime", default="6h"),
    benchmark:
        (
            RESULTS
            + "benchmarks/solve_sector_network/base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}_{tyndp_scenario}_{wiggle}"
        )
    conda:
        "../envs/environment.yaml"
    script:
        "../scripts/solve_network.py"


rule solve_gasprice_hike_network:
    params:
        options=config_provider("solving", "options"),
        solving=config_provider("solving"),
        foresight=config_provider("foresight"),
        co2_sequestration_potential=config_provider(
            "sector", "co2_sequestration_potential", default=200
        ),
        custom_extra_functionality=input_custom_extra_functionality,
    input:
        network=(
            RESULTS +
            "networks/base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}_{tyndp_scenario}_{wiggle}.nc"
        ),
    output:
        network=RESULTS
        + "networks/base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}_{tyndp_scenario}_{wiggle}_{hike}.nc",
        # config=RESULTS
        # + "configs/config.base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}_{tyndp_scenario}_{wiggle}_{hike}.yaml",
    log:
        solver=RESULTS
        + "logs/base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}_{tyndp_scenario}_{wiggle}_{hike}_solver.log",
        memory=RESULTS
        + "logs/base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}_{tyndp_scenario}_{wiggle}_{hike}_memory.log",
        python=RESULTS
        + "logs/base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}_{tyndp_scenario}_{wiggle}_{hike}_python.log",
    threads: solver_threads
    resources:
        mem_mb=config_provider("solving", "mem_mb"),
        runtime=config_provider("solving", "runtime", default="6h"),
    benchmark:
        (
            RESULTS
            + "benchmarks/solve_gasprice_hike_network/base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}_{tyndp_scenario}_{wiggle}_{hike}"
        )
    conda:
        "../envs/environment.yaml"
    script:
        "../scripts/solve_gasprice_hike_network.py"