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
        # gas_heating_progress_factors=resources(
        #     "gas_heating_progress_factors_base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}_{tyndp_scenario}_{wiggle}.csv"
        # )
    output:
        network=RESULTS
        + "networks/base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}_{tyndp_scenario}_{wiggle}.nc",
        config=RESULTS
        + "configs/config.base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}_{tyndp_scenario}_{wiggle}.yaml",
        upper_mu=RESULTS
        + "duals/base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}_{tyndp_scenario}_{wiggle}_upper_mu.csv",
        lower_mu=RESULTS
        + "duals/base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}_{tyndp_scenario}_{wiggle}_lower_mu.csv",
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


def gasprice_hike_input(w):
    stem = (
        RESULTS
        + "networks/base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}_{tyndp_scenario}_{wiggle}"
    ).format(**w)
    inputs = {"network": stem + ".nc"}
    hike = str(w.hike)
    # premium run: needs the brownfield/queue-free solve whose fleet it freezes in
    if hike.endswith("qp"):
        inputs["brownfield_network"] = stem + f"_{hike[:-2]}q.nc"
    elif hike.endswith("p"):
        inputs["brownfield_network"] = stem + f"_{hike[:-1]}b.nc"
    return inputs


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
        unpack(gasprice_hike_input),
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


localrules:
    solve_hike_networks,


rule solve_hike_networks:
    input:
        expand(
            RESULTS
            + "networks/base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}_{tyndp_scenario}_{wiggle}_{hike}.nc",
            **config["scenario"],
            run=config["run"]["name"],
        ),