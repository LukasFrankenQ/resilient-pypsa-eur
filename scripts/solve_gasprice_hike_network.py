
import logging
import pandas as pd
import numpy as np
import importlib
import pathlib
import pypsa

from scripts._helpers import (
    configure_logging,
    set_scenario_config,
    update_config_from_wildcards,
)
from scripts.solve_network import prepare_network, solve_network


def remove_flexibility_options(n, current_year):
    logger.info("Removing decentral TES and BEV DSM from the network.")
    n.remove("Store", n.stores.query("carrier == 'EV battery'").index)
    carriers_to_drop = [
        "urban decentral water tanks charger",
        "urban decentral water tanks discharger",
        "urban decentral water tanks",
        "rural water tanks charger",
        "rural water tanks discharger",
        "rural water tanks",
    ]
    n.remove("Link", n.links.query(f"carrier in {carriers_to_drop}").index)
    n.remove("Store", n.stores.query(f"carrier in {carriers_to_drop}").index)
    n.remove("Bus", n.buses.query(f"carrier in {carriers_to_drop}").index)

    if current_year == 2030:
        logger.info("Removing decentral TES and batteries from the network.")
        carriers_to_drop = [
            "home battery charger",
            "home battery discharger",
            "home battery",
            "battery charger",
            "battery discharger",
            "battery",
        ]
        n.remove(
            "Link",
            n.links.query(
                f"carrier in {carriers_to_drop} and build_year == {current_year}"
            ).index,
        )
        n.remove(
            "Store",
            n.stores.query(
                f"carrier in {carriers_to_drop} and build_year == {current_year}"
            ).index,
        )


def _unfix_bottlenecks(new, deci, name, extendable_i):
    if name == "links":
        # Links that have 0-cost and are extendable
        virtual_links = [
            "land transport oil",
            "land transport fuel cell",
            "solid biomass for industry",
            "gas for industry",
            "industry methanol",
            "naphtha for industry",
            "process emissions",
            "coal for industry",
            "H2 for industry",
            "shipping methanol",
            "shipping oil",
            "kerosene for aviation",
            "agriculture machinery oil",
            "co2 sequestered",
        ]

        _idx = new.loc[new.carrier.isin(virtual_links)].index.intersection(extendable_i)
        new.loc[_idx, "p_nom_extendable"] = True

        # Bottleneck links can be extended, but not reduced to fix infeasibilities due to numerical inconsistencies
        bottleneck_links = [
            "electricity distribution grid",
            "HVC to air",  # waste CHP would get used as a flexible energy source otherwise
            "SMR",
            # Boilers create bottlenecks AND should be extendable for fixed_profile_scaling constraints to be applied correctly
            "rural gas boiler",
            "urban decentral gas boiler",
            # Biomass for 2035 when gas is banned
            "rural biomass boiler",
            "urban decentral biomass boiler",
        ]
        _idx = new.loc[new.carrier.isin(bottleneck_links)].index.intersection(
            extendable_i
        )
        new.loc[_idx, "p_nom_extendable"] = True
        new.loc[_idx, "p_nom_min"] = deci.loc[_idx, "p_nom_opt"]
        # OCGT as last resort to avoid load shedding
        # allowed only in DE
        # (previously the model sometimes expanded waste CHPs)
        _idx = new.loc[
            (new.carrier == "OCGT") & (new.index.str.startswith("DE"))
        ].index.intersection(extendable_i)
        new.loc[_idx, "p_nom_extendable"] = True
        new.loc[_idx, "p_nom_min"] = deci.loc[_idx, "p_nom_opt"]

    if name == "generators":
        fuels = [
            "lignite",
            "coal",
            "oil primary",
            "uranium",
            "gas primary",
        ]
        vents = [
            "urban central heat vent",
            "rural heat vent",
            "urban decentral heat vent",
        ]
        _idx = new.loc[new.carrier.isin(fuels + vents)].index.intersection(extendable_i)
        new.loc[_idx, "p_nom_extendable"] = True

    return

'''
def _load_attr_from_file(filename: str, attr_name: str) -> object:
    """
    Load attribute attr_name from a local python file given by filename (including '.py').
    """
    if not filename.endswith(".py"):
        raise ValueError("filename must include the '.py' extension")
    module_stem = pathlib.Path(filename).stem
    _spec_path = pathlib.Path(__file__).resolve().parent / filename
    _spec = importlib.util.spec_from_file_location(
        f"scripts.pypsa_de.{module_stem}", _spec_path
    )
    _mod = importlib.util.module_from_spec(_spec)
    assert _spec is not None and _spec.loader is not None
    _spec.loader.exec_module(_mod)
    return getattr(_mod, attr_name)


_unfix_bottlenecks = _load_attr_from_file(
    "prepare_regret_network.py", "_unfix_bottlenecks"
)
remove_flexibility_options = _load_attr_from_file(
    "modify_prenetwork.py", "remove_flexibility_options"
)
'''

nominal_attrs = {
    "generators": "p_nom",
    "lines": "s_nom",
    "links": "p_nom",
    "stores": "e_nom",
    }

def fix_capacities(n_lt, no_flex=False):
    n = n_lt.copy()


    for name, attr in nominal_attrs.items():
        new = getattr(n, name)
        lt = getattr(n_lt, name)

        extendable_i = new.query(f"{attr}_extendable").index

        new.loc[extendable_i, attr + "_extendable"] = False
        new.loc[extendable_i, attr] = new.loc[extendable_i, attr + "_opt"]

        _unfix_bottlenecks(new, lt, name, extendable_i)

        # The CO2 constraints on atmosphere and sequestration need extendable stores to work correctly
        if name == "stores":
            logger.info("Freeing co2 atmosphere and sequestered stores.")
            # there is only one co2 atmosphere store which should always be extendable, hence no intersection with extendable_i needed
            _idx = new.query("carrier == 'co2'").index
            new.loc[_idx, "e_nom_extendable"] = True
            # co2 sequestered stores from previous planning horizons should not be extendable
            _idx = new.query("carrier == 'co2 sequestered'").index.intersection(
                extendable_i
            )
            new.loc[_idx, "e_nom_extendable"] = True

        # Above several assets are switched to extendable again, for these the p_nom value is restored to the value from the decision network

        _idx = new.query(f"{attr}_extendable").index

        new.loc[_idx, attr] = lt.loc[_idx, attr]

    if no_flex:
        logger.info("Realization network is from a run without flexibility.")
        remove_flexibility_options(n)
    return n


def add_load_shedding(
    n: pypsa.Network,
    marginal_cost: float=10000,
) -> None:
    """
    Adds load shedding to the network.
    """
    n.add("Carrier", "load", color="#dd2e23", nice_name="Load Shedding")
    buses_i = pd.Index(n.loads.bus.unique())

    logger.info(f"Adding load shedding to buses with carriers {n.buses.carrier[buses_i].unique()}.")
    logger.info(f"Load shedding marginal cost: {marginal_cost} EUR/MWh.")
    n.add(
        "Generator",
        buses_i,
        " load",
        bus=buses_i,
        carrier="load",
        marginal_cost=marginal_cost,
        p_nom_extendable=True,
    )    

    n.add(
        "Generator",
        buses_i,
        " load negative",
        bus=buses_i,
        carrier="load",
        marginal_cost=-marginal_cost,
        p_nom_extendable=True,
        p_min_pu=-1,
        p_max_pu=0,
    )    


logger = logging.getLogger(__name__)

if __name__ == "__main__":

    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake(
            "solve_sector_network",
            opts="",
            clusters="50",
            configfiles="config.basicrun.yaml",
            sector_opts="168H-T-H-B-I-A-dist1",
            planning_horizons="2030",
            tyndp_scenario="NT",
            wiggle=3000,
            hike=50,
        )
    
    configure_logging(snakemake)  # pylint: disable=E0606
    set_scenario_config(snakemake)
    update_config_from_wildcards(snakemake.config, snakemake.wildcards)

    solve_opts = snakemake.params.options

    np.random.seed(solve_opts.get("seed", 123))

    n_lt = pypsa.Network(snakemake.input.network)

    n = fix_capacities(n_lt)

    carrier = 'heat200-500 industry solid biomass'

    '''
    print('before')
    print(n.links.loc[
        n.links.carrier == carrier,
        ['p_nom', 'p_nom_opt', 'p_nom_extendable', 'p_set', 'p_nom_min', 'p_nom_max', 'capital_cost']
        ]
        )
    '''

    for c in ['Link']:
        df = n.components[c].static
        mask = (~df.p_nom_extendable) & (df.p_set == 0.) & (df.p_nom != 0.)
        df.loc[mask, 'p_set'] = np.nan

        threshold = 0.1
        mask = df.p_nom_opt < threshold
        df.loc[mask, 'p_nom_opt'] = 0.
        df.loc[mask, 'p_nom'] = 0.
        df.loc[mask, 'p_nom_extendable'] = True
        df.loc[mask, 'p_nom_min'] = 0.
        df.loc[mask, 'p_nom_max'] = threshold

    '''
    print('after')
    print(n.links.loc[
        n.links.carrier == carrier,
        ['p_nom', 'p_nom_opt', 'p_nom_extendable', 'p_set', 'p_nom_min', 'p_nom_max']
        ]
        )
    '''

    # n.optimize.fix_optimal_capacities()
    # n.optimize.add_load_shedding()
    # fix_all_optimal_capacities(n)
    # set_minimum_investment(n, snakemake.wildcards.planning_horizons)
    # add_load_shedding(n)

    gasprice_markup = float(snakemake.wildcards['hike'])
    # logger.info(f"Applying gas price markup of {gasprice_markup}")
    mask = n.generators.carrier == 'gas'
    n.generators.loc[mask, 'marginal_cost'] += gasprice_markup

    prepare_network(
        n,
        solve_opts,
        foresight=snakemake.params.foresight,
        planning_horizons=snakemake.wildcards.planning_horizons,
        co2_sequestration_potential=snakemake.params.co2_sequestration_potential,
        )
    
    gas_consumption = float(snakemake.wildcards['wiggle'])

    solve_network(
        n,
        config=snakemake.config,
        params=snakemake.params,
        solving=snakemake.params.solving,
        log_fn=snakemake.log.solver,
        rule_name=snakemake.rule,
        hike_run=True,
        gas_consumption=gas_consumption,
    )

    n.meta = dict(snakemake.config, **dict(wildcards=dict(snakemake.wildcards)))
    n.export_to_netcdf(snakemake.output[0])
