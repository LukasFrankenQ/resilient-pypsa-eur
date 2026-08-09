
import logging
import pandas as pd
import numpy as np
import importlib
import pathlib
import pypsa

from scripts._benchmark import memory_logger
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


def _unfix_bottlenecks(new, deci, name, extendable_i, unconstrained=False):
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
        if unconstrained:
            # Unconstrained gas run: forbid biomass-boiler capacity expansion so
            # the model cannot build its way off gas (a gas-avoidance loophole).
            bottleneck_links = [
                c
                for c in bottleneck_links
                if c not in ("rural biomass boiler", "urban decentral biomass boiler")
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

def fix_capacities(n_lt, no_flex=False, unconstrained=False):
    n = n_lt.copy()


    for name, attr in nominal_attrs.items():
        new = getattr(n, name)
        lt = getattr(n_lt, name)

        extendable_i = new.query(f"{attr}_extendable").index

        new.loc[extendable_i, attr + "_extendable"] = False
        new.loc[extendable_i, attr] = new.loc[extendable_i, attr + "_opt"]

        _unfix_bottlenecks(new, lt, name, extendable_i, unconstrained=unconstrained)

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


def pin_installed_heating_dispatch(
    n,
    n_lt,
    carriers=("oil boiler", "biomass boiler"),
    tolerance=1e-2,
):
    """Lock oil and biomass boiler dispatch to the main-solve time series.

    Residential oil / biomass boilers are installed equipment — homeowners cannot
    switch heating source in response to a gas-price shock. We therefore pin their
    per-unit dispatch to the decision network's p0, within a small tolerance.

    Gas boilers are intentionally left flex (they remain bottleneck-extendable
    via `_unfix_bottlenecks`) so the hike solve has numerical headroom. Boilers
    are set p_nom_extendable=True with p_nom_min=p_nom_max=p_nom_opt so they
    bypass `force_boiler_profiles_existing_per_boiler` (which only targets
    non-extendable boilers) while still being capacity-locked.
    """
    mask = (
        n.links.carrier.str.contains("|".join(carriers))
        & ~n.links.carrier.str.contains("urban central")
    )
    names = n.links.index[mask]
    pinned = 0
    for name in names:
        if name not in n_lt.links_t.p0.columns:
            continue
        p_nom_opt = n_lt.links.at[name, "p_nom_opt"]
        if p_nom_opt <= 1e-3:
            continue
        profile = (n_lt.links_t.p0[name] / p_nom_opt).clip(0, 1)
        n.links.at[name, "p_nom"] = p_nom_opt
        n.links.at[name, "p_nom_min"] = p_nom_opt
        n.links.at[name, "p_nom_max"] = p_nom_opt
        n.links.at[name, "p_nom_extendable"] = True
        n.links_t.p_max_pu[name] = (profile + tolerance).clip(0, 1)
        n.links.at[name, "p_min_pu"] = 0.0
        n.links_t.p_min_pu[name] = (profile - tolerance).clip(lower=0.0)
        pinned += 1
    logger.info(
        f"Pinned dispatch of {pinned} installed heating boilers "
        f"(carriers={list(carriers)}) to main-solve time series."
    )


# Optional realism lever for the unconstrained ("endo") gas run: also strip
# perfect-foresight flexibility (decentral TES, BEV DSM, and 2030 batteries) so
# the optimiser cannot time-shift its way off gas. Off by default - it is a
# larger scenario change than the gas-boiler floor and biomass-expansion lock.
STRIP_FLEXIBILITY = False


def pin_installed_gas_boiler_floor(n, n_lt, tolerance=1e-2):
    """Floor installed gas-boiler dispatch at its main-solve level (unconstrained run).

    In an unconstrained gas run the fixed-gas-consumption equality is dropped, so
    gas dispatch responds to price. Buildings with an installed gas boiler cannot,
    however, swap to a heat pump in response to a price shock. To stop the model
    over-stating its freedom to shed gas, we floor each decentral gas boiler's
    per-unit dispatch at the decision network's profile (minus a small tolerance)
    while leaving headroom above; capacity is locked to ``p_nom_opt``.

    Urban central (district-heating) gas boilers are excluded - there the heat
    source genuinely is dispatchable. Mirrors ``pin_installed_heating_dispatch``
    but sets only the lower bound (``p_min_pu``), not a tight band.
    """
    mask = (
        n.links.carrier.str.contains("gas boiler")
        & ~n.links.carrier.str.contains("urban central")
    )
    names = n.links.index[mask]
    pinned = 0
    for name in names:
        if name not in n_lt.links_t.p0.columns:
            continue
        p_nom_opt = n_lt.links.at[name, "p_nom_opt"]
        if p_nom_opt <= 1e-3:
            continue
        profile = (n_lt.links_t.p0[name] / p_nom_opt).clip(0, 1)
        n.links.at[name, "p_nom"] = p_nom_opt
        n.links.at[name, "p_nom_min"] = p_nom_opt
        n.links.at[name, "p_nom_max"] = p_nom_opt
        n.links.at[name, "p_nom_extendable"] = True
        n.links.at[name, "p_min_pu"] = 0.0
        n.links_t.p_min_pu[name] = (profile - tolerance).clip(lower=0.0)
        pinned += 1
    logger.info(
        f"Floored dispatch of {pinned} installed gas boilers to main-solve profile "
        "(unconstrained gas run)."
    )


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


def add_capacity_market_batteries(n, gw, hours=4.0):
    """Inject out-of-market ("capacity market") utility-scale batteries.

    Adds ``gw`` GW of grid-delivered discharge power, distributed over AC
    buses proportional to mean electricity demand, under a dedicated
    carrier "battery CM" (own bus + charger/discharger links + cyclic
    ``hours``-duration store per node). All capacities are fixed
    (non-extendable) with zero capital cost: in the operation-only hike
    solve the asset exists regardless of energy-market revenues, i.e. it
    is "out of money" by construction.

    Sizing convention per node (P = allocated grid-side power in MW):
      * discharger p_nom = P / eff  -> delivers exactly P to the AC bus
      * charger    p_nom = P       -> symmetric grid connection
      * store      e_nom = hours * P / eff -> hours * P deliverable energy
    """
    elec_loads = n.loads.query("carrier == 'electricity'")
    profiles = n.loads_t.p_set.reindex(columns=elec_loads.index).dropna(
        axis=1, how="all"
    )
    weights = profiles.mean()
    weights /= weights.sum()
    # electricity-load names coincide with the AC bus names ("AL0 0" etc.)
    # NB: bus references and names are passed as plain lists throughout -
    # pypsa >= 1.0 label-aligns pd.Index/pd.Series attribute values against
    # the new component names, silently yielding empty bus references.
    nodes = list(weights.index)

    eff_s = n.links.loc[n.links.carrier == "battery charger", "efficiency"]
    eff = float(eff_s.iloc[0]) if len(eff_s) else 0.96

    p_grid = (gw * 1e3 * weights).to_numpy()  # MW delivered per node

    for carrier in ["battery CM", "battery CM charger", "battery CM discharger"]:
        if carrier not in n.carriers.index:
            n.add("Carrier", carrier, color="#b8ea04", nice_name=carrier)

    cm_buses = [f"{b} battery CM" for b in nodes]
    n.add(
        "Bus",
        cm_buses,
        carrier="battery CM",
        x=n.buses.loc[nodes, "x"].to_numpy(),
        y=n.buses.loc[nodes, "y"].to_numpy(),
        country=n.buses.loc[nodes, "country"].to_list(),
    )

    n.add(
        "Link",
        [f"{b} battery CM charger" for b in nodes],
        bus0=nodes,
        bus1=cm_buses,
        carrier="battery CM charger",
        efficiency=eff,
        p_nom=p_grid,
        p_nom_opt=p_grid,
        p_nom_extendable=False,
        capital_cost=0.0,
    )
    n.add(
        "Link",
        [f"{b} battery CM discharger" for b in nodes],
        bus0=cm_buses,
        bus1=nodes,
        carrier="battery CM discharger",
        efficiency=eff,
        p_nom=p_grid / eff,
        p_nom_opt=p_grid / eff,
        p_nom_extendable=False,
        capital_cost=0.0,
    )
    n.add(
        "Store",
        [f"{b} battery CM" for b in nodes],
        bus=cm_buses,
        carrier="battery CM",
        e_nom=hours * p_grid / eff,
        e_nom_opt=hours * p_grid / eff,
        e_nom_extendable=False,
        e_cyclic=True,
        capital_cost=0.0,
    )

    logger.info(
        f"Added {gw} GW / {gw * hours} GWh of out-of-market 'battery CM' "
        f"capacity across {len(nodes)} AC buses (load-proportional, "
        f"round-trip efficiency {eff**2:.3f})."
    )


logger = logging.getLogger(__name__)

if __name__ == "__main__":

    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake(
            "solve_gasprice_hike_network",
            opts="",
            clusters="50",
            configfiles="config.basicrun.yaml",
            sector_opts="168H-T-H-B-I-A-dist1",
            planning_horizons="2030",
            tyndp_scenario="free",
            wiggle=2000,
            hike="10u-cm10",
        )
    
    configure_logging(snakemake)  # pylint: disable=E0606
    set_scenario_config(snakemake)
    update_config_from_wildcards(snakemake.config, snakemake.wildcards)

    solve_opts = snakemake.params.options

    np.random.seed(solve_opts.get("seed", 123))

    n_lt = pypsa.Network(snakemake.input.network)

    # An unconstrained gas run drops the fixed-gas-consumption equality
    # (gas_consumption=None below) so gas dispatch responds to price. To stop the
    # model over-stating its freedom to shed gas, the unconstrained branch also
    # floors installed gas-boiler dispatch and forbids biomass-boiler expansion.
    #
    # Two ways to request it:
    #   * wiggle="endo"  - legacy: input is an endogenously-built planning network
    #                      (investment+operation without a gas budget).
    #   * hike="<n>u"    - operation-only on an EXISTING numeric-wiggle system:
    #                      the "u" suffix (e.g. "50u") keeps fix_capacities active
    #                      (capacities locked from ..._free_{wiggle}.nc) but drops
    #                      the gas equality, so only dispatch re-optimises.
    #   * hike="<n>u-cm<gw>" - like "<n>u", but additionally injects <gw> GW of
    #                      out-of-market ("capacity market") 4h batteries before
    #                      the operation-only solve (see
    #                      add_capacity_market_batteries), e.g. "10u-cm20".
    hike_str = str(snakemake.wildcards["hike"])
    cm_gw = 0.0
    if "-cm" in hike_str:
        hike_str, _cm = hike_str.split("-cm", 1)
        cm_gw = float(_cm)
    unconstrained = (str(snakemake.wildcards["wiggle"]) == "endo"
                     or hike_str.endswith("u"))
    if unconstrained:
        logger.info("Unconstrained gas run: dropping the fixed-gas-consumption "
                    "equality and applying gas-stickiness realism levers.")

    n = fix_capacities(n_lt, unconstrained=unconstrained)

    pin_installed_heating_dispatch(n, n_lt)

    if unconstrained:
        pin_installed_gas_boiler_floor(n, n_lt)
        if STRIP_FLEXIBILITY:
            remove_flexibility_options(n, int(snakemake.wildcards.planning_horizons))

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

    # After the small-capacity cleanup above, so the injected links cannot be
    # caught by the p_nom_opt < threshold reset.
    if cm_gw > 0:
        add_capacity_market_batteries(n, cm_gw)

    gasprice_markup = float(hike_str.rstrip("u"))
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
    
    gas_consumption = None if unconstrained else float(snakemake.wildcards['wiggle'])

    logging_frequency = snakemake.config.get("solving", {}).get(
        "mem_logging_frequency", 30
    )
    with memory_logger(
        filename=getattr(snakemake.log, "memory", None), interval=logging_frequency
    ) as mem:
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

    logger.info(f"Maximum memory usage: {mem.mem_usage}")

    n.meta = dict(snakemake.config, **dict(wildcards=dict(snakemake.wildcards)))
    n.export_to_netcdf(snakemake.output[0])
