"""Validate the model's heating fuel mix against real-world data.

For the network ``base_s_50_lv1.25_3H-T-H-B-I-A-dist1_2030_free_4000`` this
script computes the share of heat supplied by four fuel categories
— ELECTRIC (heat pumps + resistive heating), OIL, BIOMASS and GAS — across the
three heat segments the model resolves:

    * urban central     -> district heating
    * urban decentral   -> individual heating in urban areas
    * rural             -> individual heating in rural areas

It reports both absolute totals and percentages, and then runs one *direct
comparison per data source*. Each source carries a ``description`` that gives
the context needed to read the comparison correctly: what the dataset measures,
who publishes it, its geographic and temporal scope, and the main caveats when
holding it against a 2030 model.

Heat-pump accounting (the ambient-heat convention)
--------------------------------------------------
A heat pump delivers heat = electricity drawn + ambient heat upgraded. How that
delivered heat is booked differs between datasets, so the model must follow
*each source's* convention for the comparison to be fair. Every source therefore
declares ``hp_basis``:

    * ``"delivered"``    — the whole heat output counts as ELECTRIC.
      Used where the source is on a delivered/rated-output basis (the
      existing-heating capacities are rated thermal output; the UK census and
      the Norway profile treat heat pumps as electric heating).
    * ``"final_energy"`` — only the electricity the heat pump *consumes* counts
      as ELECTRIC; the ambient heat it upgrades is reported separately in its own
      'ambient' column (and split out of the source's renewables row too). This
      matches how Eurostat and ODYSSEE book heat pumps (electricity in the
      electricity row, ambient heat in the renewables row), so the model is put
      on the same basis before comparing. On these panels the electric bar is
      further split into heat-pump electricity input vs. resistive heating.

The model's heat-pump electricity input is computed exactly from the link flows
(``links_t.p0``); ambient = delivered − electricity input.

Scope discipline
----------------
Every comparison is scoped on the model side to match the source:

    * Eurostat / ODYSSEE 2023 figures are EU-27       -> model filtered to EU-27.
    * UK and Norway sources                            -> model filtered to GB / NO.
    * The existing-heating (EC Mapping study) input    -> full model scope
      (the study itself covers EU + UK + Norway + Switzerland).

Comparing a 2030 model against ~2012-2024 actuals is intentional: the actuals
are validation anchors for the starting point and direction of travel, not
2030 end-points. A credible 2030 trajectory sits *above* the actuals on
electric/biomass and *below* on gas/oil.

Outputs
-------
    * A printed report (this is the primary deliverable).
    * ``heat_share_validation.csv`` next to this script — every comparison,
      long format, model vs. source, totals and percentages, plus the
      heat-pump / resistive / ambient breakdown.
    * ``heat_share_validation.pdf`` (one grouped-bar panel per comparison;
      panels that distinguish heat pumps split the electric bar into heat pump
      vs. resistive), saved next to this script and to ``../gas_resilience/imgs``.
"""
from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pypsa
import yaml

# ── Paths ────────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = Path(__file__).resolve().parent
NETWORK_DIR = REPO_ROOT / "results" / "networks"
SIBLING_IMGS = REPO_ROOT.parent / "gas_resilience" / "imgs"
SIBLING_IMGS.mkdir(parents=True, exist_ok=True)

PREFIX = "base_s_50_lv1.25_3H-T-H-B-I-A-dist1_2030"


def network_path(scenario: str = "free", wiggle: int = 4000, hike: int | None = None) -> Path:
    """Construct a network path (see claude_instructions/OPEN_MODEL.md)."""
    hike_str = f"_{hike}" if hike is not None else ""
    return NETWORK_DIR / f"{PREFIX}_{scenario}_{wiggle}{hike_str}.nc"


NETWORK = network_path(scenario="free", wiggle=4000)
EXISTING_HEATING_CSV = REPO_ROOT / "resources" / "existing_heating_distribution_base_s_50_2030.csv"

# Four fuel categories the whole analysis is expressed in.
FUELS = ["electric", "gas", "oil", "biomass"]

# Fine categories carried internally so the electric bar can be split into
# heat-pump vs. resistive, and so heat-pump ambient heat can be re-allocated
# depending on hp_basis. They aggregate to FUELS via FINE_TO_FUEL below.
FINE = ["hp", "resistive", "gas", "oil", "biomass", "ambient", "other"]

# Display categories -> the FINE keys that feed them. Most comparisons use the
# four FUELS (which exclude the model's 'other' bucket). The district-heating
# comparison uses an extended set so coal and waste can be shown explicitly even
# though the model supplies zero (coal is assumed phased out by 2030); there,
# 'waste/other' captures the model's 'other' bucket.
CATEGORY_FINE = {
    "electric": ("hp", "resistive"),     # heat-pump electricity input + resistive
    "gas": ("gas",),
    "oil": ("oil",),
    "biomass": ("biomass",),             # solid biomass only (boilers + CHP)
    "ambient": ("ambient",),             # heat-pump ambient heat (own column)
    "coal": (),                          # no coal heating carrier in the model
    "waste/other": ("other",),           # solar thermal / waste CHP / industrial waste heat
}
# Comparisons that split heat pumps (final-energy basis) show 'ambient' as its
# own column right after 'biomass', so true biomass and heat-pump ambient heat
# are no longer conflated on either side.
FE_CATEGORIES = ["electric", "gas", "oil", "biomass", "ambient"]
DH_CATEGORIES = ["electric", "gas", "oil", "biomass", "ambient", "coal", "waste/other"]

# EU-27 ISO-2 codes (model lacks CY and MT, which drop out of the intersection).
EU27 = {
    "AT", "BE", "BG", "HR", "CY", "CZ", "DK", "EE", "FI", "FR", "DE", "GR",
    "HU", "IE", "IT", "LV", "LT", "LU", "MT", "NL", "PL", "PT", "RO", "SK",
    "SI", "ES", "SE",
}

HEAT_SEGMENTS = {
    "urban central": "urban central heat",
    "urban decentral": "urban decentral heat",
    "rural": "rural heat",
}


# ── Fuel mapping ─────────────────────────────────────────────────────────────
def fine_of(carrier: str) -> str:
    """Map a model heat-supply carrier to a FINE category.

    Order matters: 'boiler' contains the substring 'oil', so heat-pump /
    resistive and the explicit 'oil boiler' are matched before the generic
    'gas'/'biomass' keywords. 'other' captures solar thermal, waste CHP and
    industrial waste heat (electrolysis, Haber-Bosch, Sabatier, methanol).
    """
    if "heat pump" in carrier:
        return "hp"
    if "resistive heater" in carrier:
        return "resistive"
    if "oil boiler" in carrier:
        return "oil"
    if "biomass" in carrier:  # biomass boiler, solid biomass CHP (+ CC)
        return "biomass"
    if "gas" in carrier:  # gas boiler, gas CHP
        return "gas"
    return "other"


# Technologies in the existing-heating input table -> FINE category.
EXISTING_TECH_FINE = {
    "gas boiler": "gas",
    "oil boiler": "oil",
    "biomass boiler": "biomass",
    "resistive heater": "resistive",
    "air heat pump": "hp",
    "ground heat pump": "hp",
}


# ── Model-side extraction ────────────────────────────────────────────────────
def hp_electricity_input(n: pypsa.Network, segments: list[str], countries: set[str] | None) -> float:
    """Electricity (MWh) drawn by heat pumps delivering to the given segments
    and countries — the exact electricity input behind their heat output."""
    heat_buses = n.buses.index[n.buses.carrier.isin([HEAT_SEGMENTS[s] for s in segments])]
    hp = n.links[n.links.carrier.str.contains("heat pump") & n.links.bus1.isin(heat_buses)]
    if countries is not None:
        hp = hp[hp.bus1.str[:2].isin(countries)]
    if hp.empty:
        return 0.0
    w = n.snapshot_weightings.generators
    return float(n.links_t.p0[hp.index].mul(w, axis=0).sum().sum())


def model_supply_fine(n: pypsa.Network, segments: list[str], countries: set[str] | None,
                      hp_basis: str) -> pd.Series:
    """Heat *supplied* (MWh) by FINE category, summed over the given heat
    segments and country subset. Only positive flows into the heat bus count
    as supply. ``hp_basis`` controls heat-pump booking (see module docstring).
    """
    fine = pd.Series(0.0, index=FINE)
    for seg in segments:
        eb = n.statistics.energy_balance(
            bus_carrier=HEAT_SEGMENTS[seg], groupby=["bus", "carrier"]
        ).reset_index()
        eb.columns = ["component", "bus", "carrier", "value"]
        eb["ct"] = eb.bus.str[:2]
        if countries is not None:
            eb = eb[eb.ct.isin(countries)]
        eb = eb[eb.value > 0]
        eb["fine"] = eb.carrier.map(fine_of)
        fine = fine.add(eb.groupby("fine").value.sum(), fill_value=0.0)

    if hp_basis == "final_energy":
        # Re-book heat pumps: electricity input stays electric ('hp'); the
        # ambient heat it upgrades moves to the renewables bucket ('ambient').
        hp_delivered = fine["hp"]
        elec_in = hp_electricity_input(n, segments, countries)
        elec_in = min(elec_in, hp_delivered)  # guard against rounding
        fine["hp"] = elec_in
        fine["ambient"] += hp_delivered - elec_in
    elif hp_basis != "delivered":
        raise ValueError(f"unknown hp_basis: {hp_basis!r}")
    return fine


def existing_capacity_fine(segments: list[str], countries: set[str] | None) -> pd.Series:
    """Installed existing-heating *capacity* (MW) by FINE category. Capacities
    are rated thermal output, so heat pumps are on a 'delivered' basis (no
    ambient split) — match with hp_basis='delivered'.
    """
    df = pd.read_csv(EXISTING_HEATING_CSV, header=[0, 1], index_col=0)
    stk = df.stack([0, 1], future_stack=True).rename("cap").reset_index()
    stk.columns = ["node", "heat_name", "tech", "cap"]
    stk["ct"] = stk.node.str[:2]
    stk["fine"] = stk.tech.map(EXISTING_TECH_FINE)

    def seg_of(hn: str) -> str | None:
        if "urban decentral" in hn:
            return "urban decentral"
        if "rural" in hn:
            return "rural"
        if "urban central" in hn:
            return "urban central"
        return None

    stk["seg"] = stk.heat_name.map(seg_of)
    sel = stk[stk.seg.isin(segments)]
    if countries is not None:
        sel = sel[sel.ct.isin(countries)]
    return sel.groupby("fine").cap.sum().reindex(FINE).fillna(0.0)


def fine_to_categories(fine: pd.Series, categories: list[str]) -> pd.Series:
    """Aggregate a FINE series to the given display categories."""
    out = pd.Series(0.0, index=categories)
    for cat in categories:
        out[cat] = sum(fine.get(k, 0.0) for k in CATEGORY_FINE[cat])
    return out


def to_pct(s: pd.Series) -> pd.Series:
    tot = s.sum()
    return (100 * s / tot) if tot else s


# ── Data sources ─────────────────────────────────────────────────────────────
def renorm(d: dict[str, float]) -> dict[str, float]:
    tot = sum(d.values())
    return {k: 100 * v / tot for k, v in d.items()}


# Estimated share of EU-27 residential space heating that is heat-pump AMBIENT
# heat (the part Eurostat/ODYSSEE bury inside the renewables row). Not published
# as a clean figure — ambient heat is the small, fast-growing slice of renewables
# while biomass is the bulk. ~5% in 2023 is a defensible mid-estimate (Eurostat
# renewable heating & cooling was 26% in 2023, dominated by biomass; heat-pump
# ambient is a few % of household space heating). Adjust here if a firmer figure
# is found; it is subtracted from the source renewables and shown as 'ambient'.
AMBIENT_DATA_PCT = 5.0


# Source fuel shares (%) renormalised to the four model fuels. 'biomass' is the
# renewables bucket (it includes heat-pump ambient heat on the final_energy
# basis, matching Eurostat/ODYSSEE). 'source_split' optionally gives the
# heat-pump vs. resistive split of the electric share, for the figure.
DATA_SOURCES = [
    {
        "key": "existing_heating_input",
        "title": "Existing-heating capacities used in the model (EC Mapping study, 2012)",
        "segments": ["urban decentral", "rural"],
        "countries": None,            # full model scope (study covers EU + UK + NO + CH)
        "hp_basis": "delivered",      # capacities are rated thermal output
        "split_electric": True,       # show heat pump vs. resistive in the figure
        "description": (
            "The brownfield heating capacities the model starts from, taken from the "
            "EC study 'Mapping and analyses of the current and future (2020-2030) "
            "heating/cooling fuel deployment'. Reference year ~2012, BUILDINGS ONLY "
            "(excludes district heating), expressed as installed capacity (GW). That is "
            "why only urban-decentral and rural are compared. The ELECTRIC bar is split "
            "into heat pump vs. resistive: in this stock electric is mostly RESISTIVE "
            "direct heating (~14% of capacity, large in FR/ES/Nordics) while heat pumps "
            "are only ~7% — so the high electric share is not over-installed heat pumps. "
            "Heat pumps are on a delivered (rated-output) basis here to match the "
            "capacity figures, so no ambient split is applied. NOTE the capacity-vs-"
            "energy mismatch: resistive heaters are low-utilisation peakers (big GW, "
            "small TWh), so the supplied share need not equal the capacity share."
        ),
        "source": "existing_capacity",
        "source_label": "existing capacity",
    },
    {
        "key": "eurostat_2023_individual",
        "title": "Eurostat residential space-heating fuel mix, EU-27, 2023",
        "segments": ["urban decentral", "rural"],
        "countries": "EU27",
        "hp_basis": "final_energy",
        "split_electric": True,
        "categories": FE_CATEGORIES,   # ambient shown as its own column
        "description": (
            "Eurostat 'Energy consumption in households' (dataset nrg_d_hhq), 'Share of "
            "fuels in residential space heating, 2023', scope EU-27. Raw 2023 shares: "
            "natural gas 34.3%, renewables 33.0%, oil 12.7%, derived heat (district) "
            "10.1%, electricity ~7%, solid fossil ~3%. To match the model's INDIVIDUAL "
            "heating (urban-decentral + rural) derived heat is removed (district, "
            "validated separately) and coal — absent from the model's building stock — "
            "dropped; the rest are renormalised. Heat pumps are split on the final-energy "
            "basis: ELECTRICITY INPUT counts as electric (matching Eurostat's electricity "
            "row) and AMBIENT heat is shown in its own column. Eurostat buries ambient "
            f"heat inside the renewables row, so ~{AMBIENT_DATA_PCT:.0f} pp of the 33.0% "
            "is moved out to 'ambient' (an estimate — ambient heat is the small, "
            "fast-growing slice of renewables, biomass the bulk), leaving the rest as "
            "true biomass. The electric bar is split into heat-pump electricity input "
            "(bottom) and resistive (top): the model split is exact; Eurostat does not "
            "report it, so the source split is inferred from the ambient estimate via "
            "the model's COP (elec input = ambient/(COP-1)). 2023 actuals — a 2030 model "
            "should show more electric/ambient and less gas/oil."
        ),
        "source": renorm({"gas": 34.3, "biomass": 33.0 - AMBIENT_DATA_PCT,
                          "ambient": AMBIENT_DATA_PCT, "oil": 12.7, "electric": 7.0}),
        "source_label": "Eurostat 2023",
    },
    {
        "key": "odyssee_2023_individual",
        "title": "ODYSSEE-MURE household space-heating fuel mix, EU-27, 2023",
        "segments": ["urban decentral", "rural"],
        "countries": "EU27",
        "hp_basis": "final_energy",
        "split_electric": True,
        "categories": FE_CATEGORIES,   # ambient shown as its own column
        "description": (
            "ODYSSEE-MURE (Enerdata/ADEME, EU-funded energy-efficiency database), "
            "'Heating energy consumption by energy sources', scope EU-27, year 2023 — an "
            "independent cross-check on Eurostat. Raw 2023 shares: natural gas 37%, "
            "renewables 28%, oil 14%, district heat 11%, electricity 7% (~3% residual "
            "coal). District heat removed and the rest renormalised. Heat pumps split on "
            "the final-energy basis: electricity input -> electric, ambient heat -> its "
            f"own column. ~{AMBIENT_DATA_PCT:.0f} pp of the 28% renewables is moved out "
            "to 'ambient' (same estimate as Eurostat), leaving true biomass. The electric "
            "bar is split into heat-pump electricity input (bottom) vs. resistive (top); "
            "the source split is inferred from the ambient estimate via the model's COP. "
            "2023 actuals vs a 2030 model."
        ),
        "source": renorm({"gas": 37.0, "biomass": 28.0 - AMBIENT_DATA_PCT,
                          "ambient": AMBIENT_DATA_PCT, "oil": 14.0, "electric": 7.0}),
        "source_label": "ODYSSEE 2023",
    },
    {
        "key": "eu_district_heating_mix",
        "title": "EU district-heating fuel mix (European Commission via RAP, 2017)",
        "segments": ["urban central"],
        "countries": "EU27",
        "hp_basis": "final_energy",
        "split_electric": True,
        "categories": DH_CATEGORIES,   # show coal & waste explicitly (model = 0)
        "description": (
            "Fuel mix feeding EU district-heating networks, European Commission data via "
            "the Regulatory Assistance Project ('How clean is Europe's district "
            "heating?'). Scope EU, reference year 2017 (latest itemised public "
            "breakdown). Shares shown here on their native categories (no "
            "renormalisation): natural gas 32%, coal & lignite 26%, biomass & biofuels "
            "22%, renewable waste 7% + other 13% (= waste/other 20%). COAL is shown "
            "explicitly even though the model supplies 0% — that is the key finding: the "
            "model has no coal district heating (assumed phased out by 2030), so the "
            "quarter-coal of the 2017 starting point is carried by gas in the model, "
            "inflating its gas share. Large heat pumps are on the final-energy basis. "
            "Context (Euroheat & Power DHC Market Outlook 2025, 2023 data): renewables + "
            "waste heat have risen to ~44%, biomass ~35%, gas and coal both shrinking — "
            "so the real system is moving toward the model's low-coal mix, just via "
            "renewables/waste rather than gas."
        ),
        "source": {"electric": 0.0, "gas": 32.0, "oil": 0.0, "biomass": 22.0,
                   "ambient": 0.0, "coal": 26.0, "waste/other": 20.0},
        "source_label": "EU DH 2017",
    },
    {
        "key": "uk_census_2021",
        "title": "UK individual heating — Census 2021 main-fuel shares (GB)",
        "segments": ["urban decentral", "rural"],
        "countries": {"GB"},
        "hp_basis": "delivered",   # dwelling counts treat heat pumps as electric; HP share tiny
        "split_electric": False,
        "description": (
            "UK Census 2021 (ONS) / House of Commons Library briefing CBP-9838: share of "
            "dwellings by main central-heating fuel in Great Britain, 2021 — the "
            "UK-inclusive validation (model scope extends beyond the EU-27). Raw shares "
            "of HOMES: mains gas 83%, electricity 9.3%, heating oil 4.4%, solid fuel "
            "1.2%, LPG 0.7%. Renormalised to the four model fuels (LPG -> gas, solid "
            "fuel -> biomass). Caveat: a count of dwellings, not energy, so it is an "
            "approximate anchor; heat pumps are negligible in the UK so they are left on "
            "a delivered basis (electric). The pattern to confirm is the UK's "
            "overwhelming gas dependence."
        ),
        "source": renorm({"gas": 83.0 + 0.7, "electric": 9.3, "oil": 4.4, "biomass": 1.2}),
        "source_label": "UK Census 2021",
    },
    {
        "key": "norway_energifakta",
        "title": "Norway individual heating — electricity-dominated (Energifakta Norge)",
        "segments": ["urban decentral", "rural"],
        "countries": {"NO"},
        "hp_basis": "delivered",   # Energifakta frames HP heat as electricity ('direct + heat pumps')
        "split_electric": False,
        "description": (
            "Energifakta Norge (Norwegian Ministry of Energy official portal), ~2024 — "
            "the second non-EU anchor. Norwegian heating is dominated by electricity "
            "(direct + heat pumps), biomass (wood) second, essentially NO gas heating; "
            "district heating covers only ~10-15% of demand (21.9 TWh of heat-pump "
            "output in 2024). The official framing counts heat-pump output as "
            "electricity, so heat pumps are on a delivered basis (electric) to match. No "
            "clean energy-weighted national split is published, so the source shares are "
            "an indicative profile, not an exact statistic — read qualitatively. The "
            "point to confirm is that the model keeps Norway electricity-led; an "
            "appreciable model gas share would be a red flag."
        ),
        "source": renorm({"electric": 60.0, "biomass": 35.0, "oil": 4.0, "gas": 1.0}),
        "source_label": "Norway (indicative)",
    },
]


# ── Plotting helpers ─────────────────────────────────────────────────────────
def load_fuel_colors() -> dict[str, str]:
    """Representative carrier colours from config for the fuel categories and
    the heat-pump / resistive split."""
    with open(REPO_ROOT / "config" / "plotting.default.yaml") as f:
        tc = yaml.safe_load(f)["plotting"]["tech_colors"]
    rep = {
        "electric": "air heat pump", "gas": "gas boiler",
        "oil": "oil boiler", "biomass": "biomass boiler",
        "hp": "air heat pump", "resistive": "resistive heater",
        "coal": "coal", "waste/other": "waste CHP",
    }
    colors = {}
    for name, key in rep.items():
        if key not in tc:
            raise KeyError(f"colour for {key!r} missing from tech_colors")
        colors[name] = tc[key]
    # 'ambient' is a derived analytical category (heat-pump ambient heat), not a
    # model carrier, so it has no tech_colors entry — assign a distinct shade.
    colors["ambient"] = "#7fd4e8"
    return colors


def style_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)


# ── Main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    if not NETWORK.exists():
        raise FileNotFoundError(f"network not found: {NETWORK}")
    print(f"Loading {NETWORK.name}\n")
    n = pypsa.Network(NETWORK)

    model_ct = set(n.buses.country[n.buses.carrier == "AC"].unique())
    scope_map = {"EU27": EU27 & model_ct}

    def resolve(countries):
        if countries is None:
            return None
        if isinstance(countries, str):
            return scope_map[countries]
        return set(countries)

    fuel_colors = load_fuel_colors()
    records = []
    panels = []

    # ── Model overview (all segments, full scope, both hp conventions) ────────
    print("=" * 78)
    print("MODEL HEAT SUPPLY OVERVIEW  —  full model scope (all countries)")
    print(f"network: {NETWORK.name}")
    print("=" * 78)
    for seg in HEAT_SEGMENTS:
        fine = model_supply_fine(n, [seg], None, hp_basis="delivered")
        fuels = fine_to_categories(fine, FUELS)
        twh = fuels / 1e6
        pct = to_pct(fuels)
        print(f"\n{seg} heat  (4-fuel total {twh.sum():.1f} TWh; "
              f"other {fine['other'] / 1e6:.1f} TWh)  [heat pumps on delivered basis]")
        print("  " + "  ".join(f"{f:>8}: {twh[f]:7.1f} TWh ({pct[f]:5.1f}%)" for f in FUELS))
        print(f"  electric breakdown: heat pump {fine['hp'] / 1e6:.1f} TWh, "
              f"resistive {fine['resistive'] / 1e6:.1f} TWh")

    # ── One comparison per data source ────────────────────────────────────────
    for src in DATA_SOURCES:
        countries = resolve(src["countries"])
        scope_lbl = (
            "EU-27" if src["countries"] == "EU27"
            else "full model scope" if src["countries"] is None
            else "+".join(sorted(src["countries"]))
        )
        seg_lbl = " + ".join(src["segments"])
        categories = src.get("categories", FUELS)

        # Model side, on this source's heat-pump basis.
        model_fine = model_supply_fine(n, src["segments"], countries, hp_basis=src["hp_basis"])
        model_cats = fine_to_categories(model_fine, categories)
        model_twh = (model_cats / 1e6)
        model_pct = to_pct(model_cats)

        # Source side.
        if src["source"] == "existing_capacity":
            src_fine = existing_capacity_fine(src["segments"], countries)
            src_cats = fine_to_categories(src_fine, categories)
            source_pct = to_pct(src_cats)
            source_totals = src_cats / 1e3  # GW
            source_unit = "GW (capacity)"
            src_hp, src_res = src_fine["hp"], src_fine["resistive"]
        else:
            source_pct = pd.Series(src["source"]).reindex(categories).fillna(0.0)
            source_totals = None
            source_unit = "% only"
            src_hp = src_res = np.nan
            if src["split_electric"]:
                # The statistics don't separate heat-pump electricity from resistive,
                # but the two are tied to the (estimated) ambient heat through the COP:
                # elec_input = ambient / (COP - 1). Use the model's COP for this scope
                # so model and source split on the same physics.
                model_cop = ((model_fine["hp"] + model_fine["ambient"]) / model_fine["hp"]
                             if model_fine["hp"] > 0 else np.nan)
                amb_pct = float(source_pct.get("ambient", 0.0))
                if np.isfinite(model_cop) and model_cop > 1 and amb_pct > 0:
                    src_hp = amb_pct / (model_cop - 1)
                    src_res = max(source_pct["electric"] - src_hp, 0.0)
                else:
                    src_hp, src_res = 0.0, float(source_pct["electric"])

        print("\n" + "=" * 78)
        print(f"COMPARISON: {src['title']}")
        print("=" * 78)
        print(f"scope: {scope_lbl}   segments: {seg_lbl}   heat-pump basis: {src['hp_basis']}")
        print(textwrap.fill(src["description"], width=76,
                            initial_indent="  ", subsequent_indent="  "))
        print()
        header = f"  {'category':>11} | {'MODEL %':>9}  {'MODEL TWh':>10} | {src['source_label']:>18} %"
        print(header)
        print("  " + "-" * (len(header) - 2))
        for f in categories:
            print(f"  {f:>11} | {model_pct[f]:8.1f}%  {model_twh[f]:9.1f}  | {source_pct[f]:18.1f}")
            records.append({
                "comparison": src["key"], "scope": scope_lbl, "segments": seg_lbl,
                "hp_basis": src["hp_basis"], "category": f,
                "model_pct": round(float(model_pct[f]), 2),
                "model_TWh": round(float(model_twh[f]), 2),
                "source_label": src["source_label"],
                "source_pct": round(float(source_pct[f]), 2),
                "source_total": (round(float(source_totals[f]), 2)
                                 if source_totals is not None else None),
                "source_unit": source_unit,
                "model_hp_TWh": round(float(model_fine["hp"] / 1e6), 2),
                "model_resistive_TWh": round(float(model_fine["resistive"] / 1e6), 2),
                "model_ambient_TWh": round(float(model_fine["ambient"] / 1e6), 2),
                # Heat-pump share of the electric bar (fraction), unit-independent
                # so it is comparable across capacity- and energy-based panels.
                "model_hp_frac_of_electric": (
                    round(float(model_fine["hp"] / (model_fine["hp"] + model_fine["resistive"])), 3)
                    if (model_fine["hp"] + model_fine["resistive"]) > 0 else None),
                "source_hp_frac_of_electric": (
                    round(float(src_hp / (src_hp + src_res)), 3)
                    if (np.isfinite(src_hp) and np.isfinite(src_res) and (src_hp + src_res) > 0)
                    else None),
            })
        # Electric breakdown line.
        ambient_dest = "'ambient' column" if "ambient" in categories else "biomass/renewables"
        print(f"  electric (model): heat pump {model_fine['hp'] / 1e6:.1f} TWh + "
              f"resistive {model_fine['resistive'] / 1e6:.1f} TWh"
              + (f"; ambient heat {model_fine['ambient'] / 1e6:.1f} TWh -> {ambient_dest}"
                 if src["hp_basis"] == "final_energy" else ""))
        gap = (model_pct - source_pct).abs()
        worst = gap.idxmax()
        print(f"  -> largest gap: {worst} ({model_pct[worst]:.1f}% model vs "
              f"{source_pct[worst]:.1f}% source, {gap[worst]:.1f} pp)")

        panels.append({
            "src": src, "scope": scope_lbl, "categories": categories,
            "model_pct": model_pct, "source_pct": source_pct,
            "model_hp": float(model_fine["hp"]), "model_res": float(model_fine["resistive"]),
            "src_hp": float(src_hp), "src_res": float(src_res),
        })

    # ── CSV ───────────────────────────────────────────────────────────────────
    out_csv = SCRIPT_DIR / "heat_share_validation.csv"
    pd.DataFrame(records).to_csv(out_csv, index=False)
    print(f"\nwrote {out_csv}")

    # ── Figure ────────────────────────────────────────────────────────────────
    # Each rendered panel is a list of bar "groups" (model + one or more sources),
    # drawn as a grouping per carrier. The two EU-27 individual-heating sources
    # (Eurostat + ODYSSEE) share the same model side, so they collapse into one
    # three-way panel: model, Eurostat, ODYSSEE. UK Census and Norway follow as
    # ordinary two-way panels. (Existing-capacity and EU district heating dropped.)
    by_key = {p["src"]["key"]: p for p in panels}
    euro, odys = by_key["eurostat_2023_individual"], by_key["odyssee_2023_individual"]

    def group(p, source=False, hatch=None, label=None):
        return {
            "pct": p["source_pct"] if source else p["model_pct"],
            "hp": p["src_hp"] if source else p["model_hp"],
            "res": p["src_res"] if source else p["model_res"],
            "hatch": hatch, "label": label,
        }

    def two_way(p):
        src = p["src"]
        return {
            "title": f"{src['source_label']}\n({p['scope']}, HP {src['hp_basis']})",
            "categories": p["categories"], "split_electric": src["split_electric"],
            "groups": [group(p, label="model"),
                       group(p, source=True, hatch="///", label=src["source_label"])],
        }

    fig_panels = [
        {
            "title": "EU-27 individual heating\n(model vs Eurostat & ODYSSEE, final energy)",
            "categories": euro["categories"], "split_electric": euro["src"]["split_electric"],
            "groups": [
                group(euro, label="model"),
                group(euro, source=True, hatch="///", label="Eurostat 2023"),
                group(odys, source=True, hatch="xxx", label="ODYSSEE 2023"),
            ],
        },
        two_way(by_key["uk_census_2021"]),
        two_way(by_key["norway_energifakta"]),
    ]

    ncols = len(fig_panels)
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 4))
    axes = np.atleast_1d(axes).ravel()
    for panel_idx, (ax, p) in enumerate(zip(axes, fig_panels)):
        ax.text(0.02, 0.98, chr(ord("a") + panel_idx), transform=ax.transAxes,
                fontsize=12, fontweight="bold", va="top", ha="left")
        categories = p["categories"]
        x = np.arange(len(categories))
        groups = p["groups"]
        n = len(groups)
        gw = 0.8 / n            # per-group slot width within the carrier cluster
        bw = gw * 0.9
        for gi, g in enumerate(groups):
            offset = (gi - (n - 1) / 2) * gw
            hatch = g["hatch"]
            for j, f in enumerate(categories):
                if p["split_electric"] and f == "electric" and (g["hp"] + g["res"]) > 0:
                    # Split the electric bar into heat pump (bottom) + resistive (top).
                    elec = g["pct"][f]
                    hp_pct = elec * g["hp"] / (g["hp"] + g["res"])
                    res_pct = elec - hp_pct
                    ax.bar(x[j] + offset, hp_pct, bw, color=fuel_colors["hp"],
                           edgecolor="k", linewidth=0.5, hatch=hatch)
                    ax.bar(x[j] + offset, res_pct, bw, bottom=hp_pct, color=fuel_colors["resistive"],
                           edgecolor="k", linewidth=0.5, hatch=hatch)
                else:
                    ax.bar(x[j] + offset, g["pct"][f], bw, color=fuel_colors[f],
                           edgecolor="k", linewidth=0.5, hatch=hatch)
        ax.set_xticks(x)
        ax.set_xticklabels(categories, fontsize=8, rotation=30, ha="right")
        ax.set_ylabel("share of heat [%]")
        ax.set_title(p["title"], fontsize=9)
        # Legend: one entry per group (model/source), plus HP/resistive where split.
        handles = [plt.Rectangle((0, 0), 1, 1, fc="0.6", ec="k", hatch=g["hatch"]) for g in groups]
        labels = [g["label"] for g in groups]
        if p["split_electric"]:
            handles += [
                plt.Rectangle((0, 0), 1, 1, fc=fuel_colors["hp"], ec="k"),
                plt.Rectangle((0, 0), 1, 1, fc=fuel_colors["resistive"], ec="k"),
            ]
            labels += ["• heat pump", "• resistive"]
        ax.legend(handles, labels, frameon=True, fontsize=7)
        style_ax(ax)
    fig.suptitle("Heating fuel-mix validation  —  solid = model, hatched = source",
                 fontsize=12, y=1.0)
    fig.tight_layout()

    out_pdf = SCRIPT_DIR / "heat_share_validation.pdf"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(SIBLING_IMGS / "heat_share_validation.pdf", bbox_inches="tight")
    print(f"wrote {out_pdf}")
    print(f"wrote {SIBLING_IMGS / 'heat_share_validation.pdf'}")


if __name__ == "__main__":
    main()
