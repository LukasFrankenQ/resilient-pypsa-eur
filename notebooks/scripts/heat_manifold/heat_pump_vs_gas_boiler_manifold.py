"""
LCOH: heat pump vs. gas boiler across heat applications and gas-price /
carbon-price regimes.

Idea
----
For a given peak heat demand the equipment is sized to the peak, so the
*load factor* (CF) of the demand profile sets the full-load hours over
which the (annualized) capex is spread:

    full_load_hours = CF * 8760

    CF = 1.0   -> perfectly flat demand          (capex spread over all hours)
    CF = 0.25  -> ~ residential heat demand
    CF -> 0    -> single sharp peak              (capex spread over ~nothing)

Levelized cost of heat (EUR/MWh_heat):

    LCOH_hp = FIX_hp / (CF * 8760) + p_el / COP                 + VOM_hp
    LCOH_gb = FIX_gb / (CF * 8760) + (p_gas + co2*EF) / eta_gb  + VOM_gb

with FIX = investment * (annuity + FOM) the annualized fixed cost per MW of
(thermal output) capacity, and the CO2 price applied to the gas boiler via
the gas CO2 intensity EF (model assumption).

Each row is one heat application (a heat-pump / gas-boiler pair); each column
fixes the gas price and CO2 price. The x-axis is the demand load factor;
the y-axis is the electricity/gas price ratio (1..4). Filled contours show
LCOH_hp - LCOH_gb (BrBG, centered on 0); the solid line is the break-even.

A 500 EUR/kW_el grid connection cost is added to every heat pump. The
heat-pump capex here is per kW_th (heat output), so the per-kW_el grid cost is
converted to kW_th by dividing by the COP (matching the model, which adds the
grid cost on the electrical side of the heat-pump link).

Techno-economic data are read from the PyPSA-Eur cost table; the cost keys are
the ones the model uses for each application (see prepare_sector_network.py).
"""

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pypsa
from matplotlib.colors import TwoSlopeNorm

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------- #
# configuration
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parents[3]          # repo root (pypsa-eur)
COSTS_FILE = ROOT / "resources" / "costs_2030.csv"

DISCOUNT_RATE = 0.07           # fill value when a tech has no discount rate
GRID_INVEST_EL = 500.0         # EUR/kW_el grid connection capacity (heat pumps)

# axis ranges
CF_MIN, CF_MAX = 0.1, 1.0      # load factor of the demand profile
RATIO_MIN, RATIO_MAX = 0.0, 5.0  # electricity price / gas price

# gas price [EUR/MWh] and CO2 price [EUR/tCO2]
REGIME = dict(gas=25.6, co2=100.0)

# one heat application per panel (heat pump vs. gas boiler).
#  hp / gb        -> cost-table keys (for the LCOH model)
#  heat / hp_link / gb_link -> network carriers (for the data overlay)
ROWS = [
    dict(row="industry heat <100 °C\n(medium-temp HP vs gas boiler)",
         hp="industrial heat pump medium temperature", gb="gas boiler steam",
         heat="heat<100 industry",
         hp_link="heat<100 industry industrial heat pump medium temperature",
         gb_link="heat<100 industry gas",
         capex_csv="industry_lowtemp_heat_pump_literature_capex.csv"),
    dict(row="industry heat 100–200 °C\n(high-temp HP vs gas boiler)",
         hp="industrial heat pump high temperature", gb="gas boiler steam",
         heat="heat100-200 industry",
         hp_link="heat100-200 industry industrial heat pump high temperature",
         gb_link="heat100-200 industry gas",
         capex_csv="industry_hightemp_heat_pump_literature_capex.csv"),
    dict(row="urban individual heating\n(air HP vs gas boiler)",
         hp="decentral air-sourced heat pump", gb="decentral gas boiler",
         heat="urban decentral heat",
         hp_link="urban decentral air heat pump",
         gb_link="urban decentral gas boiler",
         capex_csv="air_heat_pump_literature_capex.csv"),
    dict(row="rural heating\n(ground HP vs gas boiler)",
         hp="decentral ground-sourced heat pump", gb="decentral gas boiler",
         heat="rural heat",
         hp_link="rural ground heat pump",
         gb_link="rural gas boiler",
         capex_csv="ground_heat_pump_literature_capex.csv"),
]

NETWORK = (ROOT / "results" / "networks"
           / "base_s_50_lv1.25_3H-T-H-B-I-A-dist1_2030_free_2000.nc")
# literature heat-pump CAPEX tables (column eur_per_kw_thermal -- same basis as
# the cost-table investment used in the LCOH model)
LIT_DIR = Path(__file__).resolve().parent.parent / "heat_capex_opex_lcohs"

OUT_NAME = "heat_pump_vs_gas_boiler_lcoh.pdf"
HERE = Path(__file__).resolve().parent
IMG_DIR = ROOT.parent / "gas_resilience" / "imgs"


# --------------------------------------------------------------------------- #
# techno-economic data
# --------------------------------------------------------------------------- #
def annuity(rate: float, lifetime: float) -> float:
    """Capital recovery factor."""
    return rate / (1.0 - (1.0 + rate) ** -lifetime)


def load_tech(costs: pd.DataFrame, tech: str) -> dict:
    sub = costs.loc[tech].set_index("parameter")["value"]
    inv = float(sub["investment"]) * 1e3          # EUR/kW -> EUR/MW (of output)
    fom = float(sub["FOM"]) / 100.0               # %/yr -> fraction/yr
    eta = float(sub["efficiency"])                # COP (hp) or boiler efficiency
    lifetime = float(sub["lifetime"])
    vom = float(sub["VOM"]) if "VOM" in sub.index else 0.0
    dr = float(sub["discount rate"]) if "discount rate" in sub.index else DISCOUNT_RATE
    fixed = inv * (annuity(dr, lifetime) + fom)   # EUR/MW/yr
    return dict(inv=inv, fom=fom, eta=eta, lifetime=lifetime, vom=vom, dr=dr,
                fixed=fixed)


costs = pd.read_csv(COSTS_FILE).set_index("technology")
co2_ef = float(
    costs.loc["gas"].set_index("parameter").loc["CO2 intensity", "value"]
)  # tCO2/MWh_th gas


def build_pair(hp_tech: str, gb_tech: str):
    """Load a heat-pump / gas-boiler pair and add the grid cost to the HP.

    The grid connection cost is per kW_el; the heat-pump capex is per kW_th, so
    convert el -> th by dividing by the COP, then annualise with the heat
    pump's own discount rate and lifetime.
    """
    hp = load_tech(costs, hp_tech)
    gb = load_tech(costs, gb_tech)
    grid_inv_th = GRID_INVEST_EL * 1e3 / hp["eta"]          # EUR/MW_th
    hp["inv"] += grid_inv_th
    hp["fixed"] += grid_inv_th * annuity(hp["dr"], hp["lifetime"])
    return hp, gb


def hp_fixed_from_capex(hp, capex_eur_per_kw_th):
    """Annualized HP fixed cost (EUR/MW_th/yr) for a given *thermal* overnight
    CAPEX [EUR/kW_th], including the grid connection cost. Same construction as
    build_pair, so directly comparable to hp['fixed'] (the cost-table value)."""
    a = annuity(hp["dr"], hp["lifetime"])
    inv_th = capex_eur_per_kw_th * 1e3                       # EUR/MW_th
    grid_th = GRID_INVEST_EL * 1e3 / hp["eta"]              # EUR/MW_th
    return inv_th * (a + hp["fom"]) + grid_th * a


# --------------------------------------------------------------------------- #
# LCOH model
# --------------------------------------------------------------------------- #
def lcoh_hp(hp, cf, p_el):
    return hp["fixed"] / (cf * 8760.0) + p_el / hp["eta"] + hp["vom"]


def lcoh_gb(gb, cf, p_gas, co2):
    return gb["fixed"] / (cf * 8760.0) + (p_gas + co2 * co2_ef) / gb["eta"] + gb["vom"]


def breakeven_ratio(hp, gb, cf, p_gas, co2):
    """Electricity/gas price ratio at which LCOH_hp == LCOH_gb, vs CF."""
    rhs = (
        (gb["fixed"] - hp["fixed"]) / (cf * 8760.0)
        + (p_gas + co2 * co2_ef) / gb["eta"]
        + gb["vom"] - hp["vom"]
    )
    return hp["eta"] * rhs / p_gas      # break-even p_el / p_gas


# grids
cf_grid = np.linspace(CF_MIN, CF_MAX, 400)
ratio_grid = np.linspace(RATIO_MIN, RATIO_MAX, 400)
CF, RATIO = np.meshgrid(cf_grid, ratio_grid)

# LCOH difference per application; positive -> heat pump more expensive
pairs = [build_pair(r["hp"], r["gb"]) for r in ROWS]
p_el = RATIO * REGIME["gas"]
diffs = [lcoh_hp(hp, CF, p_el) - lcoh_gb(gb, CF, REGIME["gas"], REGIME["co2"])
         for (hp, gb) in pairs]

alld = np.concatenate([d.ravel() for d in diffs])
vmin, vmax = alld.min(), alld.max()
norm = TwoSlopeNorm(vcenter=0.0, vmin=vmin, vmax=vmax)
levels = np.linspace(vmin, vmax, 257)


# --------------------------------------------------------------------------- #
# network overlay: one marker per location per application
#
# For each location:
#   x = load factor of the (total) heat demand on that heat bus
#   y = (demand-weighted avg electricity price on the heat pump's bus0)
#       / (demand-weighted avg gas price on the gas boiler's bus0)
#       i.e. each price is weighted by the heat-demand profile, so it reflects
#       the price faced when heat is actually being served.
#   colour: the LCOH difference (LCOH_hp - LCOH_gb) itself -- the same metric
#           and colour scale as the filled contours -- but evaluated at the
#           bus's *actual local* electricity and gas prices. So a marker shows
#           the real cost gap for that location, and departs from the regime
#           background wherever the local gas price differs from the regime.
DEMAND_FLOOR_FRAC = 0.02   # drop buses with < 2 % of the largest bus's annual
#                            heat demand (per application) from the markers
# --------------------------------------------------------------------------- #
def compute_points(n, row, hp_d, gb_d):
    w = n.snapshot_weightings.objective
    mp = n.buses_t.marginal_price
    locs = n.buses.index[n.buses.carrier == "AC"]
    pts = []
    for loc in locs:
        heat_bus = f"{loc} {row['heat']}"
        hp_l = n.links[(n.links.carrier == row["hp_link"]) & (n.links.bus1 == heat_bus)]
        gb_l = n.links[(n.links.carrier == row["gb_link"]) & (n.links.bus1 == heat_bus)]
        if hp_l.empty or gb_l.empty:
            continue
        hp_l, gb_l = hp_l.index[0], gb_l.index[0]

        # total heat demand profile on the bus (static + time-varying loads)
        loads = n.loads.index[n.loads.bus == heat_bus]
        if len(loads) == 0:
            continue
        prof = np.zeros(len(n.snapshots))
        for lid in loads:
            if lid in n.loads_t.p_set.columns:
                prof = prof + n.loads_t.p_set[lid].values
            else:
                prof = prof + n.loads.at[lid, "p_set"]
        peak = prof.max()
        if peak <= 0:
            continue
        cf = prof.mean() / peak

        # prices: demand-weighted average marginal price (weight = heat demand
        # per snapshot, i.e. profile x snapshot weighting)
        dweight = prof * w.values
        el_bus = n.links.at[hp_l, "bus0"]      # AC (industry) or low voltage
        gas_bus = n.links.at[gb_l, "bus0"]     # gas
        p_el = float((mp[el_bus].values * dweight).sum() / dweight.sum())
        p_gas = float((mp[gas_bus].values * dweight).sum() / dweight.sum())
        if p_gas <= 0:
            continue
        ratio = p_el / p_gas

        # marker value: same metric as the contours (LCOH_hp - LCOH_gb), but at
        # the bus's actual local prices and load factor
        lcoh_diff = (lcoh_hp(hp_d, cf, p_el)
                     - lcoh_gb(gb_d, cf, p_gas, REGIME["co2"]))

        ann_demand = float((prof * w.values).sum())   # MWh/a heat demand
        pts.append((cf, ratio, lcoh_diff, ann_demand))

    # drop buses with very small heat demand relative to the largest bus
    if pts:
        dmax = max(p[3] for p in pts)
        pts = [p for p in pts if p[3] >= DEMAND_FLOOR_FRAC * dmax]
    return [(cf, ratio, val) for cf, ratio, val, _ in pts]


print("loading network:", NETWORK.name)
n = pypsa.Network(str(NETWORK))
points = [compute_points(n, row, hp, gb)
          for row, (hp, gb) in zip(ROWS, pairs)]
for row, pts in zip(ROWS, points):
    vals = [v for *_, v in pts]
    print(f"  {row['heat']:22s}: {len(pts)} buses, "
          f"LCOH diff range [{min(vals):.0f}, {max(vals):.0f}] EUR/MWh")


# --------------------------------------------------------------------------- #
# plot
# --------------------------------------------------------------------------- #
ncols = len(ROWS)
LIT_LOW_COLOR = "tab:blue"    # low literature-CAPEX break-even line
LIT_HIGH_COLOR = "tab:green"  # high literature-CAPEX break-even line
fig, axes = plt.subplots(1, ncols, figsize=(4.2 * ncols, 5.0),
                         sharex=True, sharey=True)
fig.subplots_adjust(left=0.055, right=0.99, top=0.80, bottom=0.10, wspace=0.08)

cf_handle = None
for j, (row, (hp, gb)) in enumerate(zip(ROWS, pairs)):
    ax = axes[j]
    cf_handle = ax.contourf(CF, RATIO, diffs[j], levels=levels, norm=norm,
                            cmap="BrBG_r", extend="both", antialiased=True)
    cf_handle.set_edgecolor("face")   # remove antialiased band seams

    r_be = breakeven_ratio(hp, gb, cf_grid, REGIME["gas"], REGIME["co2"])
    ax.plot(cf_grid, r_be, color="red", lw=1.4, zorder=5)
    # break-even if the CO2 price were zero
    r_be0 = breakeven_ratio(hp, gb, cf_grid, REGIME["gas"], 0.0)
    ax.plot(cf_grid, r_be0, color="red", lw=1.4, ls="--", zorder=5)

    # break-even for the low/high heat-pump CAPEX from the literature (2026),
    # using eur_per_kw_thermal (same basis as the cost-table investment)
    lit = pd.read_csv(LIT_DIR / row["capex_csv"])["eur_per_kw_thermal"].astype(float)
    r_lit = {}
    for cap, col, key in [(lit.min(), LIT_LOW_COLOR, "low"),
                          (lit.max(), LIT_HIGH_COLOR, "high")]:
        hp_lit = {**hp, "fixed": hp_fixed_from_capex(hp, cap)}
        r_lit[key] = breakeven_ratio(hp_lit, gb, cf_grid, REGIME["gas"], REGIME["co2"])
        ax.plot(cf_grid, r_lit[key], color=col, lw=1.2, zorder=4)
    ax.fill_between(cf_grid, r_lit["low"], r_lit["high"],
                    color="0.7", alpha=0.08, zorder=3)

    # network overlay: one marker per location, coloured by its local-price
    # LCOH difference (same metric/scale as the contours), black-edged for
    # visibility against the matching background
    if points[j]:
        xs, ys, vals = zip(*points[j])
        ax.scatter(xs, ys, c=vals, cmap="BrBG_r", norm=norm, marker="X",
                   s=70, edgecolors="black", linewidths=0.6,
                   zorder=6, clip_on=False)

    ax.set_xlim(CF_MIN, CF_MAX)
    ax.set_ylim(RATIO_MIN, RATIO_MAX)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlabel("demand load factor")

    # title as a framed label in the top-left corner (first line only)
    ax.text(0.04, 0.95, row["row"].split("\n")[0], transform=ax.transAxes,
            ha="left", va="top", fontsize=9.5,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="0.5", alpha=0.9))

    if j == 0:
        ax.set_ylabel("electricity price / gas price  [ratio]")
        ax.set_yticks([0, 1, 2, 3, 4, 5])
        ax.set_yticklabels(["0", "1", "2", "3", "4", "5"])

# shared horizontal colorbar above the axes (narrow, thin), with directional
# labels; shifted left to leave room for the demand-profile insets, so that the
# colorbar + the two insets are centered together
cax = fig.add_axes([0.255, 0.905, 0.22, 0.014])   # [left, bottom, width, height]
cbar = fig.colorbar(cf_handle, cax=cax, orientation="horizontal")
cbar.ax.axvline(norm(0.0), color="black", lw=1.0)
ticks = np.arange(np.ceil(vmin / 20) * 20, np.floor(vmax / 20) * 20 + 1, 20)
ticks = np.union1d(ticks, [0.0])
cbar.set_ticks(ticks)
cbar.ax.set_xticklabels([f"{t:.0f}" for t in ticks])
cbar.set_label("LCOH difference  [EUR/MWh]", fontsize=9)

LBL = "0.15"
cbar.ax.text(0.0, 1.55, "◄ heat pump favoured", ha="left", va="bottom",
             fontsize=9, color=LBL, transform=cbar.ax.transAxes)
cbar.ax.text(1.0, 1.55, "gas boiler favoured ►", ha="right", va="bottom",
             fontsize=9, color=LBL, transform=cbar.ax.transAxes)

# demand-profile insets next to the colorbar: low vs high load factor
res_prof = n.loads_t.p_set["DE0 0 urban decentral heat"].values
res_prof = res_prof / res_prof.max()
xs = np.arange(len(res_prof))
flat_prof = np.ones_like(res_prof)
for left, prof, title in [
    (0.50, res_prof, "residential heat profile\n(demand load factor ≈ 0.25)"),
    (0.635, flat_prof, "flat industry profile\n(demand load factor ≈ 1)"),
]:
    axp = fig.add_axes([left, 0.85, 0.11, 0.11])
    axp.plot(xs, prof, color="navy", lw=0.5)
    axp.set_ylim(0, 1.15)
    axp.set_xticks([])
    axp.set_yticks([])
    for sp in axp.spines.values():
        sp.set_edgecolor("0.6")
        sp.set_linewidth(0.6)
    axp.set_title(title, fontsize=7, pad=3)

# legend for the markers, framed (alpha 0.4) in the bottom-left of the left ax
from matplotlib.lines import Line2D  # noqa: E402

handles = [
    Line2D([], [], marker="X", color="0.6", lw=0, markersize=9,
           markeredgecolor="black", markeredgewidth=0.6,
           label="model LCOE difference"),
    Line2D([], [], color="red", lw=1.4,
           label=f"break-even, CO₂ {REGIME['co2']:g} EUR/tCO₂"),
    Line2D([], [], color="red", lw=1.4, ls="--",
           label="break-even without carbon pricing"),
    Line2D([], [], color=LIT_LOW_COLOR, lw=1.2,
           label="break-even, low 2026 literature value"),
    Line2D([], [], color=LIT_HIGH_COLOR, lw=1.2,
           label="break-even, high 2026 literature value"),
]
axes[0].legend(handles=handles, loc="lower left", fontsize=8, frameon=True,
               framealpha=0.9, handletextpad=0.5, borderpad=0.4)

# --------------------------------------------------------------------------- #
# save (PDF) -- next to the script and a copy into gas_resilience/imgs
# --------------------------------------------------------------------------- #
for d in (HERE, IMG_DIR):
    d.mkdir(parents=True, exist_ok=True)
    fig.savefig(d / OUT_NAME, bbox_inches="tight", pad_inches=0.5)
    print("saved", d / OUT_NAME)
