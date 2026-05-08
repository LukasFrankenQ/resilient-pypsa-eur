"""
Deep analysis of a solved PyPSA-Eur network.

Produces three families of plots for any solved `.nc` file under
`./results/networks/`:

  1. Total annual energy balance per bus_carrier (stacked bar, TWh).
  2. Load-weighted average marginal price per bus_carrier (bar, EUR/MWh).
  3. Time-disaggregated area-based energy balance per carrier group (GW).

PDFs are written to the sibling `gas_resilience` repo
(`<repo>/../gas_resilience/imgs/model_overview/<network_stem>/`) as prescribed
by `claude_instructions/PLOTTING.md`.

Usage
-----
    python model_overview.py                       # default target network
    python model_overview.py <filename-in-results>
    python model_overview.py /absolute/path/to/network.nc
"""
from __future__ import annotations

import logging
import re
import sys
import warnings
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pypsa
import yaml
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from frontend_export import export_frontend_data  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger("model_overview")

REPO_ROOT = Path(__file__).resolve().parents[3]
NETWORK_DIR = REPO_ROOT / "results" / "networks"
CONFIG_PATH = REPO_ROOT / "config.basicrun.yaml"
IMGS_ROOT = REPO_ROOT.parent / "gas_resilience" / "imgs" / "model_overview"

DEFAULT_NETWORK = (
    # "base_s_50_lv1.25_3H-T-H-B-I-A-dist1-gas+Generator+m1.5_2030_free_3000.nc"
    "base_s_50_lv1.25_3H-T-H-B-I-A-dist1_2030_free_2000.nc"
)

BALANCE_BUS_CARRIERS_SINGLE = [
    "H2", "NH3", "gas", "methanol", "oil", "solid biomass", "biogas",
    "co2 stored", "co2",
]
BALANCE_CARRIER_GROUPS = {
    "electricity": ["AC", "low voltage"],
    "heat": [
        "urban central heat", "urban decentral heat", "rural heat",
        "residential urban decentral heat", "residential rural heat",
        "services urban decentral heat", "services rural heat",
    ],
    **{c: [c] for c in BALANCE_BUS_CARRIERS_SINGLE},
}

TOTAL_BALANCE_BUS_CARRIERS = [
    "AC", "low voltage", "H2", "gas", "oil", "methanol", "solid biomass",
    "urban central heat", "rural heat",
]
TOTAL_BALANCE_CO2_BUS_CARRIERS = ["co2", "co2 stored"]

PRICE_BUS_CARRIERS = [
    "AC", "low voltage", "H2", "gas", "oil", "methanol", "solid biomass",
    "urban central heat", "urban decentral heat", "rural heat",
]


def extract_wiggle_twh(stem: str) -> str | None:
    """Trailing integer in the network stem encodes total gas consumption (TWh)."""
    m = re.search(r"_(\d+)$", stem)
    return m.group(1) if m else None


def resolve_network_path(arg: str | None) -> Path:
    if arg is None:
        path = NETWORK_DIR / DEFAULT_NETWORK
    elif "/" in arg or arg.startswith("."):
        path = Path(arg).expanduser().resolve()
    else:
        path = NETWORK_DIR / arg
    if not path.exists():
        raise FileNotFoundError(f"Network not found: {path}")
    return path


def load_tech_colors() -> dict:
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)
    return config["plotting"]["tech_colors"]


THEMATIC_PALETTES = {
    # heat delivery / boilers / heat pumps -> reds
    "heat":     ["#c63c3c", "#d95555", "#e37676", "#b02a2a", "#a52020",
                 "#ef8b8b", "#8e1616", "#d04848", "#ed9f9f"],
    # wind -> navy
    "wind":     ["#235894", "#3773b8", "#1a3e68", "#4a8bcc", "#13335a"],
    # solar -> yellow/gold
    "solar":    ["#f4b400", "#ffc845", "#d99e00", "#ffd467"],
    # hydrogen -> purple
    "hydrogen": ["#b030b0", "#9e1f9e", "#cc44cc", "#7a157a"],
    # biomass / biogas / wood -> green
    "biomass":  ["#5a9c3a", "#7fb661", "#3d7827", "#6ba845", "#a3c984",
                 "#2e5f1f"],
    # oil / kerosene / naphtha -> dark brown
    "oil":      ["#5c4a2b", "#4a3d23", "#6f5a37", "#806748", "#3a2f1c"],
    # methanol -> tan
    "methanol": ["#a46c4f", "#8b5a42", "#c28765"],
    # gas / OCGT / CCGT / CHP / SMR -> orange
    "gas":      ["#d35050", "#c46a38", "#b8813a", "#cc7843", "#a05f2f"],
    # electric / EV / BEV -> blue
    "electric": ["#2d7fcc", "#4795d9", "#1c5e99", "#5fa7e3"],
    # battery / PHS -> grey-blue
    "battery":  ["#6b7280", "#4b5563", "#9ca3af"],
    # co2 -> dark grey
    "co2":      ["#3f3f3f", "#262626", "#555555"],
}

# Order matters: first match wins. "heat" first so gas boilers become red.
THEMATIC_RULES = [
    ("heat",     ("heat", "boiler", "heat pump", "resistive", "vent")),
    ("wind",     ("onwind", "offwind", "wind")),
    ("solar",    ("solar", "pv")),
    ("hydrogen", ("h2", "hydrogen", "fuel cell", "electrolysis")),
    ("biomass",  ("biomass", "biogas", "wood", "bioliquid", "waste chp",
                  "forest residues", "landscape care", "municipal")),
    ("oil",      ("oil", "kerosene", "naphtha", "diesel", "petrol")),
    ("methanol", ("methanol",)),
    ("gas",      ("gas", "ocgt", "ccgt", "chp", "smr", "sabatier",
                  "fischer-tropsch", "haber-bosch")),
    ("electric", ("electric", "electricity", "ev ", "bev", "trans")),
    ("battery",  ("battery", "phs", "storage")),
    ("co2",      ("co2",)),
]


# Heat-specific theming: shared palette per supply category, plus a 'consumption'
# bucket for net-negative carriers (loads, vents, chargers).
HEAT_CATEGORY_RULES = [
    # (display name, match keywords on lowercased carrier; first match wins)
    ("gas",              ("gas boiler", "gas chp", "micro gas chp", "gas-fired chp")),
    ("oil",              ("oil boiler", "oil chp")),
    ("biomass",          ("biomass", "biogas", "wood")),
    ("waste / CHP",      ("waste chp", "waste heat")),
    ("electric heating", ("heat pump", "resistive")),
    ("solar thermal",    ("solar thermal",)),
    ("hydrogen",         ("h2 ", "hydrogen", "fuel cell")),
    ("storage",          ("water tank", "water pit", "aquifer",
                          "thermal energy storage", "geothermal")),
]

HEAT_PALETTES = {
    "gas":              ["#7a1414", "#a32020", "#c63c3c", "#d95555", "#e37676", "#ef8b8b"],
    "oil":              ["#1f1f1f", "#3a3a3a", "#5a5a5a", "#7a7a7a", "#9a9a9a", "#bababa"],
    "biomass":          ["#5a3a14", "#7b5a2a", "#9c7038", "#b88848", "#d4a060", "#e6c290"],
    "waste / CHP":      ["#4a5d23", "#6b8235", "#8aa047"],
    "electric heating": ["#0f5a2a", "#1f6f3a", "#2f8f4e", "#46aa66", "#6dc18b", "#a3d9b3"],
    "solar thermal":    ["#b88200", "#d99e00", "#f4b400", "#ffc845", "#ffd467"],
    "hydrogen":         ["#5a0e5a", "#7a157a", "#9e1f9e", "#b030b0", "#cc44cc", "#d666d6"],
    "storage":          ["#3f4956", "#52606d", "#6b7280", "#9ca3af", "#c2c8d0"],
    "consumption":      ["#1a3754", "#2a4a6e", "#3a6088", "#4a76a2", "#5a8cbc",
                         "#6aa2d6", "#7ab8e6", "#a8c8e0"],
    "other":            ["#6b6b6b", "#888888", "#a8a8a8"],
}


def _rename_heat_carrier(carrier: str) -> str:
    """For the heat balance plot: 'urban central' -> 'district heat',
    'urban decentral' -> 'urban individual'. Collapses 'district heat heat'
    -> 'district heat' so the standalone bus-carrier name reads naturally.
    """
    new = carrier.replace("urban central", "district heat") \
                 .replace("urban decentral", "urban individual")
    return new.replace("district heat heat", "district heat")


def _classify_heat_supply(carrier: str) -> str | None:
    c = carrier.lower()
    for cat, kws in HEAT_CATEGORY_RULES:
        if any(k in c for k in kws):
            return cat
    return None


def _shade(palette: list, idx: int, total: int) -> str:
    if total <= 1:
        return palette[len(palette) // 2]
    pos = idx / (total - 1) * (len(palette) - 1)
    return palette[int(round(pos))]


def _heat_color_and_sections(df: pd.DataFrame):
    """Build per-carrier colors and ordered legend sections for the heat plot.

    df: time-indexed DataFrame; columns are carriers; values are signed (GW).
    Returns (color_map, sections) where sections is an ordered list of
    (section_label, [carriers]) — supply categories first (in HEAT_CATEGORY_RULES
    order, then 'other'), then 'consumption' for net-negative carriers.
    """
    sums = df.sum(axis=0)
    is_demand = sums < 0

    cat_order = [name for name, _ in HEAT_CATEGORY_RULES] + ["other"]
    supply_by_cat: dict[str, list[str]] = {c: [] for c in cat_order}
    consumption: list[str] = []

    for carrier in df.columns:
        if is_demand[carrier]:
            consumption.append(carrier)
            continue
        cat = _classify_heat_supply(carrier) or "other"
        supply_by_cat[cat].append(carrier)

    for cat in supply_by_cat:
        supply_by_cat[cat].sort(key=lambda c: -df[c].abs().mean())
    consumption.sort(key=lambda c: df[c].sum())  # most negative first

    color_map: dict[str, str] = {}
    sections: list[tuple[str, list[str]]] = []
    for cat in cat_order:
        carriers = supply_by_cat[cat]
        if not carriers:
            continue
        palette = HEAT_PALETTES.get(cat, HEAT_PALETTES["other"])
        for i, c in enumerate(carriers):
            color_map[c] = _shade(palette, i, len(carriers))
        sections.append((cat, carriers))

    if consumption:
        palette = HEAT_PALETTES["consumption"]
        for i, c in enumerate(consumption):
            color_map[c] = _shade(palette, i, len(consumption))
        sections.append(("consumption", consumption))

    return color_map, sections


def _thematic_color(carrier: str) -> str:
    c = carrier.lower()
    for theme, keywords in THEMATIC_RULES:
        if any(k in c for k in keywords):
            palette = THEMATIC_PALETTES[theme]
            # stable shade selection per carrier
            idx = sum(ord(ch) for ch in carrier) % len(palette)
            return palette[idx]
    return "grey"


def color_for(carrier: str, colors: dict, missing: set) -> str:
    if carrier in colors and colors[carrier]:
        return colors[carrier]
    missing.add(carrier)
    return _thematic_color(carrier)


def style_ax(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)


# --- copied from scripts/plot_balance_timeseries.py ---
def plot_stacked_area_steplike(ax: plt.Axes, df: pd.DataFrame, colors: dict):
    df_cum = df.cumsum(axis=1)
    previous = np.zeros_like(df_cum.iloc[:, 0].values)
    for col in df_cum.columns:
        ax.fill_between(
            df_cum.index, previous, df_cum[col], step="pre", linewidth=0,
            color=colors.get(col, "grey"), label=col,
        )
        previous = df_cum[col].values


def setup_time_axis(ax: plt.Axes, timespan: pd.Timedelta):
    long = timespan > pd.Timedelta(weeks=5)
    if not long:
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MONDAY))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%e\n%b"))
        ax.xaxis.set_minor_locator(mdates.DayLocator())
        ax.xaxis.set_minor_formatter(mdates.DateFormatter("%e"))
    else:
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonthday=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%e\n%b"))
        ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonthday=15))
        ax.xaxis.set_minor_formatter(mdates.DateFormatter("%e"))
    ax.tick_params(axis="x", which="minor", labelcolor="grey")
# --- end copy ---


def _balance_frame(
    n: pypsa.Network, bus_carriers: list, divisor: float,
) -> pd.DataFrame:
    """Per-bus-carrier total energy balances as columns, carriers as rows."""
    frames = {}
    for bc in bus_carriers:
        try:
            eb = n.statistics.energy_balance(bus_carrier=bc, nice_names=False)
        except Exception as err:
            logger.warning(f"energy_balance({bc}): {err}")
            continue
        if eb.empty:
            continue
        frames[bc] = eb.groupby("carrier").sum().div(divisor)
    if not frames:
        return pd.DataFrame()
    return pd.DataFrame(frames).fillna(0.0)


def _plot_stacked_totals(
    df: pd.DataFrame, unit: str, title: str, out_path: Path,
    colors: dict, missing: set, other_threshold: float = 30.0,
) -> None:
    """Vertical stacked bar per column of df (one column = one bus_carrier).

    Carriers whose largest absolute contribution is below `other_threshold`
    (in the plot's unit) are collapsed into an "other" row to keep the legend
    readable.
    """
    df = df.loc[df.abs().sum(axis=1) > 1e-6]
    small = df.abs().max(axis=1) < other_threshold
    if small.any():
        other = df.loc[small].sum(axis=0)
        df = df.loc[~small]
        df.loc["other"] = other
    carriers = df.abs().sum(axis=1).sort_values(ascending=False).index
    df = df.loc[carriers]

    fig, ax = plt.subplots(figsize=(max(7.5, 0.9 * len(df.columns) + 2.5), 7.5))
    x = np.arange(len(df.columns))
    pos_bottom = np.zeros(len(df.columns))
    neg_bottom = np.zeros(len(df.columns))
    seen = []
    for carrier in carriers:
        row = df.loc[carrier].values
        col = "grey" if carrier == "other" else color_for(carrier, colors, missing)
        pos = np.where(row > 0, row, 0.0)
        neg = np.where(row < 0, row, 0.0)
        if pos.any():
            ax.bar(x, pos, bottom=pos_bottom, color=col, width=0.7,
                   label=carrier if carrier not in seen else None)
            pos_bottom = pos_bottom + pos
            seen.append(carrier)
        if neg.any():
            ax.bar(x, neg, bottom=neg_bottom, color=col, width=0.7,
                   label=carrier if carrier not in seen else None)
            neg_bottom = neg_bottom + neg
            seen.append(carrier)

    ax.axhline(0, color="grey", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(df.columns, rotation=30, ha="right")
    ax.set_ylabel(f"energy balance [{unit}]")
    ax.set_title(title)
    style_ax(ax)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5),
              fontsize=7, frameon=True, ncol=1)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_total_energy_balance(
    n: pypsa.Network, bus_carriers: list, out_dir: Path,
    colors: dict, missing: set,
) -> None:
    df = _balance_frame(n, bus_carriers, divisor=1e6)  # TWh/a
    if df.empty:
        logger.info("no total-balance data for energy carriers")
        return
    _plot_stacked_totals(
        df, unit="TWh/a",
        title="Total energy balance by bus carrier",
        out_path=out_dir / "01_total_energy_balance.pdf",
        colors=colors, missing=missing,
    )


def _classify_gas_demand(carrier: str) -> str:
    c = carrier.lower()
    if "industry" in c or "smr" in c or "dri" in c:
        return "industry"
    if "boiler" in c:
        return "heat"
    if c in {"ocgt", "ccgt"} or "chp" in c:
        return "power"
    return "other"


def plot_gas_sector_balance(
    n: pypsa.Network, out_dir: Path, colors: dict, missing: set,
) -> None:
    """Gas bus_carrier balance with demand ordered industry → heat → power."""
    try:
        eb = n.statistics.energy_balance(bus_carrier="gas", nice_names=False)
    except Exception as err:
        logger.warning(f"energy_balance(gas): {err}")
        return
    if eb.empty:
        logger.info("no gas bus_carrier found; skipping gas sector balance")
        return

    s = eb.groupby("carrier").sum().div(1e6)  # TWh/a
    s = s[s.abs() > 1e-6]
    supply = s[s > 0].sort_values(ascending=False)

    demand = s[s < 0]
    sector_order = ["industry", "heat", "power", "other"]
    groups = {sec: [] for sec in sector_order}
    for carrier, val in demand.items():
        groups[_classify_gas_demand(carrier)].append((carrier, val))
    for sec in groups:
        groups[sec].sort(key=lambda kv: kv[1])  # most negative first within sector

    fig, ax = plt.subplots(figsize=(6.0, 7.5))

    # Supply stack (positive)
    bottom = 0.0
    for carrier, val in supply.items():
        col = color_for(carrier, colors, missing)
        ax.bar(0, val, bottom=bottom, color=col, width=0.55, label=carrier)
        bottom += val
    supply_top = bottom

    # Demand stack (negative), industry first (closest to zero), then heat, power, other
    bottom = 0.0
    sector_spans = {}  # sec -> (y_from, y_to, total)
    for sec in sector_order:
        items = groups[sec]
        if not items:
            continue
        sec_start = bottom
        for carrier, val in items:
            col = color_for(carrier, colors, missing)
            ax.bar(0, val, bottom=bottom, color=col, width=0.55, label=carrier)
            bottom += val
        sector_spans[sec] = (sec_start, bottom, bottom - sec_start)

    ax.axhline(0, color="grey", linewidth=0.7)
    ax.set_xticks([0])
    ax.set_xticklabels(["gas"])
    ax.set_ylabel("gas balance [TWh/a]")
    ax.set_title("Gas energy balance by sector")
    style_ax(ax)

    # Sector brackets on the right of the bar
    for sec, (y0, y1, _) in sector_spans.items():
        mid = (y0 + y1) / 2
        ax.annotate(
            sec,
            xy=(0.32, mid), xytext=(0.60, mid),
            fontsize=9, fontweight="bold", va="center", ha="left",
            annotation_clip=False,
            arrowprops=dict(
                arrowstyle="-[, widthB=" + f"{max(0.3, abs(y1 - y0) / max(supply_top, abs(bottom)) * 8):.2f}" + ", lengthB=0.5",
                lw=1.2, color="black",
            ),
        )

    ax.set_xlim(-0.6, 1.6)
    ax.legend(loc="center left", bbox_to_anchor=(1.35, 0.5),
              fontsize=7, frameon=True, ncol=1)
    fig.savefig(out_dir / "04_gas_sector_balance.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_total_co2_balance(
    n: pypsa.Network, bus_carriers: list, out_dir: Path,
    colors: dict, missing: set,
) -> None:
    df = _balance_frame(n, bus_carriers, divisor=1e6)  # Mt/a
    if df.empty:
        logger.info("no total-balance data for co2 carriers")
        return
    _plot_stacked_totals(
        df, unit="Mt/a",
        title="Total CO2 balance by bus carrier",
        out_path=out_dir / "01_total_co2_balance.pdf",
        colors=colors, missing=missing,
    )


def carrier_price_weighted(n: pypsa.Network, bus_carrier: str) -> float:
    """Load-weighted average commodity price (EUR/MWh). NaN if no demand."""
    carrier_buses = n.buses.index[n.buses.carrier == bus_carrier]
    if carrier_buses.empty:
        return float("nan")
    prices = n.buses_t.marginal_price[carrier_buses]
    weights_t = n.snapshot_weightings.generators

    load_ts = pd.DataFrame(0.0, index=n.snapshots, columns=carrier_buses)
    loads_here = n.loads[n.loads.bus.isin(carrier_buses)]
    if not loads_here.empty:
        for load_name, row in loads_here.iterrows():
            if load_name in n.loads_t.p_set.columns:
                load_ts[row.bus] += n.loads_t.p_set[load_name]
            elif load_name in n.loads_t.p.columns:
                load_ts[row.bus] += n.loads_t.p[load_name]
            else:
                load_ts[row.bus] += row.p_set
    else:
        consuming_links = n.links[n.links.bus0.isin(carrier_buses)]
        for link_name, row in consuming_links.iterrows():
            if link_name in n.links_t.p0.columns:
                load_ts[row.bus0] += n.links_t.p0[link_name].clip(lower=0)

    num = (prices * load_ts).multiply(weights_t, axis=0).sum().sum()
    den = load_ts.multiply(weights_t, axis=0).sum().sum()
    return float("nan") if den == 0 else num / den


def plot_marginal_prices(
    n: pypsa.Network, bus_carriers: list, out_dir: Path,
    colors: dict, missing: set,
) -> None:
    available = set(n.buses.carrier.unique())
    records = {}
    for bc in bus_carriers:
        if bc not in available:
            continue
        p = carrier_price_weighted(n, bc)
        if np.isfinite(p):
            records[bc] = p
    if not records:
        logger.warning("no prices computed")
        return
    s = pd.Series(records).sort_values()

    fig, ax = plt.subplots(figsize=(7.5, max(3.5, 0.32 * len(s) + 1.5)))
    bar_colors = [color_for(c, colors, missing) for c in s.index]
    ax.barh(s.index, s.values, color=bar_colors)
    for carrier, val in s.items():
        ax.text(val, carrier, f"  {val:,.1f}", va="center", fontsize=8)
    ax.set_xlabel("load-weighted avg marginal price [EUR/MWh]")
    ax.set_title("Average commodity prices")
    style_ax(ax)
    fig.tight_layout()
    fig.savefig(out_dir / "02_marginal_prices.pdf")
    plt.close(fig)


def plot_balance_timeseries_group(
    balance: pd.DataFrame, group_name: str, bus_carriers: list,
    out_dir: Path, colors: dict, missing: set,
    max_threshold: float = 5.0, mean_threshold: float = 1.0,
    wiggle_twh: str | None = None,
) -> None:
    mask = balance.index.get_level_values("bus_carrier").isin(bus_carriers)
    if not mask.any():
        logger.info(f"no data for group '{group_name}' (bus_carriers={bus_carriers})")
        return
    df = balance[mask].groupby("carrier").sum().div(1e3).T  # GW, snapshots on rows

    below = df.columns[
        (df.abs().max() < max_threshold) & (df.abs().mean() < mean_threshold)
    ].tolist()
    if below:
        df = df.T.groupby(
            df.columns.map(lambda c: "other" if c in below else c)
        ).sum().T

    # Upsample to hourly then daily mean (handles subsampled snapshots)
    df = df.resample("1h").ffill().resample("D").mean()

    is_heat = group_name == "heat"
    if is_heat:
        df = df.rename(columns=_rename_heat_carrier)
        if df.columns.duplicated().any():
            df = df.T.groupby(level=0).sum().T
        color_map, sections = _heat_color_and_sections(df)
        color_map["other"] = "grey"
        ordered_cols = [c for _, cs in sections for c in cs if c in df.columns]
        # keep any leftover columns (shouldn't normally happen) at the end
        ordered_cols += [c for c in df.columns if c not in ordered_cols]
        df = df.loc[:, ordered_cols]
    else:
        df = df.loc[:, (df / df.abs().max().replace(0, np.nan)).var().sort_values().index]
        color_map = {c: color_for(c, colors, missing) for c in df.columns}
        color_map["other"] = "grey"

    pos = df.where(df > 0).fillna(0.0)
    neg = df.where(df < 0).fillna(0.0)

    fig, ax = plt.subplots(figsize=(10, 4.5), layout="constrained")
    plot_stacked_area_steplike(ax, pos, color_map)
    plot_stacked_area_steplike(ax, neg, color_map)
    ax.set_xlim(df.index[0], df.index[-1])
    setup_time_axis(ax, df.index[-1] - df.index[0])
    ax.axhline(0, color="grey", linewidth=0.5)
    ylim = np.ceil(max(-neg.sum(axis=1).min(), pos.sum(axis=1).max()) / 50) * 50
    ax.set_ylim(-ylim, ylim)
    unit = "kt/h" if "co2" in group_name.lower() else "GW"
    ax.set_ylabel(f"{group_name} balance [{unit}]")
    style_ax(ax)

    if wiggle_twh is not None:
        ax.text(
            0.985, 0.96, f"{wiggle_twh} TWh\ntotal gas consumption",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=8, fontweight="bold", color="#222",
            linespacing=1.05,
        )

    if is_heat:
        handles, labels, header_idx = [], [], []
        for label, carriers in sections:
            header_idx.append(len(labels))
            handles.append(Patch(facecolor="none", edgecolor="none"))
            labels.append(label.upper())
            for c in carriers:
                handles.append(Patch(facecolor=color_map[c], edgecolor="none"))
                labels.append(c)
        leg = fig.legend(
            handles=handles, labels=labels,
            loc="outside right upper", fontsize=7, frameon=True,
            handlelength=1.5, handletextpad=0.6, labelspacing=0.35,
        )
        for i in header_idx:
            leg.get_texts()[i].set_fontweight("bold")
    else:
        handles, labels = ax.get_legend_handles_labels()
        half = len(handles) // 2
        fig.legend(
            handles=handles[:half], labels=labels[:half],
            loc="outside right upper", fontsize=7, frameon=True,
        )
    safe = group_name.replace(" ", "_").replace("/", "_")
    fig.savefig(out_dir / f"03_balance_timeseries_{safe}.pdf")
    plt.close(fig)

    if group_name == "heat":
        export_frontend_data("03_balance_timeseries_heat", {
            "y_label": f"{group_name} balance [{unit}]",
            "unit": unit,
            "wiggle_TWh": wiggle_twh,
            "time_index_iso": [t.isoformat() for t in df.index],
            "carriers": list(df.columns),
            "values_GW": {c: df[c].astype(float).tolist() for c in df.columns},
            "colors": {c: color_map.get(c) for c in df.columns},
        })


def main(argv: list[str]) -> None:
    arg = argv[1] if len(argv) > 1 else None
    network_path = resolve_network_path(arg)
    stem = network_path.stem
    wiggle_twh = extract_wiggle_twh(stem)

    out_dir = IMGS_ROOT / stem
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"network: {network_path.name}")
    logger.info(f"output:  {out_dir}")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        n = pypsa.Network(network_path)

    colors = load_tech_colors()
    missing: set = set()
    available_bus_carriers = set(n.buses.carrier.unique())

    # 1) Total energy balance — all energy bus carriers on one stacked-bar plot
    energy_bcs = [bc for bc in TOTAL_BALANCE_BUS_CARRIERS
                  if bc in available_bus_carriers]
    plot_total_energy_balance(n, energy_bcs, out_dir, colors, missing)
    co2_bcs = [bc for bc in TOTAL_BALANCE_CO2_BUS_CARRIERS
               if bc in available_bus_carriers]
    if co2_bcs:
        plot_total_co2_balance(n, co2_bcs, out_dir, colors, missing)

    # 2) Load-weighted avg marginal prices
    plot_marginal_prices(n, PRICE_BUS_CARRIERS, out_dir, colors, missing)

    # Specialty: gas balance grouped by demand sector
    if "gas" in available_bus_carriers:
        plot_gas_sector_balance(n, out_dir, colors, missing)

    # 3) Time-disaggregated area balance per carrier group
    logger.info("computing time-resolved energy balance (this can take a minute)...")
    balance = n.statistics.energy_balance(groupby_time=False, nice_names=False)
    for group, bus_carriers in BALANCE_CARRIER_GROUPS.items():
        present = [bc for bc in bus_carriers if bc in available_bus_carriers]
        if not present:
            logger.info(f"skip group '{group}' — no matching bus carriers")
            continue
        plot_balance_timeseries_group(
            balance, group, present, out_dir, colors, missing,
            wiggle_twh=wiggle_twh,
        )

    if missing:
        logger.info(
            f"{len(missing)} carrier(s) missing from tech_colors "
            f"(using thematic fallback): {sorted(missing)}"
        )
    logger.info("done.")


if __name__ == "__main__":
    main(sys.argv)
