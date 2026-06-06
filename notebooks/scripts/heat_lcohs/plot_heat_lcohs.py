"""Heat-LCOH distribution sweep across the gas-budget wiggle axis.

For each wiggle network in `results/networks/` (3H, free, lv1.25 implicit,
wiggles 0, 250, …, 4000):

  * compute a per-bus weighted LCOH for each (demand, supply-category) pair,
    where supply categories aggregate one or more carriers
    (e.g. electric heating = heat pumps + resistive heaters).
  * across all buses, compute the heat-output-weighted mean and standard
    deviation of the bus LCOHs.

Plots a 7x4 panel:
  rows = {urban decentral heat, rural heat, urban central heat,
          industry <100°C, industry 100-200°C, industry 200-500°C,
          industry >500°C}
  cols = the (up to 4) techs that supply most heat to that demand.

For demands with fewer than 4 supply techs (heat200-500: 3, heat>500: 2)
the trailing axes are hidden. Within a row the y-axis is shared across
columns. Each panel shows the mean line plus +/-1 sigma (darker) and
+/-2 sigma (fainter) shaded bands across the wiggle axis.

LCOH formula matches `notebooks/scripts/res_heat_rural/res_heat_rural.py`
for single-output techs (boilers, heat pumps):

    lcoh_b = (fuel_cost_b + capex_b + opex_b) / delivered_heat_b

For CHPs we use the by-product method: subtract electricity revenue
(elec output × elec-bus marginal price) from total cost before dividing
by heat output. This gives a 'net heat cost' comparable to single-output
techs.

Costs and flows are pulled directly from `n.links` / `n.links_t` so that
techs whose input bus is global (e.g. oil boilers attached to "EU oil")
are routed to the right heat-output region rather than being silently
dropped by a naive `bus + " oil"` lookup.

Filtering: at the link level we drop links with capacity factor
(annual heat output / (p_nom_opt × hours × |efficiency_at_heat_port|))
below CF_MIN. At the cell level (demand × column × wiggle) we drop the
point if fewer than MIN_BUSES valid buses remain after filtering.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pypsa
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = Path(__file__).resolve().parent
NETWORK_DIR = REPO_ROOT / "results" / "networks"
SIBLING_IMGS = REPO_ROOT.parent / "gas_resilience" / "imgs"
SIBLING_IMGS.mkdir(parents=True, exist_ok=True)

PREFIX = "base_s_50_lv1.25_3H-T-H-B-I-A-dist1_2030_free"
WIGGLES = list(range(0, 4001, 250))  # 0, 250, ..., 4000

CF_MIN = 0.02              # drop links whose annual capacity factor is below this
MIN_BUSES = 3              # cell hidden if fewer valid buses remain
MIN_CELL_OUTPUT_TWH = 10.0 # cell hidden if total annual heat output below this

idx = pd.IndexSlice

# Tech specification: (tech_carrier, heat_port, fuel_port, elec_port)
# heat_port  : link port at which heat is delivered (1 for boilers/HPs,
#              2 for CHPs whose convention is bus0=fuel, bus1=elec, bus2=heat)
# fuel_port  : link port at which fuel is withdrawn (typically 0)
# elec_port  : if non-None, port at which the by-product electricity is
#              delivered; the LCOH calculation subtracts elec revenue
# A column may aggregate several techs (e.g. electric heating = HP + resistive).

ROWS: list[tuple[str, list[tuple[str, list[tuple[str, int, int, int | None]], str]]]] = [
    ("urban decentral heat", [
        ("Electric heating", [
            ("urban decentral air heat pump", 1, 0, None),
            ("urban decentral resistive heater", 1, 0, None),
        ], "#2f8f4e"),
        ("Biomass boiler", [
            ("urban decentral biomass boiler", 1, 0, None),
        ], "#9c7038"),
        ("Gas boiler", [
            ("urban decentral gas boiler", 1, 0, None),
        ], "#c63c3c"),
        ("Oil boiler", [
            ("urban decentral oil boiler", 1, 0, None),
        ], "#5a5a5a"),
    ]),
    ("rural heat", [
        ("Electric heating", [
            ("rural air heat pump", 1, 0, None),
            ("rural ground heat pump", 1, 0, None),
            ("rural resistive heater", 1, 0, None),
        ], "#2f8f4e"),
        ("Biomass boiler", [
            ("rural biomass boiler", 1, 0, None),
        ], "#9c7038"),
        ("Gas boiler", [
            ("rural gas boiler", 1, 0, None),
        ], "#c63c3c"),
        ("Oil boiler", [
            ("rural oil boiler", 1, 0, None),
        ], "#5a5a5a"),
    ]),
    ("urban central heat", [
        ("Biomass CHP", [
            ("urban central solid biomass CHP", 2, 0, 1),
        ], "#9c7038"),
        ("Gas boiler", [
            ("urban central gas boiler", 1, 0, None),
        ], "#c63c3c"),
        ("Gas CHP", [
            ("urban central gas CHP", 2, 0, 1),
        ], "#7a1414"),
        ("Heat pump", [
            ("urban central air heat pump", 1, 0, None),
        ], "#2f8f4e"),
    ]),
    ("heat<100 industry", [
        ("Heat pump", [
            ("heat<100 industry industrial heat pump medium temperature", 1, 0, None),
        ], "#2f8f4e"),
        ("Biomass", [
            ("heat<100 industry solid biomass", 1, 0, None),
        ], "#9c7038"),
        ("Gas", [
            ("heat<100 industry gas", 1, 0, None),
        ], "#c63c3c"),
        ("Electric boiler", [
            ("heat<100 industry electric boiler steam", 1, 0, None),
        ], "#1f6f3a"),
    ]),
    ("heat100-200 industry", [
        ("Heat pump", [
            ("heat100-200 industry industrial heat pump high temperature", 1, 0, None),
        ], "#2f8f4e"),
        ("Biomass", [
            ("heat100-200 industry solid biomass", 1, 0, None),
        ], "#9c7038"),
        ("Gas", [
            ("heat100-200 industry gas", 1, 0, None),
        ], "#c63c3c"),
        ("Electric boiler", [
            ("heat100-200 industry electric boiler steam", 1, 0, None),
        ], "#1f6f3a"),
    ]),
    ("heat200-500 industry", [
        ("Biomass", [
            ("heat200-500 industry solid biomass", 1, 0, None),
        ], "#9c7038"),
        ("Gas", [
            ("heat200-500 industry gas", 1, 0, None),
        ], "#c63c3c"),
        ("Hydrogen", [
            ("heat200-500 industry hydrogen", 1, 0, None),
        ], "#9e1f9e"),
    ]),
    ("heat>500 industry", [
        ("Gas", [
            ("heat>500 industry gas", 1, 0, None),
        ], "#c63c3c"),
        ("Hydrogen", [
            ("heat>500 industry hydrogen", 1, 0, None),
        ], "#9e1f9e"),
    ]),
]

ROW_LABELS = {
    "urban decentral heat":  "Urban Individual Heat",
    "rural heat":            "Rural Heat",
    "urban central heat":    "District Heat",
    "heat<100 industry":     "Industry <100°C Heat",
    "heat100-200 industry":  "Industry 100-200°C Heat",
    "heat200-500 industry":  "Industry 200-500°C Heat",
    "heat>500 industry":     "Industry >500°C Heat",
}

NCOLS = 4


def network_path(wiggle: int) -> Path:
    return NETWORK_DIR / f"{PREFIX}_{wiggle}.nc"


def _link_efficiency_at_port(link: pd.Series, port: int) -> float:
    """Efficiency of `link` at output port `port` (signed; we use |·| for CF)."""
    if port == 1:
        return float(link.get("efficiency", 1.0))
    return float(link.get(f"efficiency{port}", 1.0))


def per_bus_lcoh(
    n: pypsa.Network,
    demand: str,
    techs: list[tuple[str, int, int, int | None]],
    buses: list[str],
) -> pd.DataFrame:
    """For one network and one (demand, column-of-techs), return a DataFrame
    indexed by AC bus (region key) with columns
    'cost' [EUR/a]  = capex + opex + fuel_cost - elec_revenue   (sum over techs)
    'output' [MWh/a heat] = aggregated annual heat output       (sum over techs)

    Operates on Links directly so that techs whose input bus is global
    (e.g. 'EU oil') are routed to the correct heat-output region. Links
    with capacity factor below CF_MIN are dropped (filters out the
    'huge capex / tiny dispatch' regime that produces astronomical LCOHs).
    """
    w = n.snapshot_weightings["generators"]
    hours_total = float(w.sum())
    suffix = " " + demand

    cost = pd.Series(0.0, index=buses)
    output = pd.Series(0.0, index=buses)
    seen = pd.Series(False, index=buses)

    for tech_carrier, heat_port, fuel_port, elec_port in techs:
        L = n.links[n.links.carrier == tech_carrier]
        if L.empty:
            continue

        heat_p_col = f"p{heat_port}"
        fuel_p_col = f"p{fuel_port}"
        elec_p_col = f"p{elec_port}" if elec_port is not None else None

        for link_name, link in L.iterrows():
            heat_bus = link[f"bus{heat_port}"]
            if not isinstance(heat_bus, str) or not heat_bus.endswith(suffix):
                continue
            bus_region = heat_bus[: -len(suffix)]
            if bus_region not in cost.index:
                continue

            if heat_p_col not in n.links_t or link_name not in n.links_t[heat_p_col].columns:
                continue
            if fuel_p_col not in n.links_t or link_name not in n.links_t[fuel_p_col].columns:
                continue

            heat_t = -n.links_t[heat_p_col][link_name]
            heat_year = float(heat_t.mul(w).sum())
            if heat_year <= 0:
                continue

            # capacity-factor filter: dispatch / nominal energy at heat port
            p_nom_opt = float(link.p_nom_opt)
            eta_heat = abs(_link_efficiency_at_port(link, heat_port))
            denom = p_nom_opt * hours_total * eta_heat
            if denom <= 0:
                continue
            cf = heat_year / denom
            if cf < CF_MIN:
                continue

            p_fuel_t = n.links_t[fuel_p_col][link_name]
            fuel_withdrawal = p_fuel_t.abs()  # positive flow magnitude

            capex_link = float(link.capital_cost) * p_nom_opt
            opex_link = float(link.marginal_cost) * float(fuel_withdrawal.mul(w).sum())

            fuel_bus = link[f"bus{fuel_port}"]
            fuel_cost = float(
                n.buses_t.marginal_price[fuel_bus]
                .mul(fuel_withdrawal)
                .mul(w)
                .sum()
            )

            elec_revenue = 0.0
            if elec_port is not None and elec_p_col in n.links_t and \
                    link_name in n.links_t[elec_p_col].columns:
                elec_t = -n.links_t[elec_p_col][link_name]  # output positive
                elec_bus = link[f"bus{elec_port}"]
                if elec_bus in n.buses_t.marginal_price.columns:
                    elec_revenue = float(
                        n.buses_t.marginal_price[elec_bus]
                        .mul(elec_t)
                        .mul(w)
                        .sum()
                    )

            cost.loc[bus_region] += capex_link + opex_link + fuel_cost - elec_revenue
            output.loc[bus_region] += heat_year
            seen.loc[bus_region] = True

    cost.loc[~seen] = np.nan
    output.loc[~seen] = np.nan
    return pd.DataFrame({"cost": cost, "output": output})


def weighted_stats(lcoh: pd.Series, weight: pd.Series) -> tuple[float, float, int, float]:
    """Heat-output-weighted mean and standard deviation across buses."""
    mask = lcoh.notna() & weight.notna() & (weight > 0)
    if not mask.any():
        return np.nan, np.nan, 0, 0.0
    x = lcoh[mask].astype(float).values
    w = weight[mask].astype(float).values
    W = w.sum()
    mean = (w * x).sum() / W
    var = (w * (x - mean) ** 2).sum() / W
    std = float(np.sqrt(var))
    return float(mean), std, int(mask.sum()), float(W)


def main():
    n0 = pypsa.Network(network_path(WIGGLES[len(WIGGLES) // 2]))
    buses = sorted(n0.buses.loc[n0.buses.carrier == "AC"].index.unique().tolist())

    rows = []

    for wiggle in tqdm(WIGGLES, desc="networks"):
        path = network_path(wiggle)
        if not path.exists():
            print(f"missing: {path.name} -- skipping")
            continue
        n = pypsa.Network(path)

        for demand, columns in ROWS:
            for col_label, techs, _color in columns:
                df = per_bus_lcoh(n, demand, techs, buses)
                df = df.dropna()
                df = df[df["output"] > 0]

                total_TWh = float(df["output"].sum()) / 1e6
                if len(df) < MIN_BUSES or total_TWh < MIN_CELL_OUTPUT_TWH:
                    rows.append({
                        "demand": demand, "column": col_label, "wiggle": wiggle,
                        "mean": np.nan, "std": np.nan, "n_buses": len(df),
                        "total_output_TWh": total_TWh,
                    })
                    continue

                lcoh = df["cost"] / df["output"]
                mean, std, n_buses, W = weighted_stats(lcoh, df["output"])
                rows.append({
                    "demand": demand, "column": col_label, "wiggle": wiggle,
                    "mean": mean, "std": std, "n_buses": n_buses,
                    "total_output_TWh": W / 1e6,
                })

    stats = pd.DataFrame(rows).set_index(["demand", "column", "wiggle"]).sort_index()
    stats.to_csv(SCRIPT_DIR / "heat_lcohs_distribution.csv")
    print(f"wrote {SCRIPT_DIR / 'heat_lcohs_distribution.csv'}")

    # ---- plotting ----
    nrows = len(ROWS)
    fig, axes = plt.subplots(nrows, NCOLS, figsize=(16, 3.2 * nrows * 2 / 3),
                             sharex=True, sharey="row")
    if nrows == 1:
        axes = np.array([axes])

    visible = np.zeros((nrows, NCOLS), dtype=bool)

    for r, (demand, columns) in enumerate(ROWS):
        for c in range(NCOLS):
            ax = axes[r, c]

            if c >= len(columns):
                ax.set_visible(False)
                continue

            col_label, _techs, color = columns[c]
            try:
                sub = stats.loc[(demand, col_label)].sort_index()
            except KeyError:
                sub = pd.DataFrame(columns=["mean", "std"])

            x = sub.index.values.astype(float) if len(sub) else np.array([])
            mean = sub["mean"].values.astype(float) if len(sub) else np.array([])
            std = sub["std"].values.astype(float) if len(sub) else np.array([])

            valid = np.isfinite(mean) & np.isfinite(std) if len(sub) else np.array([], dtype=bool)
            x_v = x[valid]
            m_v = mean[valid]
            s_v = std[valid]

            if len(x_v) == 0:
                # no valid points across the whole sweep — hide this panel
                ax.set_visible(False)
                continue

            ax.fill_between(x_v, m_v - 2 * s_v, m_v + 2 * s_v,
                            color=color, alpha=0.18, linewidth=0,
                            label=r"$\pm 2\sigma$")
            ax.fill_between(x_v, m_v - s_v, m_v + s_v,
                            color=color, alpha=0.40, linewidth=0,
                            label=r"$\pm 1\sigma$")
            ax.plot(x_v, m_v, color=color, lw=2.0, label="mean")

            ax.text(
                0.97, 0.95, col_label,
                transform=ax.transAxes, ha="right", va="top", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="0.4", linewidth=0.8, alpha=0.9),
            )
            if c == 0:
                ax.set_ylabel("LCOH [EUR/MWh]")

            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.yaxis.grid(True, alpha=0.3)
            ax.set_axisbelow(True)

            visible[r, c] = True

    # Bottom-most visible row per column (skip columns with no visible axes)
    last_visible_row_per_col = {}
    for c in range(NCOLS):
        rows_visible = np.where(visible[:, c])[0]
        if len(rows_visible):
            last_visible_row_per_col[c] = int(rows_visible.max())

    # If a row has lost its first column (column 0 hidden) but other columns
    # are visible, place the y-label on the leftmost visible axis instead.
    for r, (demand, _columns) in enumerate(ROWS):
        if not visible[r, 0] and visible[r].any():
            leftmost = int(np.where(visible[r])[0].min())
            axes[r, leftmost].set_ylabel("LCOH [EUR/MWh]")

    # Big bold sector label in the top-left corner of each row's leftmost
    # visible axis.
    for r, (demand, _columns) in enumerate(ROWS):
        if not visible[r].any():
            continue
        leftmost = int(np.where(visible[r])[0].min())
        axes[r, leftmost].text(
            0.03, 1.04, ROW_LABELS[demand],
            transform=axes[r, leftmost].transAxes,
            ha="left", va="bottom",
            fontsize=11, fontweight="bold",
        )

    # x-labels and tick labels go on the bottom-most visible axis in each
    # column (sharex=True hides ticks above the bottom row by default, so
    # we re-enable them for columns whose last visible axis is not at the
    # absolute bottom row).
    for c, r in last_visible_row_per_col.items():
        ax = axes[r, c]
        ax.set_xlabel("System gas consumption [TWh]")
        ax.tick_params(axis="x", labelbottom=True)
        for lbl in ax.get_xticklabels():
            lbl.set_visible(True)

    # legend on the rightmost visible axis of the first row
    first_row_visible = np.where(visible[0])[0]
    if len(first_row_visible):
        axes[0, int(first_row_visible.max())].legend(
            frameon=True, loc="lower left", fontsize=8
        )

    fig.tight_layout()

    out_local = SCRIPT_DIR / "heat_lcohs_distribution.pdf"
    out_paper = SIBLING_IMGS / "heat_lcohs_distribution.pdf"
    fig.savefig(out_local, bbox_inches="tight")
    fig.savefig(out_paper, bbox_inches="tight")
    print(f"wrote {out_local}")
    print(f"wrote {out_paper}")


if __name__ == "__main__":
    main()
