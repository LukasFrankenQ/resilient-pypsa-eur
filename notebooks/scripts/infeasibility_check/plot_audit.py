"""Plot the UDH adequacy audit results.

Three figures, all written as PDF to
`<repo_root>/../gas_resilience/imgs/infeasibility_check/`:

  fig1 — `udh_adequacy_overview.pdf`
         per-bus required new capacity (MW), grouped by carrier, baseline
         vs oil-unpinned, sorted by baseline total addition.

  fig2 — `udh_fixed_slack_vs_demand.pdf`
         scatter of (peak demand, |min fixed slack|) per bus, two scenarios
         overlaid; the magnitude of the gap that must be closed by extendable
         supply tells you which buses have the thinnest brownfield envelope.

  fig3 — `udh_worst_bus_snapshot.pdf`
         time-series view of the bus with the largest baseline shortfall:
         stacked supply envelope (fixed + extendable potential) vs demand,
         with binding hours highlighted. Side-by-side baseline / oil-unpinned.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
CACHE_DIR = SCRIPT_DIR / "cache"
ROOT = SCRIPT_DIR.parents[2]
PAPER_DIR = ROOT.parent / "gas_resilience" / "imgs" / "infeasibility_check"
PAPER_DIR.mkdir(parents=True, exist_ok=True)

with open(ROOT / "config.basicrun.yaml") as f:
    _cfg = yaml.safe_load(f)
TECH_COLORS = _cfg["plotting"]["tech_colors"]
TECH_COLORS.setdefault("urban decentral solar thermal", "#ffbf2b")  # fallback to generic 'solar thermal'

CARRIER_ORDER = [
    "urban decentral air heat pump",
    "urban decentral biomass boiler",
    "urban decentral gas boiler",
    "urban decentral oil boiler",
    "urban decentral resistive heater",
    "urban decentral solar thermal",
]


def style_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)


def _short(bus: str) -> str:
    return bus.replace(" urban decentral heat", "")


def fig1_overview(summary: pd.DataFrame) -> None:
    cols = [c for c in summary.columns if c.startswith("add_") and c.endswith("_mw")]
    carriers = [c[len("add_"):-len("_mw")] for c in cols]

    df = summary.copy()
    df["bus_short"] = df.bus.map(_short)
    df["total_added"] = df[cols].fillna(0).sum(axis=1)

    order = (
        df[df.scenario == "baseline"]
        .sort_values("total_added", ascending=False)
        ["bus_short"].tolist()
    )

    fig, axes = plt.subplots(2, 1, figsize=(13, 8.5), sharex=True)

    for ax, scenario in zip(axes, ["baseline", "oil_unpinned"]):
        sub = df[df.scenario == scenario].set_index("bus_short").reindex(order)
        bottom = np.zeros(len(order))
        carrier_order = [c for c in CARRIER_ORDER if c in carriers]
        leftover = [c for c in carriers if c not in CARRIER_ORDER]
        for carrier in carrier_order + leftover:
            col = f"add_{carrier}_mw"
            if col not in sub.columns:
                continue
            vals = sub[col].fillna(0).values
            ax.bar(
                np.arange(len(order)),
                vals,
                bottom=bottom,
                color=TECH_COLORS.get(carrier, "grey"),
                edgecolor="white",
                linewidth=0.4,
                label=carrier.replace("urban decentral ", "UDH "),
            )
            bottom = bottom + vals
        ax.set_title(f"scenario: {scenario}", fontsize=10, loc="left")
        ax.set_ylabel("min new capacity to close UDH heat balance [MW]", fontsize=9)
        style_ax(ax)
        ax.tick_params(axis="x", rotation=90, labelsize=7)

    axes[-1].set_xticks(np.arange(len(order)))
    axes[-1].set_xticklabels(order, rotation=90)

    handles, labels = axes[0].get_legend_handles_labels()
    seen = set()
    h2, l2 = [], []
    for h, l in zip(handles, labels):
        if l not in seen:
            seen.add(l)
            h2.append(h)
            l2.append(l)
    axes[0].legend(h2, l2, frameon=True, fontsize=7, ncol=3, loc="upper right")

    fig.suptitle(
        "UDH per-bus minimum capacity addition for hourly heat-balance "
        "feasibility (3H, wiggle 1000)\n"
        "All buses feasible — bars show how much extendable capacity the "
        "LP needs to close the worst snapshot.",
        fontsize=10, y=0.995,
    )
    fig.tight_layout()
    out = PAPER_DIR / "udh_adequacy_overview.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  wrote {out}")


def fig2_slack(summary: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    for sc, marker, color in [
        ("baseline", "o", "#c0001f"),
        ("oil_unpinned", "x", "#1f78b4"),
    ]:
        sub = summary[summary.scenario == sc]
        gap = (-sub.min_fixed_slack_mw).clip(lower=0)
        ax.scatter(
            sub.demand_peak_mw, gap,
            marker=marker, color=color,
            s=40, alpha=0.75, label=sc,
        )
    lim = max(summary.demand_peak_mw.max(), (-summary.min_fixed_slack_mw).max()) * 1.05
    ax.plot([0, lim], [0, lim], color="grey", linestyle="--", linewidth=0.8,
            label="gap = peak demand")
    ax.set_xlabel("UDH peak demand [MW]")
    ax.set_ylabel("supply gap at binding snapshot\n"
                  "= demand − Σ p_max_pu·η·p_nom (fixed only) [MW]")
    ax.set_title("How much UDH demand falls outside the brownfield envelope at peak",
                 fontsize=10, loc="left")
    ax.legend(frameon=True)
    style_ax(ax)
    fig.tight_layout()
    out = PAPER_DIR / "udh_fixed_slack_vs_demand.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  wrote {out}")


def fig3_worst_bus(summary: pd.DataFrame, series: dict) -> None:
    base = summary[summary.scenario == "baseline"].set_index("bus")
    worst = base.min_fixed_slack_mw.idxmin()  # most negative
    print(f"  worst-bus (baseline): {worst}")

    fig, axes = plt.subplots(2, 1, figsize=(13, 7.5), sharex=True)
    for ax, scenario in zip(axes, ["baseline", "oil_unpinned"]):
        ts = series[(scenario, worst)]
        demand = ts["demand_t"]
        fixed = ts["fixed_supply_max_t"]
        ext_pot = ts["extendable_supply_potential_t"]

        ax.fill_between(demand.index, 0, fixed.values,
                        color="#bbbbbb", alpha=0.6,
                        label="fixed-asset max supply (η·p_max_pu·p_nom)")
        ax.fill_between(demand.index, fixed.values,
                        (fixed + ext_pot * 1.0).values,  # 1 MW per extendable shown qualitatively
                        color="#9ecae1", alpha=0.35,
                        label="extendable per-MW heat output (qualitative shape)")
        ax.plot(demand.index, demand.values, color="black", linewidth=0.7,
                label="UDH demand")

        gap_mask = (demand - fixed) > 0
        if gap_mask.any():
            ax.fill_between(
                demand.index, 0, demand.values,
                where=gap_mask.values,
                color="#c0001f", alpha=0.15,
                label="hours where fixed supply < demand",
            )

        ax.set_title(
            f"{_short(worst)}  —  scenario: {scenario}  "
            f"(min fixed slack = {ts['fixed_slack_t'].min():,.0f} MW)",
            fontsize=10, loc="left",
        )
        ax.set_ylabel("MW")
        style_ax(ax)
        if ax is axes[0]:
            ax.legend(frameon=True, fontsize=8, loc="upper right")

    axes[-1].set_xlabel("snapshot")
    fig.suptitle(
        f"Worst-shortfall UDH bus — {_short(worst)}  (3H, wiggle 1000)",
        fontsize=11, y=0.995,
    )
    fig.tight_layout()
    out = PAPER_DIR / "udh_worst_bus_snapshot.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  wrote {out}")


def main():
    summary = pd.read_parquet(CACHE_DIR / "audit_v1.parquet")
    with open(CACHE_DIR / "audit_v1_series.pkl", "rb") as f:
        series = pickle.load(f)

    print("status counts:")
    print(summary.groupby("scenario").status.value_counts().to_string())

    fig1_overview(summary)
    fig2_slack(summary)
    fig3_worst_bus(summary, series)


if __name__ == "__main__":
    main()
