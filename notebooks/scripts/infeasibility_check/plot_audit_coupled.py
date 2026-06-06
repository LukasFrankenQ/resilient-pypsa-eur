"""Plot the coupled UDH-heat + LV-electricity adequacy audit.

Three figures, all PDF, written to
`<repo_root>/../gas_resilience/imgs/infeasibility_check/`:

  fig1  `udh_lv_uplift_per_bus.pdf`
        Per-bus LV peak load decomposed into (baseline) + (electric-heating
        uplift), baseline vs oil_unpinned scenario side-by-side.

  fig2  `udh_lv_uplift_reduction.pdf`
        For each bus, the *reduction* in LV uplift achieved by unpinning
        the oil boiler — i.e. how much LV electric load uplift the recent
        oil pin commit is forcing onto the AC bus.

  fig3  `udh_lv_worst_bus_timeseries.pdf`
        For the worst-LV-uplift bus, two stacked time-series panels
        (baseline / oil_unpinned) showing baseline LV load + electric
        heating draw + distribution capacity needed.
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

BASELINE_LOAD_COLOR = "#3e7cb1"        # baseline LV electricity load
HP_COLOR = TECH_COLORS["urban decentral air heat pump"]
RES_COLOR = TECH_COLORS["urban decentral resistive heater"]
DIST_COLOR = "#7d3c98"                 # distribution capacity overlay


def style_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)


def fig1_uplift_per_bus(summary: pd.DataFrame) -> None:
    """Per-bus LV peak load decomposition, baseline vs oil_unpinned."""
    pv = summary.pivot(index="ac_loc", columns="scenario",
                       values=["lv_baseline_peak_mw", "lv_uplift_peak_mw"])
    order = (pv["lv_uplift_peak_mw"]["baseline"]
             .sort_values(ascending=False).index.tolist())
    pv = pv.reindex(order)

    fig, ax = plt.subplots(figsize=(13, 6))
    x = np.arange(len(order))
    w = 0.4
    for j, sc in enumerate(["baseline", "oil_unpinned"]):
        base = pv["lv_baseline_peak_mw"][sc].values
        up = pv["lv_uplift_peak_mw"][sc].values
        ax.bar(x + (j - 0.5) * w, base, w,
               color=BASELINE_LOAD_COLOR, alpha=0.85,
               label="LV baseline load (peak hr)" if j == 0 else None,
               edgecolor="white", linewidth=0.4)
        ax.bar(x + (j - 0.5) * w, up, w, bottom=base,
               color=RES_COLOR if sc == "baseline" else HP_COLOR,
               alpha=0.95,
               label="LV uplift (electric heating, baseline pin)" if sc == "baseline"
                     else "LV uplift (oil unpinned)",
               edgecolor="white", linewidth=0.4)

    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=90, fontsize=7)
    ax.set_ylabel("peak LV electricity load [MW]")
    ax.set_title(
        "Coupled UDH-heat + LV-electricity LP — per-bus peak LV load,\n"
        "left bar each pair: baseline (oil pinned)  |  right bar: oil unpinned",
        fontsize=10, loc="left",
    )
    ax.legend(frameon=True, fontsize=8, loc="upper right")
    style_ax(ax)
    fig.tight_layout()
    out = PAPER_DIR / "udh_lv_uplift_per_bus.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  wrote {out}")


def fig2_uplift_reduction(summary: pd.DataFrame) -> None:
    """Reduction in LV uplift when oil boiler is unpinned, per bus."""
    base = summary[summary.scenario == "baseline"].set_index("ac_loc")
    free = summary[summary.scenario == "oil_unpinned"].set_index("ac_loc")
    reduction = (base.lv_uplift_peak_mw - free.lv_uplift_peak_mw).sort_values(ascending=False)
    reduction = reduction[reduction > 1.0]  # drop noise

    fig, ax = plt.subplots(figsize=(12, 5.5))
    x = np.arange(len(reduction))
    ax.bar(x, reduction.values, color="#c0001f", alpha=0.85,
           edgecolor="white", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(reduction.index, rotation=90, fontsize=7)
    ax.set_ylabel("LV peak uplift removed by unpinning UDH oil boiler [MW]")
    ax.set_title(
        "Extra peak LV electricity demand the UDH oil-boiler pin "
        "forces onto the AC bus,\nper AC location (3H, wiggle 1000)",
        fontsize=10, loc="left",
    )

    total = reduction.sum()
    ax.text(0.99, 0.95, f"sum across all buses: {total:,.0f} MW",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=10, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="grey", alpha=0.9))
    style_ax(ax)
    fig.tight_layout()
    out = PAPER_DIR / "udh_lv_uplift_reduction.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  wrote {out}")


def fig3_worst_bus_ts(summary: pd.DataFrame, series: dict) -> None:
    base = summary[summary.scenario == "baseline"]
    worst_loc = base.loc[base.lv_uplift_peak_mw.idxmax(), "ac_loc"]
    worst_bus = base.loc[base.lv_uplift_peak_mw.idxmax(), "bus"]
    print(f"  worst LV uplift bus: {worst_loc}")

    fig, axes = plt.subplots(2, 1, figsize=(13, 7.5), sharex=True)
    for ax, scenario in zip(axes, ["baseline", "oil_unpinned"]):
        ts = series[(scenario, worst_bus)]
        baseline = ts["lv_baseline_t"]
        total = ts["lv_total_t"]
        uplift = (total - baseline).clip(lower=0)
        # Distribution inflow * efficiency (≈ what the LP demands the AC bus push out).
        dist = ts["dist_in_t"]

        ax.fill_between(baseline.index, 0, baseline.values,
                        color=BASELINE_LOAD_COLOR, alpha=0.85,
                        label="baseline LV load (electricity + industry + agri)")
        ax.fill_between(baseline.index, baseline.values, total.values,
                        color=RES_COLOR if scenario == "baseline" else HP_COLOR,
                        alpha=0.85,
                        label="electric-heating uplift (HP + resistive)")
        ax.plot(dist.index, dist.values, color=DIST_COLOR, linewidth=0.5,
                alpha=0.7, label="distribution inflow from AC bus")

        peak_t = total.idxmax()
        ax.axvline(peak_t, color="black", linestyle="--", linewidth=0.6, alpha=0.5)
        ax.text(peak_t, total.max(),
                f" peak {total.max():,.0f} MW @ {peak_t:%Y-%m-%d %H:00}",
                fontsize=7, va="bottom", ha="left")

        ax.set_title(f"{worst_loc}  —  scenario: {scenario}",
                     fontsize=10, loc="left")
        ax.set_ylabel("MW (LV bus)")
        style_ax(ax)
        if ax is axes[0]:
            ax.legend(frameon=True, fontsize=8, loc="upper right")

    axes[-1].set_xlabel("snapshot")
    fig.suptitle(
        f"LV bus load decomposition at worst-uplift location ({worst_loc}) — 3H, wiggle 1000",
        fontsize=11, y=0.995,
    )
    fig.tight_layout()
    out = PAPER_DIR / "udh_lv_worst_bus_timeseries.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  wrote {out}")


def main():
    summary = pd.read_parquet(CACHE_DIR / "audit_coupled_v1.parquet")
    with open(CACHE_DIR / "audit_coupled_v1_series.pkl", "rb") as f:
        series = pickle.load(f)

    print("status counts:")
    print(summary.groupby("scenario").status.value_counts().to_string())

    fig1_uplift_per_bus(summary)
    fig2_uplift_reduction(summary)
    fig3_worst_bus_ts(summary, series)


if __name__ == "__main__":
    main()
