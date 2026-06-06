"""
Plot a sectoral timeline of new gas-consuming demand coming online in Europe.

Single panel: stacked bars (power, industry, buildings) of positive YoY
deltas of annual gas consumption, in TWh/y.

Color heuristic: "expected gas price over the course of building."
For a project commissioned in year y, the decision-maker's observed gas-price
environment during the ~3-year planning-and-construction period is
approximated by the 3-year trailing TTF average ending at y-1, i.e.
    expected[y] = mean(TTF[y-3 .. y-1]).

Colorbar capped at 60 EUR/MWh so the bulk of the history stays legible despite
the 2022 crisis peak.

Inputs (produced by get_data.py):
  data/eurostat_gas_sectors_annual.csv
  data/ttf_prices_annual.csv

Output:
  gas_timeline.pdf                              (local, next to the script)
  ../../../../gas_resilience/imgs/gas_timeline.pdf  (sibling-repo copy)
"""
from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize


HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
ROOT = HERE.parents[2]
SIBLING_IMGS = ROOT.parent / "gas_resilience" / "imgs"

SECTOR_ORDER = ["power", "industry", "buildings"]
SECTOR_LABEL = {
    "power":     "Power generation",
    "industry":  "Industry",
    "buildings": "Buildings (residential + services)",
}
SECTOR_HATCH = {
    "power":     "",
    "industry":  "///",
    "buildings": "xxx",
}

BUILD_WINDOW = 3        # trailing years used for the price expectation
VMAX = 60               # EUR/MWh colorbar cap


def style_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)


def load_inputs():
    cons = pd.read_csv(DATA / "eurostat_gas_sectors_annual.csv", index_col="year")
    cons = cons[SECTOR_ORDER].sort_index()
    ttf = pd.read_csv(DATA / "ttf_prices_annual.csv", index_col="year").squeeze("columns")
    return cons, ttf


def positive_deltas(cons: pd.DataFrame) -> pd.DataFrame:
    return cons.diff().clip(lower=0)


def expected_price(ttf: pd.Series, years: np.ndarray, window: int) -> pd.Series:
    """
    "Expected gas price during build" = mean of trailing TTF prices over the
    `window` years ending at y-1. Where fewer than `window` TTF values are
    available (i.e. before ~2008), fall back to any trailing values that
    exist, and finally to the earliest available TTF value. Every year gets
    a colour; the caveat is that pre-2008 uses partial / proxy lookback.
    """
    ttf_sorted = ttf.dropna().sort_index()
    fallback = float(ttf_sorted.iloc[:window].mean())
    out = {}
    for y in years:
        lookback = [ttf.get(y - k, np.nan) for k in range(1, window + 1)]
        vals = [v for v in lookback if not np.isnan(v)]
        if vals:
            out[y] = float(np.mean(vals))
        else:
            out[y] = fallback
    return pd.Series(out)


def main():
    cons, ttf = load_inputs()
    deltas = positive_deltas(cons)
    years = deltas.index.to_numpy()

    exp_price = expected_price(ttf, years, BUILD_WINDOW)

    cmap = plt.get_cmap("magma_r")
    norm = Normalize(vmin=float(np.nanmin(exp_price.values)), vmax=VMAX)

    bar_face_per_year = {y: cmap(norm(p)) for y, p in exp_price.items()}

    fig, ax = plt.subplots(figsize=(12, 5.5))
    fig.subplots_adjust(right=0.86)

    bottoms = np.zeros(len(years))
    for sector in SECTOR_ORDER:
        values = deltas[sector].values
        face_colors = [bar_face_per_year[y] for y in years]
        ax.bar(
            years, values,
            bottom=bottoms,
            color=face_colors,
            edgecolor="black",
            linewidth=0.3,
            width=0.85,
            hatch=SECTOR_HATCH[sector],
        )
        bottoms = bottoms + values

    style_ax(ax)
    ax.set_xlabel("Year")
    ax.set_ylabel("New gas demand coming online [TWh/y]")
    ax.set_xlim(years.min() - 0.8, years.max() + 0.8)

    # Sector hatch legend.
    sector_handles = [
        plt.Rectangle((0, 0), 1, 1,
                      facecolor="white", edgecolor="black", linewidth=0.4,
                      hatch=SECTOR_HATCH[s], label=SECTOR_LABEL[s])
        for s in SECTOR_ORDER
    ]
    ax.legend(handles=sector_handles, loc="upper right", frameon=True, fontsize=9)

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.72])
    cbar = fig.colorbar(sm, cax=cbar_ax, extend="max")
    cbar.set_label(
        f"Expected gas price during build\n"
        f"[EUR/MWh, {BUILD_WINDOW}-yr trailing TTF mean]"
    )
    cbar.outline.set_visible(False)

    fig.suptitle(
        "New gas-consuming demand coming online in Europe (EU27 + UK + NO + CH)",
        fontsize=12,
        x=0.44,
    )

    out_local = HERE / "gas_timeline.pdf"
    fig.savefig(out_local)
    print(f"saved {out_local}")

    SIBLING_IMGS.mkdir(parents=True, exist_ok=True)
    out_sibling = SIBLING_IMGS / "gas_timeline.pdf"
    shutil.copyfile(out_local, out_sibling)
    print(f"saved {out_sibling}")


if __name__ == "__main__":
    main()
