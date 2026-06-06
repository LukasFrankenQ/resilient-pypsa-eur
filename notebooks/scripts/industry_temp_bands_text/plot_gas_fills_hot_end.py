"""
Methods figure (line 55): gas supplies the hottest part of the heat demand.

Illustrates the modelling assumption that, when the endogenised generic process
heat is sorted by temperature, the hottest cumulative share is the part assumed
to be met by gas today (``take_cumulative_amount(..., from_start=False)`` in the
build rule). Generic heat is laid out coldest (left) to hottest (right); the
solid segments are the gas-met portion, which fills in from the hot end.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import _compute as C

HERE = Path(__file__).resolve().parent
GAS_RES_IMGS = C.REPO_ROOT.parent / "gas_resilience" / "imgs"


def style_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.xaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)


def plot(df, out_paths):
    fig, ax = plt.subplots(figsize=(9, 3.0))
    left = 0.0
    for band in C.BANDS:
        color = C.BAND_COLORS[band]
        other = df.loc[band, "other"]
        gas = df.loc[band, "gas"]
        width = other + gas
        # within a band, gas meets the hotter portion: draw non-gas then gas
        ax.barh(0, other, left=left, color=color, alpha=0.28,
                edgecolor="white", linewidth=0.5)
        ax.barh(0, gas, left=left + other, color=color,
                edgecolor="white", linewidth=0.5)
        # band label centred on its segment
        ax.text(left + width / 2, 0.62, C.BAND_LABELS[band],
                ha="center", va="bottom", fontsize=9, color=color, fontweight="bold")
        # gas-met share, printed inside the gas (hot) part of the band
        ax.text(left + other + gas / 2, 0, f"{gas / width:.0%}",
                ha="center", va="center", fontsize=8.5, color="white", fontweight="bold")
        left += width

    total = df.sum().sum()
    gas_total = df["gas"].sum()

    ax.set_ylim(-0.75, 1.0)
    ax.set_yticks([])
    ax.set_xlim(0, total)
    ax.set_xlabel("Endogenised industry process heat, by temperature band (TWh/a)")
    ax.text(0.0, 0.95, "← colder", transform=ax.transAxes, ha="left", va="top",
            fontsize=8, color="0.4")
    ax.text(1.0, 0.95, "hotter →", transform=ax.transAxes, ha="right", va="top",
            fontsize=8, color="0.4")
    ax.set_title(
        f"Gas assumed to meet the hottest heat in each process "
        f"({gas_total:.0f} of {total:.0f} TWh)", fontsize=10)

    handles = [
        Patch(facecolor="0.5", label="met by gas today"),
        Patch(facecolor="0.5", alpha=0.28, label="met by other carriers"),
    ]
    ax.legend(handles=handles, frameon=True, loc="upper left", bbox_to_anchor=(0.0, -0.30),
              ncol=2)
    style_ax(ax)
    fig.tight_layout()
    for out in out_paths:
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, bbox_inches="tight")
        print(f"wrote {out}")
    plt.close(fig)


def main():
    df = C.band_heat_twh()
    plot(df, [
        HERE / "industry_gas_fills_hot_end.pdf",
        GAS_RES_IMGS / "industry_gas_fills_hot_end.pdf",
    ])


if __name__ == "__main__":
    main()
