"""
Methods figure (line 51): share of heating demand per industry product.

For each endogenised sector, the share of its total energy demand that is
generic process heat -- i.e. the total MWh of solid biomass, gas and waste heat
(the "heat", "biomass", "methane" carriers) divided by total energy demand.
This share is what gets multiplied by the heat input per tonne and split into
temperature bands.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

import _compute as C

HERE = Path(__file__).resolve().parent
GAS_RES_IMGS = C.REPO_ROOT.parent / "gas_resilience" / "imgs"

HEAT_COLOR = "#c4453c"
REST_COLOR = "#d9d9d9"


def style_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)


def plot(shares, out_paths):
    fig, ax = plt.subplots(figsize=(8, 0.32 * len(shares) + 1.0))
    y = range(len(shares))
    ax.barh(list(y), shares.values, color=HEAT_COLOR,
            edgecolor="white", linewidth=0.5, label="generic process heat")
    ax.barh(list(y), 1 - shares.values, left=shares.values, color=REST_COLOR,
            edgecolor="white", linewidth=0.5, label="other energy demand")
    ax.set_yticks(list(y))
    ax.set_yticklabels(shares.index, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.0)
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.set_xlabel("Share of sector energy demand")
    ax.legend(frameon=True, loc="lower right")
    style_ax(ax)
    fig.tight_layout()
    for out in out_paths:
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, bbox_inches="tight")
        print(f"wrote {out}")
    plt.close(fig)


def main():
    shares = C.heat_share_by_sector()
    plot(shares, [
        HERE / "industry_heat_share_by_sector.pdf",
        GAS_RES_IMGS / "industry_heat_share_by_sector.pdf",
    ])


if __name__ == "__main__":
    main()
