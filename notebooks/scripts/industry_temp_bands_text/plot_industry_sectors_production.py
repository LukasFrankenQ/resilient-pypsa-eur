"""
Methods figure (line 41): the industry sectors covered.

EU-total industry production per sector [Mt/a], from the JRC IDEES-derived
``industrial_production_per_country`` resource. This is the set of sectors whose
production outputs feed the temperature-band estimation.
"""

from pathlib import Path

import matplotlib.pyplot as plt

import _compute as C

HERE = Path(__file__).resolve().parent
GAS_RES_IMGS = C.REPO_ROOT.parent / "gas_resilience" / "imgs"

BAR_COLOR = "#3b82bd"


def style_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)


def plot(prod, out_paths):
    fig, ax = plt.subplots(figsize=(8, 0.32 * len(prod) + 1.0))
    y = range(len(prod))
    ax.barh(list(y), prod.values, color=BAR_COLOR, edgecolor="white", linewidth=0.5)
    ax.set_yticks(list(y))
    ax.set_yticklabels(prod.index, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Industry production (Mt/a)")
    style_ax(ax)
    fig.tight_layout()
    for out in out_paths:
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, bbox_inches="tight")
        print(f"wrote {out}")
    plt.close(fig)


def main():
    prod = C.eu_production()
    plot(prod, [
        HERE / "industry_sectors_production.pdf",
        GAS_RES_IMGS / "industry_sectors_production.pdf",
    ])


if __name__ == "__main__":
    main()
