"""
Methods figure (line 57): industry heat currently met by gas, by band.

EU-total endogenised industry heat per temperature band, as ingested by the
model (build rule output). Sums to ~1,300 TWh/a, with roughly 300 TWh <100 °C,
350 TWh 100-200 °C, 100 TWh 200-500 °C and 300 TWh >500 °C.
"""

from pathlib import Path

import matplotlib.pyplot as plt

import _compute as C

HERE = Path(__file__).resolve().parent
GAS_RES_IMGS = C.REPO_ROOT.parent / "gas_resilience" / "imgs"


def style_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)


def plot(band_twh, out_paths):
    fig, ax = plt.subplots(figsize=(6, 4))
    x = range(len(band_twh))
    colors = [C.BAND_COLORS[b] for b in band_twh.index]
    ax.bar(list(x), band_twh.values, color=colors, edgecolor="white", linewidth=0.5)
    for xi, v in zip(x, band_twh.values):
        ax.text(xi, v + 8, f"{v:.0f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(list(x))
    ax.set_xticklabels([C.BAND_LABELS[b] for b in band_twh.index])
    ax.set_ylabel("Heat demand met by gas (TWh/a)")
    ax.set_ylim(0, band_twh.max() * 1.18)
    ax.set_title(f"Total: {band_twh.sum():.0f} TWh/a", fontsize=10)
    style_ax(ax)
    fig.tight_layout()
    for out in out_paths:
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, bbox_inches="tight")
        print(f"wrote {out}")
    plt.close(fig)


def main():
    band_twh = C.model_band_twh()
    plot(band_twh, [
        HERE / "industry_gas_heat_by_band.pdf",
        GAS_RES_IMGS / "industry_gas_heat_by_band.pdf",
    ])


if __name__ == "__main__":
    main()
