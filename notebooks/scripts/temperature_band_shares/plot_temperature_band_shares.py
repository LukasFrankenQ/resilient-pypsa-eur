"""
Per-industry-sector temperature-band shares as implemented in
``scripts/build_industry_sector_ratios_endogenous.py``.

Reproduces the script's mapping logic offline (no snakemake needed) so the
plot reflects exactly what the build rule produces:

  - For sectors covered by ``process_temperature_band_mapper``: mean of the
    Fleiter 2025 rows the mapper points at, with the Mechanical-pulp
    runtime override applied.
  - For sectors covered by ``backup_temperature_band_shares``: the
    hardcoded dict.

Sorted so high-temperature-heavy sectors come first.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter

REPO_ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
GAS_RES_IMGS = REPO_ROOT.parent / "gas_resilience" / "imgs"

idx = pd.IndexSlice
BANDS = ["heat<100", "heat100-200", "heat200-500", "heat>500"]
BAND_COLORS = {
    "heat<100":     "#3b82bd",
    "heat100-200":  "#86c5dc",
    "heat200-500":  "#f3a260",
    "heat>500":     "#c4453c",
}
BAND_LABELS = {
    "heat<100":    "<100 °C",
    "heat100-200": "100-200 °C",
    "heat200-500": "200-500 °C",
    "heat>500":    ">500 °C",
}


def style_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)


def load_fleiter_with_override():
    """Mirror the build script: load Excel, reshape to 4 bands, apply the
    Mechanical pulp override."""
    path = REPO_ROOT / "data" / "ente202300981-sup-0001-suppdata-s1.xlsx"
    df = pd.read_excel(path, sheet_name="Industry_ESC_FEC", index_col=[0, 1, 2], header=[0, 1])
    df.columns = df.columns.droplevel(0)
    df = df.rename(
        columns={
            "<\xa0100\xa0°C": "heat<100",
            "<\xa0100–200\xa0°C": "heat100-200",
            "200–500\xa0°C": "heat200-500",
            "500–1000\xa0°C": "heat500-1000",
            ">\xa01000\xa0°C": "heat>1000",
        }
    )
    df["heat>500"] = df["heat500-1000"] + df["heat>1000"]
    df = df[BANDS]
    df.loc[idx["Paper and printing", "Paper products", "Mechanical pulp"], :] = [0.0, 1.0, 0.0, 0.0]
    return df


# Verbatim from scripts/build_industry_sector_ratios_endogenous.py:116-160
PROCESS_TEMPERATURE_BAND_MAPPER = {
    "Aluminium - secondary production": idx["Non-ferrous metals", "NE metals", ["Aluminium, secondary"]],
    "Aluminium - primary production":   idx["Non-ferrous metals", "NE metals", ["Aluminium, primary"]],
    "Other non-ferrous metals":         idx["Non-ferrous metals", "NE metals",
                                            ["Copper, primary", "Copper, secondary", "Copper further treatment",
                                             "Zinc, primary", "Zinc, secondary"]],
    "High value chemicals":            idx[:, :, ["Adipic acid", "Calcium carbide", "Carbon black",
                                                   "Nitric acid", "Soda ash", "TDI", "Titanium dioxide"]],
    "Cement":                           idx["Non-metallic mineral products", "Clinker", :],
    "Ceramics & other NMM":             idx["Non-metallic mineral products", "Ceramics", :],
    "Glass production":                 idx["Non-metallic mineral products", "Glass", :],
    "Pulp production":                  idx["Paper and printing", "Paper products", ["Chemical pulp", "Mechanical pulp"]],
    "Paper production":                 idx["Paper and printing", "Paper products", ["Paper", "Paper Electrical"]],
    "Printing and media reproduction":  idx["Paper and printing", "Paper products", ["Chemical pulp", "Mechanical pulp"]],
    "Food, beverages and tobacco":      idx["Food, drink and tobacco", :, :],
}

# Verbatim from scripts/build_industry_sector_ratios_endogenous.py:162-209
BACKUP_TEMPERATURE_BAND_SHARES = {
    "Alumina production":           {"heat<100": 0.0, "heat100-200": 0.0,  "heat200-500": 0.0,  "heat>500": 1.0},
    "Pharmaceutical products etc.": {"heat<100": 0.3, "heat100-200": 0.6,  "heat200-500": 0.1,  "heat>500": 0.0},
    "Other industrial sectors":     {"heat<100": 0.1, "heat100-200": 0.65, "heat200-500": 0.25, "heat>500": 0.0},
    "Transport equipment":          {"heat<100": 0.25,"heat100-200": 0.6,  "heat200-500": 0.15, "heat>500": 0.0},
    "Machinery equipment":          {"heat<100": 0.25,"heat100-200": 0.6,  "heat200-500": 0.15, "heat>500": 0.0},
    "Wood and wood products":       {"heat<100": 0.65,"heat100-200": 0.35, "heat200-500": 0.0,  "heat>500": 0.0},
    "Textiles and leather":         {"heat<100": 0.7, "heat100-200": 0.2,  "heat200-500": 0.1,  "heat>500": 0.0},
}


def build_sector_shares():
    fleiter = load_fleiter_with_override()
    rows = {}
    for sector, key in PROCESS_TEMPERATURE_BAND_MAPPER.items():
        sub = fleiter.loc[key]
        if isinstance(sub, pd.Series):
            shares = sub
        else:
            shares = sub.mean()
        rows[sector] = shares.reindex(BANDS).astype(float)
    for sector, shares in BACKUP_TEMPERATURE_BAND_SHARES.items():
        rows[sector] = pd.Series(shares).reindex(BANDS).astype(float)
    df = pd.DataFrame(rows).T
    df = df.div(df.sum(axis=1), axis=0)  # normalise to 1 (Rehfeldt rows e.g. don't always)
    return df


def plot(df, out_paths):
    df_sorted = df.sort_values(
        by=["heat>500", "heat200-500", "heat100-200"], ascending=[False, False, False]
    )
    fig, ax = plt.subplots(figsize=(8, 0.32 * len(df_sorted) + 1.5))
    y = np.arange(len(df_sorted))
    left = np.zeros(len(df_sorted))
    for band in BANDS:
        widths = df_sorted[band].values
        ax.barh(y, widths, left=left, color=BAND_COLORS[band],
                label=BAND_LABELS[band], edgecolor="white", linewidth=0.5)
        left = left + widths
    ax.set_yticks(y)
    ax.set_yticklabels(df_sorted.index, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.0)
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.set_xlabel("Share of process heat demand (%)")
    ax.legend(frameon=True, loc="lower right", ncol=4, bbox_to_anchor=(1.0, -0.15))
    style_ax(ax)
    fig.tight_layout()
    for out in out_paths:
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, bbox_inches="tight")
        print(f"wrote {out}")
    plt.close(fig)


def main():
    df = build_sector_shares()
    plot(df, [
        HERE / "industry_temperature_band_shares.pdf",
        GAS_RES_IMGS / "industry_temperature_band_shares.pdf",
    ])


if __name__ == "__main__":
    main()
