"""
Validate the industry process temperature-band shares used in
``scripts/build_industry_sector_ratios_endogenous.py`` against
Rehfeldt, Fleiter & Toro (2018), *A bottom-up estimation of the heating and
cooling demand in European industry*, Energy Efficiency 11:1057-1082 (the
``s12053-017-9571-y`` PDF in ``heat_band_lit/``).

Two figures are produced in the same directory as this script:
  - ``process_comparison.pdf``: per-process Fleiter 2025 vs Rehfeldt 2018
    shares (only processes that exist in both)
  - ``backup_vs_subsidiary.pdf``: hardcoded backup shares vs Rehfeldt 2018
    nearest reference (process row or Fig. 11 subsidiary subsector
    distribution)

A summary table of mismatches is printed to stdout.

Note on colours: the four temperature bands aren't model carriers, so we
don't use ``tech_colors``. A cool-to-warm sequential palette is used
instead.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent

BANDS = ["heat<100", "heat100-200", "heat200-500", "heat>500"]
BAND_COLORS = {
    "heat<100": "#3b82bd",       # cool blue
    "heat100-200": "#86c5dc",    # light blue
    "heat200-500": "#f3a260",    # warm orange
    "heat>500": "#c4453c",       # warm red
}


def style_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)


def load_fleiter_bands():
    """Read the Fleiter 2025 supplementary data the same way the build script does."""
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
    df.index = df.index.get_level_values(2).str.strip()
    return df


# Rehfeldt et al. 2018, Energy Efficiency 11:1057-1082, Table 1.
# Process-level temperature shares; "-" treated as 0; Rehfeldt's two upper
# bands (500-1000 °C and >1000 °C) collapsed into model band heat>500.
# Some Rehfeldt rows do not sum to 1 (e.g. Aluminium, secondary sums to 0.72)
# — kept verbatim as per the source.
REHFELDT_2018 = {
    # Iron and steel
    "Sinter":                    [0.01, 0.00, 0.20, 0.80],
    "Blast furnace":             [0.01, 0.11, 0.00, 0.87],
    "Electric arc furnace":      [0.00, 0.00, 0.00, 0.99],
    "Rolled steel":              [0.00, 0.00, 0.00, 1.00],
    "Coke oven":                 [0.00, 0.00, 0.00, 1.00],
    "Smelting reduction":        [0.00, 0.00, 0.00, 1.00],
    "Direct reduction":          [0.00, 0.00, 0.00, 1.00],
    # Non-ferrous metals
    "Aluminium, primary":        [0.00, 0.00, 0.20, 0.80],
    "Aluminium, secondary":      [0.00, 0.00, 0.30, 0.42],
    "Aluminium extruding":       [0.00, 0.00, 1.00, 0.00],
    "Aluminium foundries":       [0.00, 0.00, 0.00, 1.00],
    "Aluminium rolling":         [0.00, 0.00, 1.00, 0.00],
    "Copper, primary":           [0.00, 0.00, 0.00, 1.00],
    "Copper, secondary":         [0.00, 0.00, 0.00, 1.00],
    "Copper further treatment":  [0.00, 0.00, 0.00, 1.00],
    "Zinc, primary":             [0.00, 0.00, 0.00, 1.00],
    "Zinc, secondary":           [0.00, 0.00, 0.00, 1.00],
    # Pulp and paper
    "Paper":                     [0.05, 0.88, 0.05, 0.02],
    "Chemical pulp":             [0.00, 1.00, 0.00, 0.00],
    "Mechanical pulp":           [0.00, 1.00, 0.00, 0.00],
    "Recovered fibres":          [0.00, 1.00, 0.00, 0.00],
    # Non-metallic minerals
    "Container glass":           [0.02, 0.19, 0.19, 0.60],
    "Flat glass":                [0.02, 0.21, 0.43, 0.34],
    "Fibre glass":               [0.02, 0.19, 0.19, 0.60],
    "Other glass":               [0.02, 0.22, 0.22, 0.54],
    "Houseware, sanitary ware":  [0.30, 0.00, 0.00, 0.70],
    "Technical, other ceramics": [0.30, 0.15, 0.15, 0.40],
    "Tiles, plates, refractories": [0.07, 0.11, 0.07, 0.75],
    "Clinker Calcination-Dry":     [0.00, 0.00, 0.10, 0.90],
    "Clinker Calcination-Semidry": [0.00, 0.00, 0.10, 0.90],
    "Clinker Calcination-Wet":     [0.00, 0.00, 0.10, 0.90],
    "Gypsum":                    [0.00, 0.50, 0.30, 0.20],
    "Bricks":                    [0.20, 0.00, 0.00, 0.80],
    "Lime burning":              [0.00, 0.00, 0.00, 1.00],
    # Basic chemicals
    "Adipic acid":               [0.00, 0.50, 0.25, 0.25],
    "Ammonia":                   [0.00, 0.00, 0.00, 0.99],
    "Calcium carbide":           [0.00, 0.00, 0.00, 1.00],
    "Carbon black":              [0.00, 0.00, 0.00, 1.00],
    "Chlorine, diaphragma":      [0.00, 1.00, 0.00, 0.00],
    "Chlorine, membrane":        [0.00, 1.00, 0.00, 0.00],
    "Chlorine, mercury":         [0.00, 1.00, 0.00, 0.00],
    "Methanol":                  [0.00, 0.00, 0.00, 1.00],
    "Polyethylene":              [0.00, 1.00, 0.00, 0.00],
    "Polypropylene":             [0.00, 1.00, 0.00, 0.00],
    "Polysulfones":              [0.00, 1.00, 0.00, 0.00],
    "Soda ash":                  [0.30, 0.40, 0.00, 0.30],
    "TDI":                       [0.00, 1.00, 0.00, 0.00],
    "Titanium dioxide":          [0.30, 0.30, 0.23, 0.47],
    # Food, drink and tobacco
    "Sugar":                     [0.10, 0.60, 0.30, 0.00],
    "Dairy":                     [0.90, 0.10, 0.00, 0.00],
    "Brewing":                   [0.55, 0.45, 0.00, 0.00],
    "Meat processing":           [0.40, 0.60, 0.00, 0.00],
    "Bread and bakery":          [0.20, 0.33, 0.47, 0.00],
}

# Hardcoded backup shares — copied verbatim from
# scripts/build_industry_sector_ratios_endogenous.py:162-209 so the
# validation always reflects the source of truth.
BACKUP_TEMPERATURE_BAND_SHARES = {
    "Alumina production": {
        "heat<100": 0.0,
        "heat100-200": 0.0,
        "heat200-500": 0.0,
        "heat>500": 1.0,
    },
    "Pharmaceutical products etc.": {
        "heat<100": 0.3,
        "heat100-200": 0.6,
        "heat200-500": 0.1,
        "heat>500": 1.0,  # note: sums to 2.0 — see README
    },
    "Other industrial sectors": {
        "heat<100": 0.1,
        "heat100-200": 0.65,
        "heat200-500": 0.25,
        "heat>500": 0.0,
    },
    "Transport equipment": {
        "heat<100": 0.25,
        "heat100-200": 0.6,
        "heat200-500": 0.15,
        "heat>500": 0.0,
    },
    "Machinery equipment": {
        "heat<100": 0.25,
        "heat100-200": 0.6,
        "heat200-500": 0.15,
        "heat>500": 0.0,
    },
    "Wood and wood products": {
        "heat<100": 0.65,
        "heat100-200": 0.35,
        "heat200-500": 0.0,
        "heat>500": 0.0,
    },
    "Textiles and leather": {
        "heat<100": 0.7,
        "heat100-200": 0.2,
        "heat200-500": 0.1,
        "heat>500": 0.0,
    },
}

# Rehfeldt 2018, Fig. 11 ("Subsidiary industrial subsector temperature
# distribution"). Read from the figure with three bands; the 100-500 °C band
# is split evenly into 100-200 and 200-500 (a documented approximation —
# Fig. 11 does not separate them).
REHFELDT_FIG11_SUBSIDIARY = {
    "Iron and steel":              [0.10, 0.05, 0.05, 0.80],
    "Non-ferrous metals":          [0.05, 0.15, 0.15, 0.65],
    "Paper and printing":          [0.10, 0.40, 0.40, 0.10],
    "Non-metallic mineral products": [0.05, 0.05, 0.05, 0.85],
    "Chemical industry":           [0.10, 0.25, 0.25, 0.40],
    "Food, drink and tobacco":     [0.70, 0.15, 0.15, 0.00],
    "Engineering and other metal": [0.30, 0.35, 0.35, 0.00],
    "Other non-classified":        [0.25, 0.35, 0.35, 0.05],
}

# Mapping from each backup process to the most relevant Rehfeldt reference.
# Where a process-level Rehfeldt analogue exists (only Alumina <-> Aluminium
# primary), we include it alongside the subsidiary subsector distribution.
BACKUP_REHFELDT_MAP = {
    "Alumina production":            ("Aluminium, primary (Table 1)", REHFELDT_2018["Aluminium, primary"]),
    "Pharmaceutical products etc.":  ("Chemical industry (Fig. 11)",   REHFELDT_FIG11_SUBSIDIARY["Chemical industry"]),
    "Other industrial sectors":      ("Other non-classified (Fig. 11)", REHFELDT_FIG11_SUBSIDIARY["Other non-classified"]),
    "Transport equipment":           ("Engineering and other metal (Fig. 11)", REHFELDT_FIG11_SUBSIDIARY["Engineering and other metal"]),
    "Machinery equipment":           ("Engineering and other metal (Fig. 11)", REHFELDT_FIG11_SUBSIDIARY["Engineering and other metal"]),
    "Wood and wood products":        ("Other non-classified (Fig. 11)", REHFELDT_FIG11_SUBSIDIARY["Other non-classified"]),
    "Textiles and leather":          ("Other non-classified (Fig. 11)", REHFELDT_FIG11_SUBSIDIARY["Other non-classified"]),
}


def stacked_bars(ax, labels, shares, title=None):
    """Plot horizontal stacked bars (one per label) split by temperature band."""
    y = np.arange(len(labels))
    left = np.zeros(len(labels))
    for band in BANDS:
        widths = np.array([row[band] for row in shares])
        ax.barh(y, widths, left=left, color=BAND_COLORS[band], label=band, edgecolor="white", linewidth=0.4)
        left = left + widths
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, max(1.05, left.max() * 1.02))
    ax.axvline(1.0, color="black", linewidth=0.6, linestyle="--", alpha=0.4)
    if title:
        ax.set_title(title, fontsize=10, loc="left")
    style_ax(ax)


def figure_process_comparison(fleiter_df):
    """Per-process Fleiter 2025 vs Rehfeldt 2018, two stacked bars per process."""
    common = [p for p in REHFELDT_2018 if p in fleiter_df.index]
    rows = []
    labels = []
    for proc in common:
        labels.append(f"{proc} — Fleiter 2025")
        rows.append({b: float(fleiter_df.loc[proc, b]) for b in BANDS})
        labels.append(f"{proc} — Rehfeldt 2018")
        rows.append(dict(zip(BANDS, REHFELDT_2018[proc])))

    height = 0.22 * len(labels) + 1.5
    fig, ax = plt.subplots(figsize=(11, height))
    stacked_bars(ax, labels, rows, title="Per-process temperature shares — Fleiter et al. 2025 vs Rehfeldt et al. 2018")
    ax.set_xlabel("Share of process heat demand")
    ax.legend(frameon=True, loc="lower right", ncol=4, bbox_to_anchor=(1.0, -0.05))
    fig.tight_layout()
    out = HERE / "process_comparison.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def figure_backup_vs_subsidiary():
    rows = []
    labels = []
    for proc, shares in BACKUP_TEMPERATURE_BAND_SHARES.items():
        ref_name, ref_shares = BACKUP_REHFELDT_MAP[proc]
        labels.append(f"{proc} — backup (script)")
        rows.append(dict(shares))
        labels.append(f"{proc} — {ref_name}")
        rows.append(dict(zip(BANDS, ref_shares)))

    height = 0.32 * len(labels) + 1.5
    fig, ax = plt.subplots(figsize=(11, height))
    stacked_bars(ax, labels, rows, title="Hardcoded backup shares vs Rehfeldt 2018 nearest reference")
    ax.set_xlabel("Share of process heat demand (dashed line at 1.0)")
    ax.legend(frameon=True, loc="lower right", ncol=4, bbox_to_anchor=(1.0, -0.10))
    fig.tight_layout()
    out = HERE / "backup_vs_subsidiary.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def print_summary(fleiter_df):
    print("\n=== Fleiter 2025 vs Rehfeldt 2018: max absolute band share difference ===")
    rows = []
    for proc, ref in REHFELDT_2018.items():
        if proc not in fleiter_df.index:
            continue
        f = fleiter_df.loc[proc, BANDS].astype(float).values
        r = np.array(ref)
        diff = np.abs(f - r).max()
        rows.append((proc, diff))
    summary = pd.DataFrame(rows, columns=["process", "max_abs_diff"]).sort_values(
        "max_abs_diff", ascending=False
    )
    print(summary.to_string(index=False))

    print("\n=== Backup share sums (should be 1.0) ===")
    for proc, shares in BACKUP_TEMPERATURE_BAND_SHARES.items():
        s = sum(shares.values())
        flag = " <-- !!! sums to {:.2f}, not 1.0".format(s) if abs(s - 1.0) > 1e-6 else ""
        print(f"  {proc}: {s:.2f}{flag}")


def main():
    fleiter_df = load_fleiter_bands()
    figure_process_comparison(fleiter_df)
    figure_backup_vs_subsidiary()
    print_summary(fleiter_df)


if __name__ == "__main__":
    main()
