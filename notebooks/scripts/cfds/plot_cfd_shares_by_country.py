"""
Share of 2030 RES + nuclear *generation* (estimated TWh/yr) by policy regime,
per country. Weighting by TWh (capacity × tech-level capacity factor) avoids
the GW-basis overemphasis of solar, which has a much lower capacity factor
than wind/nuclear/biomass.

Three categories are stacked to 100% per country:

  1. `2-sided CfD` — windfall clawback in place (UK AR4+ CfDs, EU post-2027
     schemes under the reformed EMD, France CR + new regulated nuclear,
     Irish RESS/ORESS, Polish URE auctions, Italian FER X, offshore
     tenders in BE/NL/PT/GR/DK/NO-SN2).
  2. `Non-clawback support` — 1-sided CfD, sliding feed-in premium,
     feed-in tariff, ROC, green certificate. Subsidised but no mechanism
     to recover high-price rents (German EEG pre-2027, Dutch SDE++,
     Swedish/Norwegian Elcertifikat, UK legacy ROC, Conto Energia, FiTs).
  3. `Merchant / PPA / unsupported` — exposed to wholesale prices,
     no support, no clawback.

CAVEATS — this is an illustrative, hand-curated estimate:
  - Capacity numbers are draft-NECP-2024 / TYNDP-2024 targets and will
    slip for most countries. Shares are more robust than absolute GW.
  - Policy assignment assumes legacy operating capacity stays on its
    historical regime through 2030, and that EU EMD reform (2027)
    applies uniformly to post-2027 auctioned volumes.
  - Merchant is the residual, so any underestimate of legacy
    shows up there.
  - Nuclear extension / regulated-nuclear agreements (FR, BE) are
    treated as 2-sided because the clawback-like structure applies
    in high-price scenarios, even though the legal construct differs
    from a classic auction CfD.
  - Country set is 20 markets with usable public policy info;
    Baltics / Balkans / LU excluded. See plan for rationale.

Source tags in the CSV are compact; see the script docstring and
`data/policy_shares_2030.csv` header for the mapping to NECPs,
national regulator publications and the Oxford Institute EL56 paper.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_CSV = SCRIPT_DIR / "data" / "policy_shares_2030.csv"

sys.path.insert(0, str(REPO_ROOT / "notebooks" / "scripts"))
from frontend_export import export_frontend_data  # noqa: E402

with open(REPO_ROOT / "config.basicrun.yaml") as f:
    config = yaml.safe_load(f)
tech_colors = config["plotting"]["tech_colors"]

# Policy categories are not carriers so not in tech_colors.
# Muted teal = clawback/controlled; amber = rent leakage; grey = neutral.
CATEGORY_COLORS = {
    "2-sided CfD": "#2a9d8f",
    "Non-clawback support": "#e9c46a",
    "Merchant / PPA / unsupported": "#9aa0a6",
}
CATEGORY_ORDER = list(CATEGORY_COLORS.keys())
SHARE_COLS = ["share_2sided_cfd", "share_nonclawback", "share_merchant"]
COL_TO_CATEGORY = dict(zip(SHARE_COLS, CATEGORY_ORDER))

# Tech-level capacity factors for converting GW capacity to TWh/yr generation.
# Rough European averages — deliberately single-valued per tech since the
# shares are the primary output and coarse CFs are sufficient to correct
# the GW-basis overweighting of solar.
CAPACITY_FACTORS = {
    "solar": 0.12,
    "onwind": 0.28,
    "offwind": 0.45,
    "hydro": 0.35,
    "biomass": 0.65,
    "nuclear": 0.85,
}
HOURS_PER_YEAR = 8760

# Dominant non-clawback support scheme per country (see
# non_clawback_mechanisms.md for full details and citations).
DOMINANT_NONCLAWBACK_SCHEME = {
    "DE": "EEG",
    "FR": "CR legacy / FiT",
    "GB": "ROC",
    "IT": "Conto Energia",
    "ES": "RII",
    "PL": "Green Cert.",
    "NL": "SDE++",
    "SE": "Elcertifikat",
    "NO": "Elcert legacy",
    "FI": "2018 premium",
    "RO": "Green Cert.",
    "GR": "FiT + DAPEEP",
    "PT": "FiT",
    "CZ": "Green Bonus",
    "BE": "GSC",
    "AT": "EAG",
    "BG": "FiT",
    "DK": "Market premium",
    "HU": "METÁR",
    "IE": "REFIT",
}
# Minimum non-clawback segment height (in %) needed to fit a rotated label.
MIN_LABEL_HEIGHT_PCT = 11.0


def style_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)


def load_and_aggregate(csv_path):
    df = pd.read_csv(csv_path)

    # Sanity: shares sum to 1 per row, no missing sources.
    row_sum = df[SHARE_COLS].sum(axis=1)
    bad = df.loc[~np.isclose(row_sum, 1.0, atol=1e-6)]
    assert bad.empty, f"Rows where shares do not sum to 1:\n{bad}"
    for col in ["source_capacity", "source_policy"]:
        missing = df[col].isna() | (df[col].astype(str).str.strip() == "")
        assert not missing.any(), f"Missing values in {col}"

    # Convert capacity to estimated annual generation using tech-level CFs.
    missing_cf = set(df["tech"].unique()) - set(CAPACITY_FACTORS)
    assert not missing_cf, f"Missing capacity factor for tech(s): {missing_cf}"
    df["generation_twh"] = (
        df["capacity_2030_gw"]
        * df["tech"].map(CAPACITY_FACTORS)
        * HOURS_PER_YEAR
        / 1000.0
    )

    # Generation-weighted aggregation per country.
    weighted = df[SHARE_COLS].multiply(df["generation_twh"], axis=0)
    weighted["country"] = df["country"]
    weighted["generation_twh"] = df["generation_twh"]
    agg = weighted.groupby("country").sum()
    shares = agg[SHARE_COLS].divide(agg["generation_twh"], axis=0)
    totals = agg["generation_twh"]

    # Per-country shares should still sum to 1.
    assert np.allclose(shares.sum(axis=1), 1.0, atol=1e-6)
    return shares, totals


def main():
    shares, totals = load_and_aggregate(DATA_CSV)

    # Sort countries by total generation (biggest left).
    order = totals.sort_values(ascending=False).index
    shares = shares.loc[order]
    totals = totals.loc[order]

    fig, ax = plt.subplots(figsize=(14, 6.2))
    x = np.arange(len(order))
    bottom = np.zeros(len(order))
    segment_bounds = {}
    for col in SHARE_COLS:
        category = COL_TO_CATEGORY[col]
        values = shares[col].values * 100.0
        ax.bar(
            x,
            values,
            bottom=bottom,
            color=CATEGORY_COLORS[category],
            label=category,
            edgecolor="white",
            linewidth=0.6,
        )
        segment_bounds[col] = (bottom.copy(), bottom + values)
        bottom = bottom + values

    # Scheme-name labels inside the non-clawback segment.
    nc_bottom, nc_top = segment_bounds["share_nonclawback"]
    for xi, country in enumerate(order):
        height = nc_top[xi] - nc_bottom[xi]
        if height < MIN_LABEL_HEIGHT_PCT:
            continue
        scheme = DOMINANT_NONCLAWBACK_SCHEME.get(country)
        if not scheme:
            continue
        ax.text(
            xi,
            (nc_bottom[xi] + nc_top[xi]) / 2,
            scheme,
            ha="center",
            va="center",
            rotation=90,
            fontsize=7.5,
            color="#3d2e00",
        )

    # Annotate estimated generation above each bar.
    for xi, country in enumerate(order):
        ax.text(
            xi,
            101,
            f"{totals.loc[country]:.0f} TWh",
            ha="center",
            va="bottom",
            fontsize=7.5,
            color="#333",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(order, fontsize=9)
    ax.set_ylim(0, 110)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.set_yticklabels([f"{v}%" for v in np.arange(0, 101, 20)])
    ax.set_ylabel("Share of estimated 2030 RES + nuclear generation")
    ax.set_title(
        "2030 power generation by policy regime (RES + nuclear, TWh-weighted)",
        fontsize=11,
        loc="left",
        pad=26,
    )
    ax.text(
        0.0,
        1.012,
        "Yellow + grey segments: generator revenues exposed to wholesale prices (no clawback mechanism).",
        transform=ax.transAxes,
        fontsize=9,
        color="#555",
        ha="left",
        va="bottom",
        style="italic",
    )

    style_ax(ax)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=3,
        frameon=True,
        fontsize=9,
    )

    fig.tight_layout(rect=(0, 0.04, 1, 1))

    out_local = SCRIPT_DIR / "cfd_shares_by_country_2030.pdf"
    out_paper = REPO_ROOT.parent / "gas_resilience" / "imgs" / "cfd_shares_by_country_2030.pdf"
    fig.savefig(out_local, bbox_inches="tight")
    if out_paper.parent.exists():
        fig.savefig(out_paper, bbox_inches="tight")
        print(f"Saved: {out_paper}")
    else:
        print(f"Skipped paper copy — {out_paper.parent} does not exist")
    print(f"Saved: {out_local}")

    # Aggregate TWh per category across plotted countries.
    totals_by_cat = shares.multiply(totals, axis=0).sum(axis=0)
    print("\nAggregate generation per policy category (plotted countries):")
    for col in SHARE_COLS:
        print(f"  {COL_TO_CATEGORY[col]:>30s}: {totals_by_cat[col]:7.1f} TWh")
    print(f"  {'TOTAL':>30s}: {totals.sum():7.1f} TWh")

    export_frontend_data("cfd_shares_by_country_2030", {
        "x_label": "Country",
        "y_label": "Share of estimated 2030 RES + nuclear generation [%]",
        "category_order": CATEGORY_ORDER,
        "category_colors": CATEGORY_COLORS,
        "countries": list(order),
        "shares_pct": {
            COL_TO_CATEGORY[col]: (shares[col] * 100.0).tolist()
            for col in SHARE_COLS
        },
        "total_generation_TWh": {c: float(totals.loc[c]) for c in order},
        "dominant_nonclawback_scheme": {
            c: DOMINANT_NONCLAWBACK_SCHEME.get(c) for c in order
        },
        "aggregate_generation_TWh_by_category": {
            COL_TO_CATEGORY[col]: float(totals_by_cat[col]) for col in SHARE_COLS
        },
    })


if __name__ == "__main__":
    main()
