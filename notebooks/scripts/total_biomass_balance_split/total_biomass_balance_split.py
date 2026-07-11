"""Reproduce total_biomass_balance_split.pdf as a standalone script.

Converted from notebooks/plot_biomass_products.ipynb (cells 0-3).
Produces a 1x2 figure: solid biomass (left) and biogas (right) energy balance
across the gas-budget wiggle sweep, with consumer flows aggregated into named
end-use groups. Writes both the PDF (script dir + paper imgs/) and a JSON
dataset.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yaml
from matplotlib.patches import Patch

idx = pd.IndexSlice

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = Path(__file__).resolve().parent
SIBLING_IMGS = REPO_ROOT.parent / "gas_resilience" / "imgs"
SIBLING_IMGS.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(REPO_ROOT / "notebooks" / "scripts"))
from frontend_export import export_frontend_data  # noqa: E402
from forest_palette import pick_color  # noqa: E402

with open(REPO_ROOT / "config" / "plotting.default.yaml") as f:
    tech_colors = yaml.safe_load(f)["plotting"]["tech_colors"]

bal = pd.read_csv(
    REPO_ROOT / "results" / "csvs_3_latest" / "energy_balance.csv",
    header=list(range(5)), index_col=[0, 1, 2],
)
bal.columns = bal.columns.get_level_values(-1).astype(int)

# ── Plot ────────────────────────────────────────────────────────────────────
fig, (ax_bm, ax_bg) = plt.subplots(nrows=1, ncols=2, figsize=(16, 6), sharex=True)

for ax in (ax_bm, ax_bg):
    ax.set_facecolor("#fafafa")
    fig.patch.set_facecolor("white")
    ax.grid(axis="y", color="#dddddd", linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#666666")
    ax.spines["bottom"].set_color("#666666")
    ax.tick_params(colors="#444444", labelsize=10)

producer_handles_bm, producer_labels_bm = [], []
consumer_handles_bm, consumer_labels_bm = [], []
producer_handles_bg, producer_labels_bg = [], []
consumer_handles_bg, consumer_labels_bg = [], []

consumer_groups = {
    "biomass to liquid": "Feedstock uses",
    "biomass-to-methanol": "Feedstock uses",
    "solid biomass for industry": "Feedstock uses",
    "solid biomass for industry CC": "Feedstock uses",
    "biogas to gas": "Feedstock uses",
    "heat<100 industry solid biomass": "Industry heat <200°C",
    "heat100-200 industry solid biomass": "Industry heat <200°C",
    "heat200-500 industry solid biomass": "Industry heat 200–500°C",
    "urban central solid biomass CHP": "Central/district heating",
    "urban central solid biomass CHP CC": "Central/district heating",
    "rural biomass boiler": "Individual heating",
    "urban decentral biomass boiler": "Individual heating",
}

consumer_group_colors = {
    "Feedstock uses": "#6B8E23",
    "Industry heat <200°C": "#DAA520",
    "Industry heat 200–500°C": "#B22222",
    "Central/district heating": "#9400D3",
    "Individual heating": "#4682B4",
}

bg_idx = "biogas"
df_bg = bal.loc[idx[:, :, bg_idx]].droplevel(0)

bm_idx = "solid biomass"
df_bm = bal.loc[idx[:, :, bm_idx]].droplevel(0)

MWh_to_TWh = 1e-6
df_bg = df_bg * MWh_to_TWh
df_bm = df_bm * MWh_to_TWh

cols = df_bg.columns


def aggregate_consumers(df, groups):
    grouped = {}
    for tech in df.index.unique(0):
        data = df.loc[tech]
        neg = data.clip(upper=0)
        if (neg < 0).any():
            group = groups.get(tech, tech)
            if group not in grouped:
                grouped[group] = pd.Series(0.0, index=cols)
            grouped[group] += neg
    return grouped


consumer_bm = aggregate_consumers(df_bm, consumer_groups)
consumer_bg = aggregate_consumers(df_bg, consumer_groups)


def _plot_panel(ax, df, consumer_agg, prod_handles, prod_labels,
                cons_handles, cons_labels, agg_color, title):
    offset_pos = pd.Series(0.0, index=cols)
    offset_neg = pd.Series(0.0, index=cols)
    used_prod = {}
    producers_payload = {}
    for tech in df.index.unique(0):
        data = df.loc[tech]
        color = pick_color(tech, tech_colors)
        pos = data.clip(lower=0)
        if (pos > 0).any():
            ax.fill_between(
                cols, offset_pos.values + pos.values, offset_pos.values,
                color=color, alpha=0.75, linewidth=0.4,
                edgecolor="black", step="post",
            )
            if tech not in used_prod:
                prod_handles.append(Patch(facecolor=color, edgecolor="black", linewidth=0.4, alpha=0.75))
                prod_labels.append(tech)
                used_prod[tech] = True
            offset_pos += pos.values
            producers_payload[tech] = {
                "color": color,
                "values_TWh": pos.astype(float).tolist(),
            }
    agg_pos = offset_pos.copy()

    used_cons = set()
    consumers_payload = {}
    for group, vals in consumer_agg.items():
        color = consumer_group_colors.get(group, tech_colors.get(group, "#888888"))
        ax.fill_between(
            cols, offset_neg.values + vals.values, offset_neg.values,
            color=color, alpha=0.75, linewidth=0.4,
            edgecolor="black", step="post",
        )
        if group not in used_cons:
            cons_handles.append(Patch(facecolor=color, edgecolor="black", linewidth=0.4, alpha=0.75))
            cons_labels.append(group)
            used_cons.add(group)
        offset_neg += vals.values
        consumers_payload[group] = {
            "color": color,
            "values_TWh_negative": vals.astype(float).tolist(),
        }

    ax.fill_between(
        cols, agg_pos.values, 0,
        color=agg_color, alpha=0.12, zorder=1, step="post",
    )
    ax.axhline(0, color="#888888", linewidth=0.8, zorder=3)
    ax.set_xlim(cols[0], cols[-1])
    ax.set_title(title)
    # Gas consumption axis in bcm (10 TWh per bcm); data stays in TWh.
    from matplotlib.ticker import FuncFormatter as _FuncFormatter
    ax.xaxis.set_major_formatter(_FuncFormatter(lambda v, _pos: f"{v / 10:.0f}"))
    ax.set_xlabel("Gas consumption constraint [bcm]", fontsize=11, color="#333333")
    ax.set_ylabel("Energy [TWh]", fontsize=11, color="#333333")

    return {"producers": producers_payload, "consumers_grouped": consumers_payload}


payload_bm = _plot_panel(
    ax_bm, df_bm, consumer_bm,
    producer_handles_bm, producer_labels_bm,
    consumer_handles_bm, consumer_labels_bm,
    tech_colors.get("solid_biomass_agg", "#B8860B"),
    "Solid biomass",
)
ax_bm.legend(
    producer_handles_bm + consumer_handles_bm,
    producer_labels_bm + consumer_labels_bm,
    loc="lower center", bbox_to_anchor=(0.5, -0.45),
    ncol=2, fontsize=8, frameon=False,
    columnspacing=1.0, handletextpad=0.5,
)

payload_bg = _plot_panel(
    ax_bg, df_bg, consumer_bg,
    producer_handles_bg, producer_labels_bg,
    consumer_handles_bg, consumer_labels_bg,
    tech_colors.get("biogas_agg", "#20639B"),
    "Biogas",
)
ax_bg.legend(
    producer_handles_bg + consumer_handles_bg,
    producer_labels_bg + consumer_labels_bg,
    loc="lower center", bbox_to_anchor=(0.5, -0.32),
    ncol=2, fontsize=8, frameon=False,
    columnspacing=1.0, handletextpad=0.5,
)

plt.tight_layout()
plt.subplots_adjust(bottom=0.37)

out_local = SCRIPT_DIR / "total_biomass_balance_split.pdf"
out_paper = SIBLING_IMGS / "total_biomass_balance_split.pdf"
plt.savefig(out_local, bbox_inches="tight")
plt.savefig(out_paper, bbox_inches="tight")
print(f"wrote {out_local}")
print(f"wrote {out_paper}")

export_frontend_data("total_biomass_balance_split", {
    "x_label": "Gas consumption constraint [TWh]",
    "y_label": "Energy [TWh]",
    "wiggles_TWh": [int(c) for c in cols],
    "consumer_groups_mapping": consumer_groups,
    "consumer_group_colors": consumer_group_colors,
    "panels": {
        "solid_biomass": payload_bm,
        "biogas": payload_bg,
    },
})
