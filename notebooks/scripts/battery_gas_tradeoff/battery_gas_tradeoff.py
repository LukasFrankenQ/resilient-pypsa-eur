"""Reproduce battery_gas_tradeoff.pdf as a standalone script.

Converted from notebooks/plot_time_series_of_energy_balances.ipynb (cells 0-5).
For each gas-budget wiggle (0..4000 step 250 TWh) loads the solved 3H 2030
'free' network, extracts daily-aggregated AC + low-voltage energy balance, and
keeps two carriers: combined battery dischargers and combined-cycle gas. Plots
side-by-side violins per wiggle and writes both the PDF and a JSON dataset.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pypsa
import yaml
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[3]
NETWORK_DIR = REPO_ROOT / "results" / "networks"
SIBLING_IMGS = REPO_ROOT.parent / "gas_resilience" / "imgs"
SIBLING_IMGS.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(REPO_ROOT / "notebooks" / "scripts"))
from frontend_export import export_frontend_data  # noqa: E402

PREFIX = "base_s_50_lv1.25_3H-T-H-B-I-A-dist1_2030_free"
WIGGLES = list(range(0, 4250, 250))

with open(REPO_ROOT / "config.basicrun.yaml") as f:
    tech_colors = yaml.safe_load(f)["plotting"]["tech_colors"]

battery_techs = ["battery discharger", "home battery discharger"]
gaspower_tech = "Combined-Cycle Gas"

b_data: list[pd.Series] = []
g_data: list[pd.Series] = []

for wiggle in tqdm(WIGGLES):
    path = NETWORK_DIR / f"{PREFIX}_{wiggle}.nc"
    if not path.exists():
        print(f"missing: {path.name} — skipping")
        continue
    n = pypsa.Network(path)
    balance = n.statistics.energy_balance(
        aggregate_time=False, nice_names=True
    )
    balance = balance.loc[
        balance.index.get_level_values("bus_carrier").isin(["AC", "low voltage"])
    ]
    balance = balance.loc[
        balance.index.get_level_values(1) != "electricity distribution grid"
    ].mul(1e-3)
    balance.index = balance.index.get_level_values(1)
    balance = balance.T
    balance.index = balance.index.strftime("%Y-%m-%d")
    balance = balance.groupby(balance.index).sum()

    batteries = balance[battery_techs].sum(axis=1).rename(wiggle) * 3
    gaspower = balance[[gaspower_tech]].sum(axis=1).rename(wiggle) * 3

    b_data.append(batteries)
    g_data.append(gaspower)

b_data = pd.concat(b_data, axis=1)
g_data = pd.concat(g_data, axis=1)

# ── Plot ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 4))

offset = 0.15
battery_color = tech_colors.get("shipping", "tab:blue")
gaspower_color = tech_colors.get(gaspower_tech, "tab:orange")

for i, col in enumerate(b_data.columns):
    x_pos = i
    b_vals = b_data[col].dropna().values
    g_vals = g_data[col].dropna().values

    parts1 = ax.violinplot(b_vals, positions=[x_pos - offset], widths=0.25,
                           showmeans=False, showextrema=False, showmedians=True)
    for pc in parts1["bodies"]:
        pc.set_facecolor(battery_color)
        pc.set_alpha(0.9)
    if "cmedians" in parts1:
        parts1["cmedians"].set_color("black")
        parts1["cmedians"].set_linewidth(2)

    parts2 = ax.violinplot(g_vals, positions=[x_pos + offset], widths=0.25,
                           showmeans=False, showextrema=False, showmedians=True)
    for pc in parts2["bodies"]:
        pc.set_facecolor(gaspower_color)
        pc.set_alpha(0.9)
    if "cmedians" in parts2:
        parts2["cmedians"].set_color("black")
        parts2["cmedians"].set_linewidth(2)

xtick_positions = list(range(len(b_data.columns)))
xtick_labels = [
    str(col) if i % 2 == 0 else "" for i, col in enumerate(b_data.columns)
]
ax.set_xticks(xtick_positions)
ax.set_xticklabels(xtick_labels, rotation=0)
ax.set_xlabel("Total Gas Consumption [TWh]")
ax.set_ylabel("Total Daily Energy Output [GWh]")

battery_patch = mpatches.Patch(color=battery_color, label="Batteries")
gaspower_patch = mpatches.Patch(color=gaspower_color, label="Gas power")
ax.legend(handles=[battery_patch, gaspower_patch])

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(True, axis="y", linestyle="--", alpha=0.7)
ax.set_axisbelow(True)

plt.tight_layout()
out = SIBLING_IMGS / "battery_gas_tradeoff.pdf"
plt.savefig(out, bbox_inches="tight")
print(f"wrote {out}")


def _summarise(arr):
    a = np.asarray(arr, float)
    a = a[np.isfinite(a)]
    if len(a) == 0:
        return {"n": 0}
    return {
        "n": int(len(a)),
        "min": float(a.min()),
        "max": float(a.max()),
        "mean": float(a.mean()),
        "median": float(np.median(a)),
        "p05": float(np.percentile(a, 5)),
        "p25": float(np.percentile(a, 25)),
        "p75": float(np.percentile(a, 75)),
        "p95": float(np.percentile(a, 95)),
    }


export_frontend_data("battery_gas_tradeoff", {
    "x_label": "Total Gas Consumption [TWh]",
    "y_label": "Total Daily Energy Output [GWh]",
    "wiggles_TWh": [int(c) for c in b_data.columns],
    "colors": {"batteries": battery_color, "gas_power": gaspower_color},
    "batteries_daily_GWh": {
        int(c): b_data[c].dropna().astype(float).tolist() for c in b_data.columns
    },
    "gas_power_daily_GWh": {
        int(c): g_data[c].dropna().astype(float).tolist() for c in g_data.columns
    },
    "summary": {
        "batteries": {
            int(c): _summarise(b_data[c].values) for c in b_data.columns
        },
        "gas_power": {
            int(c): _summarise(g_data[c].values) for c in g_data.columns
        },
    },
})
