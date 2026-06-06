"""
Stacked bar chart of optimal installed power-sector capacities across the
gas-budget (`wiggle`) sweep for the 3H, lv1.25, free-scenario networks.

Iterates over networks at wiggle = 0, 250, ..., 4000 and queries
`n.statistics.optimal_capacity(bus_carrier="AC")` to get capacity by carrier
(MW), converts to GW, and renders one stacked bar per wiggle.

Outputs are written next to this script and copied to ../../../../gas_resilience/imgs/.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pypsa


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
NETWORK_DIR = ROOT / "results" / "networks"
SIBLING_IMGS = ROOT.parent / "gas_resilience" / "imgs"

PREFIX = "base_s_50_lv1.25_3H-T-H-B-I-A-dist1_2030_free"
WIGGLES = list(range(0, 4001, 250))


def network_path(wiggle: int) -> Path:
    return NETWORK_DIR / f"{PREFIX}_{wiggle}.nc"


# ============================================================
# TECH COLORS — load from config, with sensible fallbacks for
# carriers that share visuals with siblings already in the map.
# ============================================================

with open(ROOT / "config.basicrun.yaml") as f:
    tech_colors = yaml.safe_load(f)["plotting"]["tech_colors"]

# Aliases that mirror those used in main_results.py for power-sector carriers.
tech_colors.setdefault("solar", tech_colors.get("solar-hsat", "#f9d423"))
tech_colors.setdefault("wind", tech_colors.get("Onshore Wind", "#235ebc"))


# ============================================================
# THEMATIC ORDER FOR POWER-SECTOR CARRIERS
# ============================================================
# Each tuple: (group label, predicate on the carrier name).
# Carriers are grouped in this order; within a group they are
# sorted by total capacity descending.

GROUPS: list[tuple[str, callable]] = [
    ("solar",   lambda c: c.lower().startswith("solar")),
    ("wind",    lambda c: "wind" in c.lower()),
    ("hydro",   lambda c: c in {"ror", "hydro", "PHS"}
                          or "hydro" in c.lower()
                          or "run of river" in c.lower()
                          or "reservoir" in c.lower()),
    ("nuclear", lambda c: "nuclear" in c.lower()),
    ("battery", lambda c: "battery" in c.lower() or c == "V2G"),
    ("biomass", lambda c: "biomass" in c.lower() or "biogas" in c.lower()),
    ("fossil",  lambda c: c in {"Combined-Cycle Gas", "Open-Cycle Gas",
                                 "coal", "lignite", "oil", "CCGT", "OCGT",
                                 "gas turbines"}
                          or "gas chp" in c.lower()
                          or "coal" in c.lower()
                          or "lignite" in c.lower()),
]


def assign_group(carrier: str) -> str | None:
    for label, pred in GROUPS:
        if pred(carrier):
            return label
    return None


# ============================================================
# COLLECT OPTIMAL CAPACITIES
# ============================================================

records: dict[int, pd.Series] = {}
for wiggle in WIGGLES:
    fn = network_path(wiggle)
    if not fn.exists():
        print(f"  [skip] {fn.name} not found")
        continue
    print(f"  loading {fn.name}")
    n = pypsa.Network(fn)
    cap = n.statistics.optimal_capacity(bus_carrier="AC")  # MW
    cap = cap.groupby("carrier").sum()
    cap = cap[cap > 0]
    records[wiggle] = cap.div(1e3)  # GW

caps = pd.DataFrame(records).fillna(0.0)  # rows: carrier, cols: wiggle


# ============================================================
# CARRIER DROPS
# ============================================================
# Carriers that are present in the data but should not appear
# in the figure (negligible / out of scope for this view).

CARRIER_DROPS = ["Offshore Wind (Floating)"]
caps = caps.drop(index=[c for c in CARRIER_DROPS if c in caps.index])


# ============================================================
# CARRIER GROUP-SUMS
# ============================================================
# Merge variants that should appear as a single legend entry
# and a single colored stack segment.

CARRIER_MERGES: dict[str, list[str]] = {
    "Solar": ["solar", "Solar", "solar-hsat", "solar rooftop"],
    "Offshore Wind": ["Offshore Wind (AC)", "Offshore Wind (DC)"],
    "hydropower": ["ror", "hydro", "PHS",
                   "Run of River", "Pumped Hydro Storage", "Reservoir & Dam"],
    "biomass CHP": ["urban central solid biomass CHP",
                    "urban central solid biomass CHP CC"],
    "gas turbines": ["Combined-Cycle Gas", "Open-Cycle Gas"],
    "battery": ["battery discharger"],
}

for merged, parts in CARRIER_MERGES.items():
    present = [p for p in parts if p in caps.index]
    if not present:
        continue
    summed = caps.loc[present].sum(axis=0)
    caps = caps.drop(index=present)
    caps.loc[merged] = summed

# Pick colors for the merged carriers — fall back to the first part's colour.
for merged, parts in CARRIER_MERGES.items():
    if merged in tech_colors:
        continue
    for p in parts:
        if p in tech_colors:
            tech_colors[merged] = tech_colors[p]
            break


# ============================================================
# THEMATIC ORDERING
# ============================================================

group_of = {c: assign_group(c) for c in caps.index}
unmatched = [c for c, g in group_of.items() if g is None]
if unmatched:
    print(f"  dropping non-power-sector carriers: {unmatched}")
caps = caps.loc[[c for c in caps.index if group_of[c] is not None]]

group_order = [g for g, _ in GROUPS]
totals = caps.sum(axis=1)
sort_keys = pd.DataFrame({
    "group_idx": [group_order.index(group_of[c]) for c in caps.index],
    "total":     -totals.values,
}, index=caps.index)
caps = caps.loc[sort_keys.sort_values(["group_idx", "total"]).index]

missing = [c for c in caps.index if c not in tech_colors]
if missing:
    print(f"  warning: tech_colors missing for {missing} — using grey fallback")
colors = [tech_colors.get(c, "#999999") for c in caps.index]


# ============================================================
# PLOT
# ============================================================

fig, ax = plt.subplots(figsize=(12, 6.5))

x = np.array(caps.columns, dtype=float)
bar_width = 180.0
bottom = pd.Series(0.0, index=caps.columns)
for carrier, color in zip(caps.index, colors):
    y = caps.loc[carrier]
    ax.bar(x, y.values, bottom=bottom.values, color=color, width=bar_width,
           edgecolor="black", linewidth=0.6, label=carrier)
    bottom += y

ax.set_xlabel("Total Net Gas Consumption [TWh/a]")
ax.set_ylabel("Optimal installed capacity [GW]")

# Show only every second xticklabel, horizontal.
labels = [str(int(w)) if i % 2 == 0 else "" for i, w in enumerate(x)]
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=0)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.yaxis.grid(True, alpha=0.3)
ax.set_axisbelow(True)


handles, labels_ = ax.get_legend_handles_labels()
ax.legend(handles[::-1], labels_[::-1],
          frameon=True, fontsize=8, ncol=1,
          bbox_to_anchor=(1.02, 0.5), loc="center left",
          title="Carrier", title_fontsize=9)

plt.tight_layout()

out = HERE / "optimal_installed_capacities_power_sector.pdf"
plt.savefig(out, bbox_inches="tight")
SIBLING_IMGS.mkdir(parents=True, exist_ok=True)
shutil.copy(out, SIBLING_IMGS / "optimal_installed_capacities_power_sector.pdf")
print(f"Wrote {out}")
print(f"Wrote {SIBLING_IMGS / 'optimal_installed_capacities_power_sector.pdf'}")
