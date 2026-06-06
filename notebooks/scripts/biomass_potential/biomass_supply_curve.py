import sys
import pypsa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "notebooks" / "scripts"))
from frontend_export import export_frontend_data  # noqa: E402
from forest_palette import pick_color  # noqa: E402

# Load tech colors from config
with open(ROOT / "config.basicrun.yaml") as f:
    config = yaml.safe_load(f)
tech_colors = config["plotting"]["tech_colors"]

# Load network (wiggle=0)
n = pypsa.Network(ROOT / "results/networks/base_s_50_lv1.25_3H-T-H-B-I-A-dist1_2030_free_0.nc")

# Filter generators on solid biomass buses
gens = n.generators[n.generators.bus.str.contains("solid biomass", na=False)].copy()

# Keep only finite, positive e_sum_max
gens = gens[np.isfinite(gens.e_sum_max) & (gens.e_sum_max > 0)]

# Aggregate by carrier: sum potential, mean marginal cost
agg = gens.groupby("carrier").agg(
    e_sum_max=("e_sum_max", "sum"),
    marginal_cost=("marginal_cost", "mean"),
).sort_values("marginal_cost")

# Convert MWh -> TWh
agg["e_sum_max_TWh"] = agg["e_sum_max"] / 1e6

# Build supply curve steps
fig, ax = plt.subplots(figsize=(8, 5))

cumulative = 0
for carrier, row in agg.iterrows():
    width = row["e_sum_max_TWh"]
    color = pick_color(carrier, tech_colors)
    x = [cumulative, cumulative + width]
    ax.fill_between(x, 0, row["marginal_cost"], color=color, label=carrier)
    cumulative += width

ax.set_xlabel("Biomass potential [TWh]")
ax.set_ylabel("Marginal cost [EUR/MWh]")
ax.legend(loc="lower right", frameon=True)
ax.set_title("European solid biomass supply curve")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.yaxis.grid(True, alpha=0.3)
ax.set_axisbelow(True)

plt.tight_layout()
out = ROOT.parent / "gas_resilience" / "imgs" / "biomass_supply_curve.pdf"
fig.savefig(out, bbox_inches='tight')
print(f"Saved to {out}")

# Frontend data export: stepwise supply curve as ordered carrier blocks
_steps = []
_cum = 0.0
for _carrier, _row in agg.iterrows():
    _width = float(_row["e_sum_max_TWh"])
    _steps.append({
        "carrier": _carrier,
        "x_start_TWh": _cum,
        "x_end_TWh": _cum + _width,
        "width_TWh": _width,
        "marginal_cost_EUR_per_MWh": float(_row["marginal_cost"]),
        "color": pick_color(_carrier, tech_colors),
    })
    _cum += _width

export_frontend_data("biomass_supply_curve", {
    "x_label": "Biomass potential [TWh]",
    "y_label": "Marginal cost [EUR/MWh]",
    "title": "European solid biomass supply curve",
    "steps": _steps,
})

plt.show()
