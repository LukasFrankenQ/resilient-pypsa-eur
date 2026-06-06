"""Constraint-aware LCOH for UDH supply techs.

For each tech, compute Levelised Cost of Heat as a function of annual
full-load hours `H` (heat output basis). Includes:

  - own capex (annualised) + FOM
  - own marginal cost: (fuel + carbon × CO2_intensity) / efficiency
  - induced infrastructure cost for LV-coupled techs (HP, resistive heater):
      * distribution-grid capex sized to peak input dispatch
      * marginal electricity supplied by gas CCGT at the cold-snap hour
        (gas + carbon × CO2_gas) / η_CCGT  +  CCGT VOM and capex per MWh
  - separate curves for HP at cold-snap COP (≈ 1.8) and at annual-mean COP (≈ 3.6)

The tradeoff this surfaces: HP is much cheaper per MWh heat at high CF
because it's dominated by capex; oil boiler beats HP at low CF (peaking)
because oil's capex is tiny. With the distribution + cold-snap COP penalty
the crossover shifts to longer run-hours, making oil more attractive for
PEAKING service even at 100 EUR/tCO2.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[2]
PAPER_DIR = ROOT.parent / "gas_resilience" / "imgs" / "infeasibility_check"
PAPER_DIR.mkdir(parents=True, exist_ok=True)

with open(ROOT / "config.basicrun.yaml") as f:
    _cfg = yaml.safe_load(f)
TECH_COLORS = _cfg["plotting"]["tech_colors"]


def style_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)


# ── Costs (from resources/costs_2030.csv) ────────────────────────────────────
COSTS = pd.read_csv(ROOT / "resources" / "costs_2030.csv")


def get(tech, param, default=np.nan):
    row = COSTS[(COSTS.technology == tech) & (COSTS.parameter == param)]
    return float(row.value.iloc[0]) if len(row) else default


def crf(rate, life):
    return rate / (1 - (1 + rate) ** -life)


def annualised_capex_per_kW(tech_name, dr=0.07):
    """Annualised capex (EUR/kW_th/yr), incl. FOM."""
    inv = get(tech_name, "investment")  # EUR/kWth (or EUR/kW)
    fom = get(tech_name, "FOM", 0.0) / 100.0
    life = get(tech_name, "lifetime", 20.0)
    return inv * (crf(dr, life) + fom)


# Per-tech ingredients ----------------------------------------------------------
DR = 0.07
CO2_PRICE = 100.0  # EUR/tCO2 — current d30d9041 value
GAS_PRICE = get("gas", "fuel")
GAS_CO2 = get("gas", "CO2 intensity")
OIL_PRICE = get("oil", "fuel")
OIL_CO2 = get("oil", "CO2 intensity")

ETA_CCGT = 0.55  # nominal efficiency for CCGT (electric)
CCGT_VOM = get("CCGT", "VOM", 4.0)
CCGT_CAPEX_KW = get("CCGT", "investment", 800.0)
CCGT_FOM = get("CCGT", "FOM", 2.0) / 100.0
CCGT_LIFE = get("CCGT", "lifetime", 30.0)
CCGT_CAPEX_ANN = CCGT_CAPEX_KW * (crf(DR, CCGT_LIFE) + CCGT_FOM)  # EUR/kW/yr

# Distribution grid: 50,270 EUR/MW/yr (already annualised in network)
DIST_INV = get("electricity distribution grid", "investment") * 1000  # EUR/kW → EUR/MW
DIST_FOM = get("electricity distribution grid", "FOM") / 100.0
DIST_LIFE = get("electricity distribution grid", "lifetime")
DIST_CAPEX_ANN = DIST_INV * (crf(DR, DIST_LIFE) + DIST_FOM)  # EUR/MW/yr

# Per-MWh-elec marginal cost from gas CCGT at cold-snap dispatch
ELEC_MARGINAL_COLD = (GAS_PRICE + CO2_PRICE * GAS_CO2) / ETA_CCGT + CCGT_VOM
# Add CCGT capex amortised over the cold-snap operating hours of the CCGT itself
# We assume CCGT runs ~3000 hrs/yr → CCGT capex per MWh elec
CCGT_PEAKING_HR = 2000
CCGT_CAPEX_PER_MWH_ELEC = CCGT_CAPEX_ANN * 1000 / CCGT_PEAKING_HR / 1000  # EUR/MWh
# Total elec marginal incl CCGT amortisation at peak operation:
ELEC_PEAK_TOTAL = ELEC_MARGINAL_COLD + CCGT_CAPEX_PER_MWH_ELEC

# Marginal-only (no CCGT capex, mid-merit gas dispatch)
ELEC_MARGINAL_AVG = ELEC_MARGINAL_COLD  # same fuel mix at margin

print("=== Cost inputs ===")
print(f"  CO2 price          = {CO2_PRICE:.1f} EUR/tCO2 (current d30d9041)")
print(f"  Gas price          = {GAS_PRICE:.2f} EUR/MWh_th")
print(f"  Oil price          = {OIL_PRICE:.2f} EUR/MWh_th")
print(f"  Carbon adder gas   = {CO2_PRICE * GAS_CO2:.2f} EUR/MWh_fuel")
print(f"  Carbon adder oil   = {CO2_PRICE * OIL_CO2:.2f} EUR/MWh_fuel")
print(f"  CCGT marginal elec = {ELEC_MARGINAL_COLD:.2f} EUR/MWh_el")
print(f"  CCGT capex add     = {CCGT_CAPEX_PER_MWH_ELEC:.2f} EUR/MWh_el (over {CCGT_PEAKING_HR} hrs)")
print(f"  Distribution capex = {DIST_CAPEX_ANN:,.0f} EUR/MW/yr (per MW LV peak)")


# ── Tech stack ────────────────────────────────────────────────────────────────
def lcoh_oil(H):
    capex = annualised_capex_per_kW("decentral oil boiler") * 1000  # EUR/MW/yr
    eta = get("decentral oil boiler", "efficiency")
    fuel_carbon = (OIL_PRICE + CO2_PRICE * OIL_CO2) / eta
    return capex / H + fuel_carbon


def lcoh_gas(H):
    capex = annualised_capex_per_kW("decentral gas boiler") * 1000
    eta = get("decentral gas boiler", "efficiency")
    fuel_carbon = (GAS_PRICE + CO2_PRICE * GAS_CO2) / eta
    return capex / H + fuel_carbon


def lcoh_hp(H, cop, elec_cost, include_dist=True):
    capex = annualised_capex_per_kW("decentral air-sourced heat pump") * 1000
    if include_dist:
        # Per MW heat capacity, peak distribution requirement = 1/cop_at_peak.
        # Use cold-snap COP (1.8) because that's what sizes the distribution.
        capex += DIST_CAPEX_ANN / 1.8
    marginal = elec_cost / cop
    return capex / H + marginal


def lcoh_resistive(H, elec_cost, include_dist=True):
    capex = annualised_capex_per_kW("decentral resistive heater") * 1000
    if include_dist:
        # Per MW heat capacity, peak distribution requirement = 1/0.9.
        capex += DIST_CAPEX_ANN / 0.9
    marginal = elec_cost / 0.9
    return capex / H + marginal


# ── Sweep ─────────────────────────────────────────────────────────────────────
H = np.linspace(50, 8000, 400)

curves = {
    "oil boiler":            lcoh_oil(H),
    "gas boiler":            lcoh_gas(H),
    "HP @ avg COP 3.6 (avg elec mix)": lcoh_hp(H, 3.6, ELEC_MARGINAL_AVG),
    "HP @ cold-snap COP 1.8 (CCGT margin + capex)": lcoh_hp(H, 1.8, ELEC_PEAK_TOTAL),
    "resistive heater (CCGT margin + capex)":  lcoh_resistive(H, ELEC_PEAK_TOTAL),
}

print("\n=== LCOH @ representative operating hours (EUR/MWh heat) ===")
for label_H in [200, 500, 1000, 2000, 4000]:
    parts = {k: f(label_H) if callable(f) else np.interp(label_H, H, f) for k, f in
             [(k, v) for k, v in curves.items()]}
    parts = {k: v for k, v in parts.items()}
    print(f"\nat H = {label_H} hrs/yr (full-load):")
    for k, lc in [(k, np.interp(label_H, H, v)) for k, v in curves.items()]:
        print(f"  {k:<55s} {lc:>7.1f}")

# Crossover oil ↔ HP cold-snap
def crossover(H, c1, c2):
    diff = c1 - c2
    sign = np.sign(diff)
    cross = np.where(np.diff(sign) != 0)[0]
    return H[cross[0]] if len(cross) else None

co_cold = crossover(H, lcoh_hp(H, 1.8, ELEC_PEAK_TOTAL), lcoh_oil(H))
co_avg = crossover(H, lcoh_hp(H, 3.6, ELEC_MARGINAL_AVG), lcoh_oil(H))
print(f"\nCrossover HP@cold/oil: H ≈ {co_cold:.0f} hrs/yr" if co_cold else "  no cold crossover")
print(f"Crossover HP@avg /oil: H ≈ {co_avg:.0f} hrs/yr" if co_avg else "  no avg crossover")


# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))

color_map = {
    "oil boiler":                                       TECH_COLORS["urban decentral oil boiler"],
    "gas boiler":                                       TECH_COLORS["urban decentral gas boiler"],
    "HP @ avg COP 3.6 (avg elec mix)":                  TECH_COLORS["urban decentral air heat pump"],
    "HP @ cold-snap COP 1.8 (CCGT margin + capex)":     "#1a5e1a",   # darker green
    "resistive heater (CCGT margin + capex)":           TECH_COLORS["urban decentral resistive heater"],
}
ls_map = {
    "HP @ avg COP 3.6 (avg elec mix)":                  "--",
    "HP @ cold-snap COP 1.8 (CCGT margin + capex)":     "-",
    "resistive heater (CCGT margin + capex)":           "-",
    "oil boiler":                                       "-",
    "gas boiler":                                       "-",
}
for label, lc in curves.items():
    ax.plot(H, lc, label=label, color=color_map[label],
            linestyle=ls_map[label], linewidth=2.0, alpha=0.95)

# Crossover markers
if co_cold:
    ax.axvline(co_cold, color="#666", linestyle=":", linewidth=0.8, alpha=0.7)
    ax.text(co_cold, 320, f" oil = HP(cold)\n @ {co_cold:.0f} h/yr",
            fontsize=8, color="#333", va="top")

# Annotate operating-hour ranges typical for UDH peaker vs baseload
ax.axvspan(200, 1500, color="#e8d8c8", alpha=0.25, label="UDH peaker range")
ax.axvspan(2500, 5000, color="#d4e5d6", alpha=0.25, label="UDH baseload range")

ax.set_xlabel("annual full-load hours of heat output [h/yr]")
ax.set_ylabel("LCOH [EUR/MWh heat]")
ax.set_ylim(0, 400)
ax.set_xlim(0, 8000)
ax.set_title(
    "Constraint-aware LCOH for UDH supply techs (3H, wiggle 1000, CO₂ = 100 €/t)\n"
    "HP & resistive include distribution capex (sized to LV peak) and CCGT-marginal electricity cost",
    fontsize=10, loc="left",
)
ax.legend(frameon=True, fontsize=8, loc="upper right")
style_ax(ax)
fig.tight_layout()
out = PAPER_DIR / "lcoh_hp_vs_oil.pdf"
fig.savefig(out)
plt.close(fig)
print(f"\n  wrote {out}")
