"""Total system cost vs. gas consumption for four gas-price sweeps
(baseline + m=1.25 / 1.5 / 1.75 gas-marginal-cost multipliers).

Only wiggle values available in all four sweeps are used, capped at
4000 TWh. For each sweep a smooth curve is fitted whose slope is
monotonically increasing (p'' >= 0) and whose curvature decreases
(p''' <= 0), i.e. |p''| larger on the left. The curve is flexible
enough to essentially interpolate all data points. Each curve's
fitted minimum is marked; the four minima are connected with a
dashed black line. The upper panel shows full system cost incl.
actual ETS payments (`system cost`); the lower panel shows the
resource cost with ETS payments excluded (`resource cost`).
"""
import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pypsa
from matplotlib.collections import LineCollection
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize, minimize_scalar

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from frontend_export import export_frontend_data  # noqa: E402
NET_DIR = ROOT / "results" / "networks"
IMG_DIR = Path.cwd().parent / "gas_resilience" / "imgs"
IMG_DIR.mkdir(parents=True, exist_ok=True)
CACHE_PATH = Path(__file__).resolve().parent / ".cache" / "cost_vs_gas_consumption.json"
CACHE_VERSION = 6  # bump to invalidate cache on schema changes

WIGGLE_MAX = 4000
FIT_DEGREE = 6  # 7 coeffs vs ~9 data points => near-interpolation

SWEEPS = [
    {"key": "base",
     "pattern": re.compile(
         r"^base_s_50_lv1.25_3H-T-H-B-I-A-dist1_2030_free_(\d+)\.nc$"),
     "color": "#3a6ea5"},
    {"key": "m1.25",
     "pattern": re.compile(
         r"^base_s_50_lv1.25_3H-T-H-B-I-A-dist1-gas\+Generator\+m1\.25_2030_free_(\d+)\.nc$"),
     "color": "#f1c542"},
    {"key": "m1.5",
     "pattern": re.compile(
         r"^base_s_50_lv1.25_3H-T-H-B-I-A-dist1-gas\+Generator\+m1\.5_2030_free_(\d+)\.nc$"),
     "color": "#e08537"},
    {"key": "m1.75",
     "pattern": re.compile(
         r"^base_s_50_lv1.25_3H-T-H-B-I-A-dist1-gas\+Generator\+m1\.75_2030_free_(\d+)\.nc$"),
     "color": "#c42c2c"},
    {"key": "m2.0",
     "pattern": re.compile(
         r"^base_s_50_lv1.25_3H-T-H-B-I-A-dist1-gas\+Generator\+m2\.0_2030_free_(\d+)\.nc$"),
     "color": "#7a0f1a"},
]


def total_cost_be(n):
    capex = n.statistics.capex()
    opex = n.statistics.opex()
    capex = capex[capex.index.get_level_values("carrier") != "co2-ets"]
    opex = opex[opex.index.get_level_values("carrier") != "co2-ets"]
    # methanol Stores hit degenerate LP optima with huge unused e_nom_opt
    def drop_meoh_store(s):
        comp = s.index.get_level_values("component")
        carr = s.index.get_level_values("carrier")
        return s[~((comp == "Store") & carr.str.contains("methanol", case=False, na=False))]
    capex = drop_meoh_store(capex)
    opex = drop_meoh_store(opex)
    return (capex.sum() + opex.sum()) / 1e9


def co2_ets_cost_be(n):
    """Total capex + opex of components with carrier 'co2-ets' (excluded
    from total_cost_be) in B EUR."""
    capex = n.statistics.capex()
    opex = n.statistics.opex()
    capex = capex[capex.index.get_level_values("carrier") == "co2-ets"]
    opex = opex[opex.index.get_level_values("carrier") == "co2-ets"]
    return (capex.sum() + opex.sum()) / 1e9


def gas_marginal_cost(n):
    g = n.generators[n.generators.carrier == "gas"]
    return float(g.marginal_cost.mean()) if not g.empty else np.nan


def co2_price_eur_per_t(n):
    """Carbon price (~100 €/t). co2-ets generators store this as marginal_cost with
    negative sign by PyPSA-Eur convention, so take the absolute value."""
    g = n.generators[n.generators.carrier == "co2-ets"]
    return abs(float(g.marginal_cost.mean())) if not g.empty else np.nan


def gas_co2_intensity_t_per_MWh(n):
    """t CO2 / MWh_th of pure gas combustion (~0.198). Read from OCGT.efficiency2
    (bus2 = 'co2 atmosphere') — canonical reference, no carbon capture.
    """
    ocgt = n.links[n.links.carrier == "OCGT"]
    return float(ocgt.efficiency2.iloc[0]) if not ocgt.empty else np.nan


def gas_avg_marginal_price(n):
    """Withdrawal-weighted average marginal_price at carrier='gas' buses.
    Gas buses are consumed by Links (bus0=gas bus), so weight by Link p0.
    """
    gas_buses = n.buses.index[n.buses.carrier == "gas"]
    gas_buses = gas_buses.intersection(n.buses_t.marginal_price.columns)
    if gas_buses.empty:
        return np.nan
    prices = n.buses_t.marginal_price[gas_buses]
    weights_t = n.snapshot_weightings.generators
    consuming = n.links[n.links.bus0.isin(gas_buses)]
    load_ts = pd.DataFrame(0.0, index=n.snapshots, columns=gas_buses)
    for ln, row in consuming.iterrows():
        if ln in n.links_t.p0.columns:
            load_ts[row.bus0] += n.links_t.p0[ln]
    num = (prices * load_ts).multiply(weights_t, axis=0).sum().sum()
    den = load_ts.multiply(weights_t, axis=0).sum().sum()
    return float(num / den) if den > 0 else np.nan


def _none_if_nan(x):
    return None if (x is None or np.isnan(x)) else float(x)


def load_cache():
    if CACHE_PATH.exists():
        data = json.loads(CACHE_PATH.read_text())
        if data.get("version") == CACHE_VERSION:
            return data
    return {"version": CACHE_VERSION, "entries": {}}


def save_cache(cache):
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(cache, indent=2, sort_keys=True))


def get_metrics(p, cache):
    """Return cached record for network at `p`, recomputing on (mtime, size) change.
    Record contains: cost_BEUR, gas_marginal_cost, gas_marginal_price.
    """
    entries = cache["entries"]
    st = p.stat()
    entry = entries.get(p.name)
    if entry and entry["mtime"] == st.st_mtime and entry["size"] == st.st_size:
        return entry
    print(f"  loading {p.name}")
    n = pypsa.Network(p)
    rec = {
        "mtime": st.st_mtime,
        "size": st.st_size,
        "cost_BEUR": float(total_cost_be(n)),
        "co2_ets_cost_BEUR": float(co2_ets_cost_be(n)),
        "gas_marginal_cost": _none_if_nan(gas_marginal_cost(n)),
        "gas_marginal_price": _none_if_nan(gas_avg_marginal_price(n)),
        "co2_price_EUR_per_t": _none_if_nan(co2_price_eur_per_t(n)),
        "gas_co2_intensity_t_per_MWh": _none_if_nan(gas_co2_intensity_t_per_MWh(n)),
    }
    entries[p.name] = rec
    return rec


def discover_wiggles(pattern):
    ws = set()
    for p in NET_DIR.iterdir():
        m = pattern.match(p.name)
        if m:
            w = int(m.group(1))
            if w <= WIGGLE_MAX:
                ws.add(w)
    return ws


def make_curve(x, y, degree=FIT_DEGREE):
    """Fit polynomial on rescaled x in [0, 1] with:
      p''(x) >= 0  (convex: slope monotonically increasing)
      p'''(x) <= 0 (|p''| decreases as x grows -> bigger curvature on left)
    Enough degrees of freedom to essentially interpolate.
    Returns a callable y(x) valid on [x.min(), x.max()].
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    x_max = max(x.max(), 1.0)
    x_s = x / x_max
    x_grid = np.linspace(0.0, 1.0, 120)

    def loss(c):
        return np.sum((np.polyval(c, x_s) - y) ** 2)

    def conv(c):
        return np.polyval(np.polyder(c, 2), x_grid)            # >= 0

    def decr_curv(c):
        return -np.polyval(np.polyder(c, 3), x_grid)           # p''' <= 0

    x0 = np.polyfit(x_s, y, degree)
    res = minimize(
        loss, x0,
        constraints=[
            {"type": "ineq", "fun": conv},
            {"type": "ineq", "fun": decr_curv},
        ],
        method="SLSQP",
        options={"maxiter": 2000, "ftol": 1e-14},
    )
    coeffs = res.x

    # sanity: report residual
    resid = np.polyval(coeffs, x_s) - y
    print(f"  fit rms residual = {np.sqrt((resid**2).mean()):.3f} B EUR "
          f"(max abs {np.abs(resid).max():.3f})")

    def eval_curve(x_eval):
        return np.polyval(coeffs, np.asarray(x_eval, float) / x_max)

    return eval_curve


# --- discover common wiggles across all four sweeps ---
available = {sw["key"]: discover_wiggles(sw["pattern"]) for sw in SWEEPS}
common = sorted(set.intersection(*available.values()))
print(f"common wiggles (<= {WIGGLE_MAX}): {common}")

if not common:
    missing = [k for k, ws in available.items() if not ws]
    raise SystemExit(
        f"No wiggles common to all sweeps — sweeps with no matching networks: "
        f"{missing}. Need lv1.25 networks for each gas+Generator+m{{1.25,1.5,1.75}} "
        f"variant before this figure can be built."
    )

# --- load networks (cached on (mtime, size)) ---
cache = load_cache()
n_before = len(cache["entries"])
results = {}
for sw in SWEEPS:
    wiggles, costs, ets_costs = [], [], []
    gas_price = None
    for w in common:
        for p in NET_DIR.iterdir():
            m = sw["pattern"].match(p.name)
            if m and int(m.group(1)) == w:
                rec = get_metrics(p, cache)
                wiggles.append(w)
                costs.append(rec["cost_BEUR"])
                ets_costs.append(rec["co2_ets_cost_BEUR"])
                if gas_price is None and rec["gas_marginal_cost"] is not None:
                    gas_price = rec["gas_marginal_cost"]
                break
    results[sw["key"]] = dict(
        wiggles=np.asarray(wiggles, float),
        costs=np.asarray(costs, float),
        ets_costs=np.asarray(ets_costs, float),
        gas_price=gas_price,
        color=sw["color"],
    )
    print(f"  {sw['key']}: gas price = {gas_price:.1f} EUR/MWh, "
          f"n = {len(wiggles)}")

save_cache(cache)
n_after = len(cache["entries"])
print(f"cache v{CACHE_VERSION}: {n_after - n_before} new, {n_before} reused "
      f"(at {CACHE_PATH.relative_to(ROOT)})")

# --- upper panel: full system cost incl. actual ETS payments (nothing excluded) ---
results_upper = {}
for k, r in results.items():
    results_upper[k] = {**r, "costs": r["costs"] + r["ets_costs"]}


def plot_panel(ax, results_dict, minima_label, y_lo, y_hi, label_offset=(0, 0)):
    """Plot one panel; returns (minima_x, minima_y, fits).
    label_offset shifts the minima-line label by (dx_TWh, dy_BEUR)."""
    y_range = y_hi - y_lo
    minima_x, minima_y = [], []
    fits_local = {}
    for sw in SWEEPS:
        r = results_dict[sw["key"]]
        if len(r["wiggles"]) < 3:
            continue
        print(f"Fitting {sw['key']:6s} ({minima_label}) ...")
        curve = make_curve(r["wiggles"], r["costs"])
        x_dense = np.linspace(r["wiggles"].min(), r["wiggles"].max(), 600)
        y_dense = curve(x_dense)
        ax.plot(x_dense, y_dense, color=r["color"], lw=2.4, zorder=3)
        ax.scatter(r["wiggles"], r["costs"], color=r["color"],
                   s=22, zorder=4, edgecolor="white", linewidth=0.7)

        opt = minimize_scalar(
            curve, bounds=(r["wiggles"].min(), r["wiggles"].max()),
            method="bounded",
        )
        mx_, my_ = float(opt.x), float(opt.fun)
        minima_x.append(mx_)
        minima_y.append(my_)
        fits_local[sw["key"]] = {
            "x_TWh_dense": x_dense.tolist(),
            "y_BEUR_dense": y_dense.tolist(),
            "minimum": {"x_TWh": mx_, "y_BEUR": my_},
        }
        ax.scatter([mx_], [my_], color=r["color"], s=85, zorder=6,
                   edgecolor="black", linewidth=1.1)
        ax.plot([mx_, mx_], [y_lo, my_], color="red",
                alpha=0.25, lw=0.8, zorder=2)

        x_end = r["wiggles"].max()
        y_end = float(curve(x_end))
        ax.text(x_end, y_end + 0.5, f"{r['gas_price']:.1f}\n€/MWh",
                color=r["color"], va="bottom", ha="right",
                fontsize=9, fontweight="bold",
                linespacing=0.95)

    # dashed connector through minima
    mxa = np.array(minima_x)
    mya = np.array(minima_y)
    ord_ = np.argsort(mxa)
    mxa, mya = mxa[ord_], mya[ord_]
    if len(np.unique(mxa)) == len(mxa):
        slope, intercept = np.polyfit(mxa, mya, 1)
        line_fn = lambda xv: slope * xv + intercept
        fade_len = 700.0
        x_start = max(0.0, mxa.min() - fade_len)
        x_end_ = min(WIGGLE_MAX, mxa.max() + fade_len)
        x_line = np.linspace(x_start, x_end_, 400)
        y_line = line_fn(x_line)

        alpha = np.ones_like(x_line)
        left_mask = x_line < mxa.min()
        right_mask = x_line > mxa.max()
        alpha[left_mask] = np.clip(
            (x_line[left_mask] - (mxa.min() - fade_len)) / fade_len, 0.05, 1.0
        )
        alpha[right_mask] = np.clip(
            ((mxa.max() + fade_len) - x_line[right_mask]) / fade_len, 0.05, 1.0
        )

        segs = np.stack([np.column_stack([x_line[:-1], y_line[:-1]]),
                         np.column_stack([x_line[1:], y_line[1:]])], axis=1)
        seg_alpha = (alpha[:-1] + alpha[1:]) / 2
        seg_colors = np.column_stack([
            np.zeros(len(seg_alpha)), np.zeros(len(seg_alpha)),
            np.zeros(len(seg_alpha)), seg_alpha,
        ])
        lc = LineCollection(segs, colors=seg_colors, linewidths=1.5,
                            linestyles="--", zorder=5)
        ax.add_collection(lc)

        label_x = 1100 + label_offset[0]
        label_y = line_fn(label_x) - 0.02 * y_range + label_offset[1]
        ax.text(label_x, label_y,
                minima_label,
                color="black", fontsize=9, fontweight="bold",
                ha="left", va="center", alpha=1.0, zorder=10,
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                          edgecolor="0.4", linewidth=0.8, alpha=1.0))

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    ax.set_xlim(0, 4050)
    ax.set_ylim(y_lo, y_hi)
    ax.set_xticks(common)
    ax.set_xticklabels([str(w) for w in common])
    return minima_x, minima_y, fits_local


# --- data-driven y bounds per panel ---
all_lower = np.concatenate([results[sw["key"]]["costs"] for sw in SWEEPS])
all_upper = np.concatenate([results_upper[sw["key"]]["costs"] for sw in SWEEPS])
y_lo_b = float(all_lower.min()) - 5.0
y_hi_b = float(all_lower.max()) + 12.0
y_lo_t = float(all_upper.min()) - 5.0
y_hi_t = float(all_upper.max()) + 12.0

fig, (ax_top, ax_bot) = plt.subplots(2, 1, sharex=True, figsize=(7.6, 9.0))
top_mx, top_my, top_fits = plot_panel(
    ax_top, results_upper, "system cost", y_lo_t, y_hi_t,
    label_offset=(-500, 0),
)
bot_mx, bot_my, bot_fits = plot_panel(
    ax_bot, results, "resource cost", y_lo_b, y_hi_b,
)
fits = bot_fits
minima_x, minima_y = bot_mx, bot_my

# Autarky / No-LNG vlines on both panels; framed labels only at top of upper
for ax_ in (ax_top, ax_bot):
    for xv in (2000, 2750):
        ax_.axvline(xv, color="black", alpha=0.5, linestyle="--", lw=2, zorder=2)

y_range_t = y_hi_t - y_lo_t
for xv, label, ha, side in [
    (2000, "Autarky\n(2000 TWh)", "right", -1),
    (2750, "No-LNG\n(2750 TWh)",  "left",   1),
]:
    ax_top.text(xv + side * 70, y_hi_t - 0.03 * y_range_t - 10, label,
                color="black", alpha=0.95, fontsize=9,
                ha=ha, va="top", linespacing=0.95,
                zorder=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="0.4", linewidth=0.8, alpha=1.0))

ax_bot.set_xlabel("Gas Consumption [TWh/a]")
ax_top.set_ylabel("System cost [B €/a]")
ax_bot.set_ylabel("System cost without ETS payments [B €/a]")

for ax_, tag, y in [(ax_top, "a", 0.99), (ax_bot, "b", 1.02)]:
    ax_.text(-0.08, y, tag, transform=ax_.transAxes,
             fontsize=20, fontweight="bold", ha="left", va="bottom")

fig.tight_layout()
out = IMG_DIR / "cost_vs_gas_consumption.pdf"
fig.savefig(out, bbox_inches="tight")
print(f"\nwrote {out}")

_sweeps_payload = {}
for sw in SWEEPS:
    r = results[sw["key"]]
    _entry = {
        "color": r["color"],
        "gas_marginal_cost_EUR_per_MWh": (
            float(r["gas_price"]) if r["gas_price"] is not None else None
        ),
        "data_points": {
            "x_TWh": r["wiggles"].tolist(),
            "y_BEUR": r["costs"].tolist(),
        },
    }
    if sw["key"] in fits:
        _entry.update(fits[sw["key"]])
    _sweeps_payload[sw["key"]] = _entry

export_frontend_data("cost_vs_gas_consumption", {
    "x_label": "Gas Consumption [TWh/a]",
    "y_label": "Total system cost [B €]",
    "common_wiggles_TWh": list(common),
    "sweeps": _sweeps_payload,
    "cost_optimal_minima": [
        {"x_TWh": float(_x), "y_BEUR": float(_y)}
        for _x, _y in zip(minima_x, minima_y)
    ],
    "thresholds": {
        "autarky_TWh": 2000,
        "no_lng_TWh": 2750,
    },
})
