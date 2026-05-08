"""Total system cost vs. gas consumption for four gas-price sweeps
(baseline + m=1.25 / 1.5 / 1.75 gas-marginal-cost multipliers).

Only wiggle values available in all four sweeps are used, capped at
4000 TWh. For each sweep a smooth curve is fitted whose slope is
monotonically increasing (p'' >= 0) and whose curvature decreases
(p''' <= 0), i.e. |p''| larger on the left. The curve is flexible
enough to essentially interpolate all data points. Each curve's
fitted minimum is marked; the four minima are connected with a
dashed black cubic spline labelled `cost-optimal gas use`.
"""
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
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

WIGGLE_MAX = 4000
FIT_DEGREE = 6  # 7 coeffs vs ~9 data points => near-interpolation

SWEEPS = [
    {"key": "base",
     "pattern": re.compile(
         r"^base_s_50_lv1.25_168H-T-H-B-I-A-dist1_2030_free_(\d+)\.nc$"),
     "color": "#3a6ea5"},
    {"key": "m1.25",
     "pattern": re.compile(
         r"^base_s_50_lv1.25_168H-T-H-B-I-A-dist1-gas\+Generator\+m1\.25_2030_free_(\d+)\.nc$"),
     "color": "#e8b13d"},
    {"key": "m1.5",
     "pattern": re.compile(
         r"^base_s_50_lv1.25_168H-T-H-B-I-A-dist1-gas\+Generator\+m1\.5_2030_free_(\d+)\.nc$"),
     "color": "#dd6a2b"},
    {"key": "m1.75",
     "pattern": re.compile(
         r"^base_s_50_lv1.25_168H-T-H-B-I-A-dist1-gas\+Generator\+m1\.75_2030_free_(\d+)\.nc$"),
     "color": "#b62020"},
]


def total_cost_be(n):
    return (n.statistics.capex().sum() + n.statistics.opex().sum()) / 1e9


def gas_marginal_cost(n):
    g = n.generators[n.generators.carrier == "gas"]
    return float(g.marginal_cost.mean()) if not g.empty else np.nan


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

# --- load networks ---
results = {}
for sw in SWEEPS:
    wiggles, costs = [], []
    gas_price = None
    for w in common:
        for p in NET_DIR.iterdir():
            m = sw["pattern"].match(p.name)
            if m and int(m.group(1)) == w:
                print(f"Loading {sw['key']:6s} wiggle={w}")
                n = pypsa.Network(p)
                wiggles.append(w)
                costs.append(total_cost_be(n))
                if gas_price is None:
                    gas_price = gas_marginal_cost(n)
                break
    results[sw["key"]] = dict(
        wiggles=np.asarray(wiggles, float),
        costs=np.asarray(costs, float),
        gas_price=gas_price,
        color=sw["color"],
    )
    print(f"  {sw['key']}: gas price = {gas_price:.1f} EUR/MWh, "
          f"n = {len(wiggles)}")

# --- plot ---
fig, ax = plt.subplots(figsize=(7.6, 4.6))
minima_x, minima_y = [], []
fits = {}

for sw in SWEEPS:
    r = results[sw["key"]]
    if len(r["wiggles"]) < 3:
        continue
    print(f"Fitting {sw['key']} ...")
    curve = make_curve(r["wiggles"], r["costs"])
    x_dense = np.linspace(r["wiggles"].min(), r["wiggles"].max(), 600)
    y_dense = curve(x_dense)
    ax.plot(x_dense, y_dense, color=r["color"], lw=2.4, zorder=3)
    ax.scatter(r["wiggles"], r["costs"], color=r["color"],
               s=22, zorder=4, edgecolor="white", linewidth=0.7)

    res = minimize_scalar(
        curve, bounds=(r["wiggles"].min(), r["wiggles"].max()),
        method="bounded",
    )
    mx, my = float(res.x), float(res.fun)
    minima_x.append(mx)
    minima_y.append(my)
    fits[sw["key"]] = {
        "x_TWh_dense": x_dense.tolist(),
        "y_BEUR_dense": y_dense.tolist(),
        "minimum": {"x_TWh": mx, "y_BEUR": my},
    }
    ax.scatter([mx], [my], color=r["color"], s=85, zorder=6,
               edgecolor="black", linewidth=1.1)
    ax.plot([mx, mx], [830, my], color="red",
            alpha=0.25, lw=0.8, zorder=2)

    x_end = r["wiggles"].max()
    y_end = float(curve(x_end))
    ax.text(x_end, y_end + 0.5, f"{r['gas_price']:.1f}\n€/MWh",
            color=r["color"], va="bottom", ha="right",
            fontsize=9, fontweight="bold",
            linespacing=0.95)

# --- dashed connector through minima ---
mx = np.array(minima_x)
my = np.array(minima_y)
ord_ = np.argsort(mx)
mx, my = mx[ord_], my[ord_]
if len(np.unique(mx)) == len(mx):
    slope, intercept = np.polyfit(mx, my, 1)
    line = lambda xv: slope * xv + intercept
    fade_len = 700.0
    x_start = max(0.0, mx.min() - fade_len)
    x_end = min(WIGGLE_MAX, mx.max() + fade_len)
    x_line = np.linspace(x_start, x_end, 400)
    y_line = line(x_line)

    alpha = np.ones_like(x_line)
    left_mask = x_line < mx.min()
    right_mask = x_line > mx.max()
    alpha[left_mask] = np.clip(
        (x_line[left_mask] - (mx.min() - fade_len)) / fade_len, 0.05, 1.0
    )
    alpha[right_mask] = np.clip(
        ((mx.max() + fade_len) - x_line[right_mask]) / fade_len, 0.05, 1.0
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

    ax.text(1100, 910,
            "cost-optimal gas use",
            color="black", fontsize=9, fontweight="bold",
            ha="left", va="center", alpha=1.0, zorder=10,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                      edgecolor="0.4", linewidth=0.8, alpha=1.0))

for xv, label, dx in [
    (2000, "Autarky\n(2000 TWh)", -40),
    (2750, "No-LNG\n(2750 TWh)",  40),
]:
    ax.axvline(xv, color="black", alpha=0.5, linestyle="--", lw=2, zorder=2)
    ax.text(xv + dx, 962, label,
            color="black", alpha=0.85, fontsize=9,
            ha="center", va="bottom", linespacing=0.95,
            zorder=10, clip_on=False)

ax.set_xlabel("Gas Consumption [TWh/y]")
ax.set_ylabel("Total system cost [B €]")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.yaxis.grid(True, alpha=0.3)
ax.set_axisbelow(True)

ax.set_xticks(common)
ax.set_xticklabels([str(w) for w in common])
ax.set_xlim(0, 4050)
ax.set_ylim(830, 960)

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
    "x_label": "Gas Consumption [TWh/y]",
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
