"""
Heat-pump diagnostic — snapshot-resolved competition with gas boilers.

Loads base_s_50__3H-T-H-B-I-A-dist1_2030_free_2000.nc and produces:
  - hp_vs_boiler_mc_timeseries.parquet   per-region effective MC (HP & gas) per
                                         snapshot
  - hp_vs_boiler_mc_summary.csv          weighted mean, p05, p50, p95 per region
                                         per technology
  - gas_boiler_binding.csv               for each gas-boiler link: weighted-hour
                                         fraction binding at p_max_pu, total
                                         dispatch share of the must-run cap, and
                                         flag for "must-run pinned"
  - hp_vs_boiler_mc_duration.pdf         duration curves of HP-vs-boiler MC for
                                         a small set of representative regions
  - gas_boiler_binding.pdf               share of weighted hours binding /
                                         partly-dispatched / idle per region

The two scripts together (capacity_and_lcoh + dispatch_competition) explain why
heat pumps are not expanding: aggregate cost stack vs snapshot-by-snapshot
operational competition.
"""

import yaml
import numpy as np
import pandas as pd
import pypsa
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
NETWORK_DIR = ROOT / "results" / "networks"
SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_YAML = ROOT / "config.basicrun.yaml"
PREFIX = "base_s_50__3H-T-H-B-I-A-dist1_2030"

SCENARIO = "free"
WIGGLE = 2000
HIKE = None

HP_RURAL = ["rural air heat pump", "rural ground heat pump"]
HP_URBAN_DEC = ["urban decentral air heat pump"]
GB_RURAL = "rural gas boiler"
GB_URBAN_DEC = "urban decentral gas boiler"

REGION_TECHS = {
    "rural": (HP_RURAL, GB_RURAL),
    "urban decentral": (HP_URBAN_DEC, GB_URBAN_DEC),
}


def network_path(scenario, wiggle, hike=None):
    hike_str = f"_{hike}" if hike is not None else ""
    return NETWORK_DIR / f"{PREFIX}_{scenario}_{wiggle}{hike_str}.nc"


def style_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)


def co2_bus_for(bus0):
    if bus0.startswith("GB"):
        return "GB-co2 atmosphere"
    return "co2 atmosphere"


def time_eff(n, links):
    """(T, K) DataFrame of efficiency, broadcasting static when missing."""
    static = n.links.loc[links, "efficiency"].astype(float)
    out = pd.DataFrame(
        np.broadcast_to(static.values, (len(n.snapshots), len(links))),
        index=n.snapshots, columns=links, dtype=float,
    ).copy()
    tv = n.links_t.efficiency
    if not tv.empty:
        cols = [c for c in links if c in tv.columns]
        if cols:
            out.loc[:, cols] = tv[cols].astype(float)
    return out


def time_pmaxpu(n, links):
    """(T, K) DataFrame of p_max_pu, broadcasting static when missing."""
    static = n.links.loc[links, "p_max_pu"].astype(float)
    out = pd.DataFrame(
        np.broadcast_to(static.values, (len(n.snapshots), len(links))),
        index=n.snapshots, columns=links, dtype=float,
    ).copy()
    tv = n.links_t.p_max_pu
    if not tv.empty:
        cols = [c for c in links if c in tv.columns]
        if cols:
            out.loc[:, cols] = tv[cols].astype(float)
    return out


def hp_effective_mc(n, hp_carriers):
    """Per-snapshot effective marginal cost (EUR/MWh_heat) of heat pumps,
    grouped to the AC-bus location.

    For each HP link i with bus0 = "<AC bus> low voltage":
        mc_t(i) = λ_lowvoltage(bus0_i, t) / COP_t(i)
    Aggregate per location as capacity-weighted mean across all HP links
    (air + ground variants).
    """
    idx = n.links.index[n.links.carrier.isin(hp_carriers)]
    if len(idx) == 0:
        return pd.DataFrame()
    bus0 = n.links.loc[idx, "bus0"]
    cop = time_eff(n, list(idx))

    # electricity price at each link's bus0
    valid = bus0[bus0.isin(n.buses_t.marginal_price.columns)]
    lam = n.buses_t.marginal_price[valid.values]
    lam.columns = valid.index
    lam = lam.reindex(columns=idx, fill_value=np.nan)

    mc_per_link = lam.div(cop.replace(0.0, np.nan))

    # capacity-weighted aggregate per AC location
    location = bus0.str.replace(" low voltage", "", regex=False)
    p_nom_opt = n.links.loc[idx, "p_nom_opt"].astype(float)

    df_meta = pd.DataFrame({"link": idx, "loc": location.values,
                            "p_nom_opt": p_nom_opt.values})
    aggregates = {}
    for loc, sub in df_meta.groupby("loc"):
        if sub.p_nom_opt.sum() < 1e-6:
            continue
        w = sub.p_nom_opt.values / sub.p_nom_opt.sum()
        sub_mc = mc_per_link[sub.link.tolist()]
        aggregates[loc] = sub_mc.mul(w, axis=1).sum(axis=1)
    return pd.DataFrame(aggregates)


def gas_boiler_effective_mc(n, gb_carrier):
    """Per-snapshot effective MC (EUR/MWh_heat) of gas boilers per AC location:
        mc_t(i) = (λ_gas(bus0_i, t) - λ_co2(bus2_i, t) · η_co2) / η_boiler
    where λ_co2 is negative for emitters so the second term is a positive
    markup on the gas commodity price.
    Aggregated by capacity-weighted mean across links with the same AC location
    (here only one boiler per location, but same pattern).
    """
    idx = n.links.index[n.links.carrier == gb_carrier]
    if len(idx) == 0:
        return pd.DataFrame()
    bus0 = n.links.loc[idx, "bus0"]
    bus2 = n.links.loc[idx, "bus2"].astype(str)
    eff = n.links.loc[idx, "efficiency"].astype(float)
    eff2 = n.links.loc[idx, "efficiency2"].astype(float)

    # gas commodity price
    valid_gas = bus0[bus0.isin(n.buses_t.marginal_price.columns)]
    lam_gas = n.buses_t.marginal_price[valid_gas.values]
    lam_gas.columns = valid_gas.index
    lam_gas = lam_gas.reindex(columns=idx, fill_value=np.nan)

    # CO2 price (negative for emitters → markup)
    co2_price = pd.DataFrame(0.0, index=n.snapshots, columns=idx)
    for link in idx:
        b2 = bus2[link]
        if b2 in n.buses_t.marginal_price.columns:
            co2_price[link] = n.buses_t.marginal_price[b2]

    # gas-bus location, mapped back to AC region
    # bus0 is e.g. "DE0 0 gas" → AC "DE0 0"
    location = bus0.str.replace(" gas", "", regex=False)

    mc_per_link = lam_gas.sub(co2_price.mul(eff2, axis=1), fill_value=0.0).div(eff, axis=1)

    df_meta = pd.DataFrame({"link": idx, "loc": location.values})
    aggregates = {}
    for loc, sub in df_meta.groupby("loc"):
        # one link per location typically — but mean over the group
        aggregates[loc] = mc_per_link[sub.link.tolist()].mean(axis=1)
    return pd.DataFrame(aggregates)


def gas_boiler_binding(n, gb_carrier):
    """For every gas-boiler link, compare actual output vs the cap
    `efficiency × p_nom_opt × p_max_pu_t`.
    """
    idx = n.links.index[n.links.carrier == gb_carrier]
    if len(idx) == 0:
        return pd.DataFrame()

    weights = n.snapshot_weightings.generators
    p1 = n.links_t.p1.reindex(columns=idx, fill_value=0.0)
    output = -p1  # MW heat
    eff = n.links.loc[idx, "efficiency"].astype(float)
    p_nom_opt = n.links.loc[idx, "p_nom_opt"].astype(float)
    pmaxpu = time_pmaxpu(n, list(idx))
    cap = pmaxpu.mul(p_nom_opt, axis=1).mul(eff, axis=1)  # MW heat cap

    slack = cap - output
    # binding when slack < 1e-3 × cap (and cap > 0)
    cap_eps = cap.where(cap > 1e-9)
    binding_mask = (slack <= 1e-3 * cap_eps).fillna(False) & (cap > 1e-9)

    weights_arr = weights.values.reshape(-1, 1)
    total_h = weights.sum()

    binding_share = (binding_mask.astype(float).values * weights_arr).sum(axis=0) / total_h
    cap_energy = (cap.values * weights_arr).sum(axis=0)
    out_energy = (output.values * weights_arr).sum(axis=0)
    dispatch_share = np.where(cap_energy > 1e-9, out_energy / cap_energy, np.nan)

    rows = []
    for k, link in enumerate(idx):
        bus0 = n.links.at[link, "bus0"]
        loc = bus0.replace(" gas", "")
        rows.append({
            "link": link,
            "carrier": gb_carrier,
            "country_node": loc,
            "p_nom_opt": float(p_nom_opt[link]),
            "p_nom_extendable": bool(n.links.at[link, "p_nom_extendable"]),
            "binding_share_of_hours": float(binding_share[k]),
            "dispatch_share_of_cap": float(dispatch_share[k]),
            "must_run_pinned": (
                (not bool(n.links.at[link, "p_nom_extendable"]))
                and float(dispatch_share[k]) > 0.99
            ),
            "annual_output_TWh": float(out_energy[k] / 1e6),
            "annual_cap_TWh": float(cap_energy[k] / 1e6),
        })
    return pd.DataFrame(rows)


def summarize_mc(mc_df, label, weights):
    """Return per-location (mean, p05, p50, p95) for a (T, locations) df,
    weighted by snapshot weights for the mean.
    """
    if mc_df.empty:
        return pd.DataFrame()
    rows = []
    w_total = weights.sum()
    for loc in mc_df.columns:
        s = mc_df[loc].dropna()
        if s.empty:
            continue
        w_aligned = weights.reindex(s.index)
        rows.append({
            "tech": label,
            "location": loc,
            "mean_eur_per_mwh": float((s * w_aligned).sum() / w_aligned.sum()),
            "p05": float(s.quantile(0.05)),
            "p50": float(s.quantile(0.50)),
            "p95": float(s.quantile(0.95)),
        })
    return pd.DataFrame(rows)


def plot_duration_curves(hp_mc, gb_mc, locations, region_label, out_path):
    locations = [l for l in locations if l in hp_mc.columns and l in gb_mc.columns]
    if not locations:
        return
    n_loc = len(locations)
    ncols = min(3, n_loc)
    nrows = int(np.ceil(n_loc / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.0 * nrows),
                             squeeze=False)
    for ax, loc in zip(axes.flat, locations):
        hp = hp_mc[loc].dropna().sort_values(ascending=False).values
        gb = gb_mc[loc].dropna().sort_values(ascending=False).values
        x_hp = np.linspace(0, 100, len(hp))
        x_gb = np.linspace(0, 100, len(gb))
        ax.plot(x_hp, hp, color="#1f77b4", label="heat pump")
        ax.plot(x_gb, gb, color="#d62728", label="gas boiler (with CO2)")
        ax.fill_between(x_hp, hp, np.interp(x_hp, x_gb, gb),
                        where=(np.interp(x_hp, x_gb, gb) > hp),
                        color="#1f77b4", alpha=0.15,
                        label="HP cheaper")
        ax.set_title(loc, fontsize=10)
        ax.set_xlabel("% of hours (sorted high→low)")
        ax.set_ylabel("EUR/MWh_heat")
        style_ax(ax)
    for ax in axes.flat[n_loc:]:
        ax.axis("off")
    axes.flat[0].legend(frameon=True, fontsize=8, loc="upper right")
    fig.suptitle(f"HP vs gas boiler effective MC duration — {region_label} "
                 f"({SCENARIO}, wiggle={WIGGLE})", y=1.02, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_binding(binding_df, out_path):
    """Stacked bar per region: binding_share / partly-dispatched / idle."""
    if binding_df.empty:
        return
    df = binding_df.copy()
    df["region"] = df["carrier"].apply(
        lambda c: "rural" if c.startswith("rural") else "urban decentral"
    )
    # Aggregate to region by capacity-weighted average of (binding_share,
    # dispatch_share)
    regs = []
    for reg, sub in df.groupby("region"):
        w = sub.p_nom_opt.values
        wsum = w.sum() if w.sum() > 0 else 1.0
        regs.append({
            "region": reg,
            "binding": (sub.binding_share_of_hours * w).sum() / wsum,
            "dispatch_share": (sub.dispatch_share_of_cap * w).sum() / wsum,
            "must_run_link_share": (sub.must_run_pinned.astype(float) * w).sum() / wsum,
        })
    reg_df = pd.DataFrame(regs).set_index("region")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    ax = axes[0]
    x = np.arange(len(reg_df))
    binding = reg_df["binding"].values
    other = 1 - binding
    ax.bar(x, binding, color="#d62728", label="binding @ p_max_pu",
           edgecolor="white")
    ax.bar(x, other, bottom=binding, color="#cccccc",
           label="not binding", edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(reg_df.index)
    ax.set_ylabel("share of weighted hours")
    ax.set_ylim(0, 1.05)
    ax.set_title("Gas-boiler hours binding at p_max_pu")
    style_ax(ax)
    ax.legend(frameon=True, loc="upper right", fontsize=9)

    ax = axes[1]
    ax.bar(x, reg_df["dispatch_share"].values, color="#9467bd",
           edgecolor="white")
    ax.axhline(1.0, color="black", linestyle="--", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(reg_df.index)
    ax.set_ylabel("Σ output / Σ cap (must-run dispatch share)")
    ax.set_ylim(0, 1.1)
    ax.set_title("Gas-boiler dispatch share of must-run cap")
    style_ax(ax)

    fig.suptitle(f"Gas-boiler must-run binding ({SCENARIO}, wiggle={WIGGLE})",
                 y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main():
    npath = network_path(SCENARIO, WIGGLE, HIKE)
    print(f"loading {npath}")
    n = pypsa.Network(str(npath))
    print(f"  snapshots={len(n.snapshots)}  weight={n.snapshot_weightings.generators.iloc[0]}")
    weights = n.snapshot_weightings.generators

    # 1. HP vs gas-boiler effective MC time-series
    print("\n[1/3] computing per-snapshot effective MC")
    pieces = []
    summaries = []
    for region, (hp_carriers, gb_carrier) in REGION_TECHS.items():
        hp_mc = hp_effective_mc(n, hp_carriers)
        gb_mc = gas_boiler_effective_mc(n, gb_carrier)
        # Align on common locations
        common = hp_mc.columns.intersection(gb_mc.columns)
        hp_mc = hp_mc[common]
        gb_mc = gb_mc[common]
        # tag location with region for the output frame
        for loc in common:
            piece = pd.DataFrame({
                "snapshot": n.snapshots,
                "region": region,
                "location": loc,
                "hp_mc": hp_mc[loc].values,
                "gb_mc": gb_mc[loc].values,
            })
            piece["gap"] = piece["gb_mc"] - piece["hp_mc"]
            piece["hp_cheaper"] = piece["gap"] > 0
            pieces.append(piece)

        summaries.append(summarize_mc(hp_mc, f"HP ({region})", weights))
        summaries.append(summarize_mc(gb_mc, f"gas boiler ({region})", weights))

    ts_df = pd.concat(pieces, ignore_index=True)
    ts_path = SCRIPT_DIR / "hp_vs_boiler_mc_timeseries.parquet"
    ts_df.to_parquet(ts_path)
    print(f"  wrote {ts_path.name}  rows={len(ts_df)}")

    summary_df = pd.concat(summaries, ignore_index=True)
    sum_path = SCRIPT_DIR / "hp_vs_boiler_mc_summary.csv"
    summary_df.to_csv(sum_path, index=False)
    print(f"  wrote {sum_path.name}  rows={len(summary_df)}")
    print()
    print(summary_df.groupby("tech")[["mean_eur_per_mwh", "p50"]]
          .describe()[("mean_eur_per_mwh", "mean")].to_string()
          if False else "  per-tech mean of location means:")
    print(summary_df.groupby("tech")[["mean_eur_per_mwh"]].mean().round(2).to_string())

    # Cross-region cheaper-share
    print("\n  weighted share of hours where HP_mc < gas_boiler_mc:")
    for region, sub in ts_df.groupby("region"):
        w_aligned = weights.reindex(sub["snapshot"]).values
        share = (sub["hp_cheaper"].astype(float).values * w_aligned).sum() / w_aligned.sum()
        print(f"    {region}: {share:.2%}")

    # 2. Gas-boiler must-run binding
    print("\n[2/3] gas boiler must-run binding")
    binding_pieces = []
    for gb in [GB_RURAL, GB_URBAN_DEC]:
        binding_pieces.append(gas_boiler_binding(n, gb))
    binding_df = pd.concat(binding_pieces, ignore_index=True)
    binding_path = SCRIPT_DIR / "gas_boiler_binding.csv"
    binding_df.to_csv(binding_path, index=False)
    print(f"  wrote {binding_path.name}  rows={len(binding_df)}")
    print()
    print(binding_df.groupby("carrier")[
        ["binding_share_of_hours", "dispatch_share_of_cap", "must_run_pinned"]
    ].mean().round(3).to_string())

    # Cross-link cheaper-AND-binding intersection (capacity-weighted-hours)
    print("\n  hours when HP cheaper AND gas boiler binding (capacity-weighted):")
    for region, (_, gb_carrier) in REGION_TECHS.items():
        bdf = binding_df[binding_df.carrier == gb_carrier]
        # build per-link binding_t mask same way as in gas_boiler_binding
        idx = n.links.index[n.links.carrier == gb_carrier]
        p1 = n.links_t.p1.reindex(columns=idx, fill_value=0.0)
        eff = n.links.loc[idx, "efficiency"].astype(float)
        p_nom_opt = n.links.loc[idx, "p_nom_opt"].astype(float)
        pmaxpu = time_pmaxpu(n, list(idx))
        cap = pmaxpu.mul(p_nom_opt, axis=1).mul(eff, axis=1)
        slack = cap - (-p1)
        cap_eps = cap.where(cap > 1e-9)
        binding_t = (slack <= 1e-3 * cap_eps).fillna(False) & (cap > 1e-9)
        binding_t.columns = [c.replace(f" {gb_carrier}", "") for c in binding_t.columns]

        sub = ts_df[ts_df.region == region]
        sub = sub.set_index(["snapshot", "location"])["hp_cheaper"].unstack("location")
        common = binding_t.columns.intersection(sub.columns)
        if len(common) == 0:
            continue
        bt = binding_t[common].reindex(index=sub.index)
        intersect = (sub[common] & bt)
        weights_aligned = weights.reindex(intersect.index).values.reshape(-1, 1)
        share = (intersect.astype(float).values * weights_aligned).sum() / (
            weights_aligned.sum() * len(common)
        )
        print(f"    {region}: {share:.2%}  (across {len(common)} locations)")

    # 3. Plots
    print("\n[3/3] plots")
    # representative locations: pick those with non-empty data and large p_nom_opt
    for region, (hp_carriers, gb_carrier) in REGION_TECHS.items():
        sub = ts_df[ts_df.region == region]
        if sub.empty:
            continue
        # rank locations by total HP capacity
        idx_hp = n.links.index[n.links.carrier.isin(hp_carriers)]
        cap_per_loc = (
            n.links.loc[idx_hp]
            .assign(loc=n.links.loc[idx_hp, "bus0"].str.replace(" low voltage", "", regex=False))
            .groupby("loc").p_nom_opt.sum()
        )
        ranked = cap_per_loc.sort_values(ascending=False).head(6).index.tolist()
        out_pdf = SCRIPT_DIR / f"hp_vs_boiler_mc_duration_{region.replace(' ', '_')}.pdf"
        hp_mc_df = (sub.set_index(["snapshot", "location"])["hp_mc"]
                    .unstack("location"))
        gb_mc_df = (sub.set_index(["snapshot", "location"])["gb_mc"]
                    .unstack("location"))
        plot_duration_curves(hp_mc_df, gb_mc_df, ranked, region, out_pdf)
        print(f"  wrote {out_pdf.name}")

    binding_pdf = SCRIPT_DIR / "gas_boiler_binding.pdf"
    plot_binding(binding_df, binding_pdf)
    print(f"  wrote {binding_pdf.name}")


if __name__ == "__main__":
    main()
