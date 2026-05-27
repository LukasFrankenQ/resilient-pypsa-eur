import os
import sys
import yaml
import numpy as np
import pandas as pd
import pypsa
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from frontend_export import export_frontend_data  # noqa: E402

idx = pd.IndexSlice

REPO_ROOT = Path(__file__).resolve().parents[3]
TARGET_PATH = REPO_ROOT.parent / 'gas_resilience' / 'imgs'

path = REPO_ROOT / 'resources' / 'existing_heating_distribution_base_s_50_2030.csv'
model_path = REPO_ROOT / 'results' / 'networks' / 'base_s_50_lv1.25_3H-T-H-B-I-A-dist1_2030_free_4000.nc'

with open(
    REPO_ROOT / 'config' / 'plotting.default.yaml'
) as f:
    color_dict = yaml.safe_load(f)['plotting']['tech_colors']

installed = pd.read_csv(path, header=[0,1], index_col=0).mul(1e-3)
df = pd.concat([
    installed.T.loc[idx[['residential rural', 'services rural']]].groupby(level=1).sum().T,
    installed.T.loc[idx[['residential urban decentral', 'services urban decentral']]].groupby(level=1).sum().T,
    ],
    axis=1,
    keys=['rural', 'urban individual']
)

df = df.groupby(df.index.str[:2]).sum()
fig, axs = plt.subplots(2, 1, figsize=(10, 8))

for col, ax in zip(
    ['rural', 'urban individual'], axs
):
    df[col].plot(kind='bar', stacked=True, ax=ax, color=[color_dict[c] for c in df[col].columns], legend=False)
    ax.text(0.98, 0.98, col, transform=ax.transAxes, fontsize=10, verticalalignment='top', horizontalalignment='right')
    ax.set_ylabel('Installed Capacity')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlabel('')
    ax.set_ylabel('Installed Capacity [GW]')

    totals = df[col].sum()
    inset = ax.inset_axes([2/3, 0.84, 1/3, 0.06])
    _left = 0.0
    for _tech in totals.index:
        inset.barh(0, totals[_tech], left=_left, color=color_dict.get(_tech))
        _left += totals[_tech]
    inset.set_xlim(0, totals.sum())
    inset.set_ylim(-0.5, 0.5)
    inset.set_xticks([])
    inset.set_yticks([])
    for _s in inset.spines.values():
        _s.set_visible(False)
    inset.text(1.0, -1.2, f"Σ {totals.sum():.0f} GW",
               transform=inset.transAxes, ha='right', va='top', fontsize=8)

axs[0].legend(title='Technology', bbox_to_anchor=(0.5, 1.2), loc='upper center', ncol=4)

axs[-1].set_xlabel('country')

plt.savefig(TARGET_PATH / 'existing_heating_capacities.pdf', bbox_inches='tight')

_panels = {}
for _sector in ['rural', 'urban individual']:
    _sub = df[_sector]
    _panels[_sector] = {
        "countries": _sub.index.tolist(),
        "technologies": _sub.columns.tolist(),
        "installed_GW": {c: _sub[c].tolist() for c in _sub.columns},
        "colors": {c: color_dict.get(c) for c in _sub.columns},
    }

export_frontend_data("existing_heating_capacities", {
    "y_label": "Installed Capacity [GW]",
    "x_label": "country",
    "panels": _panels,
})


# ---------------------------------------------------------------------------
# Second plot: input capacities vs p_nom_min inserted into the model
# ---------------------------------------------------------------------------

n = pypsa.Network(str(model_path))

sector_label_to_model = {'rural': 'rural', 'urban individual': 'urban decentral'}
tech_list = df['rural'].columns.tolist()

model_df = {}
supply_share = {}
sw = n.snapshot_weightings.generators
for sector_label, model_prefix in sector_label_to_model.items():
    carriers = [f'{model_prefix} {t}' for t in tech_list]
    links = n.links[n.links.carrier.isin(carriers)]
    country = links.index.str[:2]
    grouped = (
        links.assign(country=country)
             .groupby(['country', 'carrier'])['p_nom_min'].sum()
             .unstack(fill_value=0.0)
             .mul(1e-3)
    )
    grouped.columns = grouped.columns.str.replace(f'{model_prefix} ', '', regex=False)
    grouped = grouped.reindex(columns=tech_list, fill_value=0.0)
    grouped = grouped.reindex(df.index, fill_value=0.0)
    model_df[sector_label] = grouped

    p1 = (-n.links_t.p1[links.index]).clip(lower=0.0)
    energy = p1.mul(sw, axis=0).sum()
    tech_energy = energy.groupby(
        links.carrier.str.replace(f'{model_prefix} ', '', regex=False)
    ).sum().reindex(tech_list, fill_value=0.0)
    total = tech_energy.sum()
    supply_share[sector_label] = (
        tech_energy / total if total > 0 else tech_energy * 0
    )

fig2, axs2 = plt.subplots(2, 1, figsize=(12, 8))
x_pos = np.arange(len(df.index))
width = 0.4

for col, ax in zip(['rural', 'urban individual'], axs2):
    in_data = df[col]
    md_data = model_df[col]

    bottoms_in = np.zeros(len(df.index))
    bottoms_md = np.zeros(len(df.index))
    for tech in tech_list:
        ax.bar(x_pos - width/2, in_data[tech].values, width,
               bottom=bottoms_in, color=color_dict.get(tech))
        ax.bar(x_pos + width/2, md_data[tech].values, width,
               bottom=bottoms_md, color=color_dict.get(tech),
               hatch='///', edgecolor='white', linewidth=0)
        bottoms_in += in_data[tech].values
        bottoms_md += md_data[tech].values

    ax.set_xticks(x_pos)
    ax.set_xticklabels(df.index, rotation=0)
    ax.text(0.98, 0.98, col, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlabel('')
    ax.set_ylabel('Installed Capacity [GW]')

    in_totals = in_data.sum()
    md_totals = md_data.sum()
    shares = supply_share[col]
    share_totals = shares * in_totals.sum()  # rescale share bar to input bar width
    inset = ax.inset_axes([2/3, 0.70, 1/3, 0.20])
    for row, (label, totals, hatch) in enumerate(
        [('input [GWth]', in_totals, None),
         ('model [GWprimary]', md_totals, '///'),
         ('supply shares', share_totals, '....')]
    ):
        _left = 0.0
        y = -row
        for _tech in tech_list:
            inset.barh(y, totals[_tech], left=_left,
                       color=color_dict.get(_tech),
                       hatch=hatch, edgecolor='white', linewidth=0)
            _left += totals[_tech]
        inset.text(-0.02, y, label, transform=inset.get_yaxis_transform(),
                   ha='right', va='center', fontsize=7)
    inset.set_xlim(0, max(in_totals.sum(), md_totals.sum(), share_totals.sum()))
    inset.set_ylim(-2.6, 0.6)
    inset.set_xticks([])
    inset.set_yticks([])
    for _s in inset.spines.values():
        _s.set_visible(False)

legend_handles = [Patch(facecolor=color_dict.get(t), label=t) for t in tech_list]
legend_handles += [
    Patch(facecolor='lightgrey', label='input [GWth]'),
    Patch(facecolor='lightgrey', hatch='///', edgecolor='white', label='model [GWprimary]'),
]
axs2[0].legend(handles=legend_handles, title='Technology',
               bbox_to_anchor=(0.5, 1.2), loc='upper center', ncol=4)
axs2[-1].set_xlabel('country')

plt.savefig(TARGET_PATH / 'existing_heating_capacities_vs_model.pdf',
            bbox_inches='tight')

_panels2 = {}
for _sector in ['rural', 'urban individual']:
    _in = df[_sector]
    _md = model_df[_sector]
    _panels2[_sector] = {
        "countries": _in.index.tolist(),
        "technologies": tech_list,
        "input_GWth": {c: _in[c].tolist() for c in tech_list},
        "model_GWprimary": {c: _md[c].tolist() for c in tech_list},
        "supply_share": {c: float(supply_share[_sector][c]) for c in tech_list},
        "colors": {c: color_dict.get(c) for c in tech_list},
    }

export_frontend_data("existing_heating_capacities_vs_model", {
    "y_label": "Installed Capacity [GW]",
    "x_label": "country",
    "model_network": model_path.name,
    "panels": _panels2,
})

plt.show()

