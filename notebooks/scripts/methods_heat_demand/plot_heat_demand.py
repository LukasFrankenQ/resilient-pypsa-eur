import os
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

idx = pd.IndexSlice

TARGET_PATH = Path('/home/lukas/Desktop/res/v1/gas_resilience/imgs/')

path = Path.cwd().parent.parent.parent / 'resources' / 'existing_heating_distribution_base_s_50_2030.csv'

with open(
    Path.cwd().parent.parent.parent / 'config' / 'plotting.default.yaml'
) as f:
    color_dict = yaml.safe_load(f)['plotting']['tech_colors']

installed = pd.read_csv(path, header=[0,1], index_col=0).mul(1e-3)
df = pd.concat([
    installed.T.loc[idx[['residential rural', 'services rural']]].groupby(level=1).sum().T,
    installed.T.loc[idx[['residential urban decentral', 'services urban decentral']]].groupby(level=1).sum().T,
    ],
    axis=1,
    keys=['rural', 'urban decentral']
)

df = df.groupby(df.index.str[:2]).sum()
fig, axs = plt.subplots(2, 1, figsize=(10, 8))

for col, ax in zip(
    ['rural', 'urban decentral'], axs
):
    df[col].plot(kind='bar', stacked=True, ax=ax, color=[color_dict[c] for c in df[col].columns], legend=False)
    ax.text(0.98, 0.98, col, transform=ax.transAxes, fontsize=10, verticalalignment='top', horizontalalignment='right')
    ax.set_ylabel('Installed Capacity')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlabel('')
    ax.set_ylabel('Installed Capacity [GW]')

axs[0].legend(title='Technology', bbox_to_anchor=(0.5, 1.2), loc='upper center', ncol=4)

axs[-1].set_xlabel('country')

plt.savefig(TARGET_PATH / 'existing_heating_capacities.pdf', bbox_inches='tight')
plt.show()

