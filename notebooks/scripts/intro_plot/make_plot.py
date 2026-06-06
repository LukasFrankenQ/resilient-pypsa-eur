import requests
import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path
import matplotlib.pyplot as plt

df_countries = pd.read_csv('gas_countries.csv', index_col=0)
df_demand = pd.read_csv('gas_demand.csv').set_index(['sector', 'geo'])['value_TWh']
df_supply = pd.read_csv('gas_supply.csv').set_index(['type', 'origin'])[['twh_2024', 'lon', 'lat']]

print('countries')
print(df_countries)

print('demand')
print(df_demand)

print('supp')
print(df_supply)



