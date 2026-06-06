"""
PyPSA-Eur Energy Network Visualisation - AEGIS Proposal
=====================================
Visualization of energy infrastructure for Germany including power lines,
power plants, industrial sites, and 14 black start procurement regions.

this script builds upon
- resources/ folder of PyPSA-Eur run with config/aegis.yaml
- geojson with shapes of 14 black start procurement regions scraped from picture from www.netztransparenz.de
- magic plotting routines adapted from FN's notebooks

IR, 21 Jan 2026
"""

# %% Imports
import pypsa
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import contextily as cx
import json
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib as mpl

mpl.rcParams["font.family"] = "Roboto"
mpl.rcParams["hatch.linewidth"] = 0.5

# %% Coordinate Reference System
crs = ccrs.epsg(3857)

# %% Load Regions and Demand Data
regions = (
    gpd.read_file("resources/aegis_presolve_lv/regions_onshore_base_s.geojson")
    .set_index("name")
    .to_crs(crs)
)
demand = xr.open_dataarray("resources/aegis_presolve_lv/electricity_demand_base_s.nc")
regions["demand"] = demand.mean(dim="time").to_pandas() * 8.760  # GWh per year
regions["demand"].fillna(0, inplace=True)
regions["demand_per_area"] = (
    regions["demand"] / regions.geometry.to_crs(3035).area * 1e6
)  # GWh per year and km²

# %% Load Power Plants Data
COUNTRIES = [
    "Sweden",
    "Netherlands",
    "Germany",
    "Denmark",
    "Norway",
    "Finland",
    "Poland",
    "Estonia",
    "Latvia",
    "Lithuania",
    "Great Britain",
    "Ireland",
    "United Kingdom",
    "Albania",
    "Austria",
    "Bosnia and Herzegovina",
    "Belgium",
    "Bulgaria",
    "Switzerland",
    "Czech Republic",
    "Czechia",
    "Spain",
    "France",
    "Greece",
    "Croatia",
    "Hungary",
    "Italy",
    "Luxembourg",
    "Montenegro",
    "North Macedonia",
    "Macediona",
    "Moldova",
    "Portugal",
    "Romania",
    "Serbia",
    "Slovenie",
    "Slovakia",
    "Ukraine",
    "Kosovo",
]

df = pd.read_csv("../powerplantmatching/powerplants.csv", index_col=0)
df = df.query(
    "Country in @COUNTRIES and (DateOut >= 2025 or DateOut.isna()) and (DateIn <= 2025 or DateIn.isna())"
)
ppl = gpd.GeoDataFrame(
    df, geometry=gpd.points_from_xy(df.lon, df.lat), crs=ccrs.PlateCarree()
).to_crs(crs)

FUELTYPE_MAP = {
    "Hard Coal": "Coal",
    "Lignite": "Coal",
    "Hydro": "Hydro",
    "Nuclear": "Nuclear",
    "Natural Gas": "Gas",
    "Oil": "Oil",
    "Solid Biomass": "Biomass",
    "Biogas": "Biomass",
    "Waste": "Waste",
    "Other": "Waste",
    "Wind": "Wind",
    "Solar": "Solar",
}
ppl["Fueltype_simplified"] = ppl.Fueltype.map(FUELTYPE_MAP)

# %% Load CO2 Sequestration Data
co2 = gpd.read_file("resources/aegis_presolve_lv/co2_sequestration_potentials.geojson")
COUNTRY_CODES = [
    "AL",
    "AT",
    "BA",
    "BE",
    "BG",
    "CH",
    "CZ",
    "DE",
    "DK",
    "EE",
    "ES",
    "FI",
    "FR",
    "GB",
    "GR",
    "HR",
    "HU",
    "IE",
    "IT",
    "LT",
    "LU",
    "LV",
    "ME",
    "MD",
    "MK",
    "NL",
    "NO",
    "PL",
    "PT",
    "RO",
    "RS",
    "SE",
    "SI",
    "SK",
    "UA",
    "XK",
]
co2 = co2.query("COUNTRYCOD in @COUNTRY_CODES and `optimistic estimate Mt` > 0").to_crs(
    crs
)
co2["geometry"] = co2.geometry.buffer(0)
co2 = co2.dissolve(by="COUNTRYCOD")
co2 = gpd.overlay(co2, regions.dissolve(), how="difference").dissolve()

# %% Load Industrial Sites Data
df = pd.read_csv("data/Industrial_Database.csv", sep=";", index_col=0)
df[["srid", "coordinates"]] = df.geom.str.split(";", expand=True)
df.drop(df.index[df.coordinates.isna()], inplace=True)
df = df.query("Country in @COUNTRIES")

SUBSECTOR_MAP = {
    "Iron and steel": "Steel",
    "Cement": "Cement",
    "Refineries": "Refining",
    "Paper and printing": "Other",
    "Chemical industry": "Chemicals",
    "Glass": "Other",
    "Non-ferrous metals": "Other",
    "Non-metallic mineral products": "Other",
    "Other non-classified": "Other",
}
df["Subsector_simplified"] = df.Subsector.map(SUBSECTOR_MAP)
print(df.Subsector_simplified.value_counts())

df["coordinates"] = gpd.GeoSeries.from_wkt(df["coordinates"])
industry = gpd.GeoDataFrame(df, geometry="coordinates", crs="EPSG:4326").to_crs(crs)


# %% Load LNG Terminal Data
def build_gem_lng_data(fn):
    df = pd.read_excel(fn, sheet_name="LNG terminals - data")
    df = df.set_index("ComboID")
    status_list = ["Operating"]
    df = df.query(
        "Status in @status_list "
        "& FacilityType == 'Import' "
        "& Country in @COUNTRIES "
        "& CapacityInMtpa != '--' "
        "& CapacityInMtpa != 0"
    )
    geometry = gpd.points_from_xy(df["Longitude"], df["Latitude"])
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
    return gdf


lng = build_gem_lng_data("data/gem/Europe-Gas-Tracker-2024-05.xlsx").to_crs(crs)


# %% Load Gas Network Data
def load_dataset(fn):
    df = gpd.read_file(fn)
    param = df.param.apply(json.loads).apply(pd.Series)
    cols = ["diameter_mm", "max_cap_M_m3_per_d"]
    method = df.method.apply(json.loads).apply(pd.Series)[cols]
    method.columns = method.columns + "_method"
    df = pd.concat([df, param, method], axis=1)
    to_drop = ["param", "uncertainty", "method", "tags"]
    to_drop = df.columns.intersection(to_drop)
    df.drop(to_drop, axis=1, inplace=True)
    return df


gas = (
    load_dataset("data/gas_network/scigrid-gas/data/IGGIELGN_PipeSegments.geojson")
    .to_crs(crs)
    .query("length_km < 1000")
)
gas = gas.loc[
    ~(
        gas.nuts_id_1.astype(str).str.contains("RU|BY")
        | gas.nuts_id_2.astype(str).str.contains("RU|BY")
        | gas.nuts_id_3.astype(str).str.contains("RU|BY")
    )
]

# %% Load PyPSA Network
n = pypsa.Network("resources/aegis_presolve_lv/networks/base.nc")

# %% Process Lines
geometry = gpd.GeoSeries.from_wkt(n.lines.geometry, crs=ccrs.PlateCarree()).to_crs(crs)
lines = gpd.GeoDataFrame(n.lines.drop(columns="geometry"), geometry=geometry, crs=crs)

geometry = gpd.GeoSeries.from_wkt(n.links.geometry, crs=ccrs.PlateCarree()).to_crs(crs)
links = gpd.GeoDataFrame(n.links.drop(columns="geometry"), geometry=geometry, crs=crs)

lines_380 = lines.query("v_nom >= 380")
lines_250 = lines.query("v_nom < 380 and v_nom >= 250")
lines_200 = lines.query("v_nom < 250 and v_nom >= 200")
lines_110 = lines.query("v_nom < 200 and v_nom >= 110")
lines_lv = lines.query("v_nom < 110")

# %% Color Maps
POWERPLANT_COLORS = {
    "Hydro": "lightseagreen",
    "Nuclear": "deeppink",
    "Solar": "goldenrod",
    "Wind": "royalblue",
    "Biomass": "forestgreen",
    "Gas": "orangered",
    "Coal": "grey",
    "Oil": "black",
    "Waste": "olive",
}

INDUSTRY_COLORS = {
    "Steel": "crimson",
    "Cement": "chocolate",
    "Refining": "mediumslateblue",
    "Chemicals": "gold",
    "Other": "dimgray",
}


# %% Plot Function
def plot_germany(
    show_power_lines=True,
    show_power_plants=False,
    show_industry=False,
    show_gas_pipelines=False,
    show_co2_areas=False,
    show_network_zones=True,
    output_format="pdf",
    output_filename="germany_network_zones_clean",
    map_extent=(2.8, 16.5, 46.5, 55.8),
    figsize=(10, 20),
    dpi=300,
):
    fig = plt.figure(figsize=figsize)
    ax = plt.axes(projection=crs)

    regions.plot(
        ax=ax,
        column="demand_per_area",
        edgecolor="lightgrey",
        linewidth=0.5,
        cmap="Greys",
        vmin=0,
        vmax=20,
        alpha=0.3,
    )

    if show_power_lines:
        lines_380.plot(
            ax=ax,
            color="mediumorchid",
            linewidth=lines_380.v_nom / 200,
            alpha=0.7,
            zorder=7,
        )
        lines_250.plot(
            ax=ax,
            color="orangered",
            linewidth=lines_250.v_nom / 200,
            alpha=0.7,
            zorder=6,
        )
        lines_200.plot(
            ax=ax, color="orange", linewidth=lines_200.v_nom / 200, alpha=0.7, zorder=5
        )
        lines_110.plot(
            ax=ax,
            color="limegreen",
            linewidth=lines_110.v_nom / 200,
            alpha=0.7,
            zorder=4,
        )
        lines_lv.plot(
            ax=ax, color="teal", linewidth=lines_lv.v_nom / 200, alpha=0.7, zorder=3
        )

        links.plot(
            ax=ax,
            color="dodgerblue",
            linewidth=links.p_nom / 500,
            alpha=0.7,
            linestyle="dashed",
            zorder=8,
        )

    if show_power_plants:
        ax.scatter(
            ppl.geometry.x,
            ppl.geometry.y,
            c=ppl.Fueltype_simplified.map(POWERPLANT_COLORS).fillna("lightgray"),
            s=ppl.Capacity / 4,
            alpha=0.5,
            edgecolor="face",
            linewidth=0,
            zorder=10,
        )

    if show_industry:
        ax.scatter(
            industry.geometry.x,
            industry.geometry.y,
            c=industry.Subsector_simplified.map(INDUSTRY_COLORS).fillna("lightgray"),
            s=industry.Emissions_ETS_2014 / 10_000,
            marker="s",
            alpha=0.7,
            edgecolor="face",
            linewidth=0,
            zorder=9,
        )

    if show_gas_pipelines:
        gas.plot(
            ax=ax,
            linestyle="dotted",
            color="sienna",
            linewidth=gas.max_cap_M_m3_per_d / 30,
            alpha=0.7,
            zorder=1,
        )

    if show_co2_areas:
        co2.plot(
            ax=ax,
            facecolor="none",
            edgecolor="grey",
            hatch="////",
            linewidth=0.5,
            zorder=0,
        )

    ax.scatter(
        lng.geometry.x,
        lng.geometry.y,
        c="orange",
        s=lng.CapacityInMtpa.astype(float) * 15,
        marker="^",
        alpha=0.8,
        edgecolor="face",
        linewidth=0,
        zorder=11,
    )

    ax.coastlines()
    ax.add_feature(cfeature.BORDERS)
    ax.gridlines(
        draw_labels=True, linewidth=1, color="lightgray", alpha=0.5, linestyle="--"
    )

    # Region presets:
    # sweden
    # ax.set_extent([7, 26, 53.5, 70], crs=ccrs.PlateCarree())
    # finland
    # ax.set_extent([17, 32.5, 58, 72], crs=ccrs.PlateCarree())
    # baltics
    # ax.set_extent([17, 29, 52.5, 61.5], crs=ccrs.PlateCarree())
    # denmark
    # ax.set_extent([4, 16, 53, 59], crs=ccrs.PlateCarree())
    # iberian peninsula
    # ax.set_extent([-14, 6, 34, 46], crs=ccrs.PlateCarree())
    # france
    # ax.set_extent([-9, 10.5, 40.5, 52.5], crs=ccrs.PlateCarree())
    # nrw
    # ax.set_extent([4, 9.7, 50, 52.5], crs=ccrs.PlateCarree())
    # berlin
    # ax.set_extent([12.5, 14, 52, 53], crs=ccrs.PlateCarree())
    # GB and Ireland
    # ax.set_extent([-12, 5, 49.5, 61.5], crs=ccrs.PlateCarree())

    ax.set_extent(map_extent, crs=ccrs.PlateCarree())

    cx.add_basemap(ax, source=cx.providers.CartoDB.Voyager, alpha=0.7, zoom=10)

    legend_items = []

    if show_power_lines:
        legend_items.extend(
            [
                mpatches.Patch(
                    color="none",
                    label=r"$\bf{Stromleitungen}$" + "\n(Breite ~ Spannung)",
                ),
                mlines.Line2D(
                    [],
                    [],
                    alpha=0.7,
                    color="mediumorchid",
                    lw=380 / 200,
                    label="≥380 kV",
                ),
                mlines.Line2D(
                    [],
                    [],
                    alpha=0.7,
                    color="orangered",
                    lw=250 / 200,
                    label="250-380 kV",
                ),
                mlines.Line2D(
                    [], [], alpha=0.7, color="orange", lw=200 / 200, label="200-250 kV"
                ),
                mlines.Line2D(
                    [],
                    [],
                    alpha=0.7,
                    color="limegreen",
                    lw=110 / 200,
                    label="110-200 kV",
                ),
                mlines.Line2D(
                    [], [], alpha=0.7, color="teal", lw=70 / 200, label="<110 kV"
                ),
                mlines.Line2D(
                    [],
                    [],
                    alpha=0.7,
                    color="dodgerblue",
                    lw=2,
                    ls="--",
                    label="HGÜ-Verbindung",
                ),
            ]
        )

    if show_power_plants:
        legend_items.extend(
            [
                mpatches.Patch(color="none", label=""),
                mpatches.Patch(
                    color="none",
                    label=r"$\bf{Kraftwerke}$" + "\n(Größe ~ Kapazität)",
                ),
                mlines.Line2D(
                    [],
                    [],
                    alpha=0.7,
                    color="royalblue",
                    marker="o",
                    lw=0,
                    markersize=8,
                    label="Wind",
                ),
                mlines.Line2D(
                    [],
                    [],
                    alpha=0.7,
                    color="goldenrod",
                    marker="o",
                    lw=0,
                    markersize=8,
                    label="Solar",
                ),
                mlines.Line2D(
                    [],
                    [],
                    alpha=0.7,
                    color="lightseagreen",
                    marker="o",
                    lw=0,
                    markersize=8,
                    label="Wasserkraft",
                ),
                mlines.Line2D(
                    [],
                    [],
                    alpha=0.7,
                    color="forestgreen",
                    marker="o",
                    lw=0,
                    markersize=8,
                    label="Biomasse",
                ),
                mlines.Line2D(
                    [],
                    [],
                    alpha=0.7,
                    color="olive",
                    marker="o",
                    lw=0,
                    markersize=8,
                    label="Abfall",
                ),
                mlines.Line2D(
                    [],
                    [],
                    alpha=0.7,
                    color="deeppink",
                    marker="o",
                    lw=0,
                    markersize=8,
                    label="Kernkraft",
                ),
                mlines.Line2D(
                    [],
                    [],
                    alpha=0.7,
                    color="black",
                    marker="o",
                    lw=0,
                    markersize=8,
                    label="Öl",
                ),
                mlines.Line2D(
                    [],
                    [],
                    alpha=0.7,
                    color="orangered",
                    marker="o",
                    lw=0,
                    markersize=8,
                    label="Gas",
                ),
                mlines.Line2D(
                    [],
                    [],
                    alpha=0.7,
                    color="grey",
                    marker="o",
                    lw=0,
                    markersize=8,
                    label="Kohle",
                ),
            ]
        )

    if show_industry:
        legend_items.extend(
            [
                mpatches.Patch(color="none", label=""),
                mpatches.Patch(
                    color="none",
                    label=r"$\bf{Industriestandorte}$" + "\n(Größe ~ Emissionen)",
                ),
                mlines.Line2D(
                    [],
                    [],
                    alpha=0.7,
                    color="crimson",
                    marker="s",
                    lw=0,
                    markersize=8,
                    label="Stahl",
                ),
                mlines.Line2D(
                    [],
                    [],
                    alpha=0.7,
                    color="chocolate",
                    marker="s",
                    lw=0,
                    markersize=8,
                    label="Zement",
                ),
                mlines.Line2D(
                    [],
                    [],
                    alpha=0.7,
                    color="mediumslateblue",
                    marker="s",
                    lw=0,
                    markersize=8,
                    label="Raffinerien",
                ),
                mlines.Line2D(
                    [],
                    [],
                    alpha=0.7,
                    color="gold",
                    marker="s",
                    lw=0,
                    markersize=8,
                    label="Chemikalien",
                ),
                mlines.Line2D(
                    [],
                    [],
                    alpha=0.7,
                    color="dimgray",
                    marker="s",
                    lw=0,
                    markersize=8,
                    label="Sonstige",
                ),
            ]
        )

    if show_gas_pipelines:
        legend_items.extend(
            [
                mpatches.Patch(color="none", label=""),
                mpatches.Patch(
                    color="none",
                    label=r"$\bf{Gasleitungen}$" + "\n(Breite ~ Durchmesser)",
                ),
                mlines.Line2D([], [], color="sienna", lw=2, ls=":", label="Pipeline"),
            ]
        )

    legend_items.extend(
        [
            mpatches.Patch(color="none", label=""),
            mpatches.Patch(
                color="none", label=r"$\bf{Choropleth-Ebene}$" + "\n(Farbe ~ Nachfrage)"
            ),
        ]
    )

    if show_network_zones:
        zones_url = "https://raw.githubusercontent.com/bobbyxng/de-beschaffungsregionen/main/geojson_nat_boundaries/de-blindleistung-shapes-nat-boundaries.geojson"
        zones = gpd.read_file(zones_url)
        zones = zones.to_crs(crs)
        zone_boundaries = zones.copy()
        zone_boundaries["geometry"] = zone_boundaries.geometry.boundary

        zone_boundaries.plot(
            ax=ax,
            color="white",
            linewidth=6,
            alpha=1.0,
            zorder=15,
        )

        zone_boundaries.plot(
            ax=ax,
            color="darkred",
            linewidth=2,
            alpha=0.9,
            zorder=16,
        )

        for idx, row in zones.iterrows():
            zone_name = row.get("name", f"Zone {idx}")
            centroid = row.geometry.centroid
            ax.text(
                centroid.x,
                centroid.y,
                zone_name,
                fontsize=6,
                fontweight="bold",
                color="darkred",
                ha="center",
                va="center",
                zorder=17,
                alpha=0.8,
                bbox=dict(
                    boxstyle="round,pad=0.2",
                    facecolor="white",
                    alpha=0.7,
                    edgecolor="none",
                ),
            )

        legend_items.append(mpatches.Patch(color="none", label=""))
        legend_items.append(
            mpatches.Patch(
                color="none",
                label=r"$\bf{Schwarzstart}$" + "\n" + r"$\bf{Beschaffungsregionen}$",
            )
        )
        legend_items.append(
            mlines.Line2D(
                [],
                [],
                alpha=0.8,
                color="darkred",
                lw=1.5,
                label="Zonengrenzen" + "\n" + "(14 Regionen)",
            )
        )

    leg = ax.legend(
        handles=legend_items,
        loc="upper left",
        frameon=True,
        framealpha=1,
        fontsize=8,
        title_fontsize=10,
        handlelength=3,
        handletextpad=0.9,
        borderpad=0.8,
        labelspacing=0.5,
    )
    leg.set_zorder(20)

    output_path = f"{output_filename}.{output_format}"
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.show()
    print(f"Saved: {output_path}")


# %% Run plot with toggles
plot_germany(
    show_power_lines=True,
    show_power_plants=True,
    show_industry=True,
    show_gas_pipelines=False,
    show_co2_areas=False,
    show_network_zones=True,
    output_format="png",
    output_filename="DE_network_system",
    map_extent=(2.2, 16, 46.5, 56),  # germany_aegis
)

# %%
