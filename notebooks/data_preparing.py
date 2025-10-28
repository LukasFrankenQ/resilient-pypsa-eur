import pandas as pd
from pathlib import Path


industry_demands = ['gas for industry', 'gas for industry CC', 'SMR', 'SMR CC']
rescom_demands = ['services rural gas boiler', 'services urban decentral gas boiler', 'urban central gas boiler', 'urban central gas CHP', 'urban central gas CHP CC']
remove_countries = ['NO', 'AL', 'BA', 'ME', 'MK', 'NO', 'RS', 'XK']


def basic_axis_formatting(ax, ylabel=None, xlim=None, ylim=None, xticks=None, xticklabels=None, xrotation=0):
    """
    Apply basic formatting to a matplotlib axis.
    """
    if ylabel is not None:
        ax.set_ylabel(ylabel)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    ax.yaxis.grid(True, which='major', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.set_axisbelow(True)

    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)

    if xticks is not None:
        ax.set_xticks(xticks)
    if xticklabels is not None:
        ax.set_xticklabels(xticklabels, rotation=xrotation)

    ax.axhline(0, color='black', linewidth=0.5)


def plot_stacked_bars(
    ax,
    data,
    stack_cols,
    colors,
    labels,
    nice_names_data=None,
    bar_width=0.7,
    y_offset=25,
    total_line_name='Total_Supply',
):
    """
    Plot stacked bars (with positive/negative support) on ax for the given data.
    Optionally overlays a total line if present.
    
    Args:
        ax (matplotlib.axes.Axes): The axis to plot on.
        data (pd.DataFrame): DataFrame with index as categories and columns as stack components.
        stack_cols (list): List of column names from data to stack.
        colors (list): List of colors corresponding to each stack_col.
        labels (list): List of labels corresponding to each stack_col.
        nice_names_data (dict, optional): Dictionary mapping labels to display names. Defaults to None.
        bar_width (float, optional): Width of the bars. Defaults to 0.7.
        y_offset (float, optional): Vertical offset for total line text annotations. Defaults to 25.
        total_line_name (str, optional): Column name for the total supply line. Defaults to 'Total_Supply'.
    
    Returns:
        pd.Index: The index of the data (quarters_plot).
    """
    quarters_plot = data.index
    pos_bottoms = [0] * len(quarters_plot)
    neg_bottoms = [0] * len(quarters_plot)
    bar_handles = []

    for i, (col, color, label) in enumerate(zip(stack_cols, colors, labels)):
        # If column missing in data, fill with zeros
        if col not in data.columns:
            values = [0] * len(quarters_plot)
        else:
            values = data[col].values
        pos_vals = [v if v >= 0 else 0 for v in values]
        neg_vals = [v if v < 0 else 0 for v in values]
        nice_label = nice_names_data.get(label, label) if nice_names_data else label

        # Plot positive part
        if any(v != 0 for v in pos_vals):
            bar = ax.bar(
                range(len(quarters_plot)),
                pos_vals,
                bottom=pos_bottoms,
                color=color,
                label=nice_label if not any(v < 0 for v in values) else None,
                width=bar_width,
                alpha=0.8,
                edgecolor='black',
                linewidth=0.5,
            )
            bar_handles.append(bar)
            pos_bottoms = [b + v for b, v in zip(pos_bottoms, pos_vals)]
        # Plot negative part
        if any(v != 0 for v in neg_vals):
            bar = ax.bar(
                range(len(quarters_plot)),
                neg_vals,
                bottom=neg_bottoms,
                color=color,
                label=nice_label if any(v < 0 for v in values) else None,
                width=bar_width,
                alpha=0.8,
                edgecolor='black',
                linewidth=0.5,
            )
            bar_handles.append(bar)
            neg_bottoms = [b + v for b, v in zip(neg_bottoms, neg_vals)]

    # Plot total supply line if present
    if total_line_name in data.columns:
        total_supply = data[total_line_name].values
        ax.plot(range(len(quarters_plot)), total_supply, 'ko-', label='Total Supply', zorder=10, alpha=0.8)
        for i, ts in enumerate(total_supply):
            ax.text(i, ts + y_offset, f"{ts:.1f}", ha='center', va='bottom', fontsize=9, color='black')

    return quarters_plot


def plot_electricity_mix(
    ax_left,
    ax_right,
    n,
    _,
    fig,
    row_idx
    ):

    add_shared_row_title(fig, ax_left, ax_right, "EU-27 + UK + Electricity Mix", row_idx)

    em = pd.read_csv(
        Path.cwd().parent / 'data' / 'europe_monthly_full_release_long_format.csv', index_col=[1,2]
    )

    em = em.loc[
        (em['Unit'] == 'TWh') &
        (em['Subcategory'] == 'Fuel')
        ]
    em = em.set_index('Variable', append=True)
    em = em.loc[em.Area.isin(['EU', 'United Kingdom'])]

    em = em.groupby([em.index.get_level_values(2), em.index.get_level_values(1).str[:7]])['Value'].sum().unstack()
    em = em.loc[:, em.columns.str.contains('2024')].T

    keepers = [
        'Pumped Hydro Storage',
        'Reservoir & Dam',
        'Offshore Wind (AC)',
        'Offshore Wind (DC)',
        'Offshore Wind (Floating)',
        'Onshore Wind',
        'Run of River',
        'Solar',
        'nuclear',
        'lignite',
        'solar-hsat',
        'Combined-Cycle Gas',
        'Open-Cycle Gas',
        'coal',
        'urban central gas CHP CC',
        'urban central solid biomass CHP',
        'urban central solid biomass CHP CC',
    ]
    grouper = {
        "Offshore Wind (AC)": "Offshore wind",
        "Offshore Wind (DC)": "Offshore wind",
        "Offshore Wind (Floating)": "Offshore wind",
        "Onshore Wind": "Onshore wind",
        "Run of River": "Hydro",
        "Pumped Hydro Storage": "Hydro",
        "Reservoir & Dam": "Hydro",
        "urban central gas CHP CC": "Gas",
        "urban central solid biomass CHP CC": "Bioenergy",
        "urban central solid biomass CHP": "Bioenergy",
        "Open-Cycle Gas": "Gas",
        "Combined-Cycle Gas": "Gas",
        "coal": "Hard coal",
        "lignite": "Lignite",
        "nuclear": "Nuclear",
        "solar": "Solar",
        "solar-hsat": "Solar",
    }
    grouper = {
        origin: grouper.get(origin, origin) for origin in keepers
    }

    weight = n.snapshot_weightings['generators'].iloc[0]
    idx = pd.IndexSlice

    # ns = n.statistics.energy_balance(aggregate_time=False).loc[idx[:,:,'AC']]
    # ns.index = ns.index.get_level_values(1)

    ns = n.statistics.energy_balance(groupby=['bus', 'carrier'], aggregate_time=False)
    ns = ns.loc[~ns.index.get_level_values(1).str.startswith(tuple(remove_countries))]
    ns = ns.loc[ns.index.get_level_values(1) != '']

    ns = ns.loc[list(map(lambda x: n.buses.loc[x, 'carrier'] == 'AC', ns.index.get_level_values(1)))]
    ns = ns.groupby(level=2).sum()

    ns = ns.groupby(grouper).sum().T
    ns = ns.groupby(ns.index.month).sum().T.mul(weight * 1e-6)
    ns.columns = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    ns = ns.T

    energy_colors = {
        'Bioenergy': '#e3d37d',
        'Nuclear': '#ff8c00',
        'Gas': '#db6a25',
        'Hard coal': '#545454',
        'Lignite': '#826837',
        'Hydro': '#298c81',
        'Other fossil': '#95A5A6',
        'Other renewables': '#27AE60',
        'Offshore wind': "#6895dd",
        'Onshore wind': "#235ebc",
        'Solar': '#ffbf2b',
    }

    ns.loc[:, pd.Index(energy_colors.keys()).difference(ns.columns)] = 0
    em.loc[:, pd.Index(energy_colors.keys()).difference(em.columns)] = 0

    ns = ns[list(energy_colors.keys())]
    em = em[list(energy_colors.keys())]
    ymax = max(ns.sum(axis=1).max(), em.sum(axis=1).max())

    for ax, data, is_left in [
        (ax_left, ns, True),
        (ax_right, em, False)
    ]:

        plot_stacked_bars(
            ax, data, ns.columns, energy_colors.values(), ns.columns, bar_width=1
        )

        basic_axis_formatting(
            ax,
            ylabel="TWh",
            xlim=(-0.5, len(ns.index) - 0.5),
            xticks=range(len(ns.index)),
            xticklabels=ns.index,
            ylim=(0, ymax),
            xrotation=90
        )

        if not is_left:
            ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1.2), frameon=False)

        # For left side, keep axis visible (for now), but could be ax.axis('off') if desired
        if is_left:
            pass  # Optionally: ax.axis('off')


def get_model_pipeline_imports(n):

    model_data = pd.DataFrame(
        index=['Q1-2024', 'Q2-2024', 'Q3-2024', 'Q4-2024'],
        columns=['Russia', 'Norway', 'North_Africa', 'Azerbaijan', 'OtherPipe'],
        data=0.
    )

    pipes_from_no = n.links.index[(n.links.bus0.str.contains('NO')) & (n.links.carrier == 'gas pipeline')]
    pipes_to_no = n.links.index[(n.links.bus1.str.contains('NO')) & (n.links.carrier == 'gas pipeline')]

    weights = n.snapshot_weightings['generators']

    imports = n.links_t.p0[pipes_from_no].sum(axis=1).mul(weights)
    exports = n.links_t.p0[pipes_to_no].sum(axis=1).mul(weights)

    imports = imports.groupby(imports.index.to_period('Q')).sum()
    exports = exports.groupby(exports.index.to_period('Q')).sum()

    model_data.loc[:, 'Norway'] = (imports - exports).values

    print('yet to implement other pipeline representation in model!')
    return model_data


# Helper to add a shared title above both axes in a row, centered horizontally
def add_shared_row_title(fig, ax_left, ax_right, title, row_idx):

    fig.canvas.draw()
    bbox_left = ax_left.get_position()
    bbox_right = ax_right.get_position()

    x = (bbox_left.x0 + bbox_right.x1) / 2

    y = bbox_left.y1 + 0.01
    fig.text(x, y, title, ha='center', va='bottom', fontsize=12, fontweight='bold')


def get_gas_supply(n):

    gas_supply = pd.DataFrame(
        0.,
        columns=['Production', 'Pipeline_Import', 'LNG_Sendout', 'Net_Storage_Withdrawal', 'Re_exports', 'Total_Supply'],
        index=['Q1-2024', 'Q2-2024', 'Q3-2024', 'Q4-2024']
    )    

    gens = n.generators.index[n.generators.carrier == 'gas']
    genweights = n.snapshot_weightings['generators']

    prod = n.generators_t.p[gens].sum(axis=1).mul(genweights)

    prod.index = pd.to_datetime(prod.index)

    prod_quarterly = prod.groupby(prod.index.to_period('Q')).sum().values
    gas_supply['Production'] = prod_quarterly * 1e-6

    stores = n.stores.index[n.stores.carrier == 'gas']
    storeweights = n.snapshot_weightings['stores']

    stor = n.stores_t.p[stores].sum(axis=1).mul(storeweights)
    stor.index = pd.to_datetime(stor.index)

    stor_quarterly = stor.groupby(stor.index.to_period('Q')).sum().values

    gas_supply['Net_Storage_Withdrawal'] = stor_quarterly * 1e-6

    gas_supply['Total_Supply'] = (
        gas_supply['Production'] +
        gas_supply['Pipeline_Import'] +
        gas_supply['LNG_Sendout'] +
        gas_supply['Net_Storage_Withdrawal'] +
        gas_supply['Re_exports']
    )

    return gas_supply


def plot_total_gas_supply(ax_left, ax_right, n, right_data, fig, row_idx):
    # Remove title, relax visuals, handle negative bars, etc.

    add_shared_row_title(fig, ax_left, ax_right, "EU-27 + UK Gas Supply", row_idx)

    right_data = right_data.loc[right_data.index.str.contains('2024')]

    stack_cols = ['Production', 'Pipeline_Import', 'LNG_Sendout', 'Net_Storage_Withdrawal', 'Re_exports']
    nice_names_data = {
        'Pipeline_Import': 'Pipeline Imports',
        'LNG_Sendout': 'LNG Liquification',
        'Net_Storage_Withdrawal': 'Storage Withdrawal',
        'Re_exports': 'Re-exports',
    }
    model_data = get_gas_supply(n)

    colors = ['#4daf4a', '#377eb8', '#ff7f00', '#984ea3', '#e41a1c']
    labels = stack_cols

    quarters = right_data.index
    x = range(len(quarters))

    # --- Compute y-limits for both left and right plots ---
    # Prepare both datasets for y-limit calculation
    # Align model_data to right_data index for fair comparison
    if hasattr(model_data, "loc") and hasattr(right_data, "index"):
        model_data_plot = model_data.loc[model_data.index.intersection(right_data.index)]
    else:
        model_data_plot = model_data

    # Helper to get all values to be plotted (stacked bars and total supply)
    def get_all_yvals(data, stack_cols):
        yvals = []
        for col in stack_cols:
            if col in data.columns:
                yvals.extend(data[col].values)
        if 'Total_Supply' in data.columns:
            yvals.extend(data['Total_Supply'].values)
        return yvals

    yvals_left = get_all_yvals(model_data_plot, stack_cols)
    yvals_right = get_all_yvals(right_data, stack_cols)

    # Remove NaNs
    yvals_left = [v for v in yvals_left if pd.notnull(v)]
    yvals_right = [v for v in yvals_right if pd.notnull(v)]

    # Compute min and max for both, then set common limits
    min_y = min(yvals_left + yvals_right) if (yvals_left + yvals_right) else 0
    max_y = max(yvals_left + yvals_right) if (yvals_left + yvals_right) else 1

    # Add a little padding
    y_range = max_y - min_y
    pad = max(10, 0.05 * y_range)
    ylim_lower = min(0, min_y - pad)
    ylim_upper = max_y + pad

    # Plot both model_data (left) and right_data (right) in a loop to avoid code duplication

    # Use the helper for both axes
    for ax, data, is_left in [
        (ax_left, model_data, True),
        (ax_right, right_data, False)
    ]:
        # For model_data, restrict to same quarters as right_data for fair comparison
        if is_left:
            # Try to align model_data index to right_data index if possible
            if hasattr(model_data, "loc") and hasattr(right_data, "index"):
                data = model_data.loc[model_data.index.intersection(right_data.index)]
            quarters_plot = data.index
        else:
            quarters_plot = right_data.index

        # Call the new helper
        plot_stacked_bars(
            ax,
            data,
            stack_cols,
            colors,
            labels,
            nice_names_data=nice_names_data,
            total_line_name='Total_Supply'
        )

        # Apply basic formatting using the helper function
        basic_axis_formatting(
            ax,
            ylabel="TWh",
            xlim=(-0.5, len(quarters_plot) - 0.5),
            ylim=(ylim_lower, ylim_upper),
            xticks=range(len(quarters_plot)),
            xticklabels=quarters_plot
        )

        # Legend to the left, no title (only for right side)
        if not is_left:
            ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1.2), frameon=False)

        # For left side, keep axis visible (for now), but could be ax.axis('off') if desired
        if is_left:
            pass  # Optionally: ax.axis('off')


def plot_eu_gas_stock(ax_left, ax_right, n, right_data, fig, row_idx):

    add_shared_row_title(fig, ax_left, ax_right, "EU-27 Gas Stocks", row_idx)

    gasstores = n.stores.index[
        (n.stores.carrier == 'gas')
        & (~n.stores.index.str[:2].isin(remove_countries))
        ]

    eustock = n.stores_t.e[gasstores].sum(axis=1)# .mul(n.snapshot_weightings['stores'])
    eustock = eustock.groupby(eustock.index.month).mean() / 1e6

    model_data = pd.Series(
        eustock.values,
        index=right_data.index,
        name='EU-27 gas model',
    )

    ylim_lower = min(0, right_data.min().min(), model_data.min())
    ylim_upper = max(right_data.max().max(), model_data.max())

    pad = 1.1

    ax_left.plot(model_data.index, model_data.values, marker='o', color='royalblue')
    ax_right.plot(right_data.index, right_data.values, marker='o', color='royalblue')

    ax_left.set_ylim(ylim_lower, ylim_upper * pad)
    ax_right.set_ylim(ylim_lower, ylim_upper * pad)

    for ax in [ax_left, ax_right]:
        basic_axis_formatting(ax)


def plot_pipeline_imports(ax_left, ax_right, n, right_data, fig, row_idx):

    add_shared_row_title(fig, ax_left, ax_right, "EU-27 + UK Pipeline Imports", row_idx)

    right_data = right_data.loc[right_data.index.str.contains('2024')]

    stack_cols = ["Russia", "Norway", "North_Africa", "Azerbaijan", "OtherPipe"]
    total_line_name = "Total_Pipeline"

    colors = ['#4daf4a', '#377eb8', '#ff7f00', '#984ea3', '#e41a1c']
    labels = stack_cols
    nice_names_data = {
        'Russia': 'Russia',
        'Norway': 'Norway',
        'North_Africa': 'North Africa',
        'Azerbaijan': 'Azerbaijan',
        'OtherPipe': 'Other Pipeline',
    }

    quarters = right_data.index
    x = range(len(quarters))

    model_data = pd.DataFrame(
        get_model_pipeline_imports(n),
        index=right_data.index,
        columns=right_data.columns,
    )

    # --- Compute y-limits for both left and right plots (with padding) ---
    max_y = max(right_data.max().max(), model_data.max().max())
    min_y = min(0, right_data.min().min(), model_data.min().min())
    y_range = max_y - min_y
    pad = max(10, 0.05 * y_range)
    ylim_lower = min(0, min_y - pad)
    ylim_upper = max_y + pad

    # Use the helper for both axes
    for ax, data, is_left in [
        (ax_left, model_data, True),
        (ax_right, right_data, False)
    ]:

        # For model_data, restrict to same quarters as right_data for fair comparison
        if is_left:
            # Try to align model_data index to right_data index if possible
            if hasattr(model_data, "loc") and hasattr(right_data, "index"):
                data = model_data.loc[model_data.index.intersection(right_data.index)]
            quarters_plot = data.index
        else:
            quarters_plot = right_data.index

        # Call the new helper
        plot_stacked_bars(
            ax,
            data,
            stack_cols,
            colors,
            labels,
            nice_names_data=nice_names_data,
            total_line_name=total_line_name
        )

        # Apply basic formatting using the helper function
        basic_axis_formatting(
            ax,
            ylabel="TWh",
            xlim=(-0.5, len(quarters_plot) - 0.5),
            ylim=(ylim_lower, ylim_upper),
            xticks=range(len(quarters_plot)),
            xticklabels=quarters_plot
        )

        # Legend to the left, no title (only for right side)
        if not is_left:
            ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1.2), frameon=False)

        # For left side, keep axis visible (for now), but could be ax.axis('off') if desired
        if is_left:
            pass  # Optionally: ax.axis('off')


def plot_residential_commercial_demand(
    ax_left,
    ax_right,
    n,
    right_data,
    fig,
    row_idx
    ):

    add_shared_row_title(fig, ax_left, ax_right, "EU-27 + UK Residential and Commercial Gas Demand", row_idx)

    ss = n.statistics.withdrawal(groupby=['carrier', 'bus'], aggregate_time=False)
    mask = n.buses.loc[ss.index.get_level_values(2).intersection(n.buses.index)].carrier == 'gas'
    mask.loc[''] = False
    mask = [mask[entry] for entry in ss.index.get_level_values(2)]

    ss = ss.loc[mask].dropna()
    ss = ss.loc[~ss.index.get_level_values(2).str[:2].isin(remove_countries)]

    rescom_consumption = ss.groupby(level=1).sum().T
    rescom_consumption = (
        rescom_consumption[rescom_consumption.columns.intersection(rescom_demands)]
        .sum(axis=1)
        .mul(n.snapshot_weightings['generators'])
    )

    model_data = rescom_consumption.groupby(rescom_consumption.index.to_period('M')).sum().mul(1e-6)
    model_data.index = right_data.index

    ylim_lower = min(0, right_data.min(), model_data.min())
    ylim_upper = max(right_data.max(), model_data.max())

    pad = 1.1

    ax_left.plot(model_data.index, model_data.values, marker='o', color='royalblue')
    ax_right.plot(right_data.index, right_data.values, marker='o', color='royalblue')

    ax_left.set_ylim(ylim_lower, ylim_upper * pad)
    ax_right.set_ylim(ylim_lower, ylim_upper * pad)

    for ax in [ax_left, ax_right]:
        basic_axis_formatting(ax)


def plot_industry_demand(
    ax_left,
    ax_right,
    n,
    right_data,
    fig,
    row_idx
    ):

    add_shared_row_title(fig, ax_left, ax_right, "EU-27 + UK Industry Gas Demand", row_idx)

    ss = n.statistics.withdrawal(groupby=['carrier', 'bus'], aggregate_time=False)
    mask = n.buses.loc[ss.index.get_level_values(2).intersection(n.buses.index)].carrier == 'gas'
    mask.loc[''] = False
    mask = [mask[entry] for entry in ss.index.get_level_values(2)]

    ss = ss.loc[mask].dropna()
    ss = ss.loc[~ss.index.get_level_values(2).str[:2].isin(remove_countries)]

    industry_consumption = ss.groupby(level=1).sum().T[industry_demands].sum(axis=1).mul(n.snapshot_weightings['generators'])

    model_data = industry_consumption.groupby(industry_consumption.index.to_period('M')).sum().mul(1e-6)
    model_data.index = right_data.index

    ylim_lower = min(0, right_data.min(), model_data.min())
    ylim_upper = max(right_data.max(), model_data.max())

    pad = 1.1

    ax_left.plot(model_data.index, model_data.values, marker='o', color='royalblue')
    ax_right.plot(right_data.index, right_data.values, marker='o', color='royalblue')

    ax_left.set_ylim(ylim_lower, ylim_upper * pad)
    ax_right.set_ylim(ylim_lower, ylim_upper * pad)

    for ax in [ax_left, ax_right]:
        basic_axis_formatting(ax)


def plot_country_consumption(ax_left, ax_right, n, right_data, fig, row_idx):


    ss = n.statistics.withdrawal(groupby=['carrier', 'bus'], aggregate_time=False)

    mask = n.buses.loc[ss.index.get_level_values(2).intersection(n.buses.index)].carrier == 'gas'
    mask.loc[''] = False
    mask = [mask[entry] for entry in ss.index.get_level_values(2)]

    ss = ss.loc[mask].dropna()
    ss = ss.loc[~ss.index.get_level_values(2).str[:2].isin(remove_countries)]
    ss = ss.loc[~ss.index.get_level_values(1).isin(['gas pipeline', 'gas pipeline new', 'biogas to gas'])]

    model_data = (
        ss.groupby(ss.index.get_level_values(2).str[:2]).sum()
        .T
        .mul(n.snapshot_weightings['generators'] * 1e-6, axis=0)
        .sum()
    )

    intersection_countries = right_data.index.intersection(model_data.index)

    model_data = model_data.loc[intersection_countries]
    right_data = right_data.loc[intersection_countries]

    right_data.plot.bar(ax=ax_right, color='forestgreen')
    model_data.plot.bar(ax=ax_left, color='forestgreen')

    ylim = max(ax_right.get_ylim()[1], model_data.max())

    add_shared_row_title(fig, ax_left, ax_right, "Country-level 2024 Gas Consumption", row_idx)

    for ax in [ax_left, ax_right]:
        basic_axis_formatting(ax)
        ax.set_ylim(0, ylim * 1.1)
