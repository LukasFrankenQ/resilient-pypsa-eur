import pandas as pd





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

    add_shared_row_title(fig, ax_left, ax_right, "Gas Supply", row_idx)

    right_data = right_data.loc[right_data.index.str.contains('2024')]

    stack_cols = ['Production', 'Pipeline_Import', 'LNG_Sendout', 'Net_Storage_Withdrawal', 'Re_exports']
    nice_names_data = {
        'Pipeline_Import': 'Pipeline Imports',
        'LNG_Sendout': 'LNG Liquification',
        'Net_Storage_Withdrawal': 'Storage Withdrawal',
        'Re_exports': 'Re-exports',
    }
    model_data = get_gas_supply(n)

    quarters = right_data.index

    colors = ['#4daf4a', '#377eb8', '#ff7f00', '#984ea3', '#e41a1c']
    labels = stack_cols

    bar_width = 0.7
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

        pos_bottoms = [0] * len(quarters_plot)
        neg_bottoms = [0] * len(quarters_plot)
        bar_handles = []

        for i, (col, color, label) in enumerate(zip(stack_cols, colors, labels)):
            # If column missing in model_data, fill with zeros
            if col not in data.columns:
                values = [0] * len(quarters_plot)
            else:
                values = data[col].values
            pos_vals = [v if v >= 0 else 0 for v in values]
            neg_vals = [v if v < 0 else 0 for v in values]
            nice_label = nice_names_data.get(label, label)

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
        if 'Total_Supply' in data.columns:
            total_supply = data['Total_Supply'].values
            ax.plot(range(len(quarters_plot)), total_supply, 'ko-', label='Total Supply', zorder=10, alpha=0.8)
            for i, ts in enumerate(total_supply):
                ax.text(i, ts + 25, f"{ts:.1f}", ha='center', va='bottom', fontsize=9, color='black')

        ax.set_ylabel("TWh")

        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)

        # Add subtle y-grid
        ax.yaxis.grid(True, which='major', linestyle='--', linewidth=0.5, alpha=0.7)
        ax.set_axisbelow(True)

        # Add space between bars and border
        ax.set_xlim(-0.5, len(quarters_plot) - 0.5)

        # Set common y-limits for both axes
        ax.set_ylim(ylim_lower, ylim_upper)

        # Legend to the left, no title (only for right side)
        if not is_left:
            ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1.2), frameon=False)

        ax.set_xticks(range(len(quarters_plot)))
        ax.set_xticklabels(quarters_plot, rotation=0)

        ax.axhline(0, color='black', linewidth=0.5)

        # For left side, keep axis visible (for now), but could be ax.axis('off') if desired
        if is_left:
            pass  # Optionally: ax.axis('off')