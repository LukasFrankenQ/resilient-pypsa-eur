# Plotting Conventions

## Colors

Load carrier colors from the config — **never hardcode colors for carriers.**

```python
import yaml

with open("./config.basicrun.yaml") as f:
    config = yaml.safe_load(f)

tech_colors = config["plotting"]["tech_colors"]

# Usage
color = tech_colors[carrier]
```

If a carrier is missing from `tech_colors`, flag it rather than inventing a color.

## Style defaults

Apply these to **every** plot unless explicitly told otherwise:

```python
import matplotlib.pyplot as plt

def style_ax(ax):
    """Apply standard style to an axes object."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(frameon=True)
```

### Rules

1. **Remove top and right spines** on every plot.
2. **Subtle y-grid** — use `alpha=0.3` and `set_axisbelow(True)` so grid lines sit behind data.
3. **Legend frame on** — always `frameon=True`.
4. **Carrier colors from config** — always use `tech_colors` for any carrier-based coloring (stacked areas, bar charts, pie charts, legends).
5. **Saving** — always save a copy of the figure as **PDF only** to (ROOT is the repo root) `ROOT.parent / 'gas_resilience' / 'imgs'`

## Anti-patterns

- **Don't use matplotlib's default color cycle for carriers.** Always map through `tech_colors`.
- **Don't leave default spines.** Every axes must have top/right spines removed.
- **Don't place grid lines on top of data.** Always call `set_axisbelow(True)`.
- **Don't create legends without frames.**
