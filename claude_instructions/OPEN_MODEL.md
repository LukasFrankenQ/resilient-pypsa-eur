# Loading PyPSA Networks

## File format and location

Networks are NetCDF (`.nc`) files in `./results/networks/`.

```python
import pypsa
n = pypsa.Network("./results/networks/<filename>.nc")
```

## Filename convention

```
base_s_50__3H-T-H-B-I-A-dist1_2030_{scenario}_{wiggle}{hike}.nc
```

### Parameters

| Parameter | Values | Default |
|---|---|---|
| `scenario` | `free`, `free+slow`, `free+medium`, `free+fast` | `free` |
| `wiggle` | Integer, 0–4000 in steps of 250 | — (always present) |
| `hike` | `""` (empty) or `_{integer}` | `""` (empty) |

- `scenario`: `free` = unconstrained. The `+slow/+medium/+fast` suffixes indicate constrained industry heat pump roll-out rates.
- `wiggle`: gas budget sweep parameter (0 = most constrained, 4000 = least constrained).
- `hike`: optional secondary sweep parameter. When present, prefixed with `_` (e.g., `_100`).

### Examples

```python
# Default scenario, wiggle=1000, no hike
"base_s_50__3H-T-H-B-I-A-dist1_2030_free_1000.nc"

# Slow heat pump roll-out, wiggle=2000, hike=500
"base_s_50__3H-T-H-B-I-A-dist1_2030_free+slow_2000_500.nc"

# Default scenario, wiggle=0, no hike
"base_s_50__3H-T-H-B-I-A-dist1_2030_free_0.nc"
```

## Constructing filenames programmatically

```python
from pathlib import Path

NETWORK_DIR = Path("./results/networks")
PREFIX = "base_s_50__3H-T-H-B-I-A-dist1_2030"

def network_path(scenario="free", wiggle=0, hike=None):
    hike_str = f"_{hike}" if hike is not None else ""
    return NETWORK_DIR / f"{PREFIX}_{scenario}_{wiggle}{hike_str}.nc"

# Load a network
n = pypsa.Network(network_path(scenario="free", wiggle=1000))
```

**Always use a helper function like this** — never construct filenames by string concatenation inline.

## Listing available networks

```python
import glob
networks = sorted(glob.glob(str(NETWORK_DIR / f"{PREFIX}_*.nc")))
```

## Anti-patterns

- **Don't hardcode full filenames.** Use the helper function or derive from parameters.
- **Don't assume a network file exists.** Check with `Path.exists()` before loading.
- **Don't parse scenario parameters from filenames with fragile splits.** If you need to extract `wiggle` or `scenario` from a filename, use a regex:

```python
import re
pattern = re.compile(
    rf"{re.escape(PREFIX)}_(?P<scenario>[^_]+(?:\+\w+)?)"
    r"_(?P<wiggle>\d+)"
    r"(?:_(?P<hike>\d+))?\.nc$"
)
match = pattern.match(Path(filepath).name)
```
