"""Canonical forest/biomass-themed colour palette.

Pleasant, diverse earthy tones for biomass feedstock carriers, shared across
the biomass figures so colours stay consistent between plots.
"""

FOREST_COLORS = {
    "industry wood": "#D4A95E",                 # honey wood
    "landscape care": "#9CAF61",                # soft sage
    "unsustainable solid biomass": "#5F7138",   # deep olive-green
    "forest residues": "#8C5E3C",               # warm bark brown
    "solid biomass": "#B5A24A",                 # warm khaki-olive (generic)
    "solid biomass import": "#E0CDA0",          # pale wood
    "municipal solid waste": "#A6B36A",         # pale moss
    "straw": "#DAA520",                         # gold
    "manure and slurry": "#7A6B4F",             # earthy taupe
    "sewage sludge": "#6F7D54",                 # muted fern
    "unsustainable biogas": "#5C5230",          # dark olive-brown
}

# On-theme moss-green fallback for any unmapped carrier.
DEFAULT = "#8A9A5B"


def pick_color(carrier, fallback_colors=None):
    """Forest palette first, then an optional fallback dict, then moss green."""
    if carrier in FOREST_COLORS:
        return FOREST_COLORS[carrier]
    if fallback_colors and fallback_colors.get(carrier):
        return fallback_colors[carrier]
    return DEFAULT
