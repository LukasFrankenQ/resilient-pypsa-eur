"""Targets of the fleet-by-shock matrix: fleet from the "<x>b" solve
dispatched at gas markup y, for x, y in HIKES and both gas scenarios.

Existing names are reused where they pose the same problem: the x = 0
column is "<y>u", the diagonal "<n>d" and the y = 0 row "<x>p". Two
duplicates ("0d40", "40d40") re-solve existing cells as a check.

usage: python deeper/matrix_targets.py > matrix_targets.txt
"""

STEM = "results/networks/base_s_50_lv1.25_3H-T-H-B-I-A-dist1_2030_free_{wiggle}_{hike}.nc"
WIGGLES = [2000, 3000]
HIKES = [0, 10, 20, 30, 40, 50, 60]
CHECKS = ["0d40", "40d40"]


def cell(x, y):
    """Hike wildcard of the run with the x fleet hit by shock y."""
    if x == 0:
        return f"{y}u"
    if y == 0:
        return f"{x}p"
    return f"{y}d" if x == y else f"{y}d{x}"


def targets():
    for wiggle in WIGGLES:
        hikes = [f"{x}b" for x in HIKES]
        hikes += [cell(x, y) for x in HIKES for y in HIKES]
        hikes += CHECKS
        for hike in dict.fromkeys(hikes):
            yield STEM.format(wiggle=wiggle, hike=hike)


if __name__ == "__main__":
    print("\n".join(targets()))
