import math
import pandas as pd


def _find_sheet_name(xls: pd.ExcelFile, contents_df: pd.DataFrame, keyword: str) -> str:
    """
    Use the 'Contents' sheet to find the sheet name whose description
    contains `keyword` (e.g. 'Wind Onshore', 'Prosumer Battery').
    """
    col = contents_df["SUPPLY INPUTS "].dropna()
    for entry in col:
        entry_str = str(entry)
        if keyword.lower() in entry_str.lower():
            # First token is the sheet prefix, e.g. "1.2." from "1.2. Wind Onshore Trajectories"
            prefix = entry_str.split()[0]
            for name in xls.sheet_names:
                if name.strip().startswith(prefix):
                    return name
    raise ValueError(f"Could not find sheet for keyword {keyword!r}")


def _find_country_col_idx(df: pd.DataFrame, header_row_idx: int = 2) -> int:
    """
    Find the column index whose header (in header_row_idx) is 'Country'.
    """
    for i in range(df.shape[1]):
        val = df.iat[header_row_idx, i]
        if isinstance(val, str) and val.strip().lower() == "country":
            return i
    raise ValueError("Could not find 'Country' column")


def _find_scenario_year_col(
    df: pd.DataFrame,
    scenario_label: str,
    year: int,
    label_row_idx: int = 1,
    header_row_idx: int = 2,
) -> int:
    """
    Given a scenario label as it appears in row `label_row_idx` (e.g. 'LOW',
    'Best Estimate', 'BEST ESTIMATE (BE)', 'Distributed Energy', etc.),
    find which column corresponds to the given `year` within that scenario block.
    """
    scenario_label_low = scenario_label.lower()
    start_idx = None

    # 1) Find the first column where the label row contains the scenario label
    for i in range(df.shape[1]):
        cell = df.iat[label_row_idx, i]
        if isinstance(cell, str) and scenario_label_low in cell.lower():
            start_idx = i
            break

    if start_idx is None:
        raise ValueError(f"Scenario label {scenario_label!r} not found in row {label_row_idx}")

    # 2) Find where this scenario block ends (next non-empty label cell)
    end_idx = df.shape[1]
    for j in range(start_idx + 1, df.shape[1]):
        cell = df.iat[label_row_idx, j]
        if isinstance(cell, str) and cell.strip() != "":
            end_idx = j
            break

    # 3) Within the scenario block, find the column whose header row has the requested year
    for j in range(start_idx, end_idx):
        v = df.iat[header_row_idx, j]
        if isinstance(v, (int, float)) and not math.isnan(v) and int(v) == int(year):
            return j

    raise ValueError(f"Year {year} not found for scenario {scenario_label!r}")


def _extract_from_sheet(
    xls: pd.ExcelFile,
    sheet_name: str,
    year: int,
    scenario_labels: dict,
) -> dict:
    """
    Core extractor for a 'trajectory' sheet.

    scenario_labels: mapping like
        {"LOW": "LOW", "BE": "Best Estimate", "HIGH": "HIGH"}
    or for nuclear:
        {"LOW": "Distributed Energy", "BE": "National Trends+", "HIGH": "Global Ambition"}
    """
    df = pd.read_excel(xls, sheet_name=sheet_name)

    label_row_idx = 1   # row with scenario names (LOW, BE, HIGH, etc.)
    header_row_idx = 2  # row with 'Country' and years

    country_col_idx = _find_country_col_idx(df, header_row_idx=header_row_idx)

    # Map each of LOW / BE / HIGH to the correct column index for this year
    col_map = {}
    for key, label in scenario_labels.items():
        col_map[key] = _find_scenario_year_col(
            df,
            scenario_label=label,
            year=year,
            label_row_idx=label_row_idx,
            header_row_idx=header_row_idx,
        )

    out = {}
    # Data rows start after the header rows
    for r in range(header_row_idx + 1, df.shape[0]):
        country = df.iat[r, country_col_idx]
        if not isinstance(country, str) or not country.strip():
            continue
        country = country.strip()

        # Keep only 4-character country codes (AL00, DE00, UKNI, etc.)
        if len(country) != 4:
            continue

        vals = {}
        for key, col_idx in col_map.items():
            val = df.iat[r, col_idx]
            if isinstance(val, float) and math.isnan(val):
                val = None
            elif isinstance(val, (int, float)):
                val = float(val)
            vals[key] = val

        out[country] = vals

    return out


def extract_supply_inputs(year: int, carrier: str, path: str) -> dict:
    """
    Extract LOW, BE (best estimate), and HIGH trajectories for a given carrier and year.

    Parameters
    ----------
    year : int
        2030 or 2040.
    carrier : str
        One of: 'onwind', 'offwind', 'solar', 'battery_prosumer', 'battery_utility', 'nuclear'.
    path : str
        Path to the Excel workbook
        "20231103 - Final Supply Inputs for TYNDP 2024 Scenarios.xlsx".

    Returns
    -------
    dict
        {country_code: {"LOW": float or None, "BE": float or None, "HIGH": float or None}}
    """
    year = int(year)
    if year not in (2030, 2040):
        raise ValueError("year must be 2030 or 2040")

    carrier = carrier.lower()

    with pd.ExcelFile(path) as xls:
        contents_df = pd.read_excel(xls, sheet_name="Contents")

        if carrier == "solar":
            sheet = _find_sheet_name(xls, contents_df, "Solar Trajectories")
            scenario_labels = {"LOW": "LOW", "BE": "Best Estimate", "HIGH": "HIGH"}
            return _extract_from_sheet(xls, sheet, year, scenario_labels)

        elif carrier == "onwind":
            sheet = _find_sheet_name(xls, contents_df, "Wind Onshore")
            # In the sheet this appears as "BEST ESTIMATE (BE)"
            scenario_labels = {"LOW": "LOW", "BE": "BEST ESTIMATE", "HIGH": "HIGH"}
            return _extract_from_sheet(xls, sheet, year, scenario_labels)

        elif carrier == "offwind":
            sheet = _find_sheet_name(xls, contents_df, "Wind Offshore")
            scenario_labels = {"LOW": "LOW", "BE": "BEST ESTIMATE", "HIGH": "HIGH"}
            return _extract_from_sheet(xls, sheet, year, scenario_labels)

        elif carrier == "nuclear":
            sheet = _find_sheet_name(xls, contents_df, "Nuclear Ex-ante capacities")
            # Interpret TYNDP scenarios as LOW / BE / HIGH:
            #   Distributed Energy   -> LOW
            #   National Trends+     -> BE (Best Estimate)
            #   Global Ambition      -> HIGH
            scenario_labels = {
                "LOW": "Distributed Energy",
                "BE": "National Trends+",
                "HIGH": "Global Ambition",
            }
            return _extract_from_sheet(xls, sheet, year, scenario_labels)

        elif carrier == "home battery":
            # Prosumer battery capacities (in MWh)
            prosumer_sheet = _find_sheet_name(xls, contents_df, "Prosumer Battery")
            scenario_labels = {"LOW": "LOW", "BE": "Best Estimate", "HIGH": "HIGH"}
            return _extract_from_sheet(xls, prosumer_sheet, year, scenario_labels)

        elif carrier == "battery":
            # Utility battery capacities (in MWh)
            utility_sheet = _find_sheet_name(xls, contents_df, "Utility Battery")
            scenario_labels = {"LOW": "LOW", "BE": "Best Estimate", "HIGH": "HIGH"}
            return _extract_from_sheet(xls, utility_sheet, year, scenario_labels)

        else:
            raise ValueError("carrier must be one of: 'onwind', 'offwind', 'solar', 'home battery', 'battery', 'nuclear'")


def to_dataframe(data):
    df = pd.DataFrame(data).T
    for col in df.columns:
        df[col] = df[col].apply(lambda x: str(x).replace('\xa0', '').replace(',', '').strip() if pd.notna(x) else x)
    df = df.astype(float).groupby(df.index.str[:2]).sum()
    df = df.rename({'UK': 'GB'}).sort_index()
    return df.rename(columns={'BE': 'NT'})


if __name__ == "__main__":

    year = int(snakemake.wildcards.planning_horizons)
    assert year in [2025, 2030, 2035, 2040], "year must be 2025, 2030, 2035, or 2040"

    path = snakemake.input.tyndp_supply_inputs

    carriers = ['onwind', 'offwind', 'solar', 'home battery', 'battery', 'nuclear']

    if year == 2025:
        # Save an empty dataframe for 2025
        caps = pd.DataFrame()
    elif year == 2035:
        # Linear interpolation between 2030 and 2040
        caps_2030 = []
        caps_2040 = []

        for c in carriers:
            caps_2030.append(to_dataframe(extract_supply_inputs(2030, c, path)))
            caps_2040.append(to_dataframe(extract_supply_inputs(2040, c, path)))

        caps_2030 = pd.concat(caps_2030, axis=1, keys=carriers)
        caps_2040 = pd.concat(caps_2040, axis=1, keys=carriers)

        # Linear interpolation: 2035 is halfway between 2030 and 2040
        caps = caps_2030 + (caps_2040 - caps_2030) * 0.5
    else:
        # For 2030 and 2040, extract directly
        caps = []
        for c in carriers:
            caps.append(to_dataframe(extract_supply_inputs(year, c, path)))
        caps = pd.concat(caps, axis=1, keys=carriers)

    caps.to_csv(snakemake.output.tyndp_capacities)
