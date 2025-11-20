import pandas as pd


def _extract_scenario_values(path, sheet_name, row_label):
    """
    Generic extractor for TYNDP 2024 scenario tables of the form:

        [0]      [1]          [2]             [3]             [4] ...
        NaN   Title row...
        NaN   NaN           Reference   National Trends(+)   ...
        NaN   NaN              2019                2030      ...

        NaN  <row_label>   value_ref   value_NT_2030   value_NT_2040 ...

    Returns a nested dict:
    {
        'National Trends': {2030: x, 2040: y, ...},
        'Distributed Energy': {...},
        'Global Ambition': {...}
    }
    """
    # Read raw (no header) to keep full structure
    df = pd.read_excel(path, sheet_name=sheet_name, header=None)

    # 1. Find the scenario header row (the one that contains "National Trends")
    scen_row_candidates = []
    for idx in range(len(df)):
        row = df.iloc[idx, :]
        if row.astype(str).str.contains('National Trends', na=False).any():
            scen_row_candidates.append(idx)

    if not scen_row_candidates:
        for idx in range(len(df)):
            row = df.iloc[idx, :]
            if row.astype(str).str.contains('Reference', na=False).any():
                scen_row_candidates.append(idx)
        # raise ValueError(f"No scenario header row found in sheet {sheet_name}")

    scen_row = scen_row_candidates[0]
    year_row = scen_row + 1

    # 2. Map column -> scenario name (propagate across empty cells)
    scenario_by_col = {}
    last_scen = None
    for col in range(2, df.shape[1]):
        val = df.iat[scen_row, col]
        if pd.notna(val):
            sval = str(val).strip()
            if sval.startswith('National Trends'):
                last_scen = 'National Trends'
            elif sval.startswith('Distributed Energy'):
                last_scen = 'Distributed Energy'
            elif sval.startswith('Global Ambition'):
                last_scen = 'Global Ambition'
            elif sval.startswith('Reference'):
                last_scen = 'Reference'
            else:
                last_scen = sval
        scenario_by_col[col] = last_scen

    # 3. Map column -> year (from the row below the scenario names)
    year_by_col = {}
    for col in range(2, df.shape[1]):
        val = df.iat[year_row, col]
        if pd.isna(val):
            continue
        try:
            year = int(val)
        except (TypeError, ValueError):
            continue
        year_by_col[col] = year

    # 4. Find the row for the requested label (column 1)
    label_col = 1
    candidates = df.index[df.iloc[:, label_col] == row_label].tolist()
    if not candidates:
        raise ValueError(f"Row label '{row_label}' not found in sheet {sheet_name}")
    r = candidates[0]

    # 5. Build result dict (ignore the "Reference" column / 2019)
    result = {'National Trends': {}, 'Distributed Energy': {}, 'Global Ambition': {}, 'Reference': {}}

    for col, scen in scenario_by_col.items():
        if scen not in result:
            continue
        year = year_by_col.get(col)
        if year is None:
            continue
        val = df.iat[r, col]
        if pd.isna(val):
            continue
        result[scen][year] = float(val)

    return result


def _extract_scenario_values_rowwise(path, sheet_name, carrier_label):
    """
    For tables like sheet '25-' where:
      - One header row contains "Scenario" and "Year"
      - Scenario names are in a column
      - Years are in a column
      - Energy carriers (e.g. 'Methane') are in that header row across columns

    Returns:
    {
        'National Trends': {2030: x, 2040: y, ...},
        'Distributed Energy': {...},
        'Global Ambition': {...}
    }
    """
    df = pd.read_excel(path, sheet_name=sheet_name, header=None)

    # 1. Find header row (contains both "Scenario" and "Year")
    header_row = None
    for i in range(len(df)):
        row_str = df.iloc[i, :].astype(str)
        if row_str.str.contains("Scenario", na=False).any() and \
           row_str.str.contains("Year", na=False).any():
            header_row = i
            break

    if header_row is None:
        raise ValueError(f"No header row with 'Scenario' and 'Year' found in sheet {sheet_name}")

    header = df.iloc[header_row, :]

    # 2. Identify scenario and year columns
    scen_col = None
    year_col = None
    for c, val in enumerate(header):
        s = str(val).strip()
        if "Scenario" in s:
            scen_col = c
        if s.startswith("Year"):
            year_col = c

    if scen_col is None or year_col is None:
        raise ValueError(f"Could not identify Scenario/Year columns in sheet {sheet_name}")

    # 3. Find the column for the requested carrier (e.g. 'Methane')
    target_col = None
    for c, val in enumerate(header):
        if pd.isna(val):
            continue
        if str(val).strip() == carrier_label:
            target_col = c
            break

    if target_col is None:
        raise ValueError(f"Carrier label '{carrier_label}' not found in header row of sheet {sheet_name}")

    # 4. Walk down the table and collect values
    result = {
        'National Trends': {},
        'Distributed Energy': {},
        'Global Ambition': {}
    }

    current_scen = None

    for r in range(header_row + 1, len(df)):
        scen_raw = df.iat[r, scen_col]

        # Update current scenario when a label appears
        if isinstance(scen_raw, str) and scen_raw.strip():
            s = scen_raw.strip()
            if "National Trends" in s:
                current_scen = "National Trends"
            elif "Distributed Energy" in s:
                current_scen = "Distributed Energy"
            elif "Global Ambition" in s:
                current_scen = "Global Ambition"
            else:
                current_scen = None  # ignore other possible rows

        if current_scen not in result:
            continue

        year_raw = df.iat[r, year_col]
        try:
            year = int(year_raw)
        except (TypeError, ValueError):
            continue

        val = df.iat[r, target_col]
        if pd.isna(val):
            continue

        if year not in result[current_scen]:
            result[current_scen][year] = float(val)

    return result
