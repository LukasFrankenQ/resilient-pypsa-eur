"""Shared helper to export figure data as JSON under <repo_root>/frontend_data/.

Each figure-producing script imports `export_frontend_data` and calls it with the
figure stem (no `.pdf`, no subdir) plus a payload dict that captures the values
actually rendered in the matplotlib figure. Pandas / numpy types are coerced
automatically by `_json_default`.
"""
from __future__ import annotations

import json
import math
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "frontend_data"


def _json_default(obj: Any):
    if isinstance(obj, (pd.Timestamp, pd.Period, datetime, date)):
        return obj.isoformat()
    if isinstance(obj, pd.DataFrame):
        return obj.reset_index().to_dict(orient="list")
    if isinstance(obj, pd.Series):
        return obj.to_dict()
    if isinstance(obj, pd.Index):
        return obj.tolist()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        val = obj.item()
        if isinstance(val, float) and not math.isfinite(val):
            return None
        return val
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, set):
        return sorted(obj)
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def export_frontend_data(name: str, payload: dict) -> Path:
    """Write `payload` as JSON to <repo_root>/frontend_data/<name>.json.

    Strips any directory prefix or `.pdf` suffix passed in `name`.
    """
    stem = Path(name).stem
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{stem}.json"
    with open(out_path, "w") as fp:
        json.dump(payload, fp, indent=2, default=_json_default)
    return out_path
