# src/data_loading.py
from pathlib import Path
from typing import Optional

import pandas as pd

from src.config import DEFAULT_RAW_FILE


def load_raw_data(path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load raw smart meter data (5-sec resolution) and standardize column names.

    Assumes the file has at least these columns:
        timeStamp, Power, Voltage, Current, PF

    We use 'timeStamp' as the true timestamp (includes seconds, e.g. 2025-02-03T23:58:55Z).
    We DO NOT drop duplicates; every row is kept as a 5-sec sample.
    """
    if path is None:
        path = DEFAULT_RAW_FILE

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Raw data file not found at: {path}")

    # Auto-detect CSV vs Excel
    if path.suffix.lower() in [".xlsx", ".xls"]:
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)

    # Check required columns
    required_cols = ["timeStamp", "Power", "Voltage", "Current", "PF"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in raw data: {missing}")

    # Parse the high-resolution timestamp
    df["timestamp"] = pd.to_datetime(df["timeStamp"], utc=True, errors="coerce")
    df["timestamp"] = df["timestamp"].dt.tz_localize(None)  # drop timezone for simplicity

    # Drop rows where timestamp could not be parsed
    df = df.dropna(subset=["timestamp"]).copy()

    # Standardize numeric columns
    df["power"] = df["Power"].astype(float)
    df["voltage"] = df["Voltage"].astype(float)
    df["current"] = df["Current"].astype(float)
    df["pf"] = df["PF"].astype(float)

    # Keep only standardized columns, sorted
    df = df[["timestamp", "power", "voltage", "current", "pf"]].sort_values("timestamp")
    df.reset_index(drop=True, inplace=True)

    return df
