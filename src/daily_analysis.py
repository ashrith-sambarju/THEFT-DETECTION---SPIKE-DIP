# src/daily_analysis.py

from typing import Tuple

import numpy as np
import pandas as pd

from src.config import (
    SAMPLE_INTERVAL_SECONDS,
    TIME_SEGMENTS,
    ANOMALY_QUANTILE,
)


# ---------- Helpers ----------

def _assign_segment(t: pd.Timestamp) -> str:
    """
    Assign a time-of-day segment label (night/morning/afternoon/evening)
    based on TIME_SEGMENTS defined in config.py.
    """
    tt = t.time()
    for seg_name, (start_t, end_t) in TIME_SEGMENTS.items():
        # inclusive start, inclusive end for simplicity
        if start_t <= tt <= end_t:
            return seg_name
    return "unknown"


def _add_energy_and_segment(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add:
      - date column
      - segment (night/morning/afternoon/evening)
      - per-sample energy in kWh

    Energy per sample (approx):
        E_kWh = Power_W * dt_seconds / (1000 * 3600)
    """
    df = df.copy()
    df["date"] = df["timestamp"].dt.date
    df["segment"] = df["timestamp"].apply(_assign_segment)

    # power in W, SAMPLE_INTERVAL_SECONDS in seconds
    # dt_hours = SAMPLE_INTERVAL_SECONDS / 3600
    dt_hours = SAMPLE_INTERVAL_SECONDS / 3600.0
    df["energy_kwh"] = (df["power"] * dt_hours) / 1000.0

    return df


# ---------- Daily Segment Energy ----------

def compute_daily_segment_energy(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily and segment-wise energy consumption.

    Returns a DataFrame with one row per day, columns:
        date,
        energy_night, energy_morning, energy_afternoon, energy_evening,
        total_energy_kwh
    """
    df = _add_energy_and_segment(df_raw)

    # Sum energy per day per segment
    seg_group = (
        df.groupby(["date", "segment"], as_index=False)["energy_kwh"].sum()
    )

    # Pivot to wide format: one column per segment
    seg_pivot = seg_group.pivot(
        index="date", columns="segment", values="energy_kwh"
    ).fillna(0.0)

    # Ensure known segment columns exist
    for seg_name in ["night", "morning", "afternoon", "evening"]:
        if seg_name not in seg_pivot.columns:
            seg_pivot[seg_name] = 0.0

    seg_pivot = seg_pivot[["night", "morning", "afternoon", "evening"]]

    seg_pivot = seg_pivot.rename(
        columns={
            "night": "energy_night_kwh",
            "morning": "energy_morning_kwh",
            "afternoon": "energy_afternoon_kwh",
            "evening": "energy_evening_kwh",
        }
    )

    seg_pivot["total_energy_kwh"] = (
        seg_pivot["energy_night_kwh"]
        + seg_pivot["energy_morning_kwh"]
        + seg_pivot["energy_afternoon_kwh"]
        + seg_pivot["energy_evening_kwh"]
    )

    seg_pivot = seg_pivot.reset_index()  # bring date back as column

    return seg_pivot


# ---------- Daily Anomaly Summary ----------

def compute_daily_anomaly_summary(
    windows_meta: pd.DataFrame,
    scores_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate window-level anomaly scores into daily-level statistics.

    Assumes:
      - windows_meta has columns: window_id, start_time, end_time, ...
      - scores_df has aligned rows with columns:
            if_score, ae_score, hybrid_score,
            if_is_anomaly, ae_is_anomaly, hybrid_is_anomaly
    """
    if len(windows_meta) != len(scores_df):
        raise ValueError(
            f"windows_meta ({len(windows_meta)}) and scores_df ({len(scores_df)}) "
            "do not have the same length."
        )

    meta = windows_meta.copy()
    meta["date"] = meta["start_time"].dt.date

    combined = pd.concat([meta.reset_index(drop=True), scores_df.reset_index(drop=True)], axis=1)

    grouped = combined.groupby("date")

    daily_rows = []
    for date, g in grouped:
        n_windows = len(g)
        n_anom = int(g["hybrid_is_anomaly"].sum())
        anomaly_ratio = n_anom / n_windows if n_windows > 0 else 0.0

        max_hybrid = float(g["hybrid_score"].max())
        mean_hybrid = float(g["hybrid_score"].mean())

        daily_rows.append(
            {
                "date": date,
                "n_windows": n_windows,
                "n_anom_windows": n_anom,
                "anomaly_ratio": anomaly_ratio,
                "max_hybrid_score": max_hybrid,
                "mean_hybrid_score": mean_hybrid,
            }
        )

    daily_df = pd.DataFrame(daily_rows)

    if not daily_df.empty:
        # Data-driven threshold: top X% by max_hybrid_score
        thresh = float(np.quantile(daily_df["max_hybrid_score"], ANOMALY_QUANTILE))
        daily_df["suspicion_threshold"] = thresh
        daily_df["is_suspicious_day"] = daily_df["max_hybrid_score"] >= thresh
    else:
        daily_df["suspicion_threshold"] = np.nan
        daily_df["is_suspicious_day"] = False

    return daily_df


# ---------- Combined Daily Summary ----------

def build_daily_summary(
    df_raw: pd.DataFrame,
    windows_meta: pd.DataFrame,
    scores_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Build:
      - daily_energy_df: per-day and per-segment energy summary
      - daily_anomaly_df: per-day anomaly statistics
      - merged_df: combined view (energy + anomalies) per day

    NOTE:
      This DOES NOT label anything as 'theft' yet.
      It only indicates 'is_suspicious_day' based on anomaly scores.
      Theft cues (sharp spikes, PF abnormality, meter freeze) will be
      handled at event-level / rules layer separately.
    """
    daily_energy_df = compute_daily_segment_energy(df_raw)
    daily_anomaly_df = compute_daily_anomaly_summary(windows_meta, scores_df)

    merged_df = pd.merge(
        daily_anomaly_df,
        daily_energy_df,
        on="date",
        how="left",
    )

    return daily_energy_df, daily_anomaly_df, merged_df
