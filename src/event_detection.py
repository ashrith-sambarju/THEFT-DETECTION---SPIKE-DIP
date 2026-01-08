# src/event_detection.py

from typing import Optional, List

import numpy as np
import pandas as pd

from src.config import WINDOW_SIZE_SECONDS, STRONG_EVENT_QUANTILE
from src.daily_analysis import _assign_segment  # reuse same time-of-day segmentation


def _find_anomalous_runs(anom_flags: np.ndarray) -> List[List[int]]:
    """
    Given a boolean array of anomaly flags (per window),
    return a list of lists, where each inner list contains
    consecutive indices that are anomalous.

    Example:
        flags = [False, True, True, False, True]
        -> [[1, 2], [4]]
    """
    runs: List[List[int]] = []
    current_run: List[int] = []

    for idx, is_anom in enumerate(anom_flags):
        if is_anom:
            current_run.append(idx)
        else:
            if current_run:
                runs.append(current_run)
                current_run = []

    # flush last run
    if current_run:
        runs.append(current_run)

    return runs


def detect_events_from_anomalies(
    windows_meta: pd.DataFrame,
    scores_df: pd.DataFrame,
    window_features_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Group anomalous windows into "events" and compute event-level statistics.

    Inputs:
        windows_meta:
            DataFrame with one row per window, columns at least:
                - window_id
                - start_time (Timestamp)
                - end_time   (Timestamp)
                - start_idx  (int, row index in raw df)
                - end_idx    (int, exclusive)
        scores_df:
            DataFrame aligned row-wise with windows_meta, columns at least:
                - if_score
                - ae_score
                - hybrid_score
                - if_is_anomaly
                - ae_is_anomaly
                - hybrid_is_anomaly
        window_features_df (optional):
            DataFrame aligned with windows_meta, containing engineered features
            like:
                - power_range
                - max_power_step
                - flat_steps_fraction
                - pf_std, pf_min, pf_max, max_pf_step
                - dip features (max_power_drop, dip_range, etc.)

    Returns:
        events_df: DataFrame with one row per event, including:
            - event_id
            - date
            - segment (night/morning/afternoon/evening) based on start_time
            - start_time, end_time
            - duration_seconds, duration_minutes
            - n_windows
            - max_hybrid_score, mean_hybrid_score
            - max_if_score, max_ae_score
            - strong_event_threshold
            - is_strong_event

            If window_features_df is provided, also aggregates:
            Spike / PF / freeze:
            - max_power_range
            - max_power_step
            - max_flat_steps_fraction
            - max_pf_std
            - global_pf_min (min pf_min across windows in event)
            - global_pf_max (max pf_max across windows in event)
            - max_pf_step

            Dip-related aggregations:
            - max_power_drop
            - max_dip_range
            - mean_sustained_low_fraction
            - total_repetitive_dip_count
            - max_zero_power_fraction
            - total_zero_power_duration_seconds
    """
    if len(windows_meta) != len(scores_df):
        raise ValueError(
            f"windows_meta ({len(windows_meta)}) and scores_df ({len(scores_df)}) "
            "must have the same length."
        )

    meta = windows_meta.reset_index(drop=True).copy()
    scores = scores_df.reset_index(drop=True).copy()

    if window_features_df is not None:
        wf = window_features_df.reset_index(drop=True).copy()
        if len(wf) != len(meta):
            raise ValueError(
                f"window_features_df ({len(wf)}) must match windows_meta ({len(meta)}) length."
            )
    else:
        wf = None

    # Identify anomaly runs using hybrid score flag
    anom_flags = scores["hybrid_is_anomaly"].to_numpy(bool)
    runs = _find_anomalous_runs(anom_flags)

    events_rows = []

    for event_id, run_indices in enumerate(runs):
        # windows in this event
        event_meta = meta.iloc[run_indices]
        event_scores = scores.iloc[run_indices]

        start_time = event_meta["start_time"].min()
        end_time = event_meta["end_time"].max()

        # Duration: from first start to last end
        duration_sec = (end_time - start_time).total_seconds()
        # For safety, if duration is zero (single window), approximate using window size
        if duration_sec <= 0:
            duration_sec = float(WINDOW_SIZE_SECONDS)
        duration_min = duration_sec / 60.0

        n_windows = len(run_indices)

        max_hybrid = float(event_scores["hybrid_score"].max())
        mean_hybrid = float(event_scores["hybrid_score"].mean())
        max_if = float(event_scores["if_score"].max())
        max_ae = float(event_scores["ae_score"].max())

        # Date and segment based on start_time
        date = start_time.date()
        segment = _assign_segment(start_time)

        row = {
            "event_id": event_id,
            "date": date,
            "segment": segment,
            "start_time": start_time,
            "end_time": end_time,
            "duration_seconds": duration_sec,
            "duration_minutes": duration_min,
            "n_windows": n_windows,
            "max_hybrid_score": max_hybrid,
            "mean_hybrid_score": mean_hybrid,
            "max_if_score": max_if,
            "max_ae_score": max_ae,
        }

        # Aggregate engineered features if provided
        if wf is not None:
            event_wf = wf.iloc[run_indices]

            # --- Spike / freeze features ---
            if "power_range" in event_wf.columns:
                row["max_power_range"] = float(event_wf["power_range"].max())
            if "max_power_step" in event_wf.columns:
                row["max_power_step"] = float(event_wf["max_power_step"].max())
            if "flat_steps_fraction" in event_wf.columns:
                row["max_flat_steps_fraction"] = float(
                    event_wf["flat_steps_fraction"].max()
                )

            # --- PF features ---
            if "pf_std" in event_wf.columns:
                row["max_pf_std"] = float(event_wf["pf_std"].max())
            if "pf_min" in event_wf.columns:
                row["global_pf_min"] = float(event_wf["pf_min"].min())
            if "pf_max" in event_wf.columns:
                row["global_pf_max"] = float(event_wf["pf_max"].max())
            if "max_pf_step" in event_wf.columns:
                row["max_pf_step"] = float(event_wf["max_pf_step"].max())

            # --- DIP FEATURES ---
            if "max_power_drop" in event_wf.columns:
                row["max_power_drop"] = float(event_wf["max_power_drop"].max())

            if "dip_range" in event_wf.columns:
                row["max_dip_range"] = float(event_wf["dip_range"].max())

            if "sustained_low_fraction" in event_wf.columns:
                row["mean_sustained_low_fraction"] = float(
                    event_wf["sustained_low_fraction"].mean()
                )

            if "repetitive_dip_count" in event_wf.columns:
                row["total_repetitive_dip_count"] = int(
                    event_wf["repetitive_dip_count"].sum()
                )

            if "zero_power_fraction" in event_wf.columns:
                row["max_zero_power_fraction"] = float(
                    event_wf["zero_power_fraction"].max()
                )

            if "zero_power_duration_seconds" in event_wf.columns:
                row["total_zero_power_duration_seconds"] = float(
                    event_wf["zero_power_duration_seconds"].sum()
                )

        events_rows.append(row)

    if not events_rows:
        # No anomalous events
        return pd.DataFrame(
            columns=[
                "event_id",
                "date",
                "segment",
                "start_time",
                "end_time",
                "duration_seconds",
                "duration_minutes",
                "n_windows",
                "max_hybrid_score",
                "mean_hybrid_score",
                "max_if_score",
                "max_ae_score",
            ]
        )

    events_df = pd.DataFrame(events_rows)

    # Data-driven threshold for "strong" events based on max_hybrid_score
    strong_thr = float(
        np.quantile(events_df["max_hybrid_score"], STRONG_EVENT_QUANTILE)
    )
    events_df["strong_event_threshold"] = strong_thr
    events_df["is_strong_event"] = events_df["max_hybrid_score"] >= strong_thr

    return events_df
