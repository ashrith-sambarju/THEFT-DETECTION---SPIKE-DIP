# src/feature_engineering.py
from typing import Tuple, List

import numpy as np
import pandas as pd

from src.config import SAMPLE_INTERVAL_SECONDS


def add_step_differences(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add step-wise differences and absolute differences for
    power, voltage, current, pf.

    These are used later to characterize:
      - sharp power spikes
      - PF abnormal changes
      - meter freeze / flatline behaviour
      - sharp power dips
    """
    df = df.sort_values("timestamp").reset_index(drop=True)

    for col in ["power", "voltage", "current", "pf"]:
        diff_col = f"{col}_diff"
        abs_diff_col = f"{col}_abs_diff"

        df[diff_col] = df[col].diff().fillna(0.0)
        df[abs_diff_col] = df[diff_col].abs()

    return df


def build_window_features(
    df_with_diffs: pd.DataFrame, windows_meta: pd.DataFrame
) -> Tuple[np.ndarray, List[str], pd.DataFrame]:
    """
    Build aggregated features per window that capture:
      - sharp power spikes        (max_power_step, power_range)
      - power dips               (max_power_drop, dip_range, low/zero fractions)
      - PF abnormality           (pf_std, pf_min, pf_max, max_pf_step)
      - meter freeze / flatline  (flat_steps_fraction)

    DOES NOT apply any threshold or label theft.
    It only computes numeric features; thresholds
    will be learned from data later.

    Returns:
        X_feat:        numpy array [n_windows, n_features]
        feature_names: list of feature names in order
        windows_meta:  same meta df but augmented if needed
    """
    feature_rows = []

    # -------------------------------
    # Global, data-driven thresholds
    # -------------------------------
    power_abs = df_with_diffs["power_abs_diff"].to_numpy()
    if len(power_abs) == 0:
        raise ValueError("Dataframe has no power_abs_diff values.")

    # 10th percentile of absolute change – small movements considered 'flat-ish'
    flat_eps = np.quantile(power_abs, 0.10)

    # For dip-related thresholds we use the full dataset as well
    all_power = df_with_diffs["power"].to_numpy()
    all_power_diff = df_with_diffs["power_diff"].to_numpy()

    # Low power / near-zero thresholds (data-driven, not fixed numbers)
    if len(all_power) > 0:
        low_power_threshold = float(np.quantile(all_power, 0.10))   # bottom 10%
        near_zero_threshold = float(np.quantile(all_power, 0.02))   # very bottom 2%
    else:
        low_power_threshold = 0.0
        near_zero_threshold = 0.0

    # Threshold for "large" negative drops (top 10% most negative diffs)
    neg_diffs = all_power_diff[all_power_diff < 0]
    if len(neg_diffs) > 0:
        # 10th percentile of negative diffs (most negative side)
        large_drop_threshold = float(np.quantile(neg_diffs, 0.10))
    else:
        large_drop_threshold = 0.0  # no negative steps in data (unlikely)

    # -------------------------------
    # Per-window feature computation
    # -------------------------------
    for _, row in windows_meta.iterrows():
        start_idx = int(row["start_idx"])
        end_idx = int(row["end_idx"])

        window_df = df_with_diffs.iloc[start_idx:end_idx]

        p = window_df["power"].to_numpy()
        v = window_df["voltage"].to_numpy()
        c = window_df["current"].to_numpy()
        pf = window_df["pf"].to_numpy()

        p_diff = window_df["power_diff"].to_numpy()
        p_diff_abs = window_df["power_abs_diff"].to_numpy()
        pf_diff_abs = window_df["pf_abs_diff"].to_numpy()

        # --- Power-related features (sharp spikes) ---
        power_mean = float(np.mean(p)) if len(p) > 0 else 0.0
        power_std = float(np.std(p)) if len(p) > 0 else 0.0
        power_min = float(np.min(p)) if len(p) > 0 else 0.0
        power_max = float(np.max(p)) if len(p) > 0 else 0.0
        power_range = power_max - power_min

        max_power_step = float(p_diff_abs.max()) if len(p_diff_abs) > 0 else 0.0

        # --- Flatness / meter freeze (based on small absolute movements) ---
        flat_steps_fraction = (
            float((p_diff_abs <= flat_eps).mean()) if len(p_diff_abs) > 0 else 0.0
        )

        # --- DIP-oriented power features ---
        # 1) Strongest drop inside the window (magnitude, positive number)
        if len(p_diff) > 0:
            min_drop = float(p_diff.min())
            max_power_drop = -min_drop if min_drop < 0 else 0.0
        else:
            max_power_drop = 0.0

        # 2) How deep the window goes relative to its own mean
        dip_range = power_mean - power_min if len(p) > 0 else 0.0

        # 3) Sustained low power fraction (relative to dataset's low-power threshold)
        if len(p) > 0:
            low_mask = p <= low_power_threshold
            sustained_low_fraction = float(low_mask.mean())
        else:
            sustained_low_fraction = 0.0

        # 4) Repetitive large dips count (how many large negative steps)
        if len(p_diff) > 0 and large_drop_threshold < 0:
            dips_mask = p_diff <= large_drop_threshold
            repetitive_dip_count = int(dips_mask.sum())
        else:
            repetitive_dip_count = 0

        # 5) Fraction of samples near zero (possible bypass / long zero periods)
        if len(p) > 0:
            zero_mask = p <= near_zero_threshold
            zero_power_fraction = float(zero_mask.mean())
            zero_power_duration_seconds = float(
                zero_mask.sum() * SAMPLE_INTERVAL_SECONDS
            )
        else:
            zero_power_fraction = 0.0
            zero_power_duration_seconds = 0.0

        # --- PF-related features (abnormality) ---
        pf_mean = float(np.mean(pf)) if len(pf) > 0 else 0.0
        pf_std = float(np.std(pf)) if len(pf) > 0 else 0.0
        pf_min = float(np.min(pf)) if len(pf) > 0 else 0.0
        pf_max = float(np.max(pf)) if len(pf) > 0 else 0.0

        max_pf_step = float(pf_diff_abs.max()) if len(pf_diff_abs) > 0 else 0.0

        # --- Voltage & Current variability (context) ---
        volt_std = float(np.std(v)) if len(v) > 0 else 0.0
        curr_std = float(np.std(c)) if len(c) > 0 else 0.0

        feature_rows.append(
            [
                # Power level / variability
                power_mean,
                power_std,
                power_min,
                power_max,
                power_range,

                # Spike features
                max_power_step,

                # Flat / freeze
                flat_steps_fraction,

                # DIP features
                max_power_drop,               # strongest downward jump (magnitude)
                dip_range,                    # mean vs min
                sustained_low_fraction,       # % below low_power_threshold
                repetitive_dip_count,         # count of large dips
                zero_power_fraction,          # % near zero
                zero_power_duration_seconds,  # time near zero in seconds

                # PF behaviour
                pf_mean,
                pf_std,
                pf_min,
                pf_max,
                max_pf_step,

                # Context
                volt_std,
                curr_std,
            ]
        )

    feature_names = [
        "power_mean",
        "power_std",
        "power_min",
        "power_max",
        "power_range",
        "max_power_step",
        "flat_steps_fraction",
        # dip features
        "max_power_drop",
        "dip_range",
        "sustained_low_fraction",
        "repetitive_dip_count",
        "zero_power_fraction",
        "zero_power_duration_seconds",
        # PF
        "pf_mean",
        "pf_std",
        "pf_min",
        "pf_max",
        "max_pf_step",
        # context
        "volt_std",
        "curr_std",
    ]

    X_feat = np.asarray(feature_rows, dtype=float)
    return X_feat, feature_names, windows_meta
