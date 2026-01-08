# src/preprocessing.py
from typing import Tuple

import numpy as np
import pandas as pd

from src.config import SAMPLES_PER_WINDOW, STEP_SIZE_SAMPLES


def create_sliding_windows(
    df: pd.DataFrame,
    samples_per_window: int = SAMPLES_PER_WINDOW,
    step_size_samples: int = STEP_SIZE_SAMPLES,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Create sliding windows from raw smart meter data.

    Each window is flattened as:
        [p1..pN, v1..vN, c1..cN, pf1..pfN]

    Returns:
        X_windows: numpy array (n_windows, samples_per_window * 4)
        meta_df:  DataFrame with:
            window_id, start_time, end_time, start_idx, end_idx
    """
    if df.empty:
        raise ValueError("Input dataframe is empty.")

    df = df.sort_values("timestamp").reset_index(drop=True)

    values = df[["power", "voltage", "current", "pf"]].to_numpy(dtype=float)
    timestamps = df["timestamp"].to_numpy()

    n_samples = len(df)
    windows = []
    start_times = []
    end_times = []
    start_idxs = []
    end_idxs = []

    for start_idx in range(0, n_samples - samples_per_window + 1, step_size_samples):
        end_idx = start_idx + samples_per_window
        window_vals = values[start_idx:end_idx]

        power_seq = window_vals[:, 0]
        volt_seq = window_vals[:, 1]
        curr_seq = window_vals[:, 2]
        pf_seq = window_vals[:, 3]

        flat = np.concatenate([power_seq, volt_seq, curr_seq, pf_seq], axis=0)
        windows.append(flat)

        start_times.append(timestamps[start_idx])
        end_times.append(timestamps[end_idx - 1])
        start_idxs.append(start_idx)
        end_idxs.append(end_idx)

    if not windows:
        raise ValueError(
            f"Not enough samples ({n_samples}) to form even one window "
            f"of size {samples_per_window}."
        )

    X_windows = np.vstack(windows)

    meta_df = pd.DataFrame(
        {
            "window_id": np.arange(len(windows)),
            "start_time": start_times,
            "end_time": end_times,
            "start_idx": start_idxs,
            "end_idx": end_idxs,  # exclusive
        }
    )

    return X_windows, meta_df
