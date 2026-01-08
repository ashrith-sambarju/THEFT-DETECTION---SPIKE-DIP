# main_run_pipeline.py

from pathlib import Path
import numpy as np
import pandas as pd

from src.config import PROCESSED_DATA_DIR
from src.data_loading import load_raw_data
from src.preprocessing import create_sliding_windows
from src.feature_engineering import add_step_differences, build_window_features
from src.anomaly_models import (
    train_anomaly_models,
    load_anomaly_models,
    compute_anomaly_scores_for_windows,
)
from src.daily_analysis import build_daily_summary
from src.event_detection import detect_events_from_anomalies
from src.rules_engine import apply_theft_cues
from src.visualization_data import build_spike_timeline_df


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def run_pipeline(
    retrain_models: bool = True,
    ae_epochs: int = 15,
    ae_batch_size: int = 256,
) -> None:
    """
    Full end-to-end pipeline:

        1. Load raw 5-sec data
        2. Add step differences (for spikes / PF / freeze)
        3. Create 10-min windows
        4. Build window-level engineered features
        5. Train or load anomaly models (Isolation Forest + Autoencoder)
        6. Compute window-level anomaly scores
        7. Build daily summaries (energy + anomalies)
        8. Detect anomaly events
        9. Apply theft-like cues (sharp spike, PF abnormality, meter freeze, strong ML)
       10. Build spike timeline data
       11. Save everything under data/processed/

    NOTE:
        This pipeline only produces "theft-like events" from signals.
        Energy-based patterns are NOT used as theft cues, only for context.
    """

    print("=== 1) Loading raw data ===")
    df_raw = load_raw_data()
    print(f"[raw] shape: {df_raw.shape}, time range: {df_raw['timestamp'].min()} -> {df_raw['timestamp'].max()}")

    _ensure_dir(PROCESSED_DATA_DIR)

    # Save cleaned raw for reference
    raw_out = PROCESSED_DATA_DIR / "raw_cleaned.csv"
    df_raw.to_csv(raw_out, index=False)
    print(f"[save] Cleaned raw data -> {raw_out}")

    print("=== 2) Adding step differences (power, voltage, current, pf) ===")
    df_diff = add_step_differences(df_raw)
        # Clean any NaNs or infinite values across all numeric columns
    df_diff = df_diff.replace([np.inf, -np.inf], np.nan)

    numeric_cols = ["power", "voltage", "current", "pf",
                    "power_diff", "voltage_diff", "current_diff", "pf_diff",
                    "power_abs_diff", "voltage_abs_diff",
                    "current_abs_diff", "pf_abs_diff"]

    before = len(df_diff)
    df_diff = df_diff.dropna(subset=numeric_cols).reset_index(drop=True)
    after = len(df_diff)
    print(f"[clean] dropped {before - after} rows due to NaN/infinite numeric values")

    print("=== 3) Creating sliding windows ===")
    X_windows, windows_meta = create_sliding_windows(df_diff)
    print(f"[windows] X_windows shape: {X_windows.shape}, windows_meta rows: {len(windows_meta)}")

    print("=== 4) Building window-engineered features ===")
    X_feat, feat_names, windows_meta = build_window_features(df_diff, windows_meta)

    # === NEW: weekday/weekend context features ===
    # We use the window start_time to encode when this window occurred.
    start_times = pd.to_datetime(windows_meta["start_time"])

    # 0–6 (Mon=0, Sun=6)
    day_of_week = start_times.dt.weekday.astype(int).to_numpy().reshape(-1, 1)
    # 0 for Mon–Fri, 1 for Sat/Sun
    is_weekend = (start_times.dt.weekday >= 5).astype(int).to_numpy().reshape(-1, 1)

    # Append these two columns to the existing feature matrix
    X_feat = np.hstack([X_feat, is_weekend, day_of_week])

    # And extend the feature-name list so downstream code sees them
    feat_names = list(feat_names) + ["is_weekend", "day_of_week"]

    print(f"[features] X_feat shape: {X_feat.shape}, n_features: {len(feat_names)}")

    print("=== 5) Training or loading anomaly models (IF + AE) ===")
    if retrain_models:
        artifacts = train_anomaly_models(
            X_windows,
            X_feat,
            epochs=ae_epochs,
            batch_size=ae_batch_size,
            save=True,
        )
        scores_df = artifacts["scores_df"]
        print("[models] Trained new models and computed scores.")
    else:
        models = load_anomaly_models()
        scores_df = compute_anomaly_scores_for_windows(X_windows, X_feat, models)
        print("[models] Loaded existing models and computed scores.")

    # Save window-level scores + meta for debugging/analysis
    windows_with_scores = pd.concat([windows_meta.reset_index(drop=True), scores_df.reset_index(drop=True)], axis=1)
    win_out = PROCESSED_DATA_DIR / "windows_with_scores.csv"
    windows_with_scores.to_csv(win_out, index=False)
    print(f"[save] Window-level scores -> {win_out}")

    print("=== 6) Building daily summaries (energy + anomalies) ===")
    daily_energy_df, daily_anomaly_df, merged_daily_df = build_daily_summary(
        df_raw,
        windows_meta,
        scores_df,
    )

    daily_energy_out = PROCESSED_DATA_DIR / "daily_energy_summary.csv"
    daily_anomaly_out = PROCESSED_DATA_DIR / "daily_anomaly_summary.csv"
    merged_daily_out = PROCESSED_DATA_DIR / "daily_merged_summary.csv"

    daily_energy_df.to_csv(daily_energy_out, index=False)
    daily_anomaly_df.to_csv(daily_anomaly_out, index=False)
    merged_daily_df.to_csv(merged_daily_out, index=False)

    print(f"[save] Daily energy summary -> {daily_energy_out}")
    print(f"[save] Daily anomaly summary -> {daily_anomaly_out}")
    print(f"[save] Daily merged summary -> {merged_daily_out}")

    print("=== 7) Detecting anomaly events from windows ===")
    # Prepare window_features_df to match what event_detection expects
    window_features_df = pd.DataFrame(X_feat, columns=feat_names)
    events_df = detect_events_from_anomalies(
        windows_meta=windows_meta,
        scores_df=scores_df,
        window_features_df=window_features_df,
    )

    events_out = PROCESSED_DATA_DIR / "events_raw.csv"
    events_df.to_csv(events_out, index=False)
    print(f"[save] Raw anomaly events -> {events_out}, n_events = {len(events_df)}")

    print("=== 8) Applying theft-like cues to events ===")
    events_with_cues_df = apply_theft_cues(events_df)

    events_cues_out = PROCESSED_DATA_DIR / "events_with_cues.csv"
    events_with_cues_df.to_csv(events_cues_out, index=False)
    print(f"[save] Events with theft-like cues -> {events_cues_out}")

    print("=== 9) Building spike timeline data for dashboard ===")
    spike_timeline_df = build_spike_timeline_df(
        daily_anomaly_df=daily_anomaly_df,
        events_df=events_with_cues_df,
    )

    spike_out = PROCESSED_DATA_DIR / "spike_timeline.csv"
    spike_timeline_df.to_csv(spike_out, index=False)
    print(f"[save] Spike timeline data -> {spike_out}")

    print("=== Pipeline finished successfully ===")


if __name__ == "__main__":
    # You can tweak these defaults as you experiment
    run_pipeline(
        retrain_models=True,   # set to False if you want to reuse saved models
        ae_epochs=15,          # maybe start with 5 for quick testing, then go higher
        ae_batch_size=256,
    )
