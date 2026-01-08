# src/config.py
from pathlib import Path
from datetime import time

# ---------- Paths ----------

# Root project dir (assumes this file is inside src/)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODELS_DIR = PROJECT_ROOT / "models"
IF_MODEL_DIR = MODELS_DIR / "isolation_forest"
AE_MODEL_DIR = MODELS_DIR / "autoencoder"

# Default raw file (you can change this filename)
DEFAULT_RAW_FILE = RAW_DATA_DIR / "house_data.csv"

# ---------- Time & Window Settings ----------

# Raw data sampling interval in seconds (your data is every 5 sec)
SAMPLE_INTERVAL_SECONDS = 5

# Window size in seconds and samples (10 minutes = 120 samples at 5 sec)
WINDOW_SIZE_SECONDS = 600
SAMPLES_PER_WINDOW = WINDOW_SIZE_SECONDS // SAMPLE_INTERVAL_SECONDS

# Step between windows (set < SAMPLES_PER_WINDOW for overlap, = for no overlap)
STEP_SIZE_SAMPLES = SAMPLES_PER_WINDOW  # no overlap for now

# ---------- Time-of-Day Segments ----------

# Using Python time objects for clarity
TIME_SEGMENTS = {
    "night":    (time(0, 0, 0),  time(6, 0, 0)),   # 00:00 - 06:00
    "morning":  (time(6, 0, 0),  time(12, 0, 0)),  # 06:00 - 12:00
    "afternoon":(time(12, 0, 0), time(18, 0, 0)),  # 12:00 - 18:00
    "evening":  (time(18, 0, 0), time(23, 59, 59)) # 18:00 - 23:59
}

# ---------- Data-Driven Threshold Logic (relative, not fixed) ----------

# Quantile for marking windows as anomalous (top X% windows)
ANOMALY_QUANTILE = 0.90      # top 3% windows are anomalies

# Quantile for very strong events (top X% events)
STRONG_EVENT_QUANTILE = 0.95 # top 1% events are "very suspicious"

# Z-score limit for segment energy being abnormal
SEGMENT_ENERGY_Z_LIMIT = 2.0 # > 2 std dev above mean = suspicious

# Random seed for reproducibility
RANDOM_STATE = 42
# Random seed for reproducibility
RANDOM_STATE = 42

# ---------- Event-Level Cue Thresholds (data-driven) ----------

SPIKE_QUANTILE = 0.95
PF_ABNORMALITY_QUANTILE = 0.95
FREEZE_QUANTILE = 0.95
