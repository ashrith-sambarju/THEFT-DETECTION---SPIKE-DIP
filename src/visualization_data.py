# src/visualization_data.py

from typing import Optional, List, Dict, Any

import pandas as pd


# -----------------------------
# 1) Day-wise spike timeline
# -----------------------------

def build_spike_timeline_df(
    daily_anomaly_df: pd.DataFrame,
    events_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Prepare a compact dataframe for the 'All Days Spike Timeline' chart.

    Inputs:
        daily_anomaly_df:
            Output of compute_daily_anomaly_summary() or build_daily_summary()
            Must contain columns:
                - date
                - max_hybrid_score
                - mean_hybrid_score
                - anomaly_ratio
                - n_anom_windows
                - is_suspicious_day

        events_df (optional):
            Events with cues (after apply_theft_cues), containing:
                - date (or start_time)
                - is_theft_like_event  (bool)

    Returns:
        spike_df: DataFrame with one row per day:
            - date
            - spike_score          (we use max_hybrid_score as primary metric)
            - mean_hybrid_score
            - anomaly_ratio
            - n_anom_windows
            - is_suspicious_day
            - n_events             (if events_df provided)
            - n_theft_like_events  (if events_df provided)
    """
    if daily_anomaly_df is None or daily_anomaly_df.empty:
        return pd.DataFrame(
            columns=[
                "date",
                "spike_score",
                "mean_hybrid_score",
                "anomaly_ratio",
                "n_anom_windows",
                "is_suspicious_day",
                "n_events",
                "n_theft_like_events",
            ]
        )

    df = daily_anomaly_df.copy()

    # main spike metric
    df["spike_score"] = df["max_hybrid_score"]

    # default event columns
    df["n_events"] = 0
    df["n_theft_like_events"] = 0

    # if we have events, aggregate per day
    if events_df is not None and not events_df.empty:
        e = events_df.copy()

        # ensure we have a date column
        if "date" not in e.columns:
            if "start_time" in e.columns:
                e["date"] = pd.to_datetime(e["start_time"]).dt.date
            else:
                # fallback: nothing to aggregate
                e["date"] = pd.NaT

        e_group = (
            e.groupby("date", as_index=False)
            .agg(
                n_events=("event_id", "count"),
                n_theft_like_events=("is_theft_like_event", lambda x: int(x.sum())),
            )
        )

        # merge into daily data
        df = df.merge(e_group, on="date", how="left")

        # if columns came from merge, fill NaNs -> 0
        if "n_events" in df.columns:
            df["n_events"] = df["n_events"].fillna(0).astype(int)
        else:
            df["n_events"] = 0

        if "n_theft_like_events" in df.columns:
            df["n_theft_like_events"] = df["n_theft_like_events"].fillna(0).astype(int)
        else:
            df["n_theft_like_events"] = 0

    # final column order
    cols = [
        "date",
        "spike_score",
        "mean_hybrid_score",
        "anomaly_ratio",
        "n_anom_windows",
        "is_suspicious_day",
        "n_events",
        "n_theft_like_events",
    ]
    df = df[cols].sort_values("date").reset_index(drop=True)

    return df


# -----------------------------
# 2) Raw time-series for a day
# -----------------------------

def get_day_timeseries(
    df_raw: pd.DataFrame,
    day: Any,
) -> pd.DataFrame:
    df = df_raw.copy()
    df["date"] = df["timestamp"].dt.date

    if isinstance(day, str):
        day = pd.to_datetime(day).date()

    df_day = df[df["date"] == day].copy()
    df_day = df_day.sort_values("timestamp")

    return df_day[["timestamp", "power", "voltage", "current", "pf"]]


# -----------------------------
# 3) Events for a selected day
# -----------------------------

def get_day_events(
    events_df: pd.DataFrame,
    day: Any,
    only_theft_like: bool = True,
) -> pd.DataFrame:
    if events_df is None or events_df.empty:
        return pd.DataFrame(
            columns=[
                "event_id",
                "date",
                "segment",
                "start_time",
                "end_time",
                "duration_minutes",
                "max_hybrid_score",
                "is_theft_like_event",
                "theft_cues",
            ]
        )

    if isinstance(day, str):
        day = pd.to_datetime(day).date()

    df = events_df.copy()
    if "date" not in df.columns:
        df["date"] = pd.to_datetime(df["start_time"]).dt.date

    df_day = df[df["date"] == day].copy()

    if only_theft_like and "is_theft_like_event" in df_day.columns:
        df_day = df_day[df_day["is_theft_like_event"]]

    df_day = df_day.sort_values("start_time").reset_index(drop=True)

    keep_cols = [
        c
        for c in [
            "event_id",
            "date",
            "segment",
            "start_time",
            "end_time",
            "duration_minutes",
            "max_hybrid_score",
            "is_theft_like_event",
            "theft_cues",
        ]
        if c in df_day.columns
    ]
    return df_day[keep_cols]


# -----------------------------
# 4) Segment energy cards
# -----------------------------

def build_segment_cards_for_day(
    daily_energy_df: pd.DataFrame,
    day: Any,
) -> List[Dict[str, Any]]:
    if daily_energy_df is None or daily_energy_df.empty:
        return []

    if isinstance(day, str):
        day = pd.to_datetime(day).date()

    df = daily_energy_df.copy()
    row = df[df["date"] == day]

    if row.empty:
        return []

    row = row.iloc[0]

    total = float(row.get("total_energy_kwh", 0.0)) or 0.0
    seg_info = [
        ("Night", "energy_night_kwh"),
        ("Morning", "energy_morning_kwh"),
        ("Afternoon", "energy_afternoon_kwh"),
        ("Evening", "energy_evening_kwh"),
    ]

    cards = []
    for label, col in seg_info:
        val = float(row.get(col, 0.0))
        pct = (val / total * 100.0) if total > 0 else 0.0
        cards.append(
            {
                "segment": label,
            "energy_kwh": val,
            "percent_of_day": pct,
            }
        )

    return cards


# -----------------------------
# 5) Helper: Merge everything for a selected day
# -----------------------------

def build_day_detail_bundle(
    df_raw: pd.DataFrame,
    daily_energy_df: pd.DataFrame,
    events_with_cues_df: pd.DataFrame,
    day: Any,
    only_theft_like_events: bool = True,
) -> Dict[str, Any]:
    df_ts = get_day_timeseries(df_raw, day)
    df_events = get_day_events(events_with_cues_df, day, only_theft_like=only_theft_like_events)
    segment_cards = build_segment_cards_for_day(daily_energy_df, day)

    return {
        "timeseries": df_ts,
        "events": df_events,
        "segment_cards": segment_cards,
    }
