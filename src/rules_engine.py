# src/rules_engine.py

import numpy as np
import pandas as pd

from src.config import (
    SPIKE_QUANTILE,
    PF_ABNORMALITY_QUANTILE,
    FREEZE_QUANTILE,
    STRONG_EVENT_QUANTILE,
)

# Data-driven quantile for "top X%" dip-based suspicion
DIP_QUANTILE = 0.90  # top 10% most suspicious dip values


def _safe_quantile(series: pd.Series, q: float) -> float:
    """
    Helper: compute a quantile safely, ignoring NaN/inf.
    Returns 0.0 if no valid values.
    """
    clean = (
        pd.to_numeric(series, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    if clean.empty:
        return 0.0
    return float(clean.quantile(q))


def apply_theft_cues(events_df: pd.DataFrame) -> pd.DataFrame:
    """
    Attach theft-like cues to each anomaly event based on:
      - Sharp power spikes
      - PF abnormality
      - Meter freeze / flatline
      - Very strong ML anomaly
      - Dip-based patterns (deep drop, sustained low, long zero, repetitive dips)
      - Dip occurring during peak hours

    Returns:
        events_df with additional columns:
            - cue_sharp_spike
            - cue_pf_abnormal
            - cue_meter_freeze
            - cue_strong_ml
            - cue_deep_dip
            - cue_sustained_low
            - cue_long_zero
            - cue_repetitive_dips
            - cue_peak_hour_dip
            - theft_cues (text list)
            - is_theft_like_event (True/False)
    """

    if events_df.empty:
        return events_df

    df = events_df.copy()

    # -----------------------------
    # 1) SHARP POWER SPIKE CUE
    # -----------------------------
    if "max_power_step" in df.columns:
        spike_thr = _safe_quantile(df["max_power_step"], SPIKE_QUANTILE)
        if spike_thr > 0:
            df["cue_sharp_spike"] = df["max_power_step"] >= spike_thr
        else:
            df["cue_sharp_spike"] = False
    else:
        df["cue_sharp_spike"] = False

    # -----------------------------
    # 2) PF ABNORMALITY CUE
    # -----------------------------
    if "max_pf_step" in df.columns:
        pf_thr = _safe_quantile(df["max_pf_step"], PF_ABNORMALITY_QUANTILE)
        df["cue_pf_abnormal"] = df["max_pf_step"] >= pf_thr
    else:
        df["cue_pf_abnormal"] = False

    # OR strong PF instability (pf_std)
    if "max_pf_std" in df.columns:
        pfstd_thr = _safe_quantile(df["max_pf_std"], PF_ABNORMALITY_QUANTILE)
        df["cue_pf_abnormal"] = df["cue_pf_abnormal"] | (df["max_pf_std"] >= pfstd_thr)

    # -----------------------------
    # 3) METER FREEZE CUE
    # -----------------------------
    if "max_flat_steps_fraction" in df.columns:
        freeze_thr = _safe_quantile(df["max_flat_steps_fraction"], FREEZE_QUANTILE)
        df["cue_meter_freeze"] = df["max_flat_steps_fraction"] >= freeze_thr
    else:
        df["cue_meter_freeze"] = False

    # -----------------------------
    # 4) STRONG ML ANOMALY
    # -----------------------------
    if "max_hybrid_score" in df.columns:
        strong_ml_thr = _safe_quantile(df["max_hybrid_score"], STRONG_EVENT_QUANTILE)
        df["cue_strong_ml"] = df["max_hybrid_score"] >= strong_ml_thr
    else:
        df["cue_strong_ml"] = False

    # -----------------------------
    # 5) DIP-BASED CUES
    #    (all data-driven using DIP_QUANTILE)
    # -----------------------------

    # 5A) Deep power dip: very large downward drop magnitude
    if "max_power_drop" in df.columns:
        deep_dip_thr = _safe_quantile(df["max_power_drop"], DIP_QUANTILE)
        if deep_dip_thr > 0:
            df["cue_deep_dip"] = df["max_power_drop"] >= deep_dip_thr
        else:
            df["cue_deep_dip"] = False
    else:
        df["cue_deep_dip"] = False

    # 5B) Sustained low power: high fraction of low-power samples
    # Expecting an event-level aggregate like mean_sustained_low_fraction
    if "mean_sustained_low_fraction" in df.columns:
        low_sustain_thr = _safe_quantile(
            df["mean_sustained_low_fraction"], DIP_QUANTILE
        )
        df["cue_sustained_low"] = (
            df["mean_sustained_low_fraction"] >= low_sustain_thr
        )
    else:
        df["cue_sustained_low"] = False

    # 5C) Long near-zero power: large fraction of near-zero samples
    # Expect an event-level metric like max_zero_power_fraction
    if "max_zero_power_fraction" in df.columns:
        zero_frac_thr = _safe_quantile(df["max_zero_power_fraction"], DIP_QUANTILE)
        df["cue_long_zero"] = df["max_zero_power_fraction"] >= zero_frac_thr
    else:
        df["cue_long_zero"] = False

    # 5D) Repetitive dips: many dip occurrences inside event
    if "total_repetitive_dip_count" in df.columns:
        rep_dip_thr = _safe_quantile(
            df["total_repetitive_dip_count"], DIP_QUANTILE
        )
        if rep_dip_thr > 0:
            df["cue_repetitive_dips"] = (
                df["total_repetitive_dip_count"] >= rep_dip_thr
            )
        else:
            df["cue_repetitive_dips"] = False
    else:
        df["cue_repetitive_dips"] = False

    # -----------------------------
    # 6) Dip During Peak Hours
    #    (meta-cue: uses existing dip cues + time-of-day)
    # -----------------------------
    # We treat "morning" and "evening" segments as peak-like
    # if a segment column exists.
    dip_any = (
        df["cue_deep_dip"]
        | df["cue_sustained_low"]
        | df["cue_long_zero"]
        | df["cue_repetitive_dips"]
    )

    if "segment" in df.columns:
        peak_mask = df["segment"].astype(str).isin(["morning", "evening"])
        df["cue_peak_hour_dip"] = dip_any & peak_mask
    else:
        df["cue_peak_hour_dip"] = False

    # -----------------------------
    # Combine cues per event
    # -----------------------------
    def gather_cues(row):
        cues = []
        # existing spike / PF / freeze / ML cues
        if row.get("cue_sharp_spike", False):
            cues.append("Sharp Power Spike")
        if row.get("cue_pf_abnormal", False):
            cues.append("PF Abnormality")
        if row.get("cue_meter_freeze", False):
            cues.append("Meter Freeze")
        if row.get("cue_strong_ml", False):
            cues.append("Strong ML Anomaly")

        # new dip-based cues
        if row.get("cue_deep_dip", False):
            cues.append("Deep Power Dip")
        if row.get("cue_sustained_low", False):
            cues.append("Sustained Low Power")
        if row.get("cue_long_zero", False):
            cues.append("Long Near-Zero Power")
        if row.get("cue_repetitive_dips", False):
            cues.append("Repetitive Dip Pattern")
        if row.get("cue_peak_hour_dip", False):
            cues.append("Dip During Peak Hours")

        return cues

    df["theft_cues"] = df.apply(gather_cues, axis=1)
    df["is_theft_like_event"] = df["theft_cues"].apply(lambda c: len(c) > 0)

    return df
