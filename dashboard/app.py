# dashboard/app.py

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
from datetime import timedelta
import calendar
import plotly.graph_objects as go
import plotly.express as px
import textwrap

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"

sys.path.append(str(BASE_DIR))

# -----------------------------
# Small helpers
# -----------------------------
             
def _add_step_differences_local(df: pd.DataFrame) -> pd.DataFrame:
  
    df = df.sort_values("timestamp").reset_index(drop=True)
    for col in ["power", "voltage", "current", "pf"]:
        diff = df[col].diff().fillna(0.0)
        df[f"{col}_diff"] = diff
        df[f"{col}_abs_diff"] = diff.abs()
    return df

@st.cache_data
def load_processed_data():
    # Raw 5-sec data (cleaned)
    df_raw = pd.read_csv(
        PROCESSED_DIR / "raw_cleaned.csv",
        parse_dates=["timestamp"],
    )

    # Daily summaries
    daily_energy = pd.read_csv(
        PROCESSED_DIR / "daily_energy_summary.csv",
        parse_dates=["date"],
    )
    daily_anomaly = pd.read_csv(
        PROCESSED_DIR / "daily_anomaly_summary.csv",
        parse_dates=["date"],
    )

    # ---------- spike_timeline base (date + spike_score only) ----------
    spike_raw = pd.read_csv(PROCESSED_DIR / "spike_timeline.csv")

    if "date" in spike_raw.columns:
        # keep date as pure date for filtering
        spike_raw["date"] = pd.to_datetime(spike_raw["date"]).dt.date
    else:
        raise ValueError("spike_timeline.csv must contain a 'date' column.")

    base_cols = ["date"]
    if "spike_score" in spike_raw.columns:
        base_cols.append("spike_score")

    # This discards any old n_events / n_theft_like_events columns
    st_base = spike_raw[base_cols].copy()

    # Events with theft-like cues
    events = pd.read_csv(
        PROCESSED_DIR / "events_with_cues.csv",
        parse_dates=["start_time", "end_time"],
    )

    # ---------- Normalize date columns ----------
    df_raw["date"] = df_raw["timestamp"].dt.date
    daily_energy["date"] = pd.to_datetime(daily_energy["date"]).dt.date
    daily_anomaly["date"] = pd.to_datetime(daily_anomaly["date"]).dt.date

    if events.empty:
        # Make sure we still return a proper events df with a date column
        events["date"] = pd.Series(dtype="object")

        st_df = st_base.copy()
        # create all count columns as 0
        for col in [
            "n_events",
            "n_theft_like_events",
            "n_spike_only_events",
            "n_dip_only_events",
            "n_spike_dip_events",
        ]:
            st_df[col] = 0
    else:
        # Ensure we have a date column on events
        if "date" not in events.columns:
            events["date"] = events["start_time"].dt.date
        else:
            events["date"] = pd.to_datetime(events["date"]).dt.date

        # ------- Spike vs Dip cue flags at EVENT level -------
        spike_cue_cols = [
            "cue_sharp_spike",
            "cue_pf_abnormal",
            "cue_meter_freeze",
        ]
        dip_cue_cols = [
            "cue_deep_dip",
            "cue_sustained_low",
            "cue_long_zero",
            "cue_repetitive_dips",
            "cue_peak_hour_dip",
        ]

        # Some cue columns might not exist if there were no such cues – add as False
        for col in spike_cue_cols + dip_cue_cols:
            if col not in events.columns:
                events[col] = False

        events["has_spike_cue"] = events[spike_cue_cols].any(axis=1)
        events["has_dip_cue"] = events[dip_cue_cols].any(axis=1)

        events["is_spike_only_event"] = events["has_spike_cue"] & ~events["has_dip_cue"]
        events["is_dip_only_event"] = events["has_dip_cue"] & ~events["has_spike_cue"]
        events["is_spike_dip_event"] = events["has_spike_cue"] & events["has_dip_cue"]

        # ------- Aggregate per day -------
        daily_counts = (
            events.groupby("date")
            .agg(
                n_events=("event_id", "count"),
                n_theft_like_events=("is_theft_like_event", "sum"),
                n_spike_only_events=("is_spike_only_event", "sum"),
                n_dip_only_events=("is_dip_only_event", "sum"),
                n_spike_dip_events=("is_spike_dip_event", "sum"),
            )
            .reset_index()
        )

        st_df = st_base.merge(daily_counts, on="date", how="left")

        # Fill missing days with 0
        for col in [
            "n_events",
            "n_theft_like_events",
            "n_spike_only_events",
            "n_dip_only_events",
            "n_spike_dip_events",
        ]:
            st_df[col] = st_df[col].fillna(0).astype(int)

    # Flag theft days
    st_df["is_theft_day"] = st_df["n_theft_like_events"] > 0

    # Day type label (Normal / Spike-only / Dip-only / Spike+Dip)
    def _classify_day(row):
        if row["n_theft_like_events"] == 0:
            return "Normal Day"

        spike_only = row["n_spike_only_events"]
        dip_only = row["n_dip_only_events"]
        spike_dip = row["n_spike_dip_events"]

        if spike_dip > 0:
            return "Spike+Dip Day"
        if spike_only > 0 and dip_only == 0:
            return "Spike-only Day"
        if dip_only > 0 and spike_only == 0:
            return "Dip-only Day"
        # Mixed situation: treat as Spike+Dip
        return "Spike+Dip Day"

    st_df["day_type"] = st_df.apply(_classify_day, axis=1)

    # ---------- Weekday / Weekend info for calendar & analysis ----------
    st_df["date_ts"] = pd.to_datetime(st_df["date"])
    st_df["weekday_idx"] = st_df["date_ts"].dt.weekday  # Monday=0, Sunday=6
    st_df["weekday_name"] = st_df["date_ts"].dt.day_name()
    st_df["week_of_month"] = ((st_df["date_ts"].dt.day - 1) // 7) + 1
    st_df["is_weekend"] = st_df["weekday_idx"] >= 5  # Sat/Sun = weekend

    return df_raw, daily_energy, daily_anomaly, st_df, events


def build_segment_cards(row: pd.Series):
    
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

def render_metric_card(title: str, value: str, subtitle: str | None = None):
    value_str = str(value)

    html = (
        '<div style="'
        'background:#161a1d;'
        'border:1px solid #2a2f33;'
        'border-radius:12px;'
        'padding:16px 18px;'
        'height:100%;'
        'display:flex;'
        'flex-direction:column;'
        'justify-content:center;'
        '">'
        f'<div style="font-size:12px;color:#9ca3af;margin-bottom:6px;'
        'text-transform:uppercase;letter-spacing:0.05em;">'
        f'{title}'
        '</div>'
        f'<div style="font-size:24px;font-weight:600;color:#e5e7eb;">'
        f'{value_str}'
        '</div>'
    )

    if subtitle:
        html += (
            f'<div style="font-size:12px;color:#6b7280;margin-top:4px;">'
            f'{subtitle}'
            '</div>'
        )

    html += '</div>'

    st.markdown(html, unsafe_allow_html=True)

def render_text_card(title: str, body_html: str, variant: str = "dark"):
   
    # Clean leading/trailing whitespace so it doesn't start with spaces
    body_html = body_html.strip()

    html = (
        '<div style="'
        'background:#161a1d;'
        'border:1px solid #2a2f33;'
        'border-radius:12px;'
        'padding:18px 20px;'
        'min-height:210px;'
        'display:flex;'
        'flex-direction:column;'
        'justify-content:flex-start;'
        '">'
        f'<div style="font-size:14px;font-weight:600;'
        'margin-bottom:10px;color:#e5e7eb;">'
        f'{title}'
        '</div>'
        f'<div style="font-size:13px;line-height:1.55;color:#e5e7eb;">'
        f'{body_html}'
        '</div>'
        '</div>'
    )

    st.markdown(html, unsafe_allow_html=True)

# -----------------------------
# Normalisation helper
# -----------------------------

def normalize_per_parameter(df_long: pd.DataFrame) -> pd.DataFrame:

    df_long = df_long.copy()

    def _norm(x: pd.Series) -> pd.Series:
        min_val = x.min()
        max_val = x.max()
        if pd.isna(min_val) or pd.isna(max_val) or max_val == min_val:
            return pd.Series(0.0, index=x.index)
        return (x - min_val) / (max_val - min_val)

    df_long["norm_value"] = (
        df_long.groupby("parameter")["value"].transform(_norm)
    )
    return df_long

# -----------------------------
# Main app
# -----------------------------

def main():
    st.set_page_config(
        page_title="Smart Meter Theft Detection - Dashboard",
        layout="wide",
    )

    st.title("Smart Meter Theft Detection Dashboard")
    st.caption(
        "Raw 5-second data → ML anomaly detection (Isolation Forest + Autoencoder) → "
        "event-level spike & dip cues + daily summaries."
    )

    # Load data for entire dataset
    (
        df_raw,
        daily_energy,
        daily_anomaly,
        spike_timeline,
        events,
    ) = load_processed_data()

    # Also keep a datetime version of daily_energy dates (for month-wise NILM)
    daily_energy_dt = daily_energy.copy()
    daily_energy_dt["date_dt"] = pd.to_datetime(daily_energy_dt["date"])

    # ================================
    # Unified Top-Level KPIs (MASTER)
    # ================================
    total_days = daily_anomaly["date"].nunique()
    total_anomaly_events = len(events)

    if "is_theft_like_event" in events.columns:
        total_theft_like_events = int(events["is_theft_like_event"].sum())
    else:
        total_theft_like_events = 0

    # Day-type counts from spike_timeline (THE master source)
    day_type_counts = spike_timeline["day_type"].value_counts()

    n_normal_days = int(day_type_counts.get("Normal Day", 0))
    n_spike_only_days = int(day_type_counts.get("Spike-only Day", 0))
    n_dip_only_days = int(day_type_counts.get("Dip-only Day", 0))
    n_spike_dip_days = int(day_type_counts.get("Spike+Dip Day", 0))

    # Days that contain at least 1 theft-like event
    n_days_with_theft_cues = int((spike_timeline["n_theft_like_events"] > 0).sum())

    # -----------------------------
    # Sidebar controls
    # -----------------------------
    st.sidebar.header("Controls")

    # Global min/max for calendar (based on full dataset)
    min_date = daily_anomaly["date"].min()
    max_date = daily_anomaly["date"].max()

    # Calendar-style day picker
    selected_date = st.sidebar.date_input(
        "Select a day",
        value=max_date,
        min_value=min_date,
        max_value=max_date,
    )

    # date_input can (rarely) return a list; normalise to a single date
    if isinstance(selected_date, list) and len(selected_date) > 0:
        selected_date = selected_date[0]

    # Only theft-like toggle
    only_theft_like = st.sidebar.checkbox(
        "Show only theft-like events (cues present)",
        value=True,
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "Thresholds and anomaly quantiles are data-driven "
        "(based on quantiles over hybrid anomaly scores and event features)."
    )

    # ---------------------------------------
    # KPI ROW 1
    # ---------------------------------------
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Days in Dataset", total_days)
    k2.metric("Total Anomaly Events", total_anomaly_events)
    k3.metric("Theft-like Events", total_theft_like_events)
    k4.metric("Days with Theft Cues", n_days_with_theft_cues)

    st.markdown("### Day-type breakdown")

    # ---------------------------------------
    # KPI ROW 2
    # ---------------------------------------
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Normal Days", n_normal_days)
    d2.metric("Spike-only Days", n_spike_only_days)
    d3.metric("Dip-only Days", n_dip_only_days)
    d4.metric("Spike+Dip Days", n_spike_dip_days)

    st.markdown("---")

    # -----------------------------
    # Tabs: Overview vs Alerts
    # -----------------------------
    tab_overview, tab_alerts = st.tabs(["Overview", "Alerts & NILM"])

    # ==========================================================
    # ALERTS TAB
    # ==========================================================
    with tab_alerts:
        st.subheader("Control Room Alerts & NILM View")
        st.caption(
            "Summarises spike/dip-based theft cues, peak-hour stress, and an "
            "appliance-wise (NILM-style) consumption estimate for the selected day."
        )

        # -----------------------------
        # Data for the selected day
        # -----------------------------
        day_events = events[events["date"] == selected_date].copy()
        day_energy = daily_energy[daily_energy["date"] == selected_date].copy()

        # Theft-like event stats for selected day
        for col in [
            "is_spike_only_event",
            "is_dip_only_event",
            "is_spike_dip_event",
            "is_theft_like_event",
        ]:
            if col not in day_events.columns:
                day_events[col] = False

        n_spike_only = int(day_events["is_spike_only_event"].sum())
        n_dip_only = int(day_events["is_dip_only_event"].sum())
        n_spike_dip = int(day_events["is_spike_dip_event"].sum())
        n_theft_like = int(day_events["is_theft_like_event"].sum())

        # Peak-time usage stats (for cards + detailed section)
        if not day_energy.empty:
            row_e = day_energy.iloc[0]
            total_energy = float(row_e.get("total_energy_kwh", 0.0) or 0.0)
            energy_morning = float(row_e.get("energy_morning_kwh", 0.0) or 0.0)
            energy_afternoon = float(row_e.get("energy_afternoon_kwh", 0.0) or 0.0)
            energy_evening = float(row_e.get("energy_evening_kwh", 0.0) or 0.0)

            peak_energy = energy_morning + energy_evening
            peak_ratio = (peak_energy / total_energy) if total_energy > 0 else 0.0
        else:
            total_energy = peak_energy = peak_ratio = 0.0
            energy_morning = energy_afternoon = energy_evening = 0.0

        # -----------------------------
        # Helper: appliance-wise breakdown (heuristic)
        # -----------------------------
        def compute_appliance_breakdown(total, e_morning, e_afternoon, e_evening):
           
            if total <= 0:
                return {
                    "AC": 0.0,
                    "Fridge": 0.0,
                    "Geyser": 0.0,
                    "Others": 0.0,
                }

            # Baseline fridge
            fridge = 0.18 * total

            # Geyser tied to morning, but capped
            geyser = min(0.5 * e_morning, 0.25 * total)

            # AC tied to afternoon+evening, but capped
            ac_potential = e_afternoon + e_evening
            ac = min(0.7 * ac_potential, 0.5 * total)

            others = total - (fridge + geyser + ac)
            if others < 0:
                others = 0.0

            # Normalise to sum exactly to total
            s = fridge + geyser + ac + others
            if s > 0:
                scale = total / s
                fridge *= scale
                geyser *= scale
                ac *= scale
                others *= scale

            return {
                "AC": float(ac),
                "Fridge": float(fridge),
                "Geyser": float(geyser),
                "Others": float(others),
            }

        # Appliance breakdown for the *selected day* (for top KPIs + default view)
        daily_breakdown = compute_appliance_breakdown(
            total_energy, energy_morning, energy_afternoon, energy_evening
        )
        if total_energy > 0:
            top_app = max(daily_breakdown.items(), key=lambda x: x[1])
            top_app_pct = (top_app[1] / total_energy) * 100.0
            top_app_label = f"{top_app[0]} – {top_app_pct:.1f}%"
        else:
            top_app_label = "No dominant appliance"

        # Risk level based on # theft-like events for the day
        if n_theft_like == 0:
            risk_level = "Low"
        elif n_theft_like <= 2:
            risk_level = "Moderate"
        else:
            risk_level = "High"

        # =============================
        # TOP STRIP – 4 METRIC CARDS
        # =============================
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            render_metric_card(
                "Risk level (selected day)",
                risk_level,
                f"{n_theft_like} theft-like events",
            )
        with c2:
            render_metric_card(
                "Theft-like events (selected day)",
                str(n_theft_like),
                "Spike / dip cues identified by rules engine",
            )
        with c3:
            if total_energy > 0:
                render_metric_card(
                    "Peak-hour share",
                    f"{peak_ratio*100:.1f}%",
                    "6–10 AM + 6–10 PM share of daily energy",
                )
            else:
                render_metric_card(
                    "Peak-hour share",
                    "–",
                    "No energy data for this day",
                )
        with c4:
            render_metric_card(
                "Top appliance (heuristic, day)",
                top_app_label,
                "Pattern-based NILM-style estimate",
            )

        st.markdown("---")

        # ================================================
        # ROW 2 – THREE CARDS FOR SPIKE / DIP CUE TYPES
        # ================================================
        col_s, col_d, col_sd = st.columns(3)

        # Spike-only card
        with col_s:
            if n_spike_only == 0:
                body_spike = (
                    "No spike-only theft cues detected for this day. "
                    "Power increases are within expected behaviour."
                )
            else:
                body_spike = f"""
<p><b>Events:</b> {n_spike_only}</p>
<ul>
    <li>Short, sharp power increases with abnormal profile.</li>
    <li>Potential signs of sudden high-load appliance usage or bypass.</li>
</ul>
<p>
    <b>Interpretation:</b> Concentrated power spikes – candidate for
    closer review if this pattern repeats on nearby days.
</p>
"""
            render_text_card("Spike-only theft cues", body_spike)

        # Dip-only card
        with col_d:
            if n_dip_only == 0:
                body_dip = (
                    "No dip-only theft cues detected for this day. "
                    "Low-consumption periods look consistent with normal usage."
                )
            else:
                body_dip = f"""
<p><b>Events:</b> {n_dip_only}</p>
<ul>
    <li>Deep or sustained drops in power during normally active hours.</li>
    <li>May indicate meter tampering, selective disconnection or diversion.</li>
</ul>
<p>
    <b>Interpretation:</b> Non-typical low-load windows – recommended
    for substation / field team review.
</p>
"""
            render_text_card("Dip-only theft cues", body_dip)

        # Spike + Dip combo card
        with col_sd:
            if n_spike_dip == 0:
                body_combo = (
                    "No combined spike+dip patterns detected for this day. "
                    "We do not see strong back-to-back anomalies."
                )
            else:
                body_combo = f"""
<p><b>Events:</b> {n_spike_dip}</p>
<ul>
    <li>Back-to-back spike followed by an abnormal drop (or vice-versa).</li>
    <li>
        Strong signal for manual inspection – pattern is unlikely under
        standard consumer behaviour.
    </li>
</ul>
<p>
    <b>Interpretation:</b> High-priority anomaly signature – suitable
    to raise as an alert to the control room team.
</p>
"""
            render_text_card("Spike + Dip theft cues", body_combo)

        st.markdown("---")

        # =========================================
        # ROW 3 – PEAK-TIME CARD + NILM SECTION
        # =========================================
        st.markdown("#### Peak-time usage (6–10 AM & 6–10 PM)")

        if total_energy <= 0:
            st.info("No energy summary available for this day.")
        else:
            if peak_ratio >= 0.6:
                peak_level = "High peak-time stress"
                peak_note = (
                    "Most of the daily energy is drawn during peak hours. "
                    "Recommend demand-side management or advisory."
                )
            elif peak_ratio >= 0.4:
                peak_level = "Moderate peak-time usage"
                peak_note = (
                    "Significant share of energy is consumed in peak windows. "
                    "Monitor and nudge for off-peak shifting."
                )
            else:
                peak_level = "Normal peak-time usage"
                peak_note = "Peak-hour usage is within a typical share of daily load."

            body_peak = f"""
<p><b>Status:</b> {peak_level}</p>
<ul>
    <li>Total energy: <b>{total_energy:.2f} kWh</b></li>
    <li>Peak-hour energy: <b>{peak_energy:.2f} kWh</b></li>
    <li>Peak-hour share: <b>{peak_ratio*100:.1f}%</b></li>
</ul>
<p>{peak_note}</p>
"""
            render_text_card("Peak-time usage assessment", body_peak)

        st.markdown("---")

        # ---- NILM-style appliance section ----
        st.markdown("#### Appliance Consumption (NILM-style – heuristic demo)")

        if day_energy.empty and total_energy <= 0:
            st.info("Cannot estimate appliance-wise split – no daily energy row.")
        else:
            energy_dt = daily_energy_dt.copy()

            granularity = st.radio(
                "View appliance-wise consumption for:",
                ["Selected Day", "Selected Month"],
                horizontal=True,
            )

            if granularity == "Selected Day":
                total = total_energy
                e_morning = energy_morning
                e_afternoon = energy_afternoon
                e_evening = energy_evening
                scope_label = f"Day: {selected_date.strftime('%Y-%m-%d')}"
            else:
                month_mask = (
                    (energy_dt["date_dt"].dt.year == selected_date.year)
                    & (energy_dt["date_dt"].dt.month == selected_date.month)
                )
                month_df = energy_dt[month_mask].copy()

                if month_df.empty:
                    total = 0.0
                    e_morning = e_afternoon = e_evening = 0.0
                else:
                    total = float(month_df["total_energy_kwh"].sum())
                    e_morning = float(month_df["energy_morning_kwh"].sum())
                    e_afternoon = float(month_df["energy_afternoon_kwh"].sum())
                    e_evening = float(month_df["energy_evening_kwh"].sum())

                scope_label = f"Month: {selected_date.strftime('%Y-%m')}"

            breakdown = compute_appliance_breakdown(
                total, e_morning, e_afternoon, e_evening
            )

            if total <= 0:
                st.info(
                    f"No energy recorded for this scope ({scope_label}) – "
                    "NILM-style split is not shown."
                )
            else:
                app_names = list(breakdown.keys())
                app_values = list(breakdown.values())
                app_perc = [v / total * 100.0 for v in app_values]

                left, right = st.columns([1.1, 1])

                with left:
                    details_html = (
                        f"<p><b>Scope:</b> {scope_label}</p>"
                        "<p>This is a pattern-based approximation for demo purposes, "
                        "not a true NILM model.</p><ul>"
                    )
                    for name, kwh, pct in zip(app_names, app_values, app_perc):
                        details_html += (
                            f"<li><b>{name}</b>: {kwh:.2f} kWh ({pct:.1f}%)</li>"
                        )
                    details_html += "</ul>"

                    render_text_card("Appliance breakdown (heuristic)", details_html)

                with right:
                    pie_df = pd.DataFrame(
                        {
                            "Appliance": app_names,
                            "Energy_kWh": app_values,
                        }
                    )

                    fig = px.pie(
                        pie_df,
                        names="Appliance",
                        values="Energy_kWh",
                        hole=0.35,
                        title=f"Appliance-wise consumption – {scope_label}",
                    )
                    fig.update_layout(
                        margin=dict(l=0, r=0, t=40, b=0),
                    )

                    st.plotly_chart(fig, use_container_width=True)

    # ==========================================================
    # OVERVIEW TAB
    # ==========================================================
    with tab_overview:
        # ==========================================================
        # 1) All Days – Theft Detection Timeline (Hybrid Anomaly Score)
        # ==========================================================
        st.subheader("All Days - Theft Detection Timeline (Hybrid Anomaly Score)")

        if not spike_timeline.empty:
            st.caption(
                "Each spike is a day. "
                "Blue line shows the hybrid anomaly score per day. "
                "Coloured points = day type based on theft cues. "
                "Orange dashed line = visual spike threshold (90th percentile)."
            )

            stp_df = spike_timeline.copy()
            spike_threshold = float(stp_df["spike_score"].quantile(0.90))

            base = alt.Chart(stp_df).encode(
                x=alt.X("date:T", title="Date"),
                y=alt.Y("spike_score:Q", title="Spike Score (max hybrid anomaly)"),
                tooltip=[
                    alt.Tooltip("date:T", title="Date"),
                    alt.Tooltip("spike_score:Q", title="Spike Score", format=",.4f"),
                    alt.Tooltip("n_events:Q", title="# Events"),
                    alt.Tooltip("n_theft_like_events:Q", title="# Theft-like Events"),
                    alt.Tooltip("day_type:N", title="Day Type"),
                ],
            )

            line = base.mark_line(color="steelblue")

            points = base.mark_circle(size=90).encode(
                color=alt.Color(
                    "day_type:N",
                    title="Day Type",
                    scale=alt.Scale(
                        domain=[
                            "Normal Day",
                            "Spike-only Day",
                            "Dip-only Day",
                            "Spike+Dip Day",
                        ],
                        range=[
                            "#9e9e9e",
                            "#ffb74d",
                            "#9575cd",
                            "#ec407a",
                        ],
                    ),
                )
            )

            threshold_df = pd.DataFrame({"threshold": [spike_threshold]})
            threshold_rule = (
                alt.Chart(threshold_df)
                .mark_rule(strokeDash=[6, 4], color="orange")
                .encode(
                    y="threshold:Q",
                    tooltip=[
                        alt.Tooltip(
                            "threshold:Q",
                            title="Spike Threshold (90th percentile)",
                            format=",.4f",
                        )
                    ],
                )
            )

            chart = (line + points + threshold_rule).properties(height=300)
            st.altair_chart(chart, width="stretch")
        else:
            st.info("No spike timeline data available.")

        st.markdown("---")

        # ==========================================================
        # 1B) All Days – Theft Cues Timeline
        # ==========================================================
        
        st.subheader("All Days – Theft Cues Timeline")

        if not spike_timeline.empty:
            st.caption(
                "Line shows counts of events per day. Use the selector to view: "
                "all theft-like events, spike-only events, dip-only events, or "
                "events that have both spike and dip cues."
            )

            mode = st.radio(
                "Event type to plot:",
                options=[
                    "All theft-like events",
                    "Spike-only events",
                    "Dip-only events",
                    "Spike+Dip events",
                ],
                index=0,
                horizontal=True,
            )

            mode_to_column = {
                "All theft-like events": "n_theft_like_events",
                "Spike-only events": "n_spike_only_events",
                "Dip-only events": "n_dip_only_events",
                "Spike+Dip events": "n_spike_dip_events",
            }
            y_col = mode_to_column[mode]

            theft_df = spike_timeline.copy()

            # STRICT filtering by pure day_type
            if mode == "All theft-like events":
                theft_df = theft_df[theft_df["n_theft_like_events"] > 0]
            elif mode == "Spike-only events":
                theft_df = theft_df[theft_df["day_type"] == "Spike-only Day"]
            elif mode == "Dip-only events":
                theft_df = theft_df[theft_df["day_type"] == "Dip-only Day"]
            elif mode == "Spike+Dip events":
                theft_df = theft_df[theft_df["day_type"] == "Spike+Dip Day"]

            if theft_df.empty:
                st.info("No days match this event type selection.")
            else:
                base_theft = alt.Chart(theft_df).encode(
                    x=alt.X("date:T", title="Date"),
                    y=alt.Y(f"{y_col}:Q", title="# Events"),
                    tooltip=[
                        alt.Tooltip("date:T", title="Date"),
                        alt.Tooltip(
                            "n_theft_like_events:Q",
                            title="Total Theft-like Events",
                        ),
                        alt.Tooltip("n_spike_only_events:Q", title="Spike-only Events"),
                        alt.Tooltip("n_dip_only_events:Q", title="Dip-only Events"),
                        alt.Tooltip("n_spike_dip_events:Q", title="Spike+Dip Events"),
                        alt.Tooltip("day_type:N", title="Day Type"),
                    ],
                )

                line_theft = base_theft.mark_line(color="steelblue")
                points_theft = base_theft.mark_circle(size=90).encode(
                    color=alt.Color(
                        "day_type:N",
                        title="Day Type",
                        scale=alt.Scale(
                            domain=[
                                "Normal Day",
                                "Spike-only Day",
                                "Dip-only Day",
                                "Spike+Dip Day",
                            ],
                            range=[
                                "#9e9e9e",   # grey - Normal
                                "#ffb74d",   # orange - Spike-only
                                "#9575cd",   # purple - Dip-only
                                "#ec407a",   # pink - Spike+Dip
                            ],
                        ),
                    )
                )

                chart_theft = (line_theft + points_theft).properties(height=260)
                st.altair_chart(chart_theft, use_container_width=True)
        else:
            st.info("No theft-cue timeline data available.")

        st.markdown("---")

                # ==========================================================
        # 1C) Weekday vs Weekend – Detailed Theft Pattern Comparison
        #      (Separate graphs + 4 lines per graph, month of selected day)
        # ==========================================================

        if not spike_timeline.empty:
            tmp = spike_timeline.copy()
            tmp["date_dt"] = pd.to_datetime(tmp["date"])

            # Map day_type -> short label
            type_to_series = {
                "Normal Day": "Normal",
                "Spike-only Day": "Spike-only",
                "Dip-only Day": "Dip-only",
                "Spike+Dip Day": "Spike+Dip",
            }
            tmp["series"] = tmp["day_type"].map(type_to_series).fillna("Other")

            # ---- filter to month of selected_date ----
            month_mask = (
                (tmp["date_dt"].dt.year == selected_date.year)
                & (tmp["date_dt"].dt.month == selected_date.month)
            )
            tmp_month = tmp[month_mask].copy()

            if tmp_month.empty:
                st.info(
                    f"No data available for "
                    f"{selected_date.strftime('%B %Y')} to compare weekdays vs weekends."
                )
            else:
                month_label = selected_date.strftime("%B %Y")

                # ---------- 1C-A) Weekday pattern (Mon–Fri) ----------
                st.subheader(f"Weekday Theft Risk Pattern (Mon–Fri) – {month_label}")

                weekday_df = tmp_month[~tmp_month["is_weekend"]].copy()
                if weekday_df.empty:
                    st.info(f"No weekday data for {month_label}.")
                else:
                    chart_weekday = (
                        alt.Chart(weekday_df)
                        .mark_line(point=True)
                        .encode(
                            x=alt.X("date_dt:T", title="Date"),
                            y=alt.Y(
                                "spike_score:Q",
                                title="Spike Score (max hybrid anomaly)",
                            ),
                            color=alt.Color(
                                "series:N",
                                title="Day Pattern",
                                scale=alt.Scale(
                                    domain=[
                                        "Normal",
                                        "Spike-only",
                                        "Dip-only",
                                        "Spike+Dip",
                                    ],
                                    range=[
                                        "#4A90E2",  # Normal
                                        "#FFB74D",  # Spike-only
                                        "#9575CD",  # Dip-only
                                        "#EC407A",  # Spike+Dip
                                    ],
                                ),
                            ),
                            tooltip=[
                                alt.Tooltip("date_dt:T", title="Date"),
                                alt.Tooltip(
                                    "spike_score:Q",
                                    title="Spike Score",
                                    format=",.4f",
                                ),
                                alt.Tooltip("day_type:N", title="Day Type"),
                                alt.Tooltip("n_events:Q", title="# Events"),
                                alt.Tooltip(
                                    "n_theft_like_events:Q",
                                    title="# Theft-like Events",
                                ),
                            ],
                        )
                        .properties(height=260)
                    )

                    st.altair_chart(chart_weekday, use_container_width=True)

                st.markdown("---")

                # ---------- 1C-B) Weekend pattern (Sat–Sun) ----------
                st.subheader(f"Weekend Theft Risk Pattern (Sat–Sun) – {month_label}")

                weekend_df = tmp_month[tmp_month["is_weekend"]].copy()
                if weekend_df.empty:
                    st.info(f"No weekend data for {month_label}.")
                else:
                    chart_weekend = (
                        alt.Chart(weekend_df)
                        .mark_line(point=True)
                        .encode(
                            x=alt.X("date_dt:T", title="Date"),
                            y=alt.Y(
                                "spike_score:Q",
                                title="Spike Score (max hybrid anomaly)",
                            ),
                            color=alt.Color(
                                "series:N",
                                title="Day Pattern",
                                scale=alt.Scale(
                                    domain=[
                                        "Normal",
                                        "Spike-only",
                                        "Dip-only",
                                        "Spike+Dip",
                                    ],
                                    range=[
                                        "#4A90E2",  # Normal
                                        "#FFB74D",  # Spike-only
                                        "#9575CD",  # Dip-only
                                        "#EC407A",  # Spike+Dip
                                    ],
                                ),
                            ),
                            tooltip=[
                                alt.Tooltip("date_dt:T", title="Date"),
                                alt.Tooltip(
                                    "spike_score:Q",
                                    title="Spike Score",
                                    format=",.4f",
                                ),
                                alt.Tooltip("day_type:N", title="Day Type"),
                                alt.Tooltip("n_events:Q", title="# Events"),
                                alt.Tooltip(
                                    "n_theft_like_events:Q",
                                    title="# Theft-like Events",
                                ),
                            ],
                        )
                        .properties(height=260)
                    )

                    st.altair_chart(chart_weekend, use_container_width=True)
        else:
            st.info("No data available to compare weekdays vs weekends.")

        # ==========================================================
        # 2) Selected Day – Overview
        # ==========================================================
        st.subheader(f"Selected Day Overview – {selected_date.strftime('%Y-%m-%d')}")

        row_energy = daily_energy[daily_energy["date"] == selected_date]
        if not row_energy.empty:
            cards = build_segment_cards(row_energy.iloc[0])
            c1, c2, c3, c4 = st.columns(4)
            cols_cards = [c1, c2, c3, c4]
            for col, card in zip(cols_cards, cards):
                col.metric(
                    label=f"{card['segment']} Energy (kWh)",
                    value=f"{card['energy_kwh']:.2f}",
                    delta=f"{card['percent_of_day']:.1f}% of day",
                )
        else:
            st.info("No segment energy data for this day.")

        st.markdown("#### 4-Parameter Load Pattern (normalized)")

        df_day = df_raw[df_raw["date"] == selected_date].copy()
        if df_day.empty:
            st.warning("No raw samples for this day.")
        else:
            cols_params = ["power", "voltage", "current", "pf"]
            df_long = df_day[["timestamp"] + cols_params].melt(
                id_vars="timestamp",
                value_vars=cols_params,
                var_name="parameter",
                value_name="value",
            )
            df_long = normalize_per_parameter(df_long)

            chart_ts = (
                alt.Chart(df_long)
                .mark_line()
                .encode(
                    x=alt.X("timestamp:T", title="Time"),
                    y=alt.Y(
                        "norm_value:Q",
                        title="Normalized Value (0-1 per parameter)",
                    ),
                    color=alt.Color(
                        "parameter:N",
                        title="Parameter",
                    ),
                    tooltip=[
                        alt.Tooltip("timestamp:T", title="Time"),
                        alt.Tooltip("parameter:N", title="Parameter"),
                        alt.Tooltip("value:Q", title="Raw Value", format=",.2f"),
                    ],
                )
                .properties(height=350)
            )

            st.altair_chart(chart_ts, use_container_width=True)

        # ==========================================================
        # 3) Events on selected day
        # ==========================================================
        st.subheader("Anomaly Events on Selected Day")

        events_day = events[events["date"] == selected_date].copy()
        if events_day.empty:
            st.info("No anomaly events detected for this day.")
        else:
            if only_theft_like and "is_theft_like_event" in events_day.columns:
                events_day = events_day[events_day["is_theft_like_event"]]

            if events_day.empty:
                st.info("No theft-like events for this day (after filtering).")
            else:
                for col in [
                    "is_spike_only_event",
                    "is_dip_only_event",
                    "is_spike_dip_event",
                    "is_theft_like_event",
                ]:
                    if col not in events_day.columns:
                        events_day[col] = False

                def _theft_type(row):
                    if row.get("is_spike_dip_event", False):
                        return "Spike+Dip"
                    if row.get("is_spike_only_event", False):
                        return "Spike"
                    if row.get("is_dip_only_event", False):
                        return "Dip"
                    if row.get("is_theft_like_event", False):
                        return "Theft-like"
                    return ""

                events_day["theft_type"] = events_day.apply(_theft_type, axis=1)

                st.markdown("#### Event Hours (start time)")

                events_day["hour"] = events_day["start_time"].dt.hour
                hour_counts = (
                    events_day.groupby("hour")
                    .size()
                    .reset_index(name="n_events")
                    .sort_values("hour")
                )

                chart_hour = (
                    alt.Chart(hour_counts)
                    .mark_line(point=True)
                    .encode(
                        x=alt.X("hour:O", title="Hour of Day"),
                        y=alt.Y("n_events:Q", title="# Events"),
                        tooltip=[
                            alt.Tooltip("hour:O", title="Hour"),
                            alt.Tooltip("n_events:Q", title="# Events"),
                        ],
                    )
                    .properties(height=200)
                )

                st.altair_chart(chart_hour, use_container_width=True)

                st.markdown("#### Event Details")

                table_cols = [
                            c
                    for c in [
                        "event_id",
                        "segment",
                        "start_time",
                        "end_time",
                        "duration_minutes",
                        "max_hybrid_score",
                        "max_power_step",
                        "max_power_drop",
                        "max_pf_step",
                        "max_flat_steps_fraction",
                        "is_theft_like_event",
                        "theft_type",
                        "theft_cues",
                    ]
                    if c in events_day.columns
                ]

                events_display = events_day[table_cols].copy()
                if "theft_cues" in events_display.columns:
                    events_display["theft_cues"] = events_display["theft_cues"].astype(
                        str
                    )

                st.dataframe(
                    events_display.sort_values("start_time"),
                    use_container_width=True,
                )

        st.markdown("---")
        st.caption("Dashboard built from ML pipeline outputs under data/processed/.")

if __name__ == "__main__":
    main()
