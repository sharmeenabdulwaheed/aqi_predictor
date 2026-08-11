"""
dashboard.py
Streamlit dashboard: current AQI, 3-day forecast, hazard alerts, historical
trend, and SHAP-based feature-importance explanation.

Reads everything from the local repo checkout — the SQLite database
(data/aqi_data.db) and the trained model files (models/*.pkl) — both of
which are kept up to date by the GitHub Actions workflows committing back
into the repo. No external service accounts are needed at all.

Run locally:
    streamlit run app/dashboard.py

Deploy for free: push to GitHub -> https://share.streamlit.io -> New app
-> point at app/dashboard.py. Streamlit Cloud auto-redeploys every time
the repo receives a new commit (i.e. every hour/day as the pipelines run).
"""

import os
import sys
import datetime as dt

import streamlit as st
import pandas as pd
import plotly.express as px

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from utils import aqi_category, CITY_NAME  # noqa: E402
from inference import get_forecast  # noqa: E402

st.set_page_config(page_title="AQI Forecaster", page_icon="\U0001F32B\uFE0F", layout="wide")


@st.cache_data(ttl=600)  # refresh every 10 minutes
def load_forecast_cached(city_name: str):
    return get_forecast(city_name)


def render_metric_row(latest_row: dict):
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Current AQI", int(latest_row.get("aqi", 0) or 0))
    col2.metric("PM2.5 (\u00b5g/m\u00b3)", round(latest_row.get("pm2_5", 0) or 0, 1))
    col3.metric("PM10 (\u00b5g/m\u00b3)", round(latest_row.get("pm10", 0) or 0, 1))
    col4.metric("Temperature (\u00b0C)", round(latest_row.get("temperature", 0) or 0, 1))
    col5.metric("Humidity (%)", round(latest_row.get("humidity", 0) or 0, 1))


def render_alert_banner(aqi):
    label, _ = aqi_category(aqi)
    if aqi is None:
        st.info("No current AQI reading available.")
    elif aqi > 200:
        st.error(f"\U0001F6A8 HAZARDOUS: Current AQI is {int(aqi)} ({label}). Avoid outdoor activity.")
    elif aqi > 150:
        st.warning(f"\u26A0\uFE0F Unhealthy air: Current AQI is {int(aqi)} ({label}). Sensitive groups should limit exposure.")
    else:
        st.success(f"Air quality is **{label}** (AQI {int(aqi)}).")


def render_forecast_section(forecast: dict, current_aqi):
    st.subheader("3-Day AQI Forecast")
    if not forecast:
        st.info("No trained forecast models found yet. Run the training pipeline first.")
        return

    labels = {"24h": "Tomorrow", "48h": "In 2 Days", "72h": "In 3 Days"}
    rows = []
    for h in ["24h", "48h", "72h"]:
        if h in forecast:
            val = forecast[h]
            cat, _ = aqi_category(val)
            rows.append({"Horizon": labels[h], "Predicted AQI": round(val, 1), "Category": cat})

    forecast_df = pd.DataFrame(rows)

    c1, c2 = st.columns([1, 1])
    with c1:
        fig = px.bar(forecast_df, x="Horizon", y="Predicted AQI", color="Category",
                     color_discrete_map={
                         "Good": "green", "Moderate": "gold",
                         "Unhealthy for Sensitive Groups": "orange",
                         "Unhealthy": "red", "Very Unhealthy": "purple",
                         "Hazardous": "maroon", "Unknown": "gray",
                     })
        fig.add_hline(y=current_aqi, line_dash="dot", annotation_text="Current AQI")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.dataframe(forecast_df, use_container_width=True, hide_index=True)

    for _, r in forecast_df.iterrows():
        if r["Predicted AQI"] > 150:
            st.warning(f"\u26A0\uFE0F {r['Horizon']}: predicted AQI {r['Predicted AQI']} \u2014 {r['Category']}")


def render_trend_chart(history_df: pd.DataFrame):
    st.subheader("Recent AQI Trend")
    history_df = history_df.copy()
    history_df["timestamp"] = pd.to_datetime(history_df["timestamp"])
    recent = history_df.tail(24 * 14)  # last ~2 weeks of hourly data
    fig = px.line(recent, x="timestamp", y="aqi", title="AQI \u2014 last 14 days")
    fig.add_hline(y=150, line_dash="dash", line_color="red",
                  annotation_text="Unhealthy threshold")
    st.plotly_chart(fig, use_container_width=True)


def render_shap_section():
    st.subheader("What's Driving This Prediction? (SHAP)")
    artifact_dir = os.path.join(os.path.dirname(__file__), "..", "artifacts")
    found_any = False
    tabs = st.tabs(["24h", "48h", "72h"])
    for tab, h in zip(tabs, ["24h", "48h", "72h"]):
        with tab:
            img_path = os.path.join(artifact_dir, f"shap_summary_{h}.png")
            if os.path.exists(img_path):
                st.image(img_path, caption=f"Feature importance for the {h} forecast model")
                found_any = True
            else:
                st.caption("SHAP summary not yet generated for this horizon. "
                           "It's produced automatically by the training pipeline.")
    if not found_any:
        st.info("Run the training pipeline at least once to generate SHAP plots.")


def main():
    st.title(f"\U0001F32B\uFE0F AQI Forecaster \u2014 {CITY_NAME}")
    st.caption("Live 3-day Air Quality Index forecast, powered by a fully serverless "
               "SQLite-backed pipeline (no external database or cloud account required).")

    try:
        result = load_forecast_cached(CITY_NAME)
    except Exception as e:
        st.error(f"Could not load data/model: {e}")
        st.info("Make sure the feature and training pipelines have run at least once "
                "(check that data/aqi_data.db and models/*.pkl exist in the repo).")
        st.stop()

    latest_row = result["latest_row"]
    render_alert_banner(result["current_aqi"])
    render_metric_row(latest_row)
    st.divider()

    left, right = st.columns([1, 1])
    with left:
        render_forecast_section(result["forecast"], result["current_aqi"])
    with right:
        render_trend_chart(result["history"])

    st.divider()
    render_shap_section()

    st.caption(f"Last refreshed: {dt.datetime.utcnow().isoformat()} UTC "
               "\u00b7 Data cached for 10 minutes.")


if __name__ == "__main__":
    main()
