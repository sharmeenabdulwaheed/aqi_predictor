"""
dashboard.py
Streamlit dashboard: current AQI, 3-day forecast, hazard alerts, historical
trend, health guidance, and model insights (metrics + SHAP).

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
import json
import datetime as dt

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from utils import aqi_category, AQI_BREAKPOINTS, CITY_NAME  # noqa: E402
from inference import get_forecast  # noqa: E402

APP_DIR = os.path.dirname(__file__)
ARTIFACT_DIR = os.path.join(APP_DIR, "..", "artifacts")

# --------------------------------------------------------------------------
# Page config + light theme styling
# --------------------------------------------------------------------------

st.set_page_config(
    page_title="AQI Forecaster",
    page_icon="\U0001F32B\uFE0F",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    .stApp { background-color: #F7F9FB; }

    /* KPI cards */
    div[data-testid="stMetric"] {
        background-color: #FFFFFF;
        border: 1px solid #E7EBEF;
        border-radius: 12px;
        padding: 14px 16px 10px 16px;
        box-shadow: 0 1px 3px rgba(16, 24, 40, 0.04);
    }
    div[data-testid="stMetricLabel"] { font-weight: 500; color: #667085; }
    div[data-testid="stMetricValue"] { font-weight: 700; color: #101828; }

    /* Section headers */
    h2, h3 { color: #101828; }

    /* Tabs */
    button[data-baseweb="tab"] { font-size: 15px; font-weight: 600; }

    /* Category pill */
    .aqi-pill {
        display: inline-block;
        padding: 6px 16px;
        border-radius: 999px;
        font-weight: 700;
        font-size: 15px;
        color: white;
    }

    /* Health advice card */
    .health-card {
        background-color: #FFFFFF;
        border: 1px solid #E7EBEF;
        border-left: 6px solid #667085;
        border-radius: 10px;
        padding: 16px 18px;
        margin-bottom: 10px;
    }

    footer { visibility: hidden; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

CATEGORY_COLOR_MAP = {
    "Good": "#2ECC71",
    "Moderate": "#F1C40F",
    "Unhealthy for Sensitive Groups": "#E67E22",
    "Unhealthy": "#E74C3C",
    "Very Unhealthy": "#8E44AD",
    "Hazardous": "#7B241C",
    "Unknown": "#95A5A6",
}

HEALTH_ADVICE = {
    "Good": {
        "icon": "\U0001F7E2",
        "summary": "Air quality is satisfactory, and air pollution poses little or no risk.",
        "tips": [
            "Enjoy outdoor activities as normal.",
            "No precautions needed for any group.",
        ],
    },
    "Moderate": {
        "icon": "\U0001F7E1",
        "summary": "Air quality is acceptable. However, there may be a risk for some "
                    "people who are unusually sensitive to air pollution.",
        "tips": [
            "Unusually sensitive individuals should consider reducing prolonged outdoor exertion.",
            "Everyone else can continue normal outdoor activities.",
        ],
    },
    "Unhealthy for Sensitive Groups": {
        "icon": "\U0001F7E0",
        "summary": "Members of sensitive groups may experience health effects. "
                    "The general public is less likely to be affected.",
        "tips": [
            "People with asthma, children, older adults: limit prolonged outdoor exertion.",
            "Consider wearing a mask (N95/KN95) if you must be outside for long periods.",
            "Keep windows closed during peak traffic hours.",
        ],
    },
    "Unhealthy": {
        "icon": "\U0001F534",
        "summary": "Everyone may begin to experience health effects; sensitive "
                    "groups may experience more serious effects.",
        "tips": [
            "Avoid prolonged or heavy outdoor exertion — everyone.",
            "Sensitive groups should stay indoors where possible.",
            "Wear an N95/KN95 mask outdoors; run an air purifier indoors if available.",
            "Keep windows and doors closed.",
        ],
    },
    "Very Unhealthy": {
        "icon": "\U0001F7E3",
        "summary": "Health alert: everyone may experience more serious health effects.",
        "tips": [
            "Avoid all outdoor physical activity.",
            "Stay indoors with windows closed and, if possible, an air purifier running.",
            "Sensitive groups should avoid going outside entirely.",
            "Wear an N95/KN95 mask if you must go out.",
        ],
    },
    "Hazardous": {
        "icon": "\u26AB",
        "summary": "Health warning of emergency conditions: the entire population "
                    "is more likely to be affected.",
        "tips": [
            "Remain indoors and keep activity levels low.",
            "Seal windows/doors; run an air purifier continuously if available.",
            "Avoid outdoor travel unless absolutely necessary.",
            "Seek medical attention if experiencing difficulty breathing, chest pain, or dizziness.",
        ],
    },
    "Unknown": {
        "icon": "\u26AA",
        "summary": "No current reading available.",
        "tips": ["Check back once the pipeline has run."],
    },
}


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------

@st.cache_data(ttl=600)  # refresh every 10 minutes
def load_forecast_cached(city_name: str):
    return get_forecast(city_name)


@st.cache_data(ttl=600)
def load_metrics_cached():
    path = os.path.join(ARTIFACT_DIR, "latest_metrics.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


# --------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------

def epa_band_shapes(fig: go.Figure, x0, x1):
    """Shade EPA AQI category bands as a background on a chart spanning x0..x1."""
    for lo, hi, label, _ in AQI_BREAKPOINTS:
        color = CATEGORY_COLOR_MAP.get(label, "#CCCCCC")
        fig.add_hrect(
            y0=lo, y1=min(hi, 500), fillcolor=color, opacity=0.07,
            layer="below", line_width=0,
        )
    return fig


def aqi_pill_html(aqi):
    label, _ = aqi_category(aqi)
    color = CATEGORY_COLOR_MAP.get(label, "#95A5A6")
    return f'<span class="aqi-pill" style="background-color:{color};">{label}</span>'


# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------

def render_header(current_aqi):
    left, right = st.columns([3, 1])
    with left:
        st.markdown(f"## \U0001F32B\uFE0F AQI Forecaster — {CITY_NAME}")
        st.caption(
            "Live 3-day Air Quality Index forecast — a fully serverless, "
            "SQLite-backed pipeline (no external database or cloud account required)."
        )
    with right:
        st.markdown(
            f"<div style='text-align:right; padding-top:10px;'>"
            f"<div style='font-size:13px; color:#667085;'>Current AQI</div>"
            f"<div style='font-size:34px; font-weight:800; color:#101828;'>{int(current_aqi or 0)}</div>"
            f"{aqi_pill_html(current_aqi)}"
            f"</div>",
            unsafe_allow_html=True,
        )


def render_alert_banner(aqi):
    label, _ = aqi_category(aqi)
    if aqi is None:
        st.info("No current AQI reading available.")
    elif aqi > 300:
        st.error(f"\U0001F6A8 HAZARDOUS: Current AQI is {int(aqi)} ({label}). Stay indoors.")
    elif aqi > 200:
        st.error(f"\U0001F6A8 VERY UNHEALTHY: Current AQI is {int(aqi)} ({label}). Avoid outdoor activity.")
    elif aqi > 150:
        st.warning(f"\u26A0\uFE0F Unhealthy air: Current AQI is {int(aqi)} ({label}). Sensitive groups should limit exposure.")
    elif aqi > 100:
        st.warning(f"\u26A0\uFE0F Current AQI is {int(aqi)} ({label}). Sensitive groups should take care.")
    else:
        st.success(f"Air quality is **{label}** (AQI {int(aqi)}).")


def render_metric_row(latest_row: dict):
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("PM2.5 (\u00b5g/m\u00b3)", round(latest_row.get("pm2_5", 0) or 0, 1))
    col2.metric("PM10 (\u00b5g/m\u00b3)", round(latest_row.get("pm10", 0) or 0, 1))
    col3.metric("Temperature (\u00b0C)", round(latest_row.get("temperature", 0) or 0, 1))
    col4.metric("Humidity (%)", round(latest_row.get("humidity", 0) or 0, 1))
    col5.metric("Wind Speed (m/s)", round(latest_row.get("wind_speed", 0) or 0, 1))


# --------------------------------------------------------------------------
# Tab 1 — Forecast
# --------------------------------------------------------------------------

def render_forecast_tab(forecast: dict, current_aqi, history_df: pd.DataFrame):
    if not forecast:
        st.info("No trained forecast models found yet. Run the training pipeline first.")
        return

    labels = {"24h": "Tomorrow", "48h": "In 2 Days", "72h": "In 3 Days"}
    rows = []
    for h in ["24h", "48h", "72h"]:
        if h in forecast:
            val = forecast[h]
            cat, _ = aqi_category(val)
            rows.append({"Horizon": labels[h], "hours": {"24h": 24, "48h": 48, "72h": 72}[h],
                         "Predicted AQI": round(val, 1), "Category": cat})
    forecast_df = pd.DataFrame(rows)

    c1, c2 = st.columns([2, 1])
    with c1:
        st.markdown("#### Forecast Timeline")
        now = pd.Timestamp.utcnow().tz_localize(None)
        x_vals = [now] + [now + pd.Timedelta(hours=r["hours"]) for r in rows]
        y_vals = [current_aqi] + [r["Predicted AQI"] for r in rows]

        fig = go.Figure()
        fig = epa_band_shapes(fig, x_vals[0], x_vals[-1])
        fig.add_trace(go.Scatter(
            x=x_vals, y=y_vals, mode="lines+markers+text",
            text=[f"{int(v)}" for v in y_vals], textposition="top center",
            line=dict(color="#101828", width=3),
            marker=dict(size=10, color=[CATEGORY_COLOR_MAP.get(aqi_category(v)[0], "#333") for v in y_vals]),
            name="AQI",
        ))
        fig.update_layout(
            plot_bgcolor="white", paper_bgcolor="white",
            height=380, margin=dict(l=10, r=10, t=20, b=10),
            yaxis_title="AQI", xaxis_title=None,
            yaxis_range=[0, max(150, max(y_vals) * 1.25)],
        )
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.markdown("#### Details")
        st.dataframe(forecast_df[["Horizon", "Predicted AQI", "Category"]],
                     use_container_width=True, hide_index=True)

    warn_rows = [r for r in rows if r["Predicted AQI"] > 150]
    if warn_rows:
        st.markdown("#### Upcoming Alerts")
        for r in warn_rows:
            st.warning(f"\u26A0\uFE0F {r['Horizon']}: predicted AQI {r['Predicted AQI']} — {r['Category']}")


# --------------------------------------------------------------------------
# Tab 2 — Trends
# --------------------------------------------------------------------------

def render_trends_tab(history_df: pd.DataFrame):
    history_df = history_df.copy()
    history_df["timestamp"] = pd.to_datetime(history_df["timestamp"])

    days_back = st.select_slider(
        "Look back", options=[3, 7, 14, 30, 90], value=14,
        format_func=lambda d: f"{d} days",
    )
    recent = history_df[history_df["timestamp"] >= history_df["timestamp"].max() - pd.Timedelta(days=days_back)]

    c1, c2 = st.columns([2, 1])
    with c1:
        st.markdown(f"#### AQI — last {days_back} days")
        fig = go.Figure()
        fig = epa_band_shapes(fig, recent["timestamp"].min(), recent["timestamp"].max())
        fig.add_trace(go.Scatter(
            x=recent["timestamp"], y=recent["aqi"], mode="lines",
            line=dict(color="#2C3E50", width=1.6), name="AQI",
        ))
        fig.add_hline(y=150, line_dash="dash", line_color="#E74C3C",
                      annotation_text="Unhealthy threshold", annotation_position="top left")
        fig.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                           height=380, margin=dict(l=10, r=10, t=20, b=10),
                           yaxis_title="AQI")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.markdown("#### Summary")
        st.metric("Average AQI", round(recent["aqi"].mean(), 1))
        st.metric("Peak AQI", round(recent["aqi"].max(), 1))
        pct_unhealthy = (recent["aqi"] > 150).mean() * 100
        st.metric("Hours > Unhealthy", f"{pct_unhealthy:.0f}%")

    st.markdown("#### Pollutant Breakdown")
    pollutant_cols = [c for c in ["pm2_5", "pm10", "no2", "so2", "co", "o3"] if c in recent.columns]
    fig2 = px.line(recent, x="timestamp", y=pollutant_cols,
                    labels={"value": "Concentration", "timestamp": "", "variable": "Pollutant"})
    fig2.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                        height=320, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig2, use_container_width=True)


# --------------------------------------------------------------------------
# Tab 3 — Health Advice
# --------------------------------------------------------------------------

def render_health_tab(current_aqi, forecast: dict):
    label, _ = aqi_category(current_aqi)
    advice = HEALTH_ADVICE.get(label, HEALTH_ADVICE["Unknown"])
    color = CATEGORY_COLOR_MAP.get(label, "#95A5A6")

    st.markdown(
        f"<div class='health-card' style='border-left-color:{color};'>"
        f"<div style='font-size:22px;'>{advice['icon']} <b>{label}</b> "
        f"<span style='color:#667085; font-weight:400;'>(current AQI {int(current_aqi or 0)})</span></div>"
        f"<p style='margin-top:8px; color:#344054;'>{advice['summary']}</p>"
        f"<ul style='color:#344054;'>" + "".join(f"<li>{t}</li>" for t in advice["tips"]) + "</ul>"
        f"</div>",
        unsafe_allow_html=True,
    )

    if forecast:
        worst_h, worst_val = max(forecast.items(), key=lambda kv: kv[1])
        worst_label, _ = aqi_category(worst_val)
        if worst_val > (current_aqi or 0) + 20:
            st.info(
                f"\U0001F4C8 Air quality is forecast to get worse — up to AQI "
                f"{int(worst_val)} ({worst_label}) within {worst_h}. Plan outdoor activities "
                f"for earlier in the window if possible."
            )
        elif worst_val < (current_aqi or 0) - 20:
            st.info(
                f"\U0001F4C9 Air quality is forecast to improve over the next {worst_h}."
            )

    st.markdown("#### AQI Scale Reference")
    ref_rows = [{"Range": f"{lo}-{hi if hi < 999 else '500+'}", "Category": lbl}
                for lo, hi, lbl, _ in AQI_BREAKPOINTS]
    ref_df = pd.DataFrame(ref_rows)
    st.dataframe(ref_df, use_container_width=True, hide_index=True)


# --------------------------------------------------------------------------
# Tab 4 — Model Insights
# --------------------------------------------------------------------------

def render_model_insights_tab():
    all_metrics = load_metrics_cached()

    st.markdown("#### Model Comparison (holdout metrics)")
    if not all_metrics:
        st.caption("No metrics found yet — run the training pipeline.")
    else:
        tabs = st.tabs(["24h", "48h", "72h"])
        for tab, h in zip(tabs, ["24h", "48h", "72h"]):
            with tab:
                if h not in all_metrics:
                    st.caption("Not trained yet.")
                    continue
                rows = [{"Model": name, **m} for name, m in all_metrics[h]["metrics"].items()]
                df = pd.DataFrame(rows).round(2)
                df["Selected"] = df["Model"] == all_metrics[h]["best_model"]
                st.dataframe(
                    df.sort_values("rmse"), use_container_width=True, hide_index=True,
                )
                fig = px.bar(df, x="Model", y="rmse", color="Selected",
                             color_discrete_map={True: "#2ECC71", False: "#B0BEC5"})
                fig.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                                   height=280, margin=dict(l=10, r=10, t=10, b=10),
                                   showlegend=False, yaxis_title="RMSE (lower is better)")
                st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.markdown("#### What's Driving Each Forecast? (SHAP)")
    tabs2 = st.tabs(["24h", "48h", "72h"])
    found_any = False
    for tab, h in zip(tabs2, ["24h", "48h", "72h"]):
        with tab:
            img_path = os.path.join(ARTIFACT_DIR, f"shap_summary_{h}.png")
            if os.path.exists(img_path):
                st.image(img_path, caption=f"Feature importance — {h} forecast model", use_container_width=True)
                found_any = True
            else:
                st.caption("SHAP summary not yet generated for this horizon.")
    if not found_any:
        st.info("Run the training pipeline at least once to generate SHAP plots.")


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------

def render_sidebar(latest_row, history_df):
    with st.sidebar:
        st.markdown("### \U0001F32B\uFE0F AQI Forecaster")
        st.caption(f"City: **{CITY_NAME}**")
        last_ts = pd.to_datetime(latest_row.get("timestamp"))
        st.caption(f"Last data point: {last_ts}")
        st.caption(f"History: {len(history_df):,} hourly rows")
        st.divider()
        st.markdown(
            "This dashboard is powered by a fully serverless pipeline: "
            "hourly feature collection and daily model training both run "
            "as free GitHub Actions, storing data in a SQLite file "
            "committed straight into the repo. No external database or "
            "cloud account is used."
        )
        st.divider()
        st.caption(f"Refreshed: {dt.datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
        st.caption("Data cached for 10 minutes.")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    try:
        result = load_forecast_cached(CITY_NAME)
    except Exception as e:
        st.error(f"Could not load data/model: {e}")
        st.info(
            "Make sure the feature and training pipelines have run at least once "
            "(check that data/aqi_data.db and models/*.pkl exist in the repo)."
        )
        st.stop()

    latest_row = result["latest_row"]
    current_aqi = result["current_aqi"]
    forecast = result["forecast"]
    history_df = result["history"]

    render_sidebar(latest_row, history_df)
    render_header(current_aqi)
    render_alert_banner(current_aqi)
    render_metric_row(latest_row)
    st.write("")

    tab1, tab2, tab3, tab4 = st.tabs([
        "\U0001F52E Forecast", "\U0001F4C8 Trends", "\U0001FA7A Health Advice", "\U0001F9E0 Model Insights",
    ])
    with tab1:
        render_forecast_tab(forecast, current_aqi, history_df)
    with tab2:
        render_trends_tab(history_df)
    with tab3:
        render_health_tab(current_aqi, forecast)
    with tab4:
        render_model_insights_tab()


if __name__ == "__main__":
    main()
