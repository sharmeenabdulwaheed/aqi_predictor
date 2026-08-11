"""
eda.py
Exploratory Data Analysis script (convert to a Jupyter notebook with
`jupytext` if you prefer a .ipynb, or just run it as-is: it saves all
plots to notebooks/eda_outputs/).

Run after the backfill script has populated data/aqi_data.db:
    python notebooks/eda.py
"""

import os
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from utils import read_history, aqi_category, CITY_NAME  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(__file__), "eda_outputs")
os.makedirs(OUT_DIR, exist_ok=True)


def load_data():
    df = read_history(CITY_NAME)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.sort_values("timestamp")


def plot_aqi_distribution(df):
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df["aqi"].dropna(), bins=40, ax=ax)
    ax.set_title("Distribution of AQI readings")
    fig.savefig(os.path.join(OUT_DIR, "aqi_distribution.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)

    df = df.copy()
    df["category"] = df["aqi"].apply(lambda x: aqi_category(x)[0])
    fig, ax = plt.subplots(figsize=(8, 5))
    df["category"].value_counts().plot(kind="bar", ax=ax)
    ax.set_title("AQI category breakdown")
    fig.savefig(os.path.join(OUT_DIR, "aqi_category_breakdown.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_time_series(df):
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(df["timestamp"], df["aqi"])
    ax.set_title("AQI over time")
    ax.set_xlabel("Date")
    ax.set_ylabel("AQI")
    fig.savefig(os.path.join(OUT_DIR, "aqi_timeseries.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_correlation_heatmap(df):
    cols = ["aqi", "pm2_5", "pm10", "no2", "so2", "co", "o3",
            "temperature", "humidity", "pressure", "wind_speed", "clouds"]
    cols = [c for c in cols if c in df.columns]
    corr = df[cols].corr()
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title("Correlation heatmap")
    fig.savefig(os.path.join(OUT_DIR, "correlation_heatmap.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_hourly_monthly_patterns(df):
    df = df.copy()
    df["hour"] = df["timestamp"].dt.hour
    df["month"] = df["timestamp"].dt.month

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=df, x="hour", y="aqi", ax=ax)
    ax.set_title("AQI by hour of day")
    fig.savefig(os.path.join(OUT_DIR, "aqi_by_hour.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=df, x="month", y="aqi", ax=ax)
    ax.set_title("AQI by month")
    fig.savefig(os.path.join(OUT_DIR, "aqi_by_month.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_acf_pacf(df):
    series = df.set_index("timestamp")["aqi"].asfreq("h").interpolate()
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    plot_acf(series.dropna(), lags=72, ax=axes[0])
    plot_pacf(series.dropna(), lags=72, ax=axes[1])
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "acf_pacf.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)


def report_missing_and_outliers(df):
    missing = df.isna().mean().sort_values(ascending=False)
    print("Missing value fraction per column:\n", missing)
    numeric_cols = df.select_dtypes("number").columns
    outlier_summary = {}
    for c in numeric_cols:
        q1, q3 = df[c].quantile([0.25, 0.75])
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outlier_summary[c] = int(((df[c] < lo) | (df[c] > hi)).sum())
    print("Outlier counts (IQR method):\n", outlier_summary)


def run():
    df = load_data()
    print(f"Loaded {len(df)} rows for EDA.")
    plot_aqi_distribution(df)
    plot_time_series(df)
    plot_correlation_heatmap(df)
    plot_hourly_monthly_patterns(df)
    plot_acf_pacf(df)
    report_missing_and_outliers(df)
    print(f"All EDA plots saved to {OUT_DIR}")


if __name__ == "__main__":
    run()
