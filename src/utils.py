"""
utils.py
Shared configuration, SQLite datastore helpers, and feature-engineering
functions used by feature_pipeline.py, backfill.py, training_pipeline.py,
and inference.py.

No external accounts are required for storage: the "feature store" is a
SQLite file committed back into the git repo by the GitHub Actions
workflows after every run. This keeps the whole stack to just your two
existing API keys (OpenWeather, AQICN) plus GitHub itself.
"""

import os
import sqlite3
import math
import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

CITY_NAME = os.environ.get("CITY_NAME", "Lahore")
CITY_LAT = float(os.environ.get("CITY_LAT", "31.5497"))
CITY_LON = float(os.environ.get("CITY_LON", "74.3436"))

# Repo-relative paths (all committed back to git by the workflows)
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DB_PATH = os.environ.get("DB_PATH", os.path.join(REPO_ROOT, "data", "aqi_data.db"))
MODELS_DIR = os.environ.get("MODELS_DIR", os.path.join(REPO_ROOT, "models"))
ARTIFACTS_DIR = os.environ.get("ARTIFACTS_DIR", os.path.join(REPO_ROOT, "artifacts"))

TABLE_NAME = "aqi_features"
MODEL_NAME = "aqi_forecaster"

# Columns the model is trained on (kept in one place so training and
# inference never drift apart).
BASE_NUMERIC_FEATURES = [
    "temperature", "humidity", "pressure", "wind_speed", "wind_deg", "clouds",
    "pm2_5", "pm10", "no2", "so2", "co", "o3",
]

TIME_FEATURES = [
    "hour_sin", "hour_cos", "day_sin", "day_cos",
    "month_sin", "month_cos", "dow_sin", "dow_cos", "is_weekend",
]

LAG_FEATURES = [
    "aqi_lag_1h", "aqi_lag_24h", "aqi_lag_72h",
    "aqi_rolling_mean_24h", "aqi_rolling_std_24h",
    "aqi_change_rate", "pm2_5_to_pm10_ratio",
]

ALL_FEATURES = BASE_NUMERIC_FEATURES + TIME_FEATURES + LAG_FEATURES
TARGET_COL = "aqi"

AQI_BREAKPOINTS = [
    (0, 50, "Good", "green"),
    (51, 100, "Moderate", "gold"),
    (101, 150, "Unhealthy for Sensitive Groups", "orange"),
    (151, 200, "Unhealthy", "red"),
    (201, 300, "Very Unhealthy", "purple"),
    (301, 999, "Hazardous", "maroon"),
]


def aqi_category(aqi: float):
    """Map a numeric AQI value to (category label, color)."""
    if aqi is None or (isinstance(aqi, float) and math.isnan(aqi)):
        return "Unknown", "gray"
    for lo, hi, label, color in AQI_BREAKPOINTS:
        if lo <= aqi <= hi:
            return label, color
    return "Hazardous", "maroon"


# --------------------------------------------------------------------------
# SQLite datastore ("feature store")
# --------------------------------------------------------------------------

RAW_COLUMNS = [
    ("city", "TEXT"),
    ("timestamp", "TEXT"),   # stored as ISO-8601 string, parsed back with pandas
    ("temperature", "REAL"),
    ("humidity", "REAL"),
    ("pressure", "REAL"),
    ("wind_speed", "REAL"),
    ("wind_deg", "REAL"),
    ("clouds", "REAL"),
    ("pm2_5", "REAL"),
    ("pm10", "REAL"),
    ("no2", "REAL"),
    ("so2", "REAL"),
    ("co", "REAL"),
    ("o3", "REAL"),
    ("aqi", "REAL"),
]


def get_connection() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    return conn


def ensure_table(conn: sqlite3.Connection = None):
    """Idempotently create the features table if it doesn't exist yet.
    A UNIQUE constraint on (city, timestamp) makes re-inserting the same
    hour a safe no-op instead of a duplicate row."""
    own_conn = conn is None
    conn = conn or get_connection()
    cols_sql = ", ".join(f"{name} {ftype}" for name, ftype in RAW_COLUMNS)
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
            {cols_sql},
            UNIQUE(city, timestamp)
        )
    """)
    conn.commit()
    if own_conn:
        return conn
    return conn


def insert_rows(df: pd.DataFrame):
    """Insert (or ignore, if the (city, timestamp) pair already exists)
    rows into the SQLite features table."""
    conn = get_connection()
    ensure_table(conn)

    df = df.copy()
    cols = [name for name, _ in RAW_COLUMNS]
    for c in cols:
        if c not in df.columns:
            df[c] = None
    df = df[cols]
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.strftime("%Y-%m-%dT%H:%M:%S")

    placeholders = ", ".join(["?"] * len(cols))
    col_names = ", ".join(cols)
    rows = list(df.itertuples(index=False, name=None))

    cur = conn.executemany(
        f"INSERT OR IGNORE INTO {TABLE_NAME} ({col_names}) VALUES ({placeholders})",
        rows,
    )
    conn.commit()
    inserted = cur.rowcount if cur.rowcount is not None else len(rows)
    conn.close()
    print(f"Inserted (or matched existing) {len(rows)} row(s) into {DB_PATH}")
    return inserted


def read_history(city_name: str = None) -> pd.DataFrame:
    """Read all rows for a city, ordered by time."""
    conn = get_connection()
    ensure_table(conn)
    city_name = city_name or CITY_NAME
    df = pd.read_sql_query(
        f"SELECT * FROM {TABLE_NAME} WHERE city = ? ORDER BY timestamp",
        conn, params=(city_name,),
    )
    conn.close()
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def read_latest_row(city_name: str = None) -> dict:
    df = read_history(city_name)
    if df.empty:
        raise ValueError(f"No rows found for city={city_name or CITY_NAME}. "
                          f"Run feature_pipeline.py or backfill.py first.")
    return df.iloc[-1].to_dict(), df


# --------------------------------------------------------------------------
# Time-based (cyclical) features
# --------------------------------------------------------------------------

def add_cyclical_time_features(df: pd.DataFrame, ts_col: str = "timestamp") -> pd.DataFrame:
    """Add sin/cos encodings of hour, day, month, day-of-week."""
    ts = pd.to_datetime(df[ts_col])
    df = df.copy()
    df["hour"] = ts.dt.hour
    df["day"] = ts.dt.day
    df["month"] = ts.dt.month
    df["day_of_week"] = ts.dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["day_sin"] = np.sin(2 * np.pi * df["day"] / 31)
    df["day_cos"] = np.cos(2 * np.pi * df["day"] / 31)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    return df


def add_lag_and_derived_features(df: pd.DataFrame, ts_col: str = "timestamp",
                                  group_col: str = "city") -> pd.DataFrame:
    """
    Add lag features, rolling stats, and AQI change rate.
    Expects one row per hour per city. Assumes df may contain multiple
    cities; features are computed per-city to avoid leakage across cities.
    """
    df = df.sort_values([group_col, ts_col]).copy()

    g = df.groupby(group_col)["aqi"]
    df["aqi_lag_1h"] = g.shift(1)
    df["aqi_lag_24h"] = g.shift(24)
    df["aqi_lag_72h"] = g.shift(72)

    df["aqi_rolling_mean_24h"] = (
        df.groupby(group_col)["aqi"].transform(lambda s: s.shift(1).rolling(24, min_periods=3).mean())
    )
    df["aqi_rolling_std_24h"] = (
        df.groupby(group_col)["aqi"].transform(lambda s: s.shift(1).rolling(24, min_periods=3).std())
    )

    df["aqi_change_rate"] = (df["aqi"] - df["aqi_lag_1h"])  # per-hour change
    df["pm2_5_to_pm10_ratio"] = df["pm2_5"] / df["pm10"].replace(0, np.nan)

    return df


def add_forecast_targets(df: pd.DataFrame, ts_col: str = "timestamp",
                          group_col: str = "city") -> pd.DataFrame:
    """
    Add the 3 forecasting targets: AQI 24h, 48h, and 72h ahead.
    Used only for TRAINING (these columns won't exist at real inference time).
    """
    df = df.sort_values([group_col, ts_col]).copy()
    g = df.groupby(group_col)["aqi"]
    df["aqi_target_24h"] = g.shift(-24)
    df["aqi_target_48h"] = g.shift(-48)
    df["aqi_target_72h"] = g.shift(-72)
    return df


def engineer_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """Full feature engineering pipeline used by both backfill and training."""
    df = add_cyclical_time_features(df)
    df = add_lag_and_derived_features(df)
    return df


def build_inference_row_from_latest(latest_row: dict) -> pd.DataFrame:
    """
    Given the single latest feature row (dict, as read from SQLite),
    assemble a 1-row DataFrame ready for model.predict().
    Missing lag features (e.g. if history is thin) are filled with the
    current AQI as a naive fallback.
    """
    row = dict(latest_row)
    for lag_col in ["aqi_lag_1h", "aqi_lag_24h", "aqi_lag_72h",
                     "aqi_rolling_mean_24h", "aqi_rolling_std_24h"]:
        val = row.get(lag_col)
        if val is None or (isinstance(val, float) and math.isnan(val)):
            row[lag_col] = row.get("aqi", 0)
    if row.get("aqi_change_rate") is None:
        row["aqi_change_rate"] = 0
    if row.get("pm2_5_to_pm10_ratio") is None and row.get("pm10"):
        row["pm2_5_to_pm10_ratio"] = row.get("pm2_5", 0) / (row.get("pm10") or 1)

    df = pd.DataFrame([row])
    df = add_cyclical_time_features(df)
    for col in ALL_FEATURES:
        if col not in df.columns:
            df[col] = 0
    return df[ALL_FEATURES]
