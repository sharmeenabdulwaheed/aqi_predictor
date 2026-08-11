"""
backfill.py
One-time (or periodic) script to populate historical (features, target)
rows into the SQLite database, using the free, no-key-required Open-Meteo
archive APIs. This is what gives the training pipeline enough history to
learn seasonal and lag-based patterns, instead of waiting months for the
hourly pipeline alone to accumulate data.

Usage:
    python src/backfill.py --start 2023-01-01 --end 2024-12-31
"""

import os
import sys
import argparse
import requests
import pandas as pd

sys.path.append(os.path.dirname(__file__))
from utils import CITY_NAME, CITY_LAT, CITY_LON, insert_rows


def fetch_air_quality_archive(lat, lon, start_date, end_date):
    url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone,us_aqi",
        "timezone": "UTC",
    }
    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()
    hourly = r.json()["hourly"]
    df = pd.DataFrame(hourly)
    df["time"] = pd.to_datetime(df["time"])
    df = df.rename(columns={
        "pm10": "pm10", "pm2_5": "pm2_5",
        "carbon_monoxide": "co", "nitrogen_dioxide": "no2",
        "sulphur_dioxide": "so2", "ozone": "o3",
        "us_aqi": "aqi",
    })
    return df


def fetch_weather_archive(lat, lon, start_date, end_date):
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "temperature_2m,relative_humidity_2m,pressure_msl,wind_speed_10m,wind_direction_10m,cloud_cover",
        "timezone": "UTC",
    }
    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()
    hourly = r.json()["hourly"]
    df = pd.DataFrame(hourly)
    df["time"] = pd.to_datetime(df["time"])
    df = df.rename(columns={
        "temperature_2m": "temperature",
        "relative_humidity_2m": "humidity",
        "pressure_msl": "pressure",
        "wind_speed_10m": "wind_speed",
        "wind_direction_10m": "wind_deg",
        "cloud_cover": "clouds",
    })
    return df


def build_backfill_dataframe(lat, lon, city_name, start_date, end_date, chunk_days=90):
    """Chunk requests to keep individual calls fast and resumable."""
    dates = pd.date_range(start_date, end_date, freq=f"{chunk_days}D")
    if len(dates) == 0 or dates[-1] < pd.Timestamp(end_date):
        dates = dates.append(pd.DatetimeIndex([pd.Timestamp(end_date)]))

    all_chunks = []
    prev = pd.Timestamp(start_date)
    for d in dates:
        s = prev.date().isoformat()
        e = min(d, pd.Timestamp(end_date)).date().isoformat()
        if s > e:
            continue
        print(f"Fetching {s} -> {e} ...")
        aq = fetch_air_quality_archive(lat, lon, s, e)
        wx = fetch_weather_archive(lat, lon, s, e)
        merged = pd.merge(aq, wx, on="time", how="inner")
        all_chunks.append(merged)
        prev = pd.Timestamp(e) + pd.Timedelta(days=1)

    df = pd.concat(all_chunks, ignore_index=True).drop_duplicates(subset="time")
    df["city"] = city_name
    df = df.rename(columns={"time": "timestamp"})
    df = df.dropna(subset=["aqi"])
    return df


def run(start_date: str, end_date: str):
    raw = build_backfill_dataframe(CITY_LAT, CITY_LON, CITY_NAME, start_date, end_date)
    print(f"Raw merged rows: {len(raw)}")

    # Only raw weather/pollutant/AQI columns are stored in SQLite — the
    # training pipeline recomputes cyclical/lag features from the full
    # history at train time (see training_pipeline.py), so nothing further
    # is needed here.
    insert_rows(raw)
    print("Backfill complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="YYYY-MM-DD")
    args = parser.parse_args()
    run(args.start, args.end)
