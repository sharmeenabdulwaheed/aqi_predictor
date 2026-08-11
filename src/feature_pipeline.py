"""
feature_pipeline.py
Runs hourly (via GitHub Actions). Fetches current weather + pollutant data
from OpenWeather and AQICN, engineers the feature row, and appends it to
the local SQLite database (data/aqi_data.db). The GitHub Actions workflow
commits the updated .db file back into the repo after this script runs.

Env vars required:
    OWM_API_KEY, AQICN_TOKEN, CITY_LAT, CITY_LON, CITY_NAME
Optional:
    ALERT_WEBHOOK_URL  (Slack/Discord/ntfy webhook for hazardous-AQI alerts)
"""

import os
import sys
import datetime as dt
import requests
import pandas as pd

sys.path.append(os.path.dirname(__file__))
from utils import (
    CITY_NAME, CITY_LAT, CITY_LON,
    insert_rows, add_cyclical_time_features, aqi_category,
)

OWM_API_KEY = os.environ.get("OWM_API_KEY")
AQICN_TOKEN = os.environ.get("AQICN_TOKEN")
ALERT_WEBHOOK_URL = os.environ.get("ALERT_WEBHOOK_URL")


def fetch_weather(lat, lon, api_key):
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {"lat": lat, "lon": lon, "appid": api_key, "units": "metric"}
    r = requests.get(url, params=params, timeout=15)
    r.raise_for_status()
    return r.json()


def fetch_pollution(lat, lon, api_key):
    url = "https://api.openweathermap.org/data/2.5/air_pollution"
    params = {"lat": lat, "lon": lon, "appid": api_key}
    r = requests.get(url, params=params, timeout=15)
    r.raise_for_status()
    return r.json()


def fetch_aqicn(lat, lon, token):
    """Use the geo-based AQICN endpoint so it works for any city, not just
    ones with a friendly URL slug."""
    url = f"https://api.waqi.info/feed/geo:{lat};{lon}/"
    params = {"token": token}
    r = requests.get(url, params=params, timeout=15)
    r.raise_for_status()
    payload = r.json()
    if payload.get("status") != "ok":
        raise RuntimeError(f"AQICN error: {payload}")
    return payload["data"]


def build_feature_row(weather_json, pollution_json, aqicn_json, city_name):
    now = dt.datetime.utcnow().replace(minute=0, second=0, microsecond=0)
    comp = pollution_json["list"][0]["components"]

    row = {
        "city": city_name,
        "timestamp": now,
        "temperature": weather_json["main"]["temp"],
        "humidity": weather_json["main"]["humidity"],
        "pressure": weather_json["main"]["pressure"],
        "wind_speed": weather_json["wind"]["speed"],
        "wind_deg": weather_json["wind"].get("deg", 0),
        "clouds": weather_json["clouds"]["all"],
        "pm2_5": comp.get("pm2_5"),
        "pm10": comp.get("pm10"),
        "no2": comp.get("no2"),
        "so2": comp.get("so2"),
        "co": comp.get("co"),
        "o3": comp.get("o3"),
        "aqi": aqicn_json.get("aqi"),
    }
    return row


def maybe_send_alert(aqi, city):
    if not ALERT_WEBHOOK_URL or aqi is None:
        return
    if aqi > 150:
        label, _ = aqi_category(aqi)
        try:
            requests.post(
                ALERT_WEBHOOK_URL,
                json={"text": f"\u26a0\ufe0f AQI Alert for {city}: {int(aqi)} ({label})"},
                timeout=10,
            )
        except requests.RequestException as e:
            print(f"Warning: failed to send alert webhook: {e}")


def run():
    if not all([OWM_API_KEY, AQICN_TOKEN]):
        raise EnvironmentError("OWM_API_KEY and AQICN_TOKEN must be set")

    weather = fetch_weather(CITY_LAT, CITY_LON, OWM_API_KEY)
    pollution = fetch_pollution(CITY_LAT, CITY_LON, OWM_API_KEY)
    aqicn = fetch_aqicn(CITY_LAT, CITY_LON, AQICN_TOKEN)

    row = build_feature_row(weather, pollution, aqicn, CITY_NAME)
    print(f"[{row['timestamp']}] {CITY_NAME}: AQI={row['aqi']} PM2.5={row['pm2_5']} temp={row['temperature']}")

    df = pd.DataFrame([row])
    df = add_cyclical_time_features(df)
    insert_rows(df)

    maybe_send_alert(row.get("aqi"), CITY_NAME)
    print("Feature row inserted successfully.")


if __name__ == "__main__":
    run()
