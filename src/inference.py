"""
inference.py
Shared inference logic used by the dashboard: loads the latest feature row
from SQLite and the 3 locally-saved horizon models, and produces a
24h/48h/72h AQI forecast.
"""

import os
import sys
import joblib

sys.path.append(os.path.dirname(__file__))
from utils import read_latest_row, build_inference_row_from_latest, MODELS_DIR, CITY_NAME

HORIZONS = ["24h", "48h", "72h"]


def load_horizon_models():
    models = {}
    for h in HORIZONS:
        path = os.path.join(MODELS_DIR, f"aqi_model_{h}.pkl")
        if os.path.exists(path):
            models[h] = joblib.load(path)
        else:
            print(f"Warning: no trained model found for horizon {h} at {path}. "
                  f"Run training_pipeline.py first.")
    return models


def get_forecast(city_name: str = CITY_NAME):
    latest_row, history_df = read_latest_row(city_name)
    models = load_horizon_models()

    X = build_inference_row_from_latest(latest_row)

    forecast = {}
    for h, model in models.items():
        pred = float(model.predict(X)[0])
        forecast[h] = pred

    return {
        "city": city_name,
        "current_aqi": latest_row.get("aqi"),
        "latest_row": latest_row,
        "forecast": forecast,
        "history": history_df,
    }


if __name__ == "__main__":
    result = get_forecast()
    print("Current AQI:", result["current_aqi"])
    print("Forecast:", result["forecast"])
