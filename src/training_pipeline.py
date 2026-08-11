"""
training_pipeline.py
Runs daily (via GitHub Actions).

1. Reads historical (features, target) rows from the SQLite database.
2. Engineers 3 forecasting targets: AQI at t+24h, t+48h, t+72h.
3. Trains/evaluates several models per horizon (Ridge, RandomForest, XGBoost).
4. Picks the best model per horizon by RMSE on a time-based holdout split.
5. Computes SHAP feature importance for each horizon's winning model.
6. Saves the 3 winning models to models/ and metrics/plots to artifacts/.
   There's no separate "model registry" service — the GitHub Actions
   workflow commits these files straight back into the repo, and the
   dashboard simply loads them from disk (same repo checkout).
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

sys.path.append(os.path.dirname(__file__))
from utils import (
    read_history, engineer_all_features, add_forecast_targets,
    ALL_FEATURES, CITY_NAME, MODELS_DIR, ARTIFACTS_DIR,
)

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

HORIZONS = {
    "24h": "aqi_target_24h",
    "48h": "aqi_target_48h",
    "72h": "aqi_target_72h",
}

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(ARTIFACTS_DIR, exist_ok=True)


def time_based_split(df: pd.DataFrame, test_frac: float = 0.2):
    """Chronological split: train on the earlier portion, test on the most
    recent slice. Avoids leaking future information into training, which a
    random split would do for a time series."""
    df = df.sort_values("timestamp")
    cutoff_idx = int(len(df) * (1 - test_frac))
    train = df.iloc[:cutoff_idx]
    test = df.iloc[cutoff_idx:]
    return train, test


def get_candidate_models():
    models = {
        "ridge": Ridge(alpha=1.0),
        "random_forest": RandomForestRegressor(
            n_estimators=300, max_depth=12, min_samples_leaf=2,
            random_state=42, n_jobs=-1,
        ),
    }
    if HAS_XGB:
        models["xgboost"] = XGBRegressor(
            n_estimators=400, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
        )
    return models


def evaluate(y_true, y_pred):
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {"rmse": rmse, "mae": mae, "r2": r2}


def train_for_horizon(df: pd.DataFrame, horizon_name: str, target_col: str):
    data = df.dropna(subset=ALL_FEATURES + [target_col]).copy()
    if len(data) < 50:
        print(f"[{horizon_name}] Not enough rows ({len(data)}) to train yet — skipping. "
              f"Run backfill.py for more history.")
        return None

    train, test = time_based_split(data)
    X_train, y_train = train[ALL_FEATURES], train[target_col]
    X_test, y_test = test[ALL_FEATURES], test[target_col]

    results = {}
    fitted_models = {}
    for name, model in get_candidate_models().items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        metrics = evaluate(y_test, preds)
        results[name] = metrics
        fitted_models[name] = model
        print(f"[{horizon_name}] {name}: RMSE={metrics['rmse']:.2f} "
              f"MAE={metrics['mae']:.2f} R2={metrics['r2']:.3f}")

    best_name = min(results, key=lambda k: results[k]["rmse"])
    best_model = fitted_models[best_name]
    print(f"[{horizon_name}] BEST model: {best_name} -> {results[best_name]}")

    return {
        "horizon": horizon_name,
        "best_model_name": best_name,
        "best_model": best_model,
        "metrics": results,
        "X_test": X_test,
        "y_test": y_test,
    }


def save_shap_summary(model, X_sample: pd.DataFrame, horizon_name: str):
    if not HAS_SHAP:
        print("shap not installed, skipping explainability plot.")
        return None
    try:
        if type(model).__name__ in ("XGBRegressor", "RandomForestRegressor"):
            explainer = shap.TreeExplainer(model)
        else:
            explainer = shap.LinearExplainer(model, X_sample)
        shap_values = explainer.shap_values(X_sample)

        plt.figure()
        shap.summary_plot(shap_values, X_sample, show=False)
        path = os.path.join(ARTIFACTS_DIR, f"shap_summary_{horizon_name}.png")
        plt.savefig(path, bbox_inches="tight", dpi=120)
        plt.close()
        print(f"Saved SHAP summary to {path}")
        return path
    except Exception as e:
        print(f"Warning: SHAP explanation failed ({e}); continuing without it.")
        return None


def save_model(model, horizon_name: str):
    path = os.path.join(MODELS_DIR, f"aqi_model_{horizon_name}.pkl")
    joblib.dump(model, path)
    print(f"Saved model to {path}")
    return path


def run():
    df = read_history(CITY_NAME)
    print(f"Loaded {len(df)} raw rows from {CITY_NAME}'s history.")
    if df.empty:
        print("No data found. Run feature_pipeline.py and/or backfill.py first.")
        return

    # The SQLite table only stores raw weather/pollutant/AQI columns; the
    # engineered features (cyclical time encodings, lags, rolling stats)
    # are recomputed here from the full history every training run. This
    # keeps the datastore schema simple and avoids ever storing stale
    # lag/rolling values that would need to be back-filled if the feature
    # engineering logic changes.
    df = engineer_all_features(df)
    df = add_forecast_targets(df)

    all_metrics = {}
    for horizon_name, target_col in HORIZONS.items():
        result = train_for_horizon(df, horizon_name, target_col)
        if result is None:
            continue

        all_metrics[horizon_name] = {
            "best_model": result["best_model_name"],
            "metrics": result["metrics"],
        }

        sample = result["X_test"].sample(min(200, len(result["X_test"])), random_state=42)
        save_shap_summary(result["best_model"], sample, horizon_name)
        save_model(result["best_model"], horizon_name)

    metrics_path = os.path.join(ARTIFACTS_DIR, "latest_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"Saved run summary to {metrics_path}")
    print(json.dumps(all_metrics, indent=2))


if __name__ == "__main__":
    run()
