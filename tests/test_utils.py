"""
Basic unit tests for feature-engineering logic and the SQLite datastore
(no external API calls — safe to run in CI without secrets).

Run with:
    pytest tests/
"""

import os
import sys
import tempfile
import numpy as np
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from utils import (
    engineer_all_features, add_forecast_targets,
    build_inference_row_from_latest, aqi_category, ALL_FEATURES,
)


def make_synthetic_df(n=240):
    ts = pd.date_range("2024-01-01", periods=n, freq="h")
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "city": "TestCity",
        "timestamp": ts,
        "temperature": rng.uniform(10, 30, n),
        "humidity": rng.uniform(20, 80, n),
        "pressure": rng.uniform(1000, 1020, n),
        "wind_speed": rng.uniform(0, 10, n),
        "wind_deg": rng.uniform(0, 360, n),
        "clouds": rng.uniform(0, 100, n),
        "pm2_5": rng.uniform(10, 150, n),
        "pm10": rng.uniform(20, 200, n),
        "no2": rng.uniform(5, 50, n),
        "so2": rng.uniform(1, 20, n),
        "co": rng.uniform(100, 1000, n),
        "o3": rng.uniform(10, 100, n),
        "aqi": rng.uniform(20, 250, n),
    })


def test_engineer_all_features_shape_and_columns():
    df = make_synthetic_df()
    feat = engineer_all_features(df)
    assert len(feat) == len(df)
    for col in ALL_FEATURES:
        assert col in feat.columns, f"missing feature column: {col}"


def test_lag_features_are_nan_at_series_start():
    df = make_synthetic_df()
    feat = engineer_all_features(df)
    assert pd.isna(feat.iloc[0]["aqi_lag_1h"])
    assert pd.isna(feat.iloc[0]["aqi_lag_24h"])


def test_forecast_targets_shift_correctly():
    df = make_synthetic_df()
    feat = engineer_all_features(df)
    targets = add_forecast_targets(feat)
    # target at row i for 24h horizon should equal aqi at row i+24
    aligned = targets["aqi"].shift(-24).reset_index(drop=True)
    got = targets["aqi_target_24h"].reset_index(drop=True)
    pd.testing.assert_series_equal(aligned, got, check_names=False)


def test_build_inference_row_has_all_features():
    df = make_synthetic_df()
    feat = engineer_all_features(df)
    row = build_inference_row_from_latest(feat.iloc[-1].to_dict())
    assert list(row.columns) == ALL_FEATURES
    assert row.shape[0] == 1
    assert not row.isna().any().any()


def test_aqi_category_boundaries():
    assert aqi_category(25)[0] == "Good"
    assert aqi_category(75)[0] == "Moderate"
    assert aqi_category(125)[0] == "Unhealthy for Sensitive Groups"
    assert aqi_category(175)[0] == "Unhealthy"
    assert aqi_category(250)[0] == "Very Unhealthy"
    assert aqi_category(350)[0] == "Hazardous"


def test_sqlite_insert_and_read_roundtrip(tmp_path, monkeypatch):
    # Point the datastore at a throwaway temp file for this test only.
    db_file = tmp_path / "test_aqi.db"
    monkeypatch.setenv("DB_PATH", str(db_file))

    import importlib
    import utils
    importlib.reload(utils)  # pick up the new DB_PATH

    df = make_synthetic_df(n=5)
    df["city"] = "RoundTripCity"
    featured = utils.engineer_all_features(df)

    utils.insert_rows(featured)
    result = utils.read_history("RoundTripCity")

    assert len(result) == 5
    assert set(["city", "timestamp", "aqi"]).issubset(result.columns)

    # Re-inserting the same rows should be a no-op, not create duplicates,
    # thanks to the UNIQUE(city, timestamp) constraint.
    utils.insert_rows(featured)
    result_after_dupe_insert = utils.read_history("RoundTripCity")
    assert len(result_after_dupe_insert) == 5
