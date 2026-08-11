# Pearls AQI Predictor — Final Report

**Author:** [Your name]
**City forecasted:** [City name]
**Repository:** [link]
**Live dashboard:** [link]

---

## 1. Problem Statement & Motivation

[Why does AQI forecasting matter for your city? Cite recent pollution
levels, health impact, seasonal patterns you're aware of, etc.]

## 2. Data Sources

| Source | Data provided | Used for |
|---|---|---|
| OpenWeatherMap | current weather, pollutant concentrations | live features |
| AQICN / WAQI | ground-truth station AQI (US EPA scale) | prediction target |
| Open-Meteo Archive API | historical weather + air quality | backfilling training data |

[Note any rate limits hit, data gaps, or quirks you discovered.]

## 3. System Architecture

[Paste/describe the architecture diagram from the README. Explain each
component: feature pipeline, feature store, training pipeline, model
registry, dashboard, and how GitHub Actions ties it together.]

## 4. Feature Engineering

Full feature list used by the models:

- **Weather:** temperature, humidity, pressure, wind speed/direction, cloud cover
- **Pollutants:** PM2.5, PM10, NO2, SO2, CO, O3
- **Time (cyclical encodings):** hour, day, month, day-of-week (sin/cos), is_weekend
- **Derived/lag:** AQI lag 1h/24h/72h, 24h rolling mean/std, AQI change rate,
  PM2.5:PM10 ratio

[Explain why cyclical encoding was chosen over raw integers, and why lag
features matter for a time-series forecasting problem.]

## 5. Exploratory Data Analysis

[Insert plots from `notebooks/eda_outputs/`: AQI distribution, category
breakdown, time series, correlation heatmap, hourly/monthly boxplots,
ACF/PACF. 3-5 sentence takeaway under each.]

## 6. Modeling

**Forecasting framing:** [direct multi-horizon models / sequence model —
explain your choice]

**Models compared:**
- Ridge Regression (linear baseline)
- Random Forest
- XGBoost
- [Optional: LSTM/GRU, SARIMA/Prophet — statistical baseline]

**Train/test split:** chronological (not random), [X]% train / [Y]% test,
to avoid leaking future information.

## 7. Results

| Horizon | Model | RMSE | MAE | R² |
|---|---|---|---|---|
| 24h | [best model] | | | |
| 48h | [best model] | | | |
| 72h | [best model] | | | |

[Discuss which model won at each horizon and why — e.g. did tree-based
models outperform linear models because of nonlinear pollutant-weather
interactions? Did accuracy degrade with longer horizons, as expected?]

## 8. Explainability (SHAP)

[Insert `artifacts/shap_summary_24h.png` etc. Interpret: which features
drove predictions most? Did PM2.5 and AQI lag dominate, as expected?]

## 9. Pipeline Automation

[Screenshot the green GitHub Actions runs for both workflows. Describe
the hourly/daily schedule, and how you validated it ran reliably over
time. Note any failures encountered and how they were resolved.]

## 10. Dashboard

[Screenshots of the deployed Streamlit dashboard: current AQI, 3-day
forecast, alert banner, trend chart, SHAP tab.]

## 11. Alerts

[Describe the hazard-threshold banner logic and, if implemented, the
webhook notification integration.]

## 12. Limitations & Future Work

- Free-tier API rate limits cap how frequently/richly data can be pulled
- Single-city scope; multi-city would require per-city feature groups
- No satellite/aerosol imagery features
- GitHub Actions cron scheduling is best-effort, not guaranteed to the minute
- [Anything else you'd improve with more time]

## 13. Appendix

- Repository: [link]
- Setup instructions: see `README.md`
- Key commands to reproduce: `backfill.py`, `training_pipeline.py`, `dashboard.py`
