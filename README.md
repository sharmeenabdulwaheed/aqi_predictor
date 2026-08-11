# Pearls AQI Predictor

A serverless, end-to-end system that predicts Air Quality Index (AQI) for
the next 3 days. **This version needs only two API keys — OpenWeather and
AQICN — and a GitHub account. No Hopsworks, no Google Cloud, no database
signup of any kind.**

```
Feature Pipeline (hourly)  ---> SQLite file (data/aqi_data.db) ---> Training Pipeline (daily)
      [GitHub Action]              committed to git repo              [GitHub Action]
                                                                              |
                                                                              v
                                                                  models/*.pkl + artifacts/
                                                                    committed to git repo
                                                                              |
                                                                              v
                                                                 Streamlit Dashboard
                                                          (reads the same repo checkout)
```

## 1. How the "serverless" storage works (no database account needed)

Instead of a hosted feature store, the **datastore is a single SQLite
file** (`data/aqi_data.db`) that lives inside this git repository. Every
hour, a GitHub Action:
1. Fetches fresh weather + AQI data,
2. Appends a row to the SQLite file,
3. Commits and pushes the updated file back into the repo.

The daily training GitHub Action does the same with `models/*.pkl` and
`artifacts/` (trained models + SHAP plots + metrics). The Streamlit
dashboard is deployed from this same repo, so it always reads the latest
committed `.db` and model files — and Streamlit Cloud automatically
redeploys whenever the repo gets a new commit.

This is genuinely free and requires no external service beyond GitHub,
which you already need for the CI/CD anyway. The trade-off (worth noting
in your report): git isn't built to be a database, so this approach is
best for a single city / moderate data volume, which fits this project
well. If you outgrow it, swapping `src/utils.py`'s storage functions for
Postgres/BigQuery/etc. later is a contained change — nothing else in the
pipeline needs to know how storage works under the hood.

## 2. What's in this repo

```
aqi-predictor/
├── .github/workflows/
│   ├── feature_pipeline.yml      # runs hourly, commits data/aqi_data.db
│   ├── training_pipeline.yml     # runs daily, commits models/ + artifacts/
│   └── tests.yml                  # runs unit tests on every push
├── src/
│   ├── utils.py                   # config, SQLite helpers, feature engineering
│   ├── feature_pipeline.py        # fetch current data -> insert into SQLite
│   ├── backfill.py                # populate historical data (Open-Meteo, free)
│   ├── training_pipeline.py       # train, evaluate, save models
│   └── inference.py               # load models + produce a forecast
├── app/
│   └── dashboard.py                # Streamlit dashboard
├── notebooks/
│   └── eda.py                      # exploratory data analysis script
├── data/aqi_data.db                # the "feature store" (created on first run)
├── models/                         # trained model files (created by training)
├── artifacts/                      # SHAP plots + metrics.json
├── report/
│   └── final_report_template.md
├── requirements.txt
├── .env.example
└── README.md
```

## 3. What you need

| Thing | Why | Link |
|---|---|---|
| OpenWeatherMap API key | current weather + pollutant concentrations | https://openweathermap.org/api |
| AQICN / WAQI token | ground-truth AQI (the prediction target, 0-500 US EPA scale) | https://aqicn.org/data-platform/token/ |
| GitHub account | hosts the repo + free CI/CD (Actions) | https://github.com |
| Streamlit Community Cloud account (free) | hosts the dashboard | https://streamlit.io/cloud |

That's it — no database, no cloud provider, no model registry service.

## 4. Local setup

```bash
git clone <your-repo-url>
cd aqi-predictor
python -m venv venv && source venv/bin/activate      # venv\Scripts\activate on Windows
pip install -r requirements.txt

cp .env.example .env
# edit .env and fill in OWM_API_KEY, AQICN_TOKEN, and your city's name/lat/lon
```

Load the `.env` file before running scripts locally:
```bash
export $(grep -v '^#' .env | xargs)   # macOS/Linux
python src/feature_pipeline.py
```
(On Windows, use a tool like `python-dotenv`'s CLI, or set the variables
manually in PowerShell with `$env:OWM_API_KEY="..."`.)

## 5. Step-by-step: bring the system up

### Step 1 — Test the feature pipeline once, manually
```bash
python src/feature_pipeline.py
```
This fetches one row of live data and appends it to `data/aqi_data.db`.
Verify it worked:
```bash
python -c "from src.utils import read_history; print(read_history().tail())"
```

### Step 2 — Backfill historical data
```bash
python src/backfill.py --start 2023-01-01 --end 2024-12-31
```
This uses the free, key-less Open-Meteo archive APIs to pull 1-2 years of
hourly weather + pollutant history for your city straight into the same
SQLite file. This is what makes real model training possible without
waiting months for the hourly pipeline alone to accumulate data.

### Step 3 — Run EDA
```bash
python notebooks/eda.py
```
Plots are saved to `notebooks/eda_outputs/`. Use these in your report.

### Step 4 — Train models
```bash
python src/training_pipeline.py
```
Trains Ridge / RandomForest / XGBoost for each of the 3 forecast horizons
(24h, 48h, 72h), picks the best model per horizon by RMSE on a
**time-based** holdout split (not random — avoids leaking future data into
training), saves a SHAP summary plot per horizon to `artifacts/`, and
saves the winning models to `models/*.pkl`.

### Step 5 — Commit the data + models, then push
```bash
git add data/aqi_data.db models/ artifacts/
git commit -m "Initial backfill + first trained models"
git push
```

### Step 6 — Set up GitHub Actions (automation)
1. Push this repo to GitHub (if you haven't already).
2. Go to **Settings → Secrets and variables → Actions**, and add:
   `OWM_API_KEY`, `AQICN_TOKEN`, `CITY_LAT`, `CITY_LON`, `CITY_NAME`, and
   optionally `ALERT_WEBHOOK_URL`.
3. Go to **Settings → Actions → General → Workflow permissions** and select
   **"Read and write permissions"** — this lets the workflows commit
   `data/aqi_data.db` and `models/` back into the repo automatically.
4. The workflows in `.github/workflows/` now run on their own: the
   feature pipeline every hour, the training pipeline once a day. You can
   also trigger either manually from the **Actions** tab
   (`workflow_dispatch`) to test them immediately instead of waiting.

### Step 7 — Run the dashboard
```bash
streamlit run app/dashboard.py
```
To deploy it for free: push to GitHub, go to share.streamlit.io, point a
new app at `app/dashboard.py`. No secrets are required for the dashboard
itself since it just reads files from the repo checkout.

## 6. Forecasting approach

AQI is predicted at 3 fixed horizons — **t+24h, t+48h, t+72h** — using
one regression model per horizon (a "direct multi-horizon" strategy). Each
model consumes the same engineered feature row (current weather,
pollutants, time-of-day/season encodings, AQI lags and rolling stats) and
is trained to predict the AQI that many hours ahead. This is simpler and
more robust with limited data than a single sequence model, while still
being extendable to an LSTM/GRU for comparison if you want a deep-learning
entry (document that experiment in `report/final_report_template.md`).

## 7. Model evaluation

Every training run logs RMSE, MAE, and R² per model per horizon (printed
to the console and saved to `artifacts/latest_metrics.json`), using a
chronological train/test split so no future information leaks into
training.

## 8. Alerts

- The dashboard shows a red/orange banner whenever current or forecast AQI
  crosses the "Unhealthy" (150) or "Hazardous" (200+) EPA thresholds.
- `feature_pipeline.py` can also push a webhook notification (Slack/
  Discord/ntfy.sh — set `ALERT_WEBHOOK_URL`) whenever the current AQI
  exceeds 150, so you're notified without opening the dashboard.

## 9. Troubleshooting

- **"No rows found for city=..."** — run `feature_pipeline.py` or
  `backfill.py` at least once before starting the dashboard.
- **Training pipeline says "Not enough rows"** — you need at least ~75
  hourly rows (more realistically a few hundred) before the 72h lag
  feature has enough history; run the backfill script for more data.
- **GitHub Action can't push (permission denied)** — check Step 6.3 above;
  the default `GITHUB_TOKEN` needs write access enabled in repo settings.
- **Two workflows racing on `git push`** — both workflows share a
  `concurrency` group and use `git pull --rebase` before pushing, so
  overlapping runs queue instead of failing outright. If you still see
  conflicts, re-run the failed job from the Actions tab.
- **GitHub Actions cron didn't fire exactly on time** — this is a known
  GitHub limitation (best-effort scheduling, can drift several minutes
  under load); use `workflow_dispatch` to trigger manually when testing.
