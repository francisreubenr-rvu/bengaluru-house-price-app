# BH — Bengaluru house price prediction (Module 5)

Machine learning model selection and cloud deployment demo for Module 5
(model selection, deployment, monitoring). A regression model predicts
Bengaluru house prices from area, bedrooms, bathrooms and location, and the
model is served with a Streamlit web app.

## Contents

| File | Purpose |
| --- | --- |
| `data/bengaluru_house_data.csv` | Raw dataset (Kaggle Bengaluru house price data, 13,320 rows) |
| `train_model.py` | Cleaning, model selection, cross-validation, artifact export |
| `banglore_home_prices_model.pickle` | Trained pipeline (StandardScaler + regressor) |
| `columns.json` | Feature order used at inference time |
| `metrics.json` | Model selection report rendered inside the app |
| `app.py` | Streamlit front end |
| `requirements.txt` | Pinned dependencies for local run and Streamlit Cloud |

## What the training script does

1. Cleans `total_sqft` (ranges and unit strings: Sq. Meter, Sq. Yards, Acres,
   Cents, Guntha, Perch, Grounds) and parses BHK from `size`.
2. Drops rows with impossible values (`bath > bhk + 2`, `total_sqft / bhk < 300`)
   and price-per-sqft outliers beyond one standard deviation inside each
   location with 10+ listings. 13,320 raw rows become 10,579.
3. Folds locations with fewer than 10 listings into `other` (198 locations kept)
   and builds the same 244-column vector the app builds at inference time.
4. Tunes five candidates with `GridSearchCV` (5-fold, RMSE) and compares them by
   cross-validation: LinearRegression, Ridge, Lasso, RandomForest,
   GradientBoosting.
5. Refits the winner on all cleaned rows and writes the pickle, `columns.json`
   and `metrics.json`.

Current selection (reproducible with `python train_model.py`):

| Model | CV RMSE (lakhs) | CV MAE | CV R² | Test RMSE | Test R² |
| --- | --- | --- | --- | --- | --- |
| GradientBoosting | 84.75 | 29.57 | 0.497 | 103.59 | 0.569 |
| RandomForest | 85.95 | 28.72 | 0.488 | 100.46 | 0.595 |
| Ridge | 96.42 | 40.67 | 0.359 | 134.82 | 0.270 |
| Lasso | 96.43 | 40.66 | 0.358 | 134.75 | 0.271 |
| LinearRegression | 96.45 | 40.82 | 0.358 | 134.77 | 0.271 |

Train on the same Python minor version you deploy on (Streamlit Cloud uses
3.12 here) so the pickle loads without a scikit-learn version mismatch.

## Run locally

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python train_model.py          # optional: regenerate the artifacts
streamlit run app.py
```

The app opens at http://localhost:8501.

## Deploy on Streamlit Community Cloud

1. Push this repository to GitHub and make it **public**
   (Streamlit Cloud reads public repos on the free tier).
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click **New app** → **Deploy a public app from GitHub**.
4. Fill in:
   - Repository: `<your-account>/<repo>`
   - Branch: `main`
   - Main file path: `app.py`
   - Advanced settings → Python version: `3.12`
5. Click **Deploy**. Streamlit installs `requirements.txt`, then serves the app
   at `https://<app-name>.streamlit.app`.
6. Every push to `main` redeploys automatically. Monitor the app from the
   dashboard (logs, resource usage, reboot) and add a `.streamlit/secrets.toml`
   if the app ever needs credentials.

## Monitoring after deployment (Module 5 checklist)

| Signal | What to watch | Where |
| --- | --- | --- |
| Latency and errors | Streamlit Cloud logs, app analytics | Streamlit dashboard |
| Data drift | Distribution of sqft, BHK, location in incoming requests | log inputs, compare to `data/bengaluru_house_data.csv` |
| Concept drift | Prediction error on recently sold listings | periodic revalidation, then rerun `train_model.py` |
| Resource usage | CPU and memory on the app page | Streamlit Cloud metrics |

Drift and error tracking can be added with Evidently AI or Prometheus +
Grafana once predictions are logged, as covered in the module slides.
