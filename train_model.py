"""Train the Bengaluru house price model (Module 5: model selection).

Pipeline followed:
  1. clean the raw Kaggle data (Bengaluru house price data)
  2. split train/test
  3. train several candidate models, tune hyperparameters with GridSearchCV
  4. compare candidates with 5-fold cross-validation on MAE / RMSE / R2
  5. refit the winner on the full training set, report holdout metrics
  6. save model pickle + columns.json + metrics.json for the Streamlit app

Run:  python train_model.py
"""

import json
import re
from pathlib import Path

import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "bengaluru_house_data.csv"
MODEL_PATH = ROOT / "banglore_home_prices_model.pickle"
COLUMNS_PATH = ROOT / "columns.json"
METRICS_PATH = ROOT / "metrics.json"

RANDOM_STATE = 42
MIN_LOCATION_COUNT = 10
UNIT_TO_SQFT = {
    "sq. meter": 10.7639,
    "sq. yards": 9.0,
    "acres": 43560.0,
    "cents": 435.6,
    "guntha": 1089.0,
    "perch": 272.25,
    "grounds": 2400.0,
}


def parse_sqft(value: str) -> float:
    """Convert '1200', '2100 - 2850', '34.46Sq. Meter' to a single sqft number."""
    text = str(value).strip().lower().replace(",", "")
    unit = 1.0
    for name, factor in UNIT_TO_SQFT.items():
        if text.endswith(name):
            unit = factor
            text = text[: -len(name)].strip()
            break
    if "-" in text:
        low, high = text.split("-")[:2]
        try:
            return (float(low) + float(high)) / 2 * unit
        except ValueError:
            return np.nan
    try:
        return float(text) * unit
    except ValueError:
        return np.nan


def parse_bhk(size: str) -> float:
    match = re.search(r"\d+", str(size))
    return float(match.group()) if match else np.nan


def load_and_clean() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    rows_raw = len(df)

    df["bhk"] = df["size"].map(parse_bhk)
    df["total_sqft"] = df["total_sqft"].map(parse_sqft)
    df["location"] = (
        df["location"].fillna("other").astype(str).str.strip().str.lower()
    )
    df = df.dropna(subset=["total_sqft", "bhk", "bath", "price"])

    # basic sanity rules
    df = df[(df["bath"] <= df["bhk"] + 2) & (df["total_sqft"] >= 300)]
    df = df[df["total_sqft"] / df["bhk"] >= 300]
    df = df[(df["price"] > 0) & (df["total_sqft"] > 0)]

    # price-per-sqft outliers inside each location (classic cleaning step)
    df["price_per_sqft"] = df["price"] * 1e5 / df["total_sqft"]
    counts = df["location"].value_counts()
    keep = pd.Series(True, index=df.index)
    for location, group in df.groupby("location"):
        if counts[location] < MIN_LOCATION_COUNT:
            continue
        mean = group["price_per_sqft"].mean()
        std = group["price_per_sqft"].std()
        keep &= ~((df["location"] == location) & (df["price_per_sqft"].between(mean - std, mean + std) == False))
    df = df[keep]

    # fold rare locations into "other"
    counts = df["location"].value_counts()
    df.loc[df["location"].isin(counts[counts < MIN_LOCATION_COUNT].index), "location"] = "other"

    df = df[["location", "total_sqft", "bath", "bhk", "price"]].reset_index(drop=True)
    print(f"rows: {rows_raw} raw -> {len(df)} clean ({len(df) / rows_raw:.1%})")
    print(f"locations kept: {df['location'].nunique()}")
    print(f"price (lakhs): mean {df['price'].mean():.1f}, median {df['price'].median():.1f}")
    return df


def build_matrix(df: pd.DataFrame, locations: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Same 244-column layout the app builds: sqft, bath, bhk, then one-hot locations."""
    x = np.zeros((len(df), 3 + len(locations)))
    x[:, 0] = df["total_sqft"].to_numpy()
    x[:, 1] = df["bath"].to_numpy()
    x[:, 2] = df["bhk"].to_numpy()
    location_index = {name: i + 3 for i, name in enumerate(locations)}
    for row, name in enumerate(df["location"].to_numpy()):
        if name in location_index:
            x[row, location_index[name]] = 1
    return x, df["price"].to_numpy()


def candidate_models() -> dict:
    return {
        "LinearRegression": (Pipeline([("scale", StandardScaler()), ("model", LinearRegression())]), {}),
        "Ridge": (
            Pipeline([("scale", StandardScaler()), ("model", Ridge(random_state=RANDOM_STATE))]),
            {"model__alpha": [0.1, 1.0, 10.0, 100.0]},
        ),
        "Lasso": (
            Pipeline([("scale", StandardScaler()), ("model", Lasso(max_iter=5000, random_state=RANDOM_STATE))]),
            {"model__alpha": [0.001, 0.01, 0.1, 1.0]},
        ),
        "RandomForest": (
            Pipeline([("model", RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1))]),
            {"model__n_estimators": [300], "model__max_depth": [None, 24], "model__min_samples_leaf": [1, 2]},
        ),
        "GradientBoosting": (
            Pipeline([("model", GradientBoostingRegressor(random_state=RANDOM_STATE))]),
            {"model__n_estimators": [300], "model__learning_rate": [0.05, 0.1], "model__max_depth": [3]},
        ),
    }


def evaluate(model, x, y) -> dict:
    predictions = model.predict(x)
    return {
        "MAE_lakhs": float(mean_absolute_error(y, predictions)),
        "RMSE_lakhs": float(mean_squared_error(y, predictions) ** 0.5),
        "R2": float(r2_score(y, predictions)),
    }


def main() -> None:
    df = load_and_clean()
    locations = sorted(df["location"].unique())
    x, y = build_matrix(df, locations)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=RANDOM_STATE
    )
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    results = []
    fitted = {}
    for name, (estimator, grid) in candidate_models().items():
        print(f"\n== {name}")
        if grid:
            search = GridSearchCV(
                estimator,
                grid,
                cv=cv,
                scoring="neg_root_mean_squared_error",
                n_jobs=-1,
            )
            search.fit(x_train, y_train)
            model = search.best_estimator_
            best_params = {k: (v.item() if hasattr(v, "item") else v) for k, v in search.best_params_.items()}
            print(f"   best params: {best_params}")
        else:
            model = estimator.fit(x_train, y_train)
            best_params = {}

        scores = cross_validate(
            model, x_train, y_train, cv=cv, n_jobs=-1,
            scoring=("neg_mean_absolute_error", "neg_root_mean_squared_error", "r2"),
        )
        cv_row = {
            "model": name,
            "best_params": best_params,
            "cv_MAE_lakhs": float(-scores["test_neg_mean_absolute_error"].mean()),
            "cv_RMSE_lakhs": float(-scores["test_neg_root_mean_squared_error"].mean()),
            "cv_R2": float(scores["test_r2"].mean()),
        }
        cv_row.update({f"test_{k}": v for k, v in evaluate(model, x_test, y_test).items()})
        results.append(cv_row)
        fitted[name] = model
        print(
            f"   CV  RMSE {cv_row['cv_RMSE_lakhs']:.2f} lakhs | R2 {cv_row['cv_R2']:.4f}"
            f"  ||  test RMSE {cv_row['test_RMSE_lakhs']:.2f} | R2 {cv_row['test_R2']:.4f}"
        )

    table = pd.DataFrame(results).sort_values("cv_RMSE_lakhs").reset_index(drop=True)
    best_name = table.loc[0, "model"]
    best_model = fitted[best_name]

    # refit winner on train+test (all cleaned rows) before shipping
    best_model.fit(x, y)
    holdout = table.loc[0]

    print("\n== ranking (by CV RMSE)")
    print(table[["model", "cv_MAE_lakhs", "cv_RMSE_lakhs", "cv_R2", "test_RMSE_lakhs", "test_R2"]].to_string(index=False))
    print(f"\nselected: {best_name}")

    MODEL_PATH.write_bytes(pickle.dumps(best_model))
    COLUMNS_PATH.write_text(json.dumps({"data_columns": ["total_sqft", "bath", "bhk"] + locations}))
    METRICS_PATH.write_text(
        json.dumps(
            {
                "selected_model": best_name,
                "rows_raw": int(pd.read_csv(DATA_PATH, usecols=["price"]).shape[0]),
                "rows_clean": int(len(df)),
                "n_locations": len(locations),
                "cv_folds": 5,
                "holdout_MAE_lakhs": round(holdout["test_MAE_lakhs"], 2),
                "holdout_RMSE_lakhs": round(holdout["test_RMSE_lakhs"], 2),
                "holdout_R2": round(holdout["test_R2"], 4),
                "cv_RMSE_lakhs": round(holdout["cv_RMSE_lakhs"], 2),
                "candidates": [
                    {
                        "model": row["model"],
                        "cv_MAE_lakhs": round(row["cv_MAE_lakhs"], 2),
                        "cv_RMSE_lakhs": round(row["cv_RMSE_lakhs"], 2),
                        "cv_R2": round(row["cv_R2"], 4),
                        "test_RMSE_lakhs": round(row["test_RMSE_lakhs"], 2),
                        "test_R2": round(row["test_R2"], 4),
                        "best_params": row["best_params"],
                    }
                    for _, row in table.iterrows()
                ],
            },
            indent=2,
        )
    )
    print(f"saved {MODEL_PATH.name}, {COLUMNS_PATH.name}, {METRICS_PATH.name}")


if __name__ == "__main__":
    main()
