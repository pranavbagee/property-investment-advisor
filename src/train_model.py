"""
train_model.py
---------------
Trains the property price-prediction model.

Uses XGBoost if it's installed (as specified in the project's software
requirements). If it isn't available in the current environment, it
transparently falls back to scikit-learn's GradientBoostingRegressor,
which has a near-identical training/prediction API, so the rest of the
pipeline (ROI, risk, SHAP, Streamlit) doesn't need to change.
"""

import json
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

NUMERIC_FEATURES = [
    "total_sqft", "bhk", "bath", "balcony", "age_years",
    "distance_to_center_km", "parking_spaces", "rera_approved",
    "nearby_schools", "nearby_hospitals",
]
CATEGORICAL_FEATURES = [
    "city", "locality_tier", "area_type", "furnishing", "facing",
]
TARGET = "price"


def build_pipeline():
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )

    if HAS_XGBOOST:
        model = XGBRegressor(
            n_estimators=400,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=42,
            n_jobs=-1,
        )
        model_name = "XGBoost"
    else:
        model = GradientBoostingRegressor(
            n_estimators=400,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.85,
            random_state=42,
        )
        model_name = "GradientBoostingRegressor (sklearn fallback for XGBoost)"

    pipe = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", model),
    ])
    return pipe, model_name


def main(base_dir="."):
    df = pd.read_csv(os.path.join(base_dir, "data", "properties.csv"))

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipe, model_name = build_pipeline()
    pipe.fit(X_train, y_train)

    preds = pipe.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    mape = mean_absolute_percentage_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    metrics = {
        "model_name": model_name,
        "mae_inr": round(float(mae), 2),
        "mape_pct": round(float(mape) * 100, 2),
        "r2_score": round(float(r2), 4),
        "n_train": len(X_train),
        "n_test": len(X_test),
    }

    print("=== Model Evaluation ===")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    models_dir = os.path.join(base_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    with open(os.path.join(models_dir, "price_model.pkl"), "wb") as f:
        pickle.dump(pipe, f)

    with open(os.path.join(models_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    with open(os.path.join(models_dir, "feature_lists.json"), "w") as f:
        json.dump({
            "numeric": NUMERIC_FEATURES,
            "categorical": CATEGORICAL_FEATURES,
            "target": TARGET,
        }, f, indent=2)

    print("\nSaved model -> models/price_model.pkl")
    print("Saved metrics -> models/metrics.json")
    return metrics


if __name__ == "__main__":
    main()
