"""
train_catboost_model.py
------------------------
Trains a second, distinct regression model for comparison against the
Linear Regression / Random Forest / XGBoost baselines already covered
in the literature review: **CatBoost**.

Why CatBoost is a genuinely different technique (not just "another
boosting library"):
  - Uses *ordered boosting*, a permutation-driven training scheme that
    reduces the target leakage / overfitting that plain gradient
    boosting (incl. XGBoost) is prone to on small-to-medium tabular data.
  - Handles categorical features (city, locality_tier, furnishing, etc.)
    *natively*, without one-hot encoding — it builds its own optimal
    target-statistics encoding internally, which usually helps when a
    categorical column (like `city`) has a big effect on price.

Fallback: this sandbox has no internet, so `catboost` can't be
installed here. If it isn't importable, this script falls back to
scikit-learn's HistGradientBoostingRegressor with native categorical
support (`categorical_features=...`), which is the closest built-in
equivalent (also histogram-based gradient boosting with native
categorical handling, no one-hot encoding). On your machine, once you
`pip install catboost`, this script automatically uses real CatBoost —
no code changes needed.
"""

import json
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.model_selection import train_test_split

try:
    from catboost import CatBoostRegressor
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

NUMERIC_FEATURES = [
    "total_sqft", "bhk", "bath", "balcony", "age_years",
    "distance_to_center_km", "parking_spaces", "rera_approved",
    "nearby_schools", "nearby_hospitals",
]
CATEGORICAL_FEATURES = [
    "city", "locality_tier", "area_type", "furnishing", "facing",
]
TARGET = "price"


class CatBoostFallbackWrapper:
    """
    Wraps HistGradientBoostingRegressor so it exposes the same
    .predict(df) interface as the real CatBoost model below, keeping
    advisor.py agnostic to which one is actually loaded.
    """
    def __init__(self, model, categorical_features):
        self.model = model
        self.categorical_features = categorical_features

    def predict(self, X: pd.DataFrame):
        X = X.copy()
        for c in self.categorical_features:
            X[c] = X[c].astype("category")
        return self.model.predict(X)

    @property
    def feature_importances_(self):
        return self.model.feature_importances_


class CatBoostWrapper:
    """Wraps a real CatBoostRegressor to accept a plain DataFrame."""
    def __init__(self, model, feature_order, cat_feature_idx):
        self.model = model
        self.feature_order = feature_order
        self.cat_feature_idx = cat_feature_idx

    def predict(self, X: pd.DataFrame):
        X = X[self.feature_order]
        return self.model.predict(X)

    @property
    def feature_importances_(self):
        return self.model.get_feature_importance()


def main():
    df = pd.read_csv("data/properties.csv")
    feature_order = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    X = df[feature_order]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    if HAS_CATBOOST:
        model_name = "CatBoost"
        cat_idx = [feature_order.index(c) for c in CATEGORICAL_FEATURES]
        raw_model = CatBoostRegressor(
            iterations=600,
            depth=6,
            learning_rate=0.05,
            loss_function="RMSE",
            cat_features=cat_idx,
            random_seed=42,
            verbose=False,
        )
        raw_model.fit(X_train, y_train)
        wrapped = CatBoostWrapper(raw_model, feature_order, cat_idx)
    else:
        model_name = "HistGradientBoostingRegressor (sklearn fallback for CatBoost)"
        X_train_c = X_train.copy()
        X_test_c = X_test.copy()
        for c in CATEGORICAL_FEATURES:
            X_train_c[c] = X_train_c[c].astype("category")
            X_test_c[c] = X_test_c[c].astype("category")

        raw_model = HistGradientBoostingRegressor(
            max_iter=600,
            max_depth=6,
            learning_rate=0.05,
            categorical_features=CATEGORICAL_FEATURES,
            random_state=42,
        )
        raw_model.fit(X_train_c, y_train)
        wrapped = CatBoostFallbackWrapper(raw_model, CATEGORICAL_FEATURES)

    preds = wrapped.predict(X_test)
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

    print("=== CatBoost Model Evaluation ===")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    with open("models/catboost_model.pkl", "wb") as f:
        pickle.dump(wrapped, f)

    with open("models/catboost_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("\nSaved model -> models/catboost_model.pkl")
    print("Saved metrics -> models/catboost_metrics.json")


if __name__ == "__main__":
    main()
