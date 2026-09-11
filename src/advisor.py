"""
advisor.py
----------
The Explainable AI Property Investment Advisor core logic.

Combines:
  1. Price prediction         (trained model from train_model.py)
  2. Future price forecasting (city-level historical appreciation)
  3. ROI estimation
  4. Investment risk classification (Low / Medium / High)
  5. Buy / Hold / Avoid recommendation
  6. Explainability            (SHAP if installed, else a transparent
                                 occlusion-based fallback explainer with
                                 the same additive, per-feature format)
"""

import json
import pickle
from dataclasses import dataclass, field
from typing import Dict

import numpy as np
import pandas as pd

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False


@dataclass
class AdvisorResult:
    predicted_price: float
    forecast_years: int
    projected_future_price: float
    annual_appreciation_used: float
    roi_pct: float
    risk_level: str
    risk_reason: str
    recommendation: str
    recommendation_reason: str
    explanation: Dict[str, float] = field(default_factory=dict)
    explanation_baseline: float = 0.0
    explanation_method: str = ""


class PropertyInvestmentAdvisor:
    def __init__(self,
                 model_path="models/price_model.pkl",
                 feature_lists_path="models/feature_lists.json",
                 appreciation_path="data/city_appreciation.csv",
                 properties_path="data/properties.csv"):
        with open(model_path, "rb") as f:
            self.pipeline = pickle.load(f)

        with open(feature_lists_path) as f:
            fl = json.load(f)
        self.numeric_features = fl["numeric"]
        self.categorical_features = fl["categorical"]
        self.all_features = self.numeric_features + self.categorical_features

        self.appreciation = pd.read_csv(appreciation_path).set_index("city")
        self.reference_df = pd.read_csv(properties_path)

        # Background sample for the fallback explainer's "baseline" values
        self._bg_numeric_medians = self.reference_df[self.numeric_features].median()
        self._bg_categorical_modes = {
            c: self.reference_df[c].mode().iloc[0] for c in self.categorical_features
        }

        if HAS_SHAP:
            self._init_shap_explainer()

    # ------------------------------------------------------------------
    # 1. Price prediction
    # ------------------------------------------------------------------
    def predict_price(self, features: dict) -> float:
        row = pd.DataFrame([features])[self.all_features]
        return float(self.pipeline.predict(row)[0])

    # ------------------------------------------------------------------
    # 2. Future price forecasting
    # ------------------------------------------------------------------
    def forecast_future_price(self, current_price: float, city: str, years: int = 5) -> tuple:
        rate = float(self.appreciation.loc[city, "avg_annual_appreciation"])
        future_price = current_price * ((1 + rate) ** years)
        return future_price, rate

    # ------------------------------------------------------------------
    # 3. ROI
    # ------------------------------------------------------------------
    @staticmethod
    def estimate_roi(current_price: float, future_price: float) -> float:
        return (future_price - current_price) / current_price * 100

    # ------------------------------------------------------------------
    # 4. Risk classification
    # ------------------------------------------------------------------
    def classify_risk(self, city: str, features: dict, predicted_price: float) -> tuple:
        volatility = float(self.appreciation.loc[city, "price_volatility"])

        # Compare predicted price to comparable properties (same city, same
        # locality tier, +-1 BHK) -- comparing within tier matters a lot,
        # since Premium vs Emerging localities differ in price by 2-3x.
        comps = self.reference_df[
            (self.reference_df["city"] == city) &
            (self.reference_df["locality_tier"] == features["locality_tier"]) &
            (self.reference_df["bhk"].between(features["bhk"] - 1, features["bhk"] + 1))
        ]
        if len(comps) >= 10:
            comp_median = comps["price"].median()
            deviation = abs(predicted_price - comp_median) / comp_median
        else:
            deviation = 0.0

        # Simple rule-based scoring: combine market volatility + price deviation
        score = volatility * 100 * 0.6 + deviation * 100 * 0.4

        if score < 6:
            level = "Low"
        elif score < 11:
            level = "Medium"
        else:
            level = "High"

        reason = (
            f"City price volatility is {volatility*100:.1f}%/yr and the predicted price "
            f"deviates {deviation*100:.1f}% from comparable {features['bhk']}-BHK properties in {city}."
        )
        return level, reason

    # ------------------------------------------------------------------
    # 5. Buy / Hold / Avoid recommendation
    # ------------------------------------------------------------------
    @staticmethod
    def recommend(roi_pct: float, risk_level: str) -> tuple:
        if roi_pct >= 35 and risk_level in ("Low", "Medium"):
            return "Buy", f"Strong projected ROI ({roi_pct:.1f}%) with acceptable risk."
        if roi_pct >= 20 and risk_level == "Low":
            return "Buy", f"Solid projected ROI ({roi_pct:.1f}%) in a low-risk market."
        if roi_pct < 10 or risk_level == "High":
            return "Avoid", f"Low projected ROI ({roi_pct:.1f}%) and/or high risk ({risk_level})."
        return "Hold", f"Moderate projected ROI ({roi_pct:.1f}%) with {risk_level.lower()} risk — worth monitoring."

    # ------------------------------------------------------------------
    # 6. Explainability
    # ------------------------------------------------------------------
    def _init_shap_explainer(self):
        model = self.pipeline.named_steps["model"]
        preprocessor = self.pipeline.named_steps["preprocess"]
        bg = self.reference_df[self.all_features].sample(
            n=min(100, len(self.reference_df)), random_state=42
        )
        bg_transformed = preprocessor.transform(bg)
        self._shap_feature_names = preprocessor.get_feature_names_out()
        try:
            self.explainer = shap.TreeExplainer(model)
        except Exception:
            self.explainer = shap.Explainer(model, bg_transformed)

    def explain(self, features: dict, top_n: int = 8) -> AdvisorResult.__annotations__:
        row = pd.DataFrame([features])[self.all_features]

        if HAS_SHAP:
            return self._explain_shap(row, top_n)
        return self._explain_fallback(features, top_n)

    def _explain_shap(self, row: pd.DataFrame, top_n: int):
        preprocessor = self.pipeline.named_steps["preprocess"]
        transformed = preprocessor.transform(row)
        shap_values = self.explainer(transformed)

        values = np.array(shap_values.values).flatten()
        names = self._shap_feature_names
        baseline = float(np.array(shap_values.base_values).flatten()[0])

        contrib = dict(zip(names, values))
        contrib = dict(sorted(contrib.items(), key=lambda kv: abs(kv[1]), reverse=True)[:top_n])
        return contrib, baseline, "SHAP (TreeExplainer)"

    def _explain_fallback(self, features: dict, top_n: int):
        """
        Occlusion-based local explanation: for each feature, replace it
        with the population median/mode (a neutral baseline) and measure
        how much the prediction shifts. This mirrors SHAP's additive,
        per-feature attribution format without requiring the shap
        package to be installed.
        """
        baseline_features = dict(features)
        for c in self.numeric_features:
            baseline_features[c] = self._bg_numeric_medians[c]
        for c in self.categorical_features:
            baseline_features[c] = self._bg_categorical_modes[c]

        baseline_pred = self.predict_price(baseline_features)
        full_pred = self.predict_price(features)

        contributions = {}
        for feat in self.all_features:
            modified = dict(features)
            if feat in self.numeric_features:
                modified[feat] = self._bg_numeric_medians[feat]
            else:
                modified[feat] = self._bg_categorical_modes[feat]
            pred_without = self.predict_price(modified)
            # How much this feature's actual value adds, relative to baseline
            contributions[feat] = full_pred - pred_without

        # Rescale so contributions + baseline ~= full prediction (additive property)
        total_contrib = sum(contributions.values())
        gap = (full_pred - baseline_pred) - total_contrib
        if contributions:
            adjust = gap / len(contributions)
            contributions = {k: v + adjust for k, v in contributions.items()}

        contributions = dict(
            sorted(contributions.items(), key=lambda kv: abs(kv[1]), reverse=True)[:top_n]
        )
        return contributions, baseline_pred, "Occlusion-based fallback (install `shap` for TreeExplainer)"

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------
    def full_assessment(self, features: dict, years: int = 5) -> AdvisorResult:
        predicted_price = self.predict_price(features)
        future_price, rate = self.forecast_future_price(predicted_price, features["city"], years)
        roi = self.estimate_roi(predicted_price, future_price)
        risk_level, risk_reason = self.classify_risk(features["city"], features, predicted_price)
        recommendation, rec_reason = self.recommend(roi, risk_level)
        contributions, baseline, method = self.explain(features)

        return AdvisorResult(
            predicted_price=predicted_price,
            forecast_years=years,
            projected_future_price=future_price,
            annual_appreciation_used=rate,
            roi_pct=roi,
            risk_level=risk_level,
            risk_reason=risk_reason,
            recommendation=recommendation,
            recommendation_reason=rec_reason,
            explanation=contributions,
            explanation_baseline=baseline,
            explanation_method=method,
        )


if __name__ == "__main__":
    advisor = PropertyInvestmentAdvisor()

    sample_property = {
        "city": "Bengaluru",
        "locality_tier": "Mid",
        "area_type": "Super built-up Area",
        "furnishing": "Semi-Furnished",
        "facing": "North",
        "total_sqft": 1450,
        "bhk": 3,
        "bath": 3,
        "balcony": 2,
        "age_years": 3,
        "distance_to_center_km": 9.5,
        "parking_spaces": 1,
        "rera_approved": 1,
        "nearby_schools": 3,
        "nearby_hospitals": 2,
    }

    result = advisor.full_assessment(sample_property, years=5)

    print("=== Property Investment Assessment ===")
    print(f"Predicted current price   : Rs. {result.predicted_price:,.0f}")
    print(f"Projected price in {result.forecast_years} yrs : Rs. {result.projected_future_price:,.0f}  "
          f"(using {result.annual_appreciation_used*100:.1f}%/yr)")
    print(f"Estimated ROI             : {result.roi_pct:.1f}%")
    print(f"Investment risk           : {result.risk_level}  -- {result.risk_reason}")
    print(f"Recommendation            : {result.recommendation}  -- {result.recommendation_reason}")
    print(f"\nExplanation method: {result.explanation_method}")
    print(f"Baseline price: Rs. {result.explanation_baseline:,.0f}")
    print("Top feature contributions to predicted price:")
    for feat, val in result.explanation.items():
        sign = "+" if val >= 0 else "-"
        print(f"  {feat:35s} {sign} Rs. {abs(val):,.0f}")
