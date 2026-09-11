"""
batch_demo.py
-------------
Runs the full advisor pipeline on a handful of varied sample properties
(to show Buy / Hold / Avoid actually varies with input), and saves a
global feature-importance chart from the trained model.
"""

import json
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from advisor import PropertyInvestmentAdvisor

SAMPLES = [
    {
        "label": "Bengaluru, 3BHK, Mid-tier, near center",
        "city": "Bengaluru", "locality_tier": "Mid", "area_type": "Super built-up Area",
        "furnishing": "Semi-Furnished", "facing": "North", "total_sqft": 1450, "bhk": 3,
        "bath": 3, "balcony": 2, "age_years": 3, "distance_to_center_km": 4.0,
        "parking_spaces": 1, "rera_approved": 1, "nearby_schools": 3, "nearby_hospitals": 2,
    },
    {
        "label": "Mumbai, 2BHK, Premium, old building",
        "city": "Mumbai", "locality_tier": "Premium", "area_type": "Carpet Area",
        "furnishing": "Unfurnished", "facing": "West", "total_sqft": 950, "bhk": 2,
        "bath": 2, "balcony": 1, "age_years": 20, "distance_to_center_km": 3.0,
        "parking_spaces": 1, "rera_approved": 0, "nearby_schools": 2, "nearby_hospitals": 1,
    },
    {
        "label": "Hyderabad, 3BHK, Emerging locality, new",
        "city": "Hyderabad", "locality_tier": "Emerging", "area_type": "Built-up Area",
        "furnishing": "Fully Furnished", "facing": "East", "total_sqft": 1600, "bhk": 3,
        "bath": 3, "balcony": 2, "age_years": 0, "distance_to_center_km": 20.0,
        "parking_spaces": 2, "rera_approved": 1, "nearby_schools": 4, "nearby_hospitals": 3,
    },
    {
        "label": "Chennai, 4BHK, Premium, far from center",
        "city": "Chennai", "locality_tier": "Premium", "area_type": "Super built-up Area",
        "furnishing": "Fully Furnished", "facing": "South", "total_sqft": 2200, "bhk": 4,
        "bath": 4, "balcony": 3, "age_years": 1, "distance_to_center_km": 18.0,
        "parking_spaces": 2, "rera_approved": 1, "nearby_schools": 1, "nearby_hospitals": 1,
    },
    {
        "label": "Pune, 1BHK, Emerging, old",
        "city": "Pune", "locality_tier": "Emerging", "area_type": "Plot Area",
        "furnishing": "Unfurnished", "facing": "South-East", "total_sqft": 550, "bhk": 1,
        "bath": 1, "balcony": 0, "age_years": 15, "distance_to_center_km": 25.0,
        "parking_spaces": 0, "rera_approved": 0, "nearby_schools": 1, "nearby_hospitals": 0,
    },
]


def main():
    advisor = PropertyInvestmentAdvisor()

    rows = []
    for sample in SAMPLES:
        label = sample.pop("label")
        result = advisor.full_assessment(sample, years=5)
        rows.append({
            "Property": label,
            "Predicted Price (Rs.)": f"{result.predicted_price:,.0f}",
            "5yr Projected (Rs.)": f"{result.projected_future_price:,.0f}",
            "ROI %": f"{result.roi_pct:.1f}",
            "Risk": result.risk_level,
            "Recommendation": result.recommendation,
        })
        sample["label"] = label  # restore for readability if reused

    summary = pd.DataFrame(rows)
    print(summary.to_string(index=False))
    summary.to_csv("models/batch_demo_results.csv", index=False)
    print("\nSaved -> models/batch_demo_results.csv")

    # --- Global feature importance chart ---
    with open("models/price_model.pkl", "rb") as f:
        pipeline = pickle.load(f)

    model = pipeline.named_steps["model"]
    preprocessor = pipeline.named_steps["preprocess"]
    feature_names = preprocessor.get_feature_names_out()
    importances = model.feature_importances_

    order = np.argsort(importances)[-15:]
    plt.figure(figsize=(8, 6))
    plt.barh(np.array(feature_names)[order], importances[order], color="#1f77b4")
    plt.xlabel("Feature Importance")
    plt.title("Top 15 Global Feature Importances - Price Prediction Model")
    plt.tight_layout()
    plt.savefig("models/feature_importance.png", dpi=150)
    print("Saved -> models/feature_importance.png")


if __name__ == "__main__":
    main()
