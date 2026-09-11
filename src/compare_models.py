"""
compare_models.py
------------------
Loads metrics from both trained models and produces a side-by-side
comparison table + bar chart -- ready to drop into a Results and
Discussion slide.
"""

import json

import matplotlib.pyplot as plt
import pandas as pd

with open("models/metrics.json") as f:
    m1 = json.load(f)

with open("models/catboost_metrics.json") as f:
    m2 = json.load(f)

rows = [
    {"Model": m1["model_name"], "MAE (Rs.)": m1["mae_inr"], "MAPE (%)": m1["mape_pct"], "R2 Score": m1["r2_score"]},
    {"Model": m2["model_name"], "MAE (Rs.)": m2["mae_inr"], "MAPE (%)": m2["mape_pct"], "R2 Score": m2["r2_score"]},
]
comparison = pd.DataFrame(rows)
print(comparison.to_string(index=False))
comparison.to_csv("models/model_comparison.csv", index=False)
print("\nSaved -> models/model_comparison.csv")

short_labels = ["XGBoost" if "XGBoost" in m1["model_name"] else "GradientBoosting",
                "CatBoost" if "CatBoost" in m2["model_name"] else "HistGradientBoosting"]

fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))

axes[0].bar(short_labels, comparison["R2 Score"], color=["#1f77b4", "#ff7f0e"])
axes[0].set_title("R² Score (higher is better)")
axes[0].set_ylim(0, 1)

axes[1].bar(short_labels, comparison["MAPE (%)"], color=["#1f77b4", "#ff7f0e"])
axes[1].set_title("MAPE % (lower is better)")

plt.tight_layout()
plt.savefig("models/model_comparison.png", dpi=150)
print("Saved -> models/model_comparison.png")
