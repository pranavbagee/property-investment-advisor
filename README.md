# Explainable AI-Based Property Investment Advisor

A working, end-to-end implementation of the system described in the project
proposal: property price prediction, future appreciation forecasting, ROI
estimation, investment risk classification, a Buy/Hold/Avoid recommendation,
and SHAP-based explainability — with a Streamlit dashboard on top.

## Project structure

```
property_advisor/
├── data/
│   ├── properties.csv            # property listings (synthetic, see note below)
│   └── city_appreciation.csv     # city-wise historical appreciation & volatility
├── models/
│   ├── price_model.pkl           # trained price prediction pipeline
│   ├── feature_lists.json        # feature schema used by the model
│   ├── metrics.json              # test-set MAE / MAPE / R²
│   ├── feature_importance.png    # global feature importance chart
│   └── batch_demo_results.csv    # sample outputs across 5 varied properties
├── src/
│   ├── generate_dataset.py       # synthetic dataset generator
│   ├── train_model.py            # trains & saves the price model
│   ├── advisor.py                # core logic: price, ROI, risk, recommendation, SHAP
│   └── batch_demo.py             # runs the pipeline on sample properties + saves chart
├── app/
│   └── app.py                    # Streamlit dashboard
└── requirements.txt
```

## About the dataset

**This sandbox has no internet access**, so the real Kaggle datasets (e.g.
Bengaluru House Price Data) couldn't be downloaded here. `generate_dataset.py`
instead creates a **synthetic but realistically-calibrated** dataset — 3,000
properties across 6 Indian cities, with the same kind of columns (city,
locality tier, sqft, BHK, bath, balcony, age, distance to center, furnishing,
etc.) and city-level appreciation/volatility numbers in the same ballpark as
real NHB RESIDEX figures. Every module (price model, ROI, risk, SHAP,
Streamlit app) runs correctly against it today.

**To switch to the real dataset once you have internet access:**
1. Download e.g. the Bengaluru House Price Data or the multi-city Indian
   House Prices Dataset (links given earlier in this conversation).
2. Rename/remap its columns to match `data/properties.csv`'s schema (see
   the column list in `train_model.py` — `NUMERIC_FEATURES` /
   `CATEGORICAL_FEATURES`), or edit those lists to match the real columns.
3. Re-run `python src/train_model.py` — nothing else changes.
4. For real appreciation data, replace `data/city_appreciation.csv` with
   rates you compute from NHB RESIDEX's quarterly index (CAGR per city).

## Running online (Streamlit Community Cloud)

The app is self-bootstrapping: `app/app.py` automatically generates the
dataset and trains the model on first run if they don't exist yet, so a
fresh deploy needs zero manual setup steps. To deploy:

1. Push this whole folder to a new **public GitHub repository**.
2. Go to https://share.streamlit.io, sign in with GitHub, click
   "New app".
3. Pick your repo/branch, set **Main file path** to `app/app.py`.
4. Click Deploy. First load takes ~1-2 minutes (generating data +
   training); after that it's instant.

Because Streamlit Cloud has real internet access, `pip install -r
requirements.txt` there will install the **real** `xgboost`, `catboost`,
and `shap` — no fallbacks, unlike this sandbox.

## How to run

```bash
pip install -r requirements.txt

cd property_advisor
python src/generate_dataset.py      # creates data/*.csv
python src/train_model.py           # trains model -> models/price_model.pkl
python src/advisor.py               # runs one sample assessment end-to-end
python src/batch_demo.py            # runs 5 varied properties + saves chart

streamlit run app/app.py            # interactive dashboard
```

## Notes on XGBoost / SHAP

This sandbox couldn't install `xgboost` or `shap` (no internet), so:
- `train_model.py` automatically falls back to scikit-learn's
  `GradientBoostingRegressor` if XGBoost isn't importable. On your machine,
  once you `pip install xgboost`, it will automatically use real XGBoost —
  no code changes needed.
- `advisor.py` falls back to a simple **occlusion-based** explainer (replace
  each feature with its population median/mode and measure the prediction
  shift) if `shap` isn't importable. It produces the same additive,
  per-feature format as SHAP. Once you `pip install shap`, it automatically
  switches to real `shap.TreeExplainer` — again, no code changes needed.

## Current model performance (on synthetic data)

See `models/metrics.json` after running `train_model.py`. On the synthetic
dataset it reaches R² ≈ 0.95 — expect this to change (likely down somewhat)
on real, noisier market data, which is normal and worth discussing in your
Results section.

## CatBoost comparison (new regression technique)

`src/train_catboost_model.py` trains **CatBoost** as a second regression
technique, for comparison against the Linear Regression / Random Forest /
XGBoost baselines in your literature review. CatBoost's real advantages
over XGBoost:
- **Ordered boosting** — reduces target leakage/overfitting vs. standard
  gradient boosting on small-to-medium tabular data.
- **Native categorical handling** — `city`, `locality_tier`, `furnishing`,
  etc. don't need one-hot encoding; CatBoost builds its own target-statistic
  encoding internally.

Run `python src/train_catboost_model.py` then `python src/compare_models.py`
to get `models/model_comparison.csv` and `models/model_comparison.png`.

**⚠️ Important — read before using these numbers in your report:**
This sandbox has no internet access, so neither `xgboost` nor `catboost`
could actually be installed here. Both training scripts silently fell back
to two *different-but-similar* scikit-learn gradient-boosting
implementations (`GradientBoostingRegressor` and
`HistGradientBoostingRegressor`), which is why the comparison you see in
`models/model_comparison.png` right now shows nearly identical results —
**that similarity is a sandbox artifact, not a real finding.** It does
NOT mean "CatBoost ≈ XGBoost" — it means "two sklearn fallbacks that are
almost the same algorithm perform almost the same." Once you're on your
own machine, run:
```bash
pip install xgboost catboost
python src/train_model.py
python src/train_catboost_model.py
python src/compare_models.py
```
and re-generate the chart with the *real* libraries before putting the
comparison in your Results and Discussion slide — the real gap between
XGBoost and CatBoost could go either way and is worth actually seeing.

## What's still a simplification

- **Risk classification** uses a transparent rule (locality-tier volatility +
  deviation from comparable properties) rather than a trained classifier.
  This is easy to swap for a supervised model once you have labeled
  historical risk/return data.
- **Buy/Hold/Avoid** uses threshold rules on ROI + risk. Documented clearly
  in `advisor.py -> recommend()` so you can justify or tune the thresholds
  in your report.
