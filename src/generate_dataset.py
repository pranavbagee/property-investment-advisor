"""
generate_dataset.py
--------------------
Creates a synthetic but realistic Indian residential property dataset.

Why synthetic: this environment has no internet access, so the real
Kaggle datasets (Bengaluru House Price Data, etc.) can't be downloaded
here. This generator mimics the same schema and realistic value ranges,
so ALL downstream code (price model, ROI, risk, SHAP, Streamlit app)
runs end-to-end today. When you have internet access, drop in the real
Kaggle CSV (rename its columns to match, see README) and everything
else keeps working unchanged.

Output: data/properties.csv (property-level listings)
        data/city_appreciation.csv (city-wise historical annual
        appreciation rate, standing in for NHB RESIDEX data)
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)

# ---------------------------------------------------------------------
# 1. City-level base price/sqft and historical appreciation & volatility
#    (Roughly calibrated to real-world ballpark figures; replace with
#    actual NHB RESIDEX-derived numbers when available.)
# ---------------------------------------------------------------------
CITY_PROFILE = {
    "Mumbai":    {"base_psf": 18500, "appreciation": 0.061, "volatility": 0.09},
    "Bengaluru": {"base_psf": 7200,  "appreciation": 0.072, "volatility": 0.07},
    "Delhi":     {"base_psf": 11500, "appreciation": 0.048, "volatility": 0.10},
    "Pune":      {"base_psf": 6800,  "appreciation": 0.066, "volatility": 0.06},
    "Chennai":   {"base_psf": 6200,  "appreciation": 0.055, "volatility": 0.07},
    "Hyderabad": {"base_psf": 6600,  "appreciation": 0.081, "volatility": 0.08},
}

LOCALITY_TIERS = {
    # tier multiplier on base price/sqft, and a distance-to-center proxy (km)
    "Premium":  {"mult": (1.4, 1.9), "dist_km": (0, 6)},
    "Mid":      {"mult": (0.9, 1.3), "dist_km": (5, 15)},
    "Emerging": {"mult": (0.55, 0.85), "dist_km": (12, 28)},
}

AREA_TYPES = ["Super built-up Area", "Built-up Area", "Plot Area", "Carpet Area"]
FURNISHING = ["Unfurnished", "Semi-Furnished", "Fully Furnished"]
FACING = ["North", "South", "East", "West", "North-East", "South-East"]

N_ROWS = 3000


def sample_city():
    return RNG.choice(list(CITY_PROFILE.keys()))


def generate_properties(n=N_ROWS) -> pd.DataFrame:
    rows = []
    for i in range(n):
        city = sample_city()
        profile = CITY_PROFILE[city]

        tier = RNG.choice(list(LOCALITY_TIERS.keys()), p=[0.25, 0.45, 0.30])
        tier_info = LOCALITY_TIERS[tier]
        locality = f"{city}-{tier}-{RNG.integers(1, 9)}"

        bhk = int(RNG.choice([1, 2, 2, 3, 3, 3, 4, 4, 5], p=None) if False else
                  RNG.choice([1, 2, 3, 4, 5], p=[0.12, 0.35, 0.33, 0.15, 0.05]))
        sqft_per_bhk = RNG.normal(550, 80)
        total_sqft = max(280, bhk * sqft_per_bhk + RNG.normal(0, 60))

        bath = max(1, min(bhk + RNG.integers(-1, 2), bhk + 1))
        balcony = max(0, min(bhk, RNG.integers(0, bhk + 1)))

        age_years = max(0, RNG.integers(0, 25))
        dist_km = RNG.uniform(*tier_info["dist_km"])

        mult = RNG.uniform(*tier_info["mult"])
        psf = profile["base_psf"] * mult

        # depreciate slightly for age, premium for lower distance to center
        age_factor = max(0.6, 1 - 0.012 * age_years)
        dist_factor = max(0.7, 1 - 0.015 * dist_km)
        noise = RNG.normal(1.0, 0.06)

        price_per_sqft = psf * age_factor * dist_factor * noise
        price = price_per_sqft * total_sqft

        area_type = RNG.choice(AREA_TYPES)
        furnishing = RNG.choice(FURNISHING, p=[0.45, 0.35, 0.20])
        facing = RNG.choice(FACING)
        parking = int(RNG.integers(0, 3))
        rera_approved = int(RNG.random() < 0.78)
        nearby_schools = int(RNG.integers(0, 6))
        nearby_hospitals = int(RNG.integers(0, 4))

        rows.append({
            "property_id": f"P{i+1:05d}",
            "city": city,
            "locality": locality,
            "locality_tier": tier,
            "area_type": area_type,
            "total_sqft": round(total_sqft, 1),
            "bhk": bhk,
            "bath": bath,
            "balcony": balcony,
            "age_years": age_years,
            "distance_to_center_km": round(dist_km, 2),
            "furnishing": furnishing,
            "facing": facing,
            "parking_spaces": parking,
            "rera_approved": rera_approved,
            "nearby_schools": nearby_schools,
            "nearby_hospitals": nearby_hospitals,
            "price": round(price, -3),  # nearest thousand, INR
        })

    return pd.DataFrame(rows)


def generate_city_appreciation() -> pd.DataFrame:
    """City-wise historical annual appreciation rate & volatility.
    Stands in for what you'd derive from NHB RESIDEX quarterly index data.
    """
    rows = []
    for city, p in CITY_PROFILE.items():
        rows.append({
            "city": city,
            "avg_annual_appreciation": p["appreciation"],
            "price_volatility": p["volatility"],
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    props = generate_properties()
    appr = generate_city_appreciation()

    props.to_csv("data/properties.csv", index=False)
    appr.to_csv("data/city_appreciation.csv", index=False)

    print(f"Wrote data/properties.csv  -> {props.shape[0]} rows, {props.shape[1]} cols")
    print(f"Wrote data/city_appreciation.csv -> {appr.shape[0]} rows")
    print("\nSample rows:")
    print(props.head(3).to_string())
