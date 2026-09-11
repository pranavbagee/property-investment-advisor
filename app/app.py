"""
app.py
------
Streamlit front-end for the Explainable AI Property Investment Advisor.

Run with:
    streamlit run app/app.py

(Run from the project root so the relative model/data paths resolve.)
"""

import sys
import os

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.append(os.path.join(PROJECT_ROOT, "src"))

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Property Investment Advisor", layout="wide")


def _bootstrap_if_needed():
    """
    First-run setup: on a fresh clone/deploy (e.g. Streamlit Community
    Cloud), data/ and models/ won't exist yet. Generate the dataset and
    train the model automatically so the app works with zero manual
    setup steps. Safe to call every time -- it skips work if the files
    already exist.
    """
    model_path = os.path.join(PROJECT_ROOT, "models", "price_model.pkl")
    properties_path = os.path.join(PROJECT_ROOT, "data", "properties.csv")

    if os.path.exists(model_path) and os.path.exists(properties_path):
        return

    with st.spinner("First-time setup: generating dataset and training the model "
                     "(only happens once)..."):
        import generate_dataset
        import train_model

        generate_dataset.save_all(PROJECT_ROOT)
        train_model.main(PROJECT_ROOT)


_bootstrap_if_needed()

from advisor import PropertyInvestmentAdvisor


@st.cache_resource
def load_advisor():
    return PropertyInvestmentAdvisor(
        model_path=os.path.join(PROJECT_ROOT, "models", "price_model.pkl"),
        feature_lists_path=os.path.join(PROJECT_ROOT, "models", "feature_lists.json"),
        appreciation_path=os.path.join(PROJECT_ROOT, "data", "city_appreciation.csv"),
        properties_path=os.path.join(PROJECT_ROOT, "data", "properties.csv"),
    )


advisor = load_advisor()

st.title("🏠 Explainable AI Property Investment Advisor")
st.caption(
    "Predicts property price, forecasts appreciation, estimates ROI, classifies "
    "investment risk, and explains every prediction with SHAP."
)

with st.sidebar:
    st.header("Property Details")

    cities = list(advisor.appreciation.index)
    city = st.selectbox("City", cities)
    locality_tier = st.selectbox("Locality Tier", ["Premium", "Mid", "Emerging"])
    area_type = st.selectbox(
        "Area Type",
        ["Super built-up Area", "Built-up Area", "Plot Area", "Carpet Area"],
    )
    furnishing = st.selectbox("Furnishing", ["Unfurnished", "Semi-Furnished", "Fully Furnished"])
    facing = st.selectbox("Facing", ["North", "South", "East", "West", "North-East", "South-East"])

    total_sqft = st.slider("Total Area (sqft)", 300, 5000, 1450, step=50)
    bhk = st.slider("BHK", 1, 5, 3)
    bath = st.slider("Bathrooms", 1, 6, 3)
    balcony = st.slider("Balconies", 0, 5, 2)
    age_years = st.slider("Property Age (years)", 0, 30, 3)
    distance_km = st.slider("Distance to City Center (km)", 0.0, 30.0, 9.5)
    parking = st.slider("Parking Spaces", 0, 3, 1)
    rera = st.checkbox("RERA Approved", value=True)
    schools = st.slider("Nearby Schools", 0, 6, 3)
    hospitals = st.slider("Nearby Hospitals", 0, 4, 2)

    years = st.slider("Forecast Horizon (years)", 1, 10, 5)

    submitted = st.button("Analyze Property", type="primary")

if submitted:
    features = {
        "city": city,
        "locality_tier": locality_tier,
        "area_type": area_type,
        "furnishing": furnishing,
        "facing": facing,
        "total_sqft": total_sqft,
        "bhk": bhk,
        "bath": bath,
        "balcony": balcony,
        "age_years": age_years,
        "distance_to_center_km": distance_km,
        "parking_spaces": parking,
        "rera_approved": int(rera),
        "nearby_schools": schools,
        "nearby_hospitals": hospitals,
    }

    result = advisor.full_assessment(features, years=years)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Predicted Price", f"₹{result.predicted_price:,.0f}")
    col2.metric(f"Projected Price ({years}y)", f"₹{result.projected_future_price:,.0f}")
    col3.metric("Estimated ROI", f"{result.roi_pct:.1f}%")
    col4.metric("Risk Level", result.risk_level)

    rec_color = {"Buy": "green", "Hold": "orange", "Avoid": "red"}[result.recommendation]
    st.markdown(
        f"### Recommendation: :{rec_color}[{result.recommendation}]\n"
        f"{result.recommendation_reason}"
    )
    st.caption(result.risk_reason)

    st.divider()
    st.subheader("Why this prediction? (Explainable AI)")
    st.caption(f"Method: {result.explanation_method}")

    exp_df = pd.DataFrame(
        [{"feature": k, "contribution": v} for k, v in result.explanation.items()]
    ).sort_values("contribution")

    st.bar_chart(exp_df.set_index("feature"))

    with st.expander("Raw feature contributions"):
        st.dataframe(exp_df.sort_values("contribution", ascending=False), use_container_width=True)

else:
    st.info("Set property details in the sidebar and click **Analyze Property** to get a prediction.")
