import streamlit as st
import pandas as pd
import joblib

# Load model and pipeline
MODEL_FILE = "model.pkl"
PIPELINE_FILE = "pipeline.pkl"

model = joblib.load(MODEL_FILE)
pipeline = joblib.load(PIPELINE_FILE)

st.set_page_config(
    page_title="California House Price Predictor",
    page_icon="🏠",
    layout="centered"
)

st.title("🏠 California House Price Prediction")
st.write("Predict **Median House Value** using Machine Learning")

st.markdown("---")

# Sidebar inputs
st.sidebar.header("Input Features")

longitude = st.sidebar.number_input("Longitude", value=-122.23)
latitude = st.sidebar.number_input("Latitude", value=37.88)
housing_median_age = st.sidebar.number_input("Housing Median Age", min_value=0, value=41)
total_rooms = st.sidebar.number_input("Total Rooms", min_value=1, value=880)
total_bedrooms = st.sidebar.number_input("Total Bedrooms", min_value=1, value=129)
population = st.sidebar.number_input("Population", min_value=1, value=322)
households = st.sidebar.number_input("Households", min_value=1, value=126)
median_income = st.sidebar.number_input("Median Income", min_value=0.0, value=8.3252)
ocean_proximity = st.sidebar.selectbox(
    "Ocean Proximity",
    ["<1H OCEAN", "INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN"]
)

# Create input dataframe
input_data = pd.DataFrame({
    "longitude": [longitude],
    "latitude": [latitude],
    "housing_median_age": [housing_median_age],
    "total_rooms": [total_rooms],
    "total_bedrooms": [total_bedrooms],
    "population": [population],
    "households": [households],
    "median_income": [median_income],
    "ocean_proximity": [ocean_proximity]
})

st.subheader("📄 Input Data")
st.dataframe(input_data)

# Prediction button
if st.button("🔮 Predict House Price"):
    transformed_data = pipeline.transform(input_data)
    prediction = model.predict(transformed_data)

    st.success(
        f"🏡 **Estimated Median House Value:** ${prediction[0]:,.2f}"
    )

st.markdown("---")
st.caption("Built with ❤️ using Streamlit & Scikit-Learn")
