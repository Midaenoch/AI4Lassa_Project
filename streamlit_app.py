import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load model and scaler
with open("models/svm_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

st.title("🧪 AI4Lassa Fever Outbreak Prediction App")
st.markdown("Enter epidemiological and environmental data to predict potential Lassa Fever outbreaks.")

selected_features = [
    'Cases', 'Any_Confirmed_Cases', 'Reports_All', 'LGA_Mean_Cases',
    'Cases_SuspectedUnconfirmed', 'Year', 'NumDiagCentres', 'LabDist',
    'Source', 'TotalPopulation_ByYear', 'LabTravelTime', 'AgriProp_ESA',
    'ForestProp_ESA', 'UrbanProp_ESA', 'TempMeanAnnual_201119_NOAA',
    'CHELSA_PrecipTotalAnnual', 'TotalRuralPop2015', 'CHELSA_TempAnnualMean',
    'CHELSA_PrecipWettestQ'
]

# Collect inputs
user_input = {}
with st.expander("🔢 Input Features"):
    for feature in selected_features:
        user_input[feature] = st.number_input(f"{feature}", format="%.6f")
# Predict
if st.button("Predict"):
    try:
        input_df = pd.DataFrame([user_input])
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)

        if prediction[0] == 1:
            st.markdown(
                """
                ### 🦠 **Prediction: Outbreak Detected**
                - ⚠️ A potential **Lassa Fever outbreak** is likely in this region based on current data.
                - 📊 Please alert relevant health authorities.
                - 🏥 Early action can help reduce spread and fatalities.
                """
            )
        else:
            st.markdown(
                """
                ### ✅ **Prediction: No Outbreak**
                - 👍 There is currently **no indication of a Lassa Fever outbreak** in this region.
                - 🧼 Continue public health monitoring and hygiene practices.
                - 🔁 Keep updating the model with new data for accuracy.
                """
            )

    except Exception as e:
        st.error(f"Prediction failed: {e}")



