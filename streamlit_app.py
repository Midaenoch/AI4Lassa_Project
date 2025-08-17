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
st.markdown("Upload your data file (CSV or Excel) or manually input data to predict potential Lassa Fever outbreaks.")

# Define expected features
selected_features = [
    'Cases', 'Any_Confirmed_Cases', 'Reports_All', 'LGA_Mean_Cases',
    'Cases_SuspectedUnconfirmed', 'Year', 'NumDiagCentres', 'LabDist',
    'Source', 'TotalPopulation_ByYear', 'LabTravelTime', 'AgriProp_ESA',
    'ForestProp_ESA', 'UrbanProp_ESA', 'TempMeanAnnual_201119_NOAA',
    'CHELSA_PrecipTotalAnnual', 'TotalRuralPop2015', 'CHELSA_TempAnnualMean',
    'CHELSA_PrecipWettestQ'
]

# 1. File upload
uploaded_file = st.file_uploader("📤 Upload CSV or Excel file", type=["csv", "xlsx"])

# 2. Manual fallback
manual_input = {}

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv'):
            data = pd.read_csv(uploaded_file)
        else:
            data = pd.read_excel(uploaded_file)

        # Validate columns
        missing_cols = [col for col in selected_features if col not in data.columns]
        if missing_cols:
            st.error(f"Missing columns in uploaded file: {', '.join(missing_cols)}")
            st.stop()

        st.success("✅ File uploaded and verified successfully!")
        st.dataframe(data[selected_features].head())

    except Exception as e:
        st.error(f"Error reading file: {e}")
        st.stop()

else:
    st.info("Or manually enter input data below 👇")
    with st.expander("🔢 Manual Input Features"):
        for feature in selected_features:
            manual_input[feature] = st.number_input(f"{feature}", format="%.2f")

# 3. Prediction button
if st.button("Predict"):
    try:
        if uploaded_file is not None:
            input_data = data[selected_features]
        else:
            input_data = pd.DataFrame([manual_input])

        scaled_input = scaler.transform(input_data)
        predictions = model.predict(scaled_input)

        for i, pred in enumerate(predictions):
            st.markdown(f"### 📍 Prediction for Row {i+1}:")
            if pred == 1:
                st.markdown(
                    """
                    #### 🦠 **Outbreak Detected**
                    - ⚠️ A potential **Lassa Fever outbreak** is likely.
                    - 🏥 Please inform relevant health bodies.
                    """
                )
            else:
                st.markdown(
                    """
                    #### ✅ **No Outbreak**
                    - 👍 No indication of an outbreak.
                    - 🧼 Maintain hygiene and monitoring.
                    """
                )

    except Exception as e:
        st.error(f"Prediction failed: {e}")


