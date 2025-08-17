Got it! You're using **Streamlit** to build a **Lassa Fever prediction app** that currently accepts **manual inputs** for multiple features.

Now you want to **replace the manual number inputs** with an option for **users to upload a CSV or Excel file**, and then have the app:

1. Read the file,
2. Extract the features,
3. Run the prediction on the data,
4. Show the results.

---

### ✅ Here's How You Can Add File Upload to Your Streamlit App

I’ll give you a **modified version** of your code that includes:

* A file uploader (`st.file_uploader`)
* Logic to handle uploaded data
* Optional fallback to manual entry if no file is uploaded

---

### 🆕 Updated Streamlit Code (with file upload support)

```python
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
```

---

### ✅ Features of This Update:

* Accepts **file upload** (CSV or Excel)
* Extracts relevant columns
* Validates missing fields
* Still supports **manual entry** as a backup
* Shows prediction results row-by-row if multiple entries are uploaded

---

Would you like:

* A **template CSV file** for users to download?
* To **display probabilities** or prediction confidence?
* Help **deploying this version** (e.g., on Streamlit Cloud)?

Let me know and I’ll assist!
