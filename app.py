"""
Patient Readmission Risk - Streamlit app.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Patient Readmission Risk", page_icon="🏥", layout="centered")


@st.cache_resource
def load_artifacts():
    model = joblib.load('readmission_model.joblib')
    preprocessor = joblib.load('preprocessor.joblib')
    metadata = joblib.load('feature_metadata.joblib')
    return model, preprocessor, metadata


model, preprocessor, metadata = load_artifacts()

st.title("Patient Readmission Risk Predictor")
st.write(
    "Estimates the likelihood that a diabetic patient will be readmitted to the "
    "hospital within 30 days of discharge, based on details from their current encounter. "
    "This is a portfolio demo trained on the UCI Diabetes 130-US Hospitals dataset, "
    "not a clinical tool."
)

st.divider()
st.subheader("Patient and encounter details")

col1, col2 = st.columns(2)

with col1:
    age = st.selectbox("Age range", metadata['category_values']['age'])
    race = st.selectbox("Race", metadata['category_values']['race'])
    gender = st.selectbox("Gender", metadata['category_values']['gender'])
    time_in_hospital = st.slider("Days in hospital", 1, 14, 4)
    num_lab_procedures = st.slider("Number of lab procedures", 0, 130, 40)
    num_procedures = st.slider("Number of procedures", 0, 6, 1)
    num_medications = st.slider("Number of medications", 1, 80, 15)

with col2:
    number_outpatient = st.slider("Prior outpatient visits (past year)", 0, 40, 0)
    number_emergency = st.slider("Prior emergency visits (past year)", 0, 40, 0)
    number_inpatient = st.slider("Prior inpatient visits (past year)", 0, 20, 0)
    number_diagnoses = st.slider("Number of diagnoses", 1, 16, 7)
    insulin = st.selectbox("Insulin status", metadata['category_values']['insulin'])
    diabetesMed = st.selectbox("On diabetes medication?", metadata['category_values']['diabetesMed'])
    change = st.selectbox("Medication changed this visit?", metadata['category_values']['change'])

st.divider()

if st.button("Predict readmission risk", type="primary"):
    # Build a single-row dataframe with sensible defaults for fields not
    # collected in this simplified form, then fill in what the user entered.
    row = {col: metadata['category_values'].get(col, [np.nan])[0] if col in metadata['categorical_cols'] else 0
           for col in metadata['all_cols']}

    row.update({
        'age': age,
        'race': race,
        'gender': gender,
        'time_in_hospital': time_in_hospital,
        'num_lab_procedures': num_lab_procedures,
        'num_procedures': num_procedures,
        'num_medications': num_medications,
        'number_outpatient': number_outpatient,
        'number_emergency': number_emergency,
        'number_inpatient': number_inpatient,
        'number_diagnoses': number_diagnoses,
        'insulin': insulin,
        'diabetesMed': diabetesMed,
        'change': change,
        'diag_1': 'Diabetes',
        'diag_2': 'Other',
        'diag_3': 'Other',
        'admission_type_id': 1,
        'discharge_disposition_id': 1,
        'admission_source_id': 7,
    })

    input_df = pd.DataFrame([row])[metadata['all_cols']]
    X_proc = preprocessor.transform(input_df).astype(np.float32)
    risk = float(model.predict_proba(X_proc)[:, 1][0])

    st.subheader("Result")
    st.metric("Estimated 30-day readmission risk", f"{risk:.1%}")

    if risk >= 0.5:
        st.warning(
            "This patient profile falls into a higher-risk band. In a real care "
            "setting, this would typically trigger a follow-up call or check-in "
            "within 48 hours of discharge."
        )
    else:
        st.success(
            "This patient profile falls into a lower-risk band based on the model. "
            "Standard discharge follow-up should still apply."
        )

    st.caption(
        "This number reflects patterns the model found in historical hospital data. "
        "It's a decision-support signal, not a diagnosis, and should never replace "
        "clinical judgment."
    )
