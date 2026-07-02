# Patient Readmission Risk Prediction

## What this project does

This project predicts whether a diabetic patient is likely to be readmitted to the hospital within 30 days of being discharged. 

The data comes from a real source: 10 years of diabetic patient encounters across 130 US hospitals, just over 100,000 records.

## Why it matters

A 30-day readmission usually means something didn't go right the first time around. It's hard on the patient, costly for the hospital, and in the US it can trigger Medicare penalties for the hospital too. If a care team can flag high-risk patients right at discharge, they can step in early with a follow-up call or check-in instead of waiting to see what happens.

## Approach

1. **Exploratory data analysis** - looked at how common 30-day readmissions actually are (around 11%, so this is an imbalanced problem) and which factors seem to matter, mainly length of stay and number of prior inpatient visits.
2. **Diagnosis code grouping** - the raw diagnosis columns held over 700 distinct medical codes each. Encoding all of those directly would have created a massive, mostly-empty feature set, so they were grouped into broad clinical categories instead (circulatory, respiratory, diabetes, and so on), which is the standard approach used in published research on this dataset.
3. **Class weighting** - since only 11% of patients get readmitted within 30 days, the network was trained with class weights so it doesn't just learn to predict "not readmitted" every time.
4. **Evaluation** - measured ROC-AUC and per-class precision and recall, since accuracy alone would be misleading on data this imbalanced.

## Results

The model reaches a test ROC-AUC of about 0.56, with recall on the readmitted class around 0.47. That's a modest result, and worth being upfront about: predicting readmission from administrative hospital data is a genuinely hard problem, and these numbers are in line with what's published elsewhere on this exact dataset. This kind of model works best as one signal feeding into a care team's decision, not as a standalone verdict.

## Files in this project

- `patient_readmission_risk.ipynb` - the full notebook, from raw data to trained model, with the neural network concepts explained in plain language before the code
- `train_and_save_model.py` - retrains the model from scratch and saves the files the app needs
- `app.py` - the Streamlit app --- https://patient-readmission-pr.streamlit.app/
- `requirements.txt` - dependencies for deployment
- `README.md` - this file


## What this demonstrates

Deep learning fundamentals: building a feedforward neural network from scratch, scaling and encoding data correctly for it, reading loss and AUC curves to catch overfitting, and handling class imbalance with class weights instead of just resampling.
