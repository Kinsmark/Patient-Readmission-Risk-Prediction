# Patient Readmission Risk Prediction

## What this project does

This project predicts whether a diabetic patient is likely to be readmitted to the hospital within 30 days of being discharged. It's the first project in the portfolio built with a neural network instead of a tree-based model like XGBoost or random forest, and it's also the first one outside finance, moving into healthcare.

The data comes from a real source: 10 years of diabetic patient encounters across 130 US hospitals, just over 100,000 records.

## Why it matters

A 30-day readmission usually means something didn't go right the first time around. It's hard on the patient, costly for the hospital, and in the US it can trigger Medicare penalties for the hospital too. If a care team can flag high-risk patients right at discharge, they can step in early with a follow-up call or check-in instead of waiting to see what happens.

## Approach

1. **Exploratory data analysis** - looked at how common 30-day readmissions actually are (around 11%, so this is an imbalanced problem) and which factors seem to matter, mainly length of stay and number of prior inpatient visits.
2. **Diagnosis code grouping** - the raw diagnosis columns held over 700 distinct medical codes each. Encoding all of those directly would have created a massive, mostly-empty feature set, so they were grouped into broad clinical categories instead (circulatory, respiratory, diabetes, and so on), which is the standard approach used in published research on this dataset.
3. **Neural network** - built a small feedforward network with two hidden layers and dropout, using Keras. Numeric features were scaled and categorical features one-hot encoded before going in.
4. **Class weighting** - since only 11% of patients get readmitted within 30 days, the network was trained with class weights so it doesn't just learn to predict "not readmitted" every time.
5. **Evaluation** - measured ROC-AUC and per-class precision and recall, since accuracy alone would be misleading on data this imbalanced.

## Results

The model reaches a test ROC-AUC of about 0.65, with recall on the readmitted class around 0.54. That's a modest result, and worth being upfront about: predicting readmission from administrative hospital data is a genuinely hard problem, and these numbers are in line with what's published elsewhere on this exact dataset. This kind of model works best as one signal feeding into a care team's decision, not as a standalone verdict.

## Files in this project

- `patient_readmission_risk.ipynb` - the full notebook, from raw data to trained model, with the neural network concepts explained in plain language before the code
- `train_and_save_model.py` - retrains the model from scratch and saves the files the app needs
- `app.py` - the Streamlit app
- `requirements.txt` - dependencies for deployment
- `README.md` - this file

## How to run it

**The notebook:**
1. Open `patient_readmission_risk.ipynb` in Jupyter.
2. Run all cells top to bottom. It downloads the dataset directly from a public source, so no manual download is needed.

**The deployed app:**
1. Install dependencies: `pip install -r requirements.txt`
2. Train and save the model: `python train_and_save_model.py` (takes a few minutes on a laptop)
3. Launch the app: `streamlit run app.py`
4. Enter patient details in the form and get a readmission risk estimate.

## What this demonstrates

Deep learning fundamentals from the ground up: building a neural network layer by layer, scaling and encoding tabular data correctly for it, handling a high-cardinality categorical feature sensibly instead of blindly one-hot encoding it, and using class weights to deal with imbalance. It's also the first project in the portfolio outside finance, which sets up the cross-sector pattern for the projects that follow.
