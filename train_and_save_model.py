"""
Trains the patient readmission risk model and saves everything the
Streamlit app needs: the trained MLPClassifier model, the fitted
preprocessor, and the list of input columns.

Run this once before launching app.py:
    python train_and_save_model.py
"""

import pandas as pd
import numpy as np
import joblib
import scipy.sparse as sp

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.neural_network import MLPClassifier
from scipy.sparse import csr_matrix
from sklearn.utils import resample
from sklearn.metrics import roc_auc_score

print("Loading dataset...")
df = pd.read_csv('diabetic_data.csv')

df['readmit_30'] = (df['readmitted'] == '<30').astype(int)

drop_cols = ['encounter_id', 'patient_nbr', 'weight', 'payer_code', 'medical_specialty', 'readmitted']
model_df = df.drop(columns=drop_cols)
model_df = model_df.replace('?', np.nan)
model_df = model_df.dropna(subset=['race'])


def bucket_diagnosis(code):
    if pd.isna(code):
        return 'Missing'
    code = str(code)
    if code.startswith('V') or code.startswith('E'):
        return 'Other'
    try:
        num = float(code)
    except ValueError:
        return 'Other'
    if 250 <= num < 251:
        return 'Diabetes'
    if 390 <= num <= 459 or num == 785:
        return 'Circulatory'
    if 460 <= num <= 519 or num == 786:
        return 'Respiratory'
    if 520 <= num <= 579 or num == 787:
        return 'Digestive'
    if 580 <= num <= 629 or num == 788:
        return 'Genitourinary'
    if 800 <= num <= 999:
        return 'Injury'
    if 710 <= num <= 739:
        return 'Musculoskeletal'
    if 140 <= num <= 239:
        return 'Neoplasms'
    return 'Other'


for col in ['diag_1', 'diag_2', 'diag_3']:
    model_df[col] = model_df[col].apply(bucket_diagnosis)

target = 'readmit_30'
X = model_df.drop(columns=[target])
y = model_df[target]

numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), numeric_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=True), categorical_cols)
])

X_train_proc = preprocessor.fit_transform(X_train).astype(np.float32)
X_test_proc = preprocessor.transform(X_test).astype(np.float32)

n_features = X_train_proc.shape[1]

model = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    activation='relu',
    alpha=1e-4,          # L2 regularization, plays a similar role to dropout
    solver='adam',
    early_stopping=True,
    validation_fraction=0.15,
    n_iter_no_change=5,
    max_iter=200,
    random_state=42
)

# MLPClassifier doesn't accept class_weight or sample_weight, so the minority
# class (readmitted within 30 days, ~11% of patients) is oversampled in the
# training set instead. The test set is left untouched.
y_train_arr = y_train.to_numpy()
X_train_proc = csr_matrix(X_train_proc)

X_majority = X_train_proc[y_train_arr == 0]
X_minority = X_train_proc[y_train_arr == 1]

X_minority_upsampled = resample(
    X_minority,
    replace=True,
    n_samples=X_majority.shape[0],
    random_state=42
)

X_train_balanced = sp.vstack([X_majority, X_minority_upsampled]).tocsr()
y_train_balanced = np.concatenate([
    np.zeros(X_majority.shape[0]),
    np.ones(X_minority_upsampled.shape[0])
])

rng = np.random.RandomState(42)
shuffle_idx = rng.permutation(len(y_train_balanced))
X_train_balanced = X_train_balanced[shuffle_idx]
y_train_balanced = y_train_balanced[shuffle_idx]

print(f"Balanced training set: {X_train_balanced.shape[0]:,} rows "
      f"({int(y_train_balanced.sum()):,} positive, "
      f"{int((y_train_balanced == 0).sum()):,} negative)")

print("Training model...")
model.fit(X_train_balanced, y_train_balanced)
print(f"Training stopped after {model.n_iter_} iterations "
      f"(best validation score: {max(model.validation_scores_):.3f})")

test_auc = roc_auc_score(y_test, model.predict_proba(X_test_proc)[:, 1])
print(f"Final test ROC-AUC: {test_auc:.3f}")

# Save everything the app needs
joblib.dump(model, 'readmission_model.joblib')
joblib.dump(preprocessor, 'preprocessor.joblib')
joblib.dump({
    'numeric_cols': numeric_cols,
    'categorical_cols': categorical_cols,
    'all_cols': X.columns.tolist(),
    'category_values': {c: sorted(X[c].dropna().unique().tolist()) for c in categorical_cols}
}, 'feature_metadata.joblib')

print("Saved: readmission_model.joblib, preprocessor.joblib, feature_metadata.joblib")
