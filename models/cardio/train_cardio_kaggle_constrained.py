import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_curve, auc

# --- PATHS ---
HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
DATA_PATH = REPO_ROOT / 'data' / 'cardio_train.csv'   # raw Kaggle "Cardiovascular Disease" dataset
MODEL_PATH = HERE / 'cardio_model_nolabs.pkl'
SCALER_PATH = HERE / 'cardio_scaler_nolabs.pkl'

print("Loading Kaggle cardiovascular disease dataset...")

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"Dataset file '{DATA_PATH}' not found. "
        f"Download the Kaggle 'Cardiovascular Disease' dataset to data/cardio_train.csv"
    )

df = pd.read_csv(DATA_PATH, sep=';')
print(f"Raw dataset shape: {df.shape}")

df = df[(df['ap_hi'] >= 70) & (df['ap_hi'] <= 220)]
df = df[(df['ap_lo'] >= 40) & (df['ap_lo'] <= 150)]
df = df[(df['ap_hi'] > df['ap_lo'])]
df = df[(df['height'] >= 140) & (df['height'] <= 200)]
df = df[(df['weight'] >= 40) & (df['weight'] <= 200)]

df['age_years'] = (df['age'] / 365.25).astype(int)
df['BMI'] = df['weight'] / ((df['height'] / 100) ** 2)
df['PP'] = df['ap_hi'] - df['ap_lo']

print(f"Cleaned dataset shape: {df.shape}")

features = ['age_years', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo', 'smoke', 'alco', 'gluc', 'BMI', 'PP']
target = 'cardio'

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

monotone_constraints = (1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1)

print("Training XGBoost classifier with monotonic constraints...")
model = xgb.XGBClassifier(
    n_estimators=300,
    learning_rate=0.03,
    max_depth=6,
    subsample=0.85,
    colsample_bytree=0.85,
    monotone_constraints=monotone_constraints,
    eval_metric='logloss',
    n_jobs=-1,
    random_state=42
)

model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

acc = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

precision, recall, _ = precision_recall_curve(y_test, y_prob)
pr_auc = auc(recall, precision)

print("\nModel Evaluation Metrics:")
print("-" * 40)
print(f"Accuracy : {acc*100:.2f}%")
print(f"ROC-AUC  : {roc_auc:.4f}")
print(f"PR-AUC   : {pr_auc:.4f}")
print("-" * 40)

print("\nLifestyle monotonicity sensitivity test:")
base_patient = pd.DataFrame([{
    'age_years': 45, 'gender': 2, 'height': 175, 'weight': 80,
    'ap_hi': 120, 'ap_lo': 80, 'smoke': 0, 'alco': 0, 'gluc': 1,
    'BMI': 80 / (1.75**2), 'PP': 40
}])

p_clean = model.predict_proba(scaler.transform(base_patient))[0][1]

p_smoke = base_patient.copy(); p_smoke['smoke'] = 1
r_smoke = model.predict_proba(scaler.transform(p_smoke))[0][1]

p_alco = base_patient.copy(); p_alco['alco'] = 1
r_alco = model.predict_proba(scaler.transform(p_alco))[0][1]

p_both = base_patient.copy(); p_both['smoke'] = 1; p_both['alco'] = 1
r_both = model.predict_proba(scaler.transform(p_both))[0][1]

print(f"Clean lifestyle risk      : {p_clean*100:.2f}%")
print(f"Smoking risk              : {r_smoke*100:.2f}% (Diff: +{(r_smoke-p_clean)*100:.2f}%)")
print(f"Alcohol risk              : {r_alco*100:.2f}% (Diff: +{(r_alco-p_clean)*100:.2f}%)")
print(f"Smoking + Alcohol risk    : {r_both*100:.2f}% (Diff: +{(r_both-p_clean)*100:.2f}%)")

joblib.dump(model, MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)

print(f"\nCardio risk model and scaler saved: {MODEL_PATH.name}, {SCALER_PATH.name}")
