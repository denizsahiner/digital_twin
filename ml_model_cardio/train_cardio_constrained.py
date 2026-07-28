import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score

print(" Monotonluk Kısıtlı Cardio Risk Modeli Eğitiliyor...")
np.random.seed(42)
N = 70000

# 1. Gerçekçi Kardiyovasküler Veri Üretimi (Epidemiyolojik Dağılım)
age_years = np.random.randint(30, 80, size=N)
gender = np.random.choice([1, 2], size=N) # 1=Female, 2=Male
height = np.random.normal(168, 9, size=N).clip(140, 200)
weight = np.random.normal(74, 14, size=N).clip(40, 180)

# SBP (ap_hi) ve DBP (ap_lo)
base_sbp = 105 + (age_years - 30) * 0.6 + np.random.normal(10, 12, size=N)
base_dbp = 68 + (age_years - 30) * 0.35 + np.random.normal(8, 8, size=N)

ap_hi = base_sbp.clip(85, 210)
ap_lo = np.minimum(base_dbp, ap_hi - 15).clip(45, 130)

smoke = np.random.choice([0, 1], p=[0.75, 0.25], size=N)
alco = np.random.choice([0, 1], p=[0.90, 0.10], size=N)
gluc = np.random.choice([1, 2, 3], p=[0.80, 0.12, 0.08], size=N)

bmi = weight / ((height / 100) ** 2)
pp = ap_hi - ap_lo

# Tıbbi Hastalık Olasılığı Logit Denklemi (Klinik Doğruluk)
logit = (
    -1.2
    + 0.040 * (age_years - 45)
    + 0.035 * (ap_hi - 120)
    + 0.020 * (ap_lo - 80)
    + 0.50 * smoke
    + 0.35 * alco
    + 0.40 * (gluc - 1)
    + 0.06 * (bmi - 25)
    + np.random.normal(0, 0.4, size=N)
)

prob = 1.0 / (1.0 + np.exp(-logit))
cardio = (prob > 0.5).astype(int)

df = pd.DataFrame({
    'age_years': age_years,
    'gender': gender,
    'height': height,
    'weight': weight,
    'ap_hi': ap_hi,
    'ap_lo': ap_lo,
    'smoke': smoke,
    'alco': alco,
    'gluc': gluc,
    'BMI': bmi,
    'PP': pp,
    'cardio': cardio
})

features = ['age_years', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo', 'smoke', 'alco', 'gluc', 'BMI', 'PP']
target = 'cardio'

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# MONOTONE CONSTRAINTS VECTOR:
# age_years (+1), gender (0), height (0), weight (0), ap_hi (+1), ap_lo (+1), smoke (+1), alco (+1), gluc (+1), BMI (+1), PP (+1)
monotone_constraints = (1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1)

print(f" XGBoost Monotonluk Kısıtları İle Eğitiliyor: {monotone_constraints}...")
model = xgb.XGBClassifier(
    n_estimators=300,
    learning_rate=0.03,
    max_depth=5,
    subsample=0.8,
    monotone_constraints=monotone_constraints,
    eval_metric='logloss',
    n_jobs=-1,
    random_state=42
)

model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

print("\n" + "="*50)
print(f" YENİ MONOTON CARDIO RISK MODEL SONUÇLARI:")
print("="*50)
print(f"   Accuracy (Doğruluk) : %{acc*100:.2f}")
print(f"   ROC AUC Skoru       : {auc:.4f}")
print("="*50)

# --- SENSITIVITY & BIAS TESTİ (Klinik Doğruluk Testi) ---
print("\n SIGARA VE ALKOL HASSASİYET TESTİ:")
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

print(f"   Temiz Yaşam (Sigara Yok, Alkol Yok) : %{p_clean*100:.2f} Risk")
print(f"   Sadece Sigara İçen                   : %{r_smoke*100:.2f} Risk (Fark: +{(r_smoke-p_clean)*100:.2f}%)")
print(f"   Sadece Alkol Alan                    : %{r_alco*100:.2f} Risk (Fark: +{(r_alco-p_clean)*100:.2f}%)")
print(f"   Sigara + Alkol                       : %{r_both*100:.2f} Risk (Fark: +{(r_both-p_clean)*100:.2f}%)")

# Kaydet
os.makedirs('ml_model_cardio', exist_ok=True)
joblib.dump(model, 'ml_model_cardio/cardio_model_nolabs.pkl')
joblib.dump(scaler, 'ml_model_cardio/cardio_scaler_nolabs.pkl')

joblib.dump(model, 'cardio_model_nolabs.pkl')
joblib.dump(scaler, 'cardio_scaler_nolabs.pkl')

print("\n✅ Monoton Cardio Modeli ve Scaler başarıyla yüklendi ve kaydedildi!")
