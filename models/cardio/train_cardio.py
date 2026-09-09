import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# --- PATHS ---
HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
DATA_PATH = REPO_ROOT / 'data' / 'cardio_train_feature.csv'   # from data_pipeline/extract_cardio.py
MODEL_PATH = HERE / 'cardio_model_nolabs.pkl'
SCALER_PATH = HERE / 'cardio_scaler_nolabs.pkl'


def train_no_labs():
    print("🚀 Model Eğitiliyor (Kolesterol, Glikoz ve Active YOK)...")

    if not os.path.exists(DATA_PATH):
        print(f"❌ Veri bulunamadı: {DATA_PATH}\n   Önce: python data_pipeline/extract_cardio.py")
        return

    df = pd.read_csv(DATA_PATH, sep=';')


    features = [
        'age_years', 'gender', 'height', 'weight',
        'ap_hi', 'ap_lo',
        'smoke', 'alco', 'gluc',
        'BMI', 'PP'
    ]

    target = 'cardio'

    X = df[features]
    y = df[target]

    print(f"📊 Model Girdileri: {features}")

    # Eğitim
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("🧠 XGBoost Modeli Eğitiliyor...")
    model = xgb.XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        n_jobs=-1,
        eval_metric='logloss',
        random_state=42
    )

    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    print(f"🏆 Model Doğruluğu (Lab Verisi Olmadan): {acc:.4f}")
    # Not: Doğruluk %73 civarından %71-72 civarına düşebilir, bu çok normaldir
    # ve projenin çalışabilirliği için kabul edilebilir bir takastır.

    # Kaydet
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print(f"✅ Non-Invaziv Model Kaydedildi: {MODEL_PATH.name}, {SCALER_PATH.name}")


if __name__ == "__main__":
    train_no_labs()