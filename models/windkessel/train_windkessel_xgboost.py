import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import xgboost as xgb
import os
from pathlib import Path

# --- PATHS ---
HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
DATA_PATH = REPO_ROOT / 'data' / 'realistic_windkessel_dataset.csv'  # from data_pipeline/generators/generate_dataset_3.py
MODEL_PATH = HERE / 'realistic_model.pkl'
SCALER_PATH = HERE / 'realistic_scaler.pkl'

def train_realistic_model():
    print("🚀 Gerçekçi Model Eğitiliyor...")
    if not os.path.exists(DATA_PATH):
        print(f"❌ Veri bulunamadı: {DATA_PATH}\n"
              f"   Önce: python data_pipeline/generators/generate_dataset_3.py")
        return
    df = pd.read_csv(DATA_PATH)

    # --- FEATURE ENGINEERING (Gürültülü verilerle yapılıyor!) ---
    # Modelin elinde sadece gürültülü (Observed) veriler var.

    # 1. Stiffness Proxy (Hatalı PP ile hesaplanır)
    df['Stiffness_Obs'] = df['PP_Obs'] / df['SV']

    # 2. Resistance Proxy (Hatalı MAP ile hesaplanır)
    df['R_Obs'] = df['MAP_Obs'] / ((df['SV'] * df['HR']) / 60.0)

    # 3. Shape Index (Eğri şekli)
    df['Shape_Index'] = (df['MAP_Obs'] - df['Dia_Obs']) / df['PP_Obs']

    # 4. Fiziksel C Tahmini (Logaritmik Decay)
    # Gürültü olduğu için +1e-6 eklemek önemli
    cycle_time = 60.0 / df['HR']
    dia_duration = cycle_time - df['Sys_Duration']
    df['C_Physics'] = dia_duration / (df['R_Obs'] * np.log(df['Sys_Obs'] / df['Dia_Obs']) + 1e-6)

    # Girdiler (Hepsi Observed/Gürültülü)
    features = ['HR', 'SV', 'Sys_Obs', 'Dia_Obs', 'MAP_Obs', 'PP_Obs',
                'Shape_Index', 'Stiffness_Obs', 'R_Obs', 'C_Physics']

    # HEDEFLER (Gerçek Değerler)
    targets = ['R_True', 'C_True', 'Zc_True']

    X = df[features]
    y = df[targets]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("🧠 XGBoost (Robust Mode)...")
    from sklearn.multioutput import MultiOutputRegressor

    model = MultiOutputRegressor(xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.02,  # Yavaş öğrensin ki gürültüyü ezberlemesin
        max_depth=6,
        subsample=0.8,  # Her ağaçta verinin %80'ini görsün (Genelleme artar)
        colsample_bytree=0.8,
        n_jobs=-1,
        random_state=42
    ))

    model.fit(X_train_scaled, y_train)

    print("\n📊 SONUÇLAR (Realistic Noise):")
    y_pred = model.predict(X_test_scaled)

    for i, col in enumerate(targets):
        r2 = r2_score(y_test.iloc[:, i], y_pred[:, i])
        mae = mean_absolute_error(y_test.iloc[:, i], y_pred[:, i])

        # Beklenen: R2 0.85 - 0.95 arası
        print(f"  🔹 {col:<7} | R2 = {r2:.4f} | MAE = {mae:.4f}")

    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print(f"✅ Kaydedildi: {MODEL_PATH.name}, {SCALER_PATH.name}")


if __name__ == "__main__":
    train_realistic_model()