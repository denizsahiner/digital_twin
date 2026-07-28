import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.multioutput import MultiOutputRegressor
import joblib
import xgboost as xgb
import os

DATA_PATH = 'datasets/realistic_windkessel_dataset.csv'


def train_realistic_model():
    print("Training Windkessel XGBoost model...")
    if not os.path.exists(DATA_PATH):
        print(f"Dataset '{DATA_PATH}' not found. Run train_fast.py to generate data.")
        return

    df = pd.read_csv(DATA_PATH)

    df['Stiffness_Obs'] = df['PP_Obs'] / df['SV']
    df['R_Obs'] = df['MAP_Obs'] / ((df['SV'] * df['HR']) / 60.0)
    df['Shape_Index'] = (df['MAP_Obs'] - df['Dia_Obs']) / df['PP_Obs']

    cycle_time = 60.0 / df['HR']
    dia_duration = cycle_time - df['Sys_Duration']
    df['C_Physics'] = dia_duration / (df['R_Obs'] * np.log(df['Sys_Obs'] / df['Dia_Obs']) + 1e-6)

    features = ['HR', 'SV', 'Sys_Obs', 'Dia_Obs', 'MAP_Obs', 'PP_Obs',
                'Shape_Index', 'Stiffness_Obs', 'R_Obs', 'C_Physics']
    targets = ['R_True', 'C_True', 'Zc_True']

    X = df[features]
    y = df[targets]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = MultiOutputRegressor(xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.02,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=-1,
        random_state=42
    ))

    model.fit(X_train_scaled, y_train)

    print("\nModel evaluation metrics:")
    print("-" * 40)
    y_pred = model.predict(X_test_scaled)

    for i, col in enumerate(targets):
        r2 = r2_score(y_test.iloc[:, i], y_pred[:, i])
        mae = mean_absolute_error(y_test.iloc[:, i], y_pred[:, i])
        print(f"{col:<8} | R2: {r2:.4f} | MAE: {mae:.4f}")
    print("-" * 40)

    os.makedirs('ml_model_windkessel', exist_ok=True)
    joblib.dump(model, 'ml_model_windkessel/realistic_model.pkl')
    joblib.dump(scaler, 'ml_model_windkessel/realistic_scaler.pkl')

    joblib.dump(model, "realistic_model.pkl")
    joblib.dump(scaler, "realistic_scaler.pkl")
    print("Model saved successfully.")


if __name__ == "__main__":
    train_realistic_model()