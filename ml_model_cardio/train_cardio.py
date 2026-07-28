import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

DATA_PATH = 'datasets/cardio_train.csv'


def train_no_labs():
    print("Training Cardio Risk model...")

    if not os.path.exists(DATA_PATH):
        print(f"Dataset file '{DATA_PATH}' not found.")
        return

    df = pd.read_csv(DATA_PATH, sep=';')
    df['age_years'] = (df['age'] / 365.25).astype(int)
    df['BMI'] = df['weight'] / ((df['height'] / 100) ** 2)
    df['PP'] = df['ap_hi'] - df['ap_lo']

    features = [
        'age_years', 'gender', 'height', 'weight',
        'ap_hi', 'ap_lo',
        'smoke', 'alco', 'gluc',
        'BMI', 'PP'
    ]

    target = 'cardio'

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

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
    print(f"Cardio Risk model accuracy: {acc:.4f}")

    os.makedirs('ml_model_cardio', exist_ok=True)
    joblib.dump(model, 'ml_model_cardio/cardio_model_nolabs.pkl')
    joblib.dump(scaler, 'ml_model_cardio/cardio_scaler_nolabs.pkl')

    joblib.dump(model, 'cardio_model_nolabs.pkl')
    joblib.dump(scaler, 'cardio_scaler_nolabs.pkl')
    print("Model saved successfully.")


if __name__ == "__main__":
    train_no_labs()