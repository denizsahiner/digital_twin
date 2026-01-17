# risk_predictor.py
import pandas as pd
import joblib
import numpy as np
import os


class RiskPredictor:
    def __init__(self, model_path, scaler_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError("Model dosyası yok! Önce eğitimi tamamla.")

        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)

    def predict(self, simulator_instance, patient_static_data):
        """
        simulator_instance: BioDigitalTwin objesi (R, C, BP buradan gelir)
        patient_static_data: Dict (Age, Gender, Height, Weight buradan gelir)
        """

        # 1. Simülatörden Dinamik Verileri Al
        sys, dia = simulator_instance.get_current_bp()
        lifestyle = simulator_instance.current_lifestyle  # {'smoke': 1, ...}

        # 2. DataFrame Oluştur (Modelin beklediği sütun sırasıyla)
        # Not: Modelinde 'active', 'cholesterol' vs. çıkarıp çıkardığına göre burayı güncelle.
        # Biz 'cardio_model_nolabs.pkl' (No Labs) kullandığını varsayıyoruz.

        input_data = pd.DataFrame([{
            'age_years': patient_static_data['age'],
            'gender': patient_static_data['gender'],
            'height': patient_static_data['height'],
            'weight': patient_static_data['weight'],

            'ap_hi': sys,
            'ap_lo': dia,

            'smoke': lifestyle.get('smoke', 0),
            'alco': lifestyle.get('alco', 0),
            'gluc': lifestyle.get('gluc', 0),

            'BMI': patient_static_data['weight'] / ((patient_static_data['height'] / 100) ** 2),
            'PP': sys - dia
        }])

        # 3. Tahmin
        scaled_input = self.scaler.transform(input_data)
        prob = self.model.predict_proba(scaled_input)[0][1]  # Hasta olma ihtimali

        return prob, sys, dia


def calculate_history_risk(self, history_list, patient_static_data):
    """
    TÜM simülasyon geçmişini alır ve her gün için 'Health Score' ekler.
    """
    if not history_list:
        return pd.DataFrame()

    df_history = pd.DataFrame(history_list)

    # --- ML Modeli İçin Girdi Matrisi Hazırla ---
    # History'deki sütun adlarını Modelin beklediği adlara çevirmeliyiz
    # Model Bekler: age_years, gender, height, weight, ap_hi, ap_lo, smoke, alco, BMI, PP

    # Statik verileri (Yaş, Boy, Kilo) tüm satırlara yay
    bmi = patient_static_data['weight'] / ((patient_static_data['height'] / 100) ** 2)

    # Girdi DataFrame'i
    input_df = pd.DataFrame({
        'age_years': patient_static_data['age'],
        'gender': patient_static_data['gender'],
        'height': patient_static_data['height'],
        'weight': patient_static_data['weight'],

        'ap_hi': df_history['SBP'],  # History'den gelir
        'ap_lo': df_history['DBP'],  # History'den gelir

        # Smoke flag history'de kayıtlıydı (Simülatör kaydediyordu)
        # Eğer history'de 'Smoke_Flag' yoksa 0 kabul et
        'smoke': df_history.get('Smoke_Flag', 0),

        # Alkol simülasyonda yoksa 0 kabul et (veya statik veriden al)
        'alco': patient_static_data.get('base_alco', 0),
        'gluc': patient_static_data.get('gluc', 0),

        'BMI': bmi,
        'PP': df_history['SBP'] - df_history['DBP']
    })

    # --- Toplu Tahmin ---
    scaled_inputs = self.scaler.transform(input_df)

    # [:, 1] -> Hasta olma olasılığı (Risk)
    risk_probs = self.model.predict_proba(scaled_inputs)[:, 1]

    # Kalp Sağlığı Skoru (100 üzerinden)
    # Risk %80 ise, Sağlık %20'dir.
    health_scores = 100 * (1.0 - risk_probs)

    # Orijinal DataFrame'e ekle
    df_history['Risk_Prob'] = risk_probs
    df_history['Heart_Health_Score'] = health_scores

    return df_history