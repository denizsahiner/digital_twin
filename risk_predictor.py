import pandas as pd
import joblib
import numpy as np
import os


class RiskPredictor:
    def __init__(self, model_path, scaler_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at '{model_path}'.")

        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)

    def predict(self, simulator_instance, patient_static_data):
        sys, dia = simulator_instance.get_current_bp()
        lifestyle = simulator_instance.current_lifestyle

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

        scaled_input = self.scaler.transform(input_data)
        prob = self.model.predict_proba(scaled_input)[0][1]

        return prob, sys, dia

    def calculate_history_risk(self, history_list, patient_static_data):
        if not history_list:
            return pd.DataFrame()

        df_history = pd.DataFrame(history_list)
        bmi = patient_static_data['weight'] / ((patient_static_data['height'] / 100) ** 2)

        input_df = pd.DataFrame({
            'age_years': patient_static_data['age'],
            'gender': patient_static_data['gender'],
            'height': patient_static_data['height'],
            'weight': patient_static_data['weight'],
            'ap_hi': df_history['SBP'],
            'ap_lo': df_history['DBP'],
            'smoke': df_history.get('Smoke_Flag', 0),
            'alco': patient_static_data.get('base_alco', 0),
            'gluc': patient_static_data.get('gluc', 0),
            'BMI': bmi,
            'PP': df_history['SBP'] - df_history['DBP']
        })

        scaled_inputs = self.scaler.transform(input_df)
        risk_probs = self.model.predict_proba(scaled_inputs)[:, 1]
        health_scores = 100.0 * (1.0 - risk_probs)

        df_history['Risk_Prob'] = risk_probs
        df_history['Heart_Health_Score'] = health_scores

        return df_history


def calculate_history_risk(predictor_instance, history_list, patient_static_data):
    return predictor_instance.calculate_history_risk(history_list, patient_static_data)