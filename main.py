from simulator import BioDigitalTwin
from risk_predictor import RiskPredictor, calculate_history_risk


def main():
    model_path = 'ml_model_cardio/cardio_model_nolabs.pkl'
    scaler_path = 'ml_model_cardio/cardio_scaler_nolabs.pkl'

    patient_static = {'age': 55, 'gender': 2, 'height': 175, 'weight': 85}
    predictor = RiskPredictor(model_path, scaler_path)

    subject = BioDigitalTwin(r=1.3, c=1.2, zc=0.07, hr=75, sv=70, active_conditions=['sigara'])

    subject.run_simulation(30, "sigara_icmek", "Smoking")
    subject.run_simulation(60, "sigarayi_birakmak", "Smoking Cessation")
    subject.run_simulation(90, "spor_yapmak", "Started Exercise")
    subject.run_simulation(10, "grip_olmak", "Flu")
    subject.run_simulation(30, "spor_yapmak", "Recovery + Exercise")

    print("\nAnalyzing simulation trajectory...")
    df_processed = calculate_history_risk(predictor, subject.history, patient_static)

    print("Simulation analysis complete.")
    print(df_processed[['Day', 'Scenario', 'SBP', 'DBP', 'Risk_Prob', 'Heart_Health_Score']].head())


if __name__ == "__main__":
    main()