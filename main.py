# main.py
from simulator import BioDigitalTwin
from risk_predictor import RiskPredictor, calculate_history_risk # calculate_history_risk eklendi
from visualizer import plot_simulation_dashboard

# ... (Diğer kodların aynı kalıyor) ...

def main():
    # 1. Ayarlar ve Hasta Tanımı
    MODEL_PATH = 'ml_model_cardio/cardio_model_nolabs.pkl' # HR destekli model önerilir
    SCALER_PATH = 'ml_model_cardio/cardio_scaler_nolabs.pkl'
    
    patient_static = {'age': 55, 'gender': 2, 'height': 175, 'weight': 85}
    predictor = RiskPredictor(MODEL_PATH, SCALER_PATH)
    
    # Başlangıç değerleri
    ahmet_bey = BioDigitalTwin(r=1.3, c=1.2, zc=0.07, hr=75, sv=70, active_conditions=['sigara'])

    # 2. Simülasyonu Çalıştır
    ahmet_bey.run_simulation(30, "sigara_icmek", "Sigara İçiyor")
    ahmet_bey.run_simulation(60, "sigarayi_birakmak", "Sigarayı Bıraktı")
    ahmet_bey.run_simulation(90, "spor_yapmak", "Spora Başladı")
    ahmet_bey.run_simulation(10, "grip_olmak", "Ağır Grip")
    ahmet_bey.run_simulation(30, "spor_yapmak", "İyileşme + Spor")

    # 3. Tüm Geçmiş İçin Risk Skorlarını Hesapla
    print("\n🧠 Yapay zeka tüm simülasyon geçmişini analiz ediyor...")
    # simulator.history listesini DataFrame'e çevirip sağlık skorlarını ekliyoruz
    df_processed = calculate_history_risk(predictor, ahmet_bey.history, patient_static)

    # 4. Dashboard'u Çiz
    print("📊 Dashboard oluşturuluyor...")
    plot_simulation_dashboard(df_processed)

if __name__ == "__main__":
    main()