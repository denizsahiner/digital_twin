import pandas as pd
import numpy as np
import joblib
import os

# --- AYARLAR ---
MIMIC_PATH = "../datasets/mimic_all_filtered.csv"
OUTPUT_PATH = "../datasets/mimic_hemodynamic_predictions_filtered.csv"
MODEL_PATH = 'realistic_model.pkl'
SCALER_PATH = 'realistic_scaler.pkl'


def predict_valid_patients():
    print("🚀 MIMIC Hemodinamik Analizi Başlıyor...")

    # 1. Veri Setini Yükle
    if not os.path.exists(MIMIC_PATH):
        print("❌ HATA: MIMIC veri dosyası bulunamadı.")
        return

    df = pd.read_csv(MIMIC_PATH)
    total_count = len(df)
    print(f"📂 Toplam Hasta Kaydı: {total_count}")

    # 2. FİLTRELEME: Sadece Geçerli SV'ye Sahip Olanlar
    # Kriter:
    # a) SV_raw (Doğrudan ölçüm) dolu olsun VEYA
    # b) CO (Kardiyak Çıktı) dolu olsun (Çünkü SV = CO/HR hesaplanır) VEYA
    # c) Eğer bunlar yoksa, SV değeri 'fillna(70)' ile atanan 70.0 olmasın.

    # Not: Senin extraction kodunda SV_raw ve CO sütunları final_df içinde var.
    # Güvenli filtreleme:
    valid_mask = ((df['SV'].notna()) & (df['CO'].notna()) & (df['SV'] != 70.0))

    # Ayrıca temel vitalleri (HR, BP) eksik olanları da atalım
    vital_mask = df[['HR', 'Systolic_BP', 'Diastolic_BP']].notna().all(axis=1)

    df_clean = df[valid_mask & vital_mask].copy()

    filtered_count = len(df_clean)
    print(f"✅ Geçerli SV ve Vital Verisi Olan Hasta Sayısı: {filtered_count}")
    print(f"🗑️  Elenen (Default SV veya Eksik Veri): {total_count - filtered_count}")

    if filtered_count == 0:
        print("⚠️ Hiç geçerli hasta kalmadı! İşlem durduruluyor.")
        return

    # 3. Model ve Scaler Yükle
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
    except:
        print("❌ Model dosyaları eksik! Önce 'train_realistic.py' çalıştır.")
        return

    print("🧠 Dijital İkiz Parametreleri (R, C, Zc) Hesaplanıyor...")

    # =========================================================================
    # 4. FEATURE ENGINEERING (Modelin Beklediği Formata Çevirme)
    # =========================================================================
    # Modelin Giriş Sırası:
    # ['HR', 'SV', 'Sys_Obs', 'Dia_Obs', 'MAP_Obs', 'PP_Obs', 'Shape_Index', 'Stiffness_Obs', 'R_Obs', 'C_Physics']

    # Değişkenleri hazırla
    hr = df_clean['HR']
    sv = df_clean['SV']
    sys_obs = df_clean['Systolic_BP']
    dia_obs = df_clean['Diastolic_BP']
    map_obs = df_clean['MAP']
    pp_obs = df_clean['PP']

    # Systole Duration (Extraction kodunda hesaplamıştın, varsa kullan yoksa hesapla)
    if 'Systole Duration' in df_clean.columns:
        sys_duration = df_clean['Systole Duration']
    else:
        cycle_len = 60.0 / hr
        sys_duration = 0.35 * np.sqrt(cycle_len)  # Basit formül yedeği

    # --- Türetilmiş Özellikler ---

    # 1. Stiffness Proxy (PP / SV)
    stiffness_obs = pp_obs / sv

    # 2. Resistance Proxy (MAP / CO)
    # CO = SV * HR / 60
    r_obs = map_obs / ((sv * hr) / 60.0)

    # 3. Shape Index
    shape_index = (map_obs - dia_obs) / pp_obs

    # 4. C_Physics (Logaritmik Decay)
    cycle_time = 60.0 / hr
    dia_duration = cycle_time - sys_duration
    # Log 0 veya sonsuz hatasını önlemek için +1e-6 ve sys>dia kontrolü teknik olarak gerekir ama MIMIC temiz varsayıyoruz
    c_physics = dia_duration / (r_obs * np.log(sys_obs / dia_obs + 1e-9) + 1e-6)

    # Girdi Matrisi Oluştur
    X_input = pd.DataFrame({
        'HR': hr,
        'SV': sv,
        'Sys_Obs': sys_obs,
        'Dia_Obs': dia_obs,
        'MAP_Obs': map_obs,
        'PP_Obs': pp_obs,
        'Shape_Index': shape_index,
        'Stiffness_Obs': stiffness_obs,
        'R_Obs': r_obs,
        'C_Physics': c_physics
    })

    # =========================================================================
    # 5. TAHMİN VE KAYIT
    # =========================================================================

    # Ölçeklendir
    X_scaled = scaler.transform(X_input)

    # Tahmin Et (R, C, Zc)
    preds = model.predict(X_scaled)

    # Sonuçları DataFrame'e ekle
    df_clean['Estimated_R'] = preds[:, 0]
    df_clean['Estimated_C'] = preds[:, 1]
    df_clean['Estimated_Zc'] = preds[:, 2]

    # Ekstra: Klinik Yorum Sütunları
    df_clean['Status_R'] = np.where(df_clean['Estimated_R'] > 1.8, 'High (Vasoconstriction)',
                                    np.where(df_clean['Estimated_R'] < 0.8, 'Low (Vasodilation)', 'Normal'))

    df_clean['Status_C'] = np.where(df_clean['Estimated_C'] < 1.0, 'Stiff (Low Compliance)', 'Elastic (Normal)')

    # Kaydet
    df_clean.to_csv(OUTPUT_PATH, index=False)
    print(f"✅ Analiz tamamlandı! Sonuçlar kaydedildi: {OUTPUT_PATH}")

    # İlk 5 hastayı göster
    print("\n🔍 Örnek Sonuçlar:")
    print(df_clean[
              ['subject_id', 'HR', 'Systolic_BP', 'SV', 'Estimated_R', 'Estimated_C', 'Status_R', 'Status_C']].head())


if __name__ == "__main__":
    predict_valid_patients()