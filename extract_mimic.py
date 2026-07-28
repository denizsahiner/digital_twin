import pandas as pd
import os
import numpy as np

# Yolları tanımla
base_path = "/home/deniz/Desktop/digital_twin_2/mimic-iv-clinical-database-demo-2.2"
output_path = "/home/deniz/Desktop/digital_twin_2/datasets/mimic_all_filtered.csv"


def extract_ultimate_mimic():
    # --- 1. HASTA BİLGİLERİ ---
    df_patients = pd.read_csv(os.path.join(base_path, "hosp/patients.csv.gz"))
    df_patients = df_patients[['subject_id', 'gender', 'anchor_age']]

    # --- 2. VİTAL BULGULAR ---
    print("ICU Vital verileri çekiliyor...")
    vitals_list = []
    relevant_ids = [220045, 220179, 220180, 220052, 225322, 220088, 226707, 226512]

    for chunk in pd.read_csv(os.path.join(base_path, "icu/chartevents.csv.gz"), chunksize=100000):
        rows = chunk[chunk['itemid'].isin(relevant_ids)][['subject_id', 'itemid', 'valuenum']]
        vitals_list.append(rows)

    df_vitals = pd.concat(vitals_list).pivot_table(index='subject_id', columns='itemid', values='valuenum',
                                                   aggfunc='mean').reset_index()

    rename_map = {
        220045: 'HR', 220179: 'Systolic_BP', 220180: 'Diastolic_BP',
        220052: 'MAP_raw', 225322: 'SV_raw', 220088: 'CO',
        226707: 'Height', 226512: 'Weight'
    }
    df_vitals.rename(columns=rename_map, inplace=True)

    # --- 3. LABORATUVAR VERİLERİ ---
    print("🧪 Laboratuvar sonuçları ekleniyor...")
    lab_ids = [50931, 51003, 50963, 50912]
    lab_list = []
    for chunk in pd.read_csv(os.path.join(base_path, "hosp/labevents.csv.gz"), chunksize=100000):
        rows = chunk[chunk['itemid'].isin(lab_ids)][['subject_id', 'itemid', 'valuenum']]
        lab_list.append(rows)

    df_labs = pd.concat(lab_list).pivot_table(index='subject_id', columns='itemid', values='valuenum',
                                              aggfunc='max').reset_index()
    df_labs.rename(columns={50931: 'Glucose', 51003: 'Troponin', 50963: 'BNP', 50912: 'Creatinine'}, inplace=True)

    # --- 4. SİGARA VE ALKOL (ICD KODLARI) ---
    print("🚬 Alışkanlık verileri işleniyor...")
    df_diag = pd.read_csv(os.path.join(base_path, "hosp/diagnoses_icd.csv.gz"))
    df_diag['icd_code'] = df_diag['icd_code'].astype(str).str.strip()

    sigara_kodlari = ('F17', '3051', 'V1582', 'Z720')
    alkol_kodlari = ('F10', '303', '3050', 'Z721')

    smokers = df_diag[df_diag['icd_code'].str.startswith(sigara_kodlari)]['subject_id'].unique()
    alcohol_users = df_diag[df_diag['icd_code'].str.startswith(alkol_kodlari)]['subject_id'].unique()

    # --- 5. BİRLEŞTİRME ---
    final_df = df_patients.merge(df_vitals, on='subject_id', how='left')
    final_df = final_df.merge(df_labs, on='subject_id', how='left')

    final_df['is_smoker'] = final_df['subject_id'].isin(smokers).astype(int)
    final_df['is_alcohol_user'] = final_df['subject_id'].isin(alcohol_users).astype(int)

    # --- 6. SMART FEATURE ENGINEERING & DATA CLEANING ---
    print("🛠️  Birim dönüşümleri ve hesaplamalar yapılıyor...")

    def get_c(name):
        return final_df[name] if name in final_df.columns else pd.Series(np.nan, index=final_df.index)

    def categorize_glucose(val):
        if pd.isna(val): return np.nan  # Veri yoksa boş bırak (veya 1 yapabilirsin)
        if val > 125:
            return 3  # Diyabetik
        elif val > 100:
            return 2  # Prediyabet
        else:
            return 1  # Normal

    if 'Glucose' in final_df.columns:
        final_df['gluc'] = final_df['Glucose'].apply(categorize_glucose)
    else:
        final_df['gluc'] = np.nan
    # Boy Düzeltme: Eğer değer < 100 ise Inçtir -> Cm'ye çevir
    final_df['Height'] = get_c('Height').apply(lambda x: x * 2.54 if 0 < x < 100 else x)

    # Kilo Düzeltme: Sadece yuvarla
    final_df['Weight'] = get_c('Weight').round(1)

    # Hemodinamik Hesaplamalar
    sbp, dbp = get_c('Systolic_BP'), get_c('Diastolic_BP')
    final_df['PP'] = sbp - dbp
    final_df['MAP'] = get_c('MAP_raw').fillna(dbp + (final_df['PP'] / 3))

    # SV (Stroke Volume)
    calculated_sv = (get_c('CO') / get_c('HR')) * 1000
    final_df['SV'] = get_c('SV_raw').fillna(calculated_sv).fillna(70)

    # BMI Hesaplama (Metrik: kg / m^2)
    # Boyu metreye çevirip karesini alıyoruz
    final_df['BMI'] = final_df['Weight'] / ((final_df['Height'] / 100) ** 2)
    final_df['BMI'] = final_df['BMI'].replace([np.inf, -np.inf], np.nan).round(1)

    return final_df


# Çalıştır ve Kaydet
data = extract_ultimate_mimic()
if not data.empty:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    data.to_csv(output_path, index=False)
    print("-" * 30)
    print(f"✅ İşlem Tamamlandı. Toplam Hasta: {len(data)}")
    # BMI ve Boy kontrolü için ekrana basalım
    print(data[['subject_id', 'Height', 'Weight', 'BMI', 'is_smoker']].head(10))