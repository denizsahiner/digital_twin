import numpy as np
import pandas as pd
from scipy.integrate import odeint
from windkessel import generate_inflow, wk3_derivative

# --- AYARLAR ---
NUM_SAMPLES = 50000  # İdeal eğitim boyutu
DT = 0.005
NUM_BEATS = 10       # Sistemin oturması için beat sayısını artırdık
np.random.seed(42)

data = []

print(f" {NUM_SAMPLES} örnekli sentetik dataset üretiliyor...")

for i in range(NUM_SAMPLES):
    # 1. Biyomekanik Parametreler (Genişletilmiş Aralıklar)
    R = np.random.uniform(0.5, 3.0)     # Direnç (Daha geniş)
    C = np.random.uniform(0.3, 2.5)     # Esneklik (Daha sert damarlar dahil)
    Zc = np.random.uniform(0.05, 0.12) * R  # Zc, toplam direncin %5-12'si

    # 2. Hemodinamik Girdiler
    HR = np.random.randint(40, 140)    # Bradycardia ve Tachycardia dahil
    SV = np.random.uniform(30.0, 110.0) # Düşük atım hacmi (Kalp yetmezliği senaryosu)

    cycle_len = 60.0 / HR
    total_time = NUM_BEATS * cycle_len
    t_eval = np.arange(0.0, total_time, DT)

    # Sistol Süresi (Dinamik ve Fizyolojik)
    systole_duration = np.clip(0.35 * np.sqrt(cycle_len), 0.15, 0.45)

    # 3. Diferansiyel Denklem Çözümü
    CO_ml_per_s = SV * HR / 60.0
    MAP_est = R * CO_ml_per_s
    p0 = [np.clip(MAP_est, 50.0, 120.0)]

    p_2element = odeint(
        wk3_derivative, y0=p0, t=t_eval,
        args=(R, C, Zc, HR, SV, systole_duration)
    ).flatten()

    q_vals = np.array([generate_inflow(t, HR, SV, systole_duration) for t in t_eval])
    p_total = p_2element + (q_vals * Zc)

    # 4. Son Döngüden Veri Çıkarma
    samples_per_beat = int(cycle_len / DT)
    last_cycle_p = p_total[-samples_per_beat:]

    # Temel Metrikler
    sys_p = np.max(last_cycle_p)
    dia_p = np.min(last_cycle_p)
    map_p = np.mean(last_cycle_p)
    pulse_p = sys_p - dia_p

    # --- KRİTİK ADIM: NOISE (GÜRÜLTÜ) EKLEME ---
    # Gerçek cihazlardaki %1-2'lik hata payını simüle ediyoruz
    sys_p += np.random.normal(0, 1.5)
    dia_p += np.random.normal(0, 1.0)
    map_p = dia_p + (sys_p - dia_p) / 3  # MAP'i gürültülü değerlerden tekrar hesapla

    data.append([
        R, C, Zc, HR, SV,
        sys_p, dia_p, map_p, pulse_p,
        systole_duration
    ])

# 5. DataFrame ve Kayıt
columns = [
    'R', 'C', 'Zc', 'HR', 'SV',
    'Systolic_BP', 'Diastolic_BP',
    'MAP', 'Pulse_Pressure', 'Systole_Duration'
]

df = pd.DataFrame(data, columns=columns)
df.to_csv('../datasets/synthetic_windkessel_dataset_v3.csv', index=False)

print(f"\n✅ Başarılı! {len(df)} satırlık dataset kaydedildi.")