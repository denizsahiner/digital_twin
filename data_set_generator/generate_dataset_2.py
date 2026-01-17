import numpy as np
import pandas as pd
from scipy.integrate import odeint
from windkessel import generate_inflow, wk3_derivative
from tqdm import tqdm

# --- AYARLAR ---
NUM_SAMPLES_TARGET = 50000
DT = 0.005
NUM_BEATS = 10
np.random.seed(42)

data = []
count = 0

print(f"🚀 {NUM_SAMPLES_TARGET} geçerli örnek üretilene kadar simülasyon sürüyor...")

with tqdm(total=NUM_SAMPLES_TARGET) as pbar:
    while len(data) < NUM_SAMPLES_TARGET:
        # 1. Parametre Üretimi
        R = np.random.uniform(0.5, 3.0)
        C = np.random.uniform(0.3, 2.5)
        Zc = np.random.uniform(0.05, 0.12) * R

        HR = np.random.randint(40, 140)
        SV = np.random.uniform(30.0, 110.0)

        cycle_len = 60.0 / HR
        systole_duration = np.clip(0.35 * np.sqrt(cycle_len), 0.15, 0.45)

        # 2. Diferansiyel Çözüm
        t_eval = np.arange(0.0, NUM_BEATS * cycle_len, DT)
        CO_ml_per_s = SV * HR / 60.0
        MAP_est = R * CO_ml_per_s
        p0 = [np.clip(MAP_est, 60.0, 110.0)]

        p_2element = odeint(wk3_derivative, y0=p0, t=t_eval,
                            args=(R, C, Zc, HR, SV, systole_duration)).flatten()

        q_vals = np.array([generate_inflow(t, HR, SV, systole_duration) for t in t_eval])
        p_total = p_2element + (q_vals * Zc)

        # 3. Son döngü değerlerini al
        samples_per_beat = int(cycle_len / DT)
        last_cycle_p = p_total[-samples_per_beat:]

        sys_p = np.max(last_cycle_p)
        dia_p = np.min(last_cycle_p)

        # --- KRİTİK FİLTRELEME (FİZYOLOJİK SINIRLAR) ---
        # Sadece yaşayan insan sınırlarında olan verileri alıyoruz
        if (75 < sys_p < 230) and (40 < dia_p < 140) and (sys_p - dia_p > 15):
            # Gürültü ekle
            sys_p += np.random.normal(0, 1.2)
            dia_p += np.random.normal(0, 0.8)
            map_p = dia_p + (sys_p - dia_p) / 3
            pulse_p = sys_p - dia_p

            data.append([
                R, C, Zc, HR, SV,
                sys_p, dia_p, map_p, pulse_p,
                systole_duration
            ])
            pbar.update(1)  # Sadece geçerli örnekte ilerle

# DataFrame Oluştur
columns = ['R', 'C', 'Zc', 'HR', 'SV', 'Systolic_BP', 'Diastolic_BP', 'MAP', 'Pulse_Pressure', 'Systole_Duration']
df = pd.DataFrame(data, columns=columns)
df.to_csv('../datasets/filtered_windkessel_dataset.csv', index=False)