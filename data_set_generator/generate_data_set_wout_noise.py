import numpy as np
import pandas as pd
from scipy.integrate import odeint
from tqdm import tqdm


# --- YARDIMCI FONKSİYONLAR ---
def generate_inflow(t, hr, sv, systole_duration):
    cycle_duration = 60.0 / hr
    local_t = t % cycle_duration
    if local_t < systole_duration:
        q_max = (sv * np.pi) / (2.0 * systole_duration)
        return q_max * np.sin(np.pi * local_t / systole_duration)
    else:
        return 0.0


def wk3_derivative(p, t, r_total, c_total, z_c, hr, sv, systole_duration):
    r_peripheral = r_total - z_c
    if r_peripheral <= 1e-6: r_peripheral = 1e-6
    q_in = generate_inflow(t, hr, sv, systole_duration)
    dp_dt = (q_in / c_total) - (p / (r_peripheral * c_total))
    return dp_dt


# --- AYARLAR ---
NUM_SAMPLES = 50000
DT = 0.005
NUM_BEATS = 10
np.random.seed(42)

data = []

print(f" {NUM_SAMPLES} adet 'TERTEMİZ' (No Noise) veri üretiliyor...")

with tqdm(total=NUM_SAMPLES) as pbar:
    while len(data) < NUM_SAMPLES:

        # 1. Parametreler (Yine Fizyolojik İlişkili Üretelim ki mantıklı olsun)
        HR = np.random.randint(50, 130)
        SV = np.random.uniform(40.0, 120.0)
        R = np.random.uniform(0.6, 2.5)

        # R ve C yine ters orantılı olsun (Doğal Fizyoloji)
        target_tau = np.random.uniform(1.0, 2.2)
        C = target_tau / R
        Zc = np.random.uniform(0.03, 0.08) * R

        cycle_len = 60.0 / HR
        systole_duration = np.clip(0.35 * np.sqrt(cycle_len), 0.18, 0.45)

        # 2. Simülasyon
        t_eval = np.arange(0.0, NUM_BEATS * cycle_len, DT)
        CO_ml_per_s = SV * HR / 60.0
        MAP_est = R * CO_ml_per_s
        p0 = [np.clip(MAP_est, 50.0, 140.0)]

        p_2element = odeint(wk3_derivative, y0=p0, t=t_eval,
                            args=(R, C, Zc, HR, SV, systole_duration)).flatten()

        q_vals = np.array([generate_inflow(t, HR, SV, systole_duration) for t in t_eval])
        p_total = p_2element + (q_vals * Zc)

        # 3. Sonuçları Al
        samples_per_beat = int(cycle_len / DT)
        last_cycle_p = p_total[-samples_per_beat:]

        sys_p = np.max(last_cycle_p)
        dia_p = np.min(last_cycle_p)
        map_true = np.mean(last_cycle_p)  # İntegral MAP

        # --- GÜRÜLTÜ EKLEMEK YOK! (Tertemiz Veri) ---
        # Sadece fizyolojik sınırlara bakıp alıyoruz
        if (80 < sys_p < 200) and (40 < dia_p < 120) and (sys_p - dia_p > 20):
            # Formül MAP (Klinik)
            pp = sys_p - dia_p
            map_formula = dia_p + (pp / 3.0)

            data.append([
                R, C, Zc, HR, SV,
                sys_p, dia_p, map_true, map_formula, pp,
                systole_duration
            ])
            pbar.update(1)

columns = ['R', 'C', 'Zc', 'HR', 'SV',
           'Systolic_BP', 'Diastolic_BP', 'MAP_True', 'MAP_Formula',
           'Pulse_Pressure', 'Systole_Duration']

df = pd.DataFrame(data, columns=columns)
df.to_csv('../datasets/clean_windkessel_dataset.csv', index=False)
print("✅ Tertemiz Dataset Hazır: ../datasets/clean_windkessel_dataset.csv")