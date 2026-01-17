import numpy as np
import pandas as pd
from scipy.integrate import odeint
from tqdm import tqdm

# --- AYARLAR ---
NUM_SAMPLES = 60000
DT = 0.005
NUM_BEATS = 10
np.random.seed(42)


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


data = []
print(f"🚀 {NUM_SAMPLES} adet 'Realistic Noise' veri üretiliyor...")

with tqdm(total=NUM_SAMPLES) as pbar:
    while len(data) < NUM_SAMPLES:

        # 1. GERÇEK FİZYOLOJİK DEĞERLER (Target - Y)
        HR = np.random.randint(50, 130)
        SV = np.random.uniform(40.0, 120.0)
        R_true = np.random.uniform(0.6, 2.5)

        # R ve C yine fiziksel bağlantılı
        target_tau = np.random.uniform(1.0, 2.2)
        C_true = target_tau / R_true
        Zc_true = np.random.uniform(0.03, 0.08) * R_true

        cycle_len = 60.0 / HR
        systole_duration = np.clip(0.35 * np.sqrt(cycle_len), 0.18, 0.45)

        # 2. ODE ÇÖZÜMÜ (Gerçek Tansiyonu Bul)
        t_eval = np.arange(0.0, NUM_BEATS * cycle_len, DT)
        CO = SV * HR / 60.0
        p0 = [R_true * CO]

        p_vals = odeint(wk3_derivative, y0=p0, t=t_eval,
                        args=(R_true, C_true, Zc_true, HR, SV, systole_duration)).flatten()

        # Stabilizasyon için son beat'i al
        samples_per_beat = int(cycle_len / DT)
        last_cycle = p_vals[-samples_per_beat:]

        real_sys = np.max(last_cycle)
        real_dia = np.min(last_cycle)
        real_map = np.mean(last_cycle)

        # 3. GÜRÜLTÜ EKLEME (OBSERVED / MEASURED VALUES - X)
        # Hastanede ölçülen değerler asla mükemmel değildir.
        # Sistolik ölçüm hatası genelde daha fazladır (±4 mmHg), Diyastolik (±2 mmHg)

        obs_sys = real_sys + np.random.normal(0, 3.0)
        obs_dia = real_dia + np.random.normal(0, 2.0)

        # MAP ölçümü de (monitörde) hafif hatalıdır
        obs_map = real_map + np.random.normal(0, 1.5)

        # Fizyolojik Filtre (Yaşayan insan mı?)
        if (80 < obs_sys < 210) and (40 < obs_dia < 130) and (obs_sys - obs_dia > 20):
            # Formül MAP (Gürültülü veri üzerinden hesaplanır)
            obs_pp = obs_sys - obs_dia

            data.append([
                # HEDEFLER (Gerçek Değerler)
                R_true, C_true, Zc_true,
                # GİRDİLER (Gürültülü Ölçümler)
                HR, SV, obs_sys, obs_dia, obs_map, obs_pp, systole_duration
            ])
            pbar.update(1)

columns = ['R_True', 'C_True', 'Zc_True',
           'HR', 'SV', 'Sys_Obs', 'Dia_Obs', 'MAP_Obs', 'PP_Obs', 'Sys_Duration']

df = pd.DataFrame(data, columns=columns)
df.to_csv('../datasets/realistic_windkessel_dataset.csv', index=False)
print("✅ Gerçekçi Dataset Hazır!")