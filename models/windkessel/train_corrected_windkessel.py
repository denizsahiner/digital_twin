import numpy as np
import pandas as pd
from scipy.integrate import odeint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
import xgboost as xgb
import joblib
from pathlib import Path

# --- PATHS ---
HERE = Path(__file__).resolve().parent
MODEL_PATH = HERE / 'realistic_model.pkl'
SCALER_PATH = HERE / 'realistic_scaler.pkl'

print(" Düzeltilmiş 3-Element Windkessel Modeli Eğitiliyor...")
np.random.seed(42)
NUM_SAMPLES = 30000
DT = 0.005

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
count = 0
print(f" {NUM_SAMPLES} adet fizyolojik doğru veri üretiliyor...")

while count < NUM_SAMPLES:
    HR = np.random.randint(50, 130)
    SV = np.random.uniform(40.0, 120.0)
    R_true = np.random.uniform(0.6, 2.5)

    target_tau = np.random.uniform(1.0, 2.2)
    C_true = target_tau / R_true
    Zc_true = np.random.uniform(0.03, 0.08) * R_true

    cycle_len = 60.0 / HR
    systole_duration = np.clip(0.35 * np.sqrt(cycle_len), 0.18, 0.45)

    t_eval = np.arange(0.0, 6 * cycle_len, DT)
    CO = SV * HR / 60.0
    p0 = [R_true * CO]

    p_rc = odeint(wk3_derivative, y0=p0, t=t_eval,
                  args=(R_true, C_true, Zc_true, HR, SV, systole_duration)).flatten()

    # Tam Aort Basıncı P_inlet = P_rc + Q_in * Zc (Zc basınç düşüşü eklendi!)
    q_in_eval = np.array([generate_inflow(t, HR, SV, systole_duration) for t in t_eval])
    p_inlet = p_rc + q_in_eval * Zc_true

    samples_per_beat = int(cycle_len / DT)
    last_cycle_inlet = p_inlet[-samples_per_beat:]

    real_sys = np.max(last_cycle_inlet)
    real_dia = np.min(last_cycle_inlet)
    real_map = np.mean(last_cycle_inlet)

    # Ölçüm gürültüsü
    obs_sys = real_sys + np.random.normal(0, 2.5)
    obs_dia = real_dia + np.random.normal(0, 1.5)
    obs_map = real_map + np.random.normal(0, 1.0)

    if (80 < obs_sys < 210) and (40 < obs_dia < 130) and (obs_sys - obs_dia > 20):
        obs_pp = obs_sys - obs_dia
        data.append([
            R_true, C_true, Zc_true,
            HR, SV, obs_sys, obs_dia, obs_map, obs_pp, systole_duration
        ])
        count += 1

columns = ['R_True', 'C_True', 'Zc_True', 'HR', 'SV', 'Sys_Obs', 'Dia_Obs', 'MAP_Obs', 'PP_Obs', 'Sys_Duration']
df = pd.DataFrame(data, columns=columns)

# Feature engineering
df['Stiffness_Obs'] = df['PP_Obs'] / df['SV']
df['R_Obs'] = df['MAP_Obs'] / ((df['SV'] * df['HR']) / 60.0)
df['Shape_Index'] = (df['MAP_Obs'] - df['Dia_Obs']) / df['PP_Obs']
cycle_time = 60.0 / df['HR']
dia_duration = cycle_time - df['Sys_Duration']
df['C_Physics'] = dia_duration / (df['R_Obs'] * np.log(df['Sys_Obs'] / (df['Dia_Obs'] + 1e-9) + 1e-9) + 1e-6)

features = ['HR', 'SV', 'Sys_Obs', 'Dia_Obs', 'MAP_Obs', 'PP_Obs', 'Shape_Index', 'Stiffness_Obs', 'R_Obs', 'C_Physics']
targets = ['R_True', 'C_True', 'Zc_True']

X = df[features]
y = df[targets]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_tr_s = scaler.fit_transform(X_train)
X_te_s = scaler.transform(X_test)

print(" XGBoost Modelleri Eğitiliyor...")
from sklearn.multioutput import MultiOutputRegressor
model = MultiOutputRegressor(xgb.XGBRegressor(
    n_estimators=600,
    learning_rate=0.03,
    max_depth=7,
    subsample=0.85,
    colsample_bytree=0.85,
    n_jobs=-1,
    random_state=42
))

model.fit(X_tr_s, y_train)

y_pred = model.predict(X_te_s)

print("\n YENİ MODEL SONUÇLARI:")
for i, col in enumerate(targets):
    r2 = r2_score(y_test.iloc[:, i], y_pred[:, i])
    mae = mean_absolute_error(y_test.iloc[:, i], y_pred[:, i])
    print(f"   {col:<7} | R2 = {r2:.4f} | MAE = {mae:.4f}")

# Modelleri kaydet
joblib.dump(model, MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)

print(f"✅ Yeni model ve scaler kaydedildi: {MODEL_PATH.name}, {SCALER_PATH.name}")
