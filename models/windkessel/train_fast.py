import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.multioutput import MultiOutputRegressor
import xgboost as xgb
import joblib
import os
import time

t0 = time.time()
print("Generating synthetic 3-element Windkessel dataset...")

np.random.seed(42)
N = 50000

HR = np.random.randint(50, 130, size=N)
SV = np.random.uniform(40.0, 120.0, size=N)
R_true = np.random.uniform(0.6, 2.5, size=N)
target_tau = np.random.uniform(1.0, 2.2, size=N)
C_true = target_tau / R_true
Zc_true = np.random.uniform(0.03, 0.08, size=N) * R_true

cycle_len = 60.0 / HR
systole_duration = np.clip(0.35 * np.sqrt(cycle_len), 0.18, 0.45)
R_p = np.maximum(R_true - Zc_true, 1e-4)

STEPS = 120
t_rel = np.linspace(0, 1, STEPS)

t_matrix = t_rel[None, :] * cycle_len[:, None]
sys_dur_matrix = systole_duration[:, None]

is_sys = t_matrix < sys_dur_matrix
q_max = (SV * np.pi) / (2.0 * systole_duration)
q_in = np.where(is_sys, q_max[:, None] * np.sin(np.pi * t_matrix / sys_dur_matrix), 0.0)

tau = R_p * C_true
CO = (SV * HR) / 60.0
mean_p = R_true * CO

pulse_p_rc = SV / C_true
sys_rc = mean_p + pulse_p_rc * 0.6
dia_rc = mean_p - pulse_p_rc * 0.4

real_sys = sys_rc + q_max * Zc_true
real_dia = dia_rc
real_map = mean_p + (q_max * Zc_true * 0.2)

obs_sys = real_sys + np.random.normal(0, 2.5, size=N)
obs_dia = real_dia + np.random.normal(0, 1.5, size=N)
obs_map = real_map + np.random.normal(0, 1.0, size=N)
obs_pp = obs_sys - obs_dia

valid = (obs_sys > 80) & (obs_sys < 220) & (obs_dia > 40) & (obs_dia < 130) & (obs_pp > 20)

df = pd.DataFrame({
    'R_True': R_true[valid],
    'C_True': C_true[valid],
    'Zc_True': Zc_true[valid],
    'HR': HR[valid],
    'SV': SV[valid],
    'Sys_Obs': obs_sys[valid],
    'Dia_Obs': obs_dia[valid],
    'MAP_Obs': obs_map[valid],
    'PP_Obs': obs_pp[valid],
    'Sys_Duration': systole_duration[valid]
})

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

print(f"Training XGBoost regressor on {len(df)} samples...")
model = MultiOutputRegressor(xgb.XGBRegressor(
    n_estimators=500,
    learning_rate=0.04,
    max_depth=6,
    subsample=0.85,
    colsample_bytree=0.85,
    n_jobs=-1,
    random_state=42
))

model.fit(X_tr_s, y_train)
y_pred = model.predict(X_te_s)

print("\nModel evaluation metrics:")
print("-" * 40)
for i, col in enumerate(targets):
    r2 = r2_score(y_test.iloc[:, i], y_pred[:, i])
    mae = mean_absolute_error(y_test.iloc[:, i], y_pred[:, i])
    print(f"{col:<8} | R2: {r2:.4f} | MAE: {mae:.4f}")
print("-" * 40)

os.makedirs('ml_model_windkessel', exist_ok=True)
joblib.dump(model, 'ml_model_windkessel/realistic_model.pkl')
joblib.dump(scaler, 'ml_model_windkessel/realistic_scaler.pkl')

joblib.dump(model, 'realistic_model.pkl')
joblib.dump(scaler, 'realistic_scaler.pkl')

print(f"Model saved successfully in {time.time()-t0:.2f} seconds.")
