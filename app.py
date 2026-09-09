from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
import joblib
import pandas as pd
import numpy as np
import h5py
import scipy.signal as sig
import scipy.optimize as opt

app = Flask(__name__, template_folder='web/templates', static_folder='web/static')
CORS(app)

# Load Models and Scalers
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
cardio_model_path = os.path.join(BASE_DIR, 'models', 'cardio', 'cardio_model_nolabs.pkl')
cardio_scaler_path = os.path.join(BASE_DIR, 'models', 'cardio', 'cardio_scaler_nolabs.pkl')
wk_model_path = os.path.join(BASE_DIR, 'models', 'windkessel', 'realistic_model.pkl')
wk_scaler_path = os.path.join(BASE_DIR, 'models', 'windkessel', 'realistic_scaler.pkl')

# Optional MIMIC-III cuffless BP .mat file (PhysioNet, not redistributed).
# Override with:  export MIMIC_MAT_PATH=/path/to/Part_3.mat
MIMIC_MAT_PATH = os.environ.get(
    'MIMIC_MAT_PATH', os.path.join(BASE_DIR, 'data', 'mimic', 'Part_3.mat')
)

try:
    cardio_model = joblib.load(cardio_model_path)
    cardio_scaler = joblib.load(cardio_scaler_path)
    wk_model = joblib.load(wk_model_path)
    wk_scaler = joblib.load(wk_scaler_path)
    models_loaded = True
    print("All machine learning models loaded successfully.")
except Exception as e:
    models_loaded = False
    print(f"Error loading models: {e}")

# --- HELPER FUNCTIONS ---
def compute_windkessel_features(hr, sv, sbp, dbp):
    pp = sbp - dbp
    map_val = dbp + (pp / 3.0)
    shape_index = (map_val - dbp) / (pp + 1e-9)
    stiffness = pp / (sv + 1e-9)
    co = (sv * hr) / 60.0
    r_obs = map_val / (co + 1e-9)
    
    cycle_time = 60.0 / hr
    sys_duration = 0.35 * np.sqrt(cycle_time)
    dia_duration = cycle_time - sys_duration
    c_physics = dia_duration / (r_obs * np.log(sbp / (dbp + 1e-9) + 1e-9) + 1e-6)
    
    return {
        'HR': hr,
        'SV': sv,
        'Sys_Obs': sbp,
        'Dia_Obs': dbp,
        'MAP_Obs': map_val,
        'PP_Obs': pp,
        'Shape_Index': shape_index,
        'Stiffness_Obs': stiffness,
        'R_Obs': r_obs,
        'C_Physics': c_physics
    }

def find_alignment_lag(ppg, abp, fs=125.0):
    ppg_std = (ppg - np.mean(ppg)) / (np.std(ppg) + 1e-9)
    abp_std = (abp - np.mean(abp)) / (np.std(abp) + 1e-9)
    
    max_lag_samples = int(0.5 * fs)
    lags = np.arange(-max_lag_samples, max_lag_samples + 1)
    corrs = []
    
    for lag in lags:
        if lag > 0:
            p = ppg_std[lag:]
            a = abp_std[:-lag]
        elif lag < 0:
            p = ppg_std[:lag]
            a = abp_std[-lag:]
        else:
            p = ppg_std
            a = abp_std
        
        n = min(1250, len(p), len(a))
        corrs.append(np.mean(p[:n] * a[:n]))
        
    corrs = np.array(corrs)
    best_idx = np.argmax(np.abs(corrs))
    best_lag = lags[best_idx]
    best_corr = corrs[best_idx]
    return int(best_lag), best_corr

def sim_3wk(t, p_start, tau, A, Ts, B):
    omega = np.pi / Ts
    p_sim = np.zeros_like(t)
    systole_mask = t < Ts
    diastole_mask = ~systole_mask
    
    t_sys = t[systole_mask]
    p_w_sys = p_start * np.exp(-t_sys/tau) + \
        (A / ((1.0/tau)**2 + omega**2)) * ((1.0/tau)*np.sin(omega*t_sys) - omega*np.cos(omega*t_sys) + omega*np.exp(-t_sys/tau))
    
    p_sim[systole_mask] = p_w_sys + B * np.sin(omega * t_sys)
    
    p_w_Ts = p_start * np.exp(-Ts/tau) + \
        (A * omega / ((1.0/tau)**2 + omega**2)) * (np.exp(-Ts/tau) + 1.0)
    
    t_dia = t[diastole_mask]
    p_sim[diastole_mask] = p_w_Ts * np.exp(-(t_dia - Ts)/tau)
    
    return p_sim

def loss_3wk(params, t, p_real):
    tau, A, Ts, B = params
    p_start = p_real[0]
    p_sim = sim_3wk(t, p_start, tau, A, Ts, B)
    return np.mean((p_real - p_sim)**2)

def fit_3wk_beat(t, p_wave):
    T_beat = t[-1]
    init_3wk = [1.2, 100.0, 0.3 * T_beat, 5.0]
    bounds_3wk = [(0.1, 5.0), (1.0, 1500.0), (0.05, 0.7 * T_beat), (0.0, 60.0)]
    res = opt.minimize(loss_3wk, init_3wk, args=(t, p_wave), bounds=bounds_3wk, method='L-BFGS-B')
    tau, A, Ts, B = res.x
    
    # Map to biophysical R, C, Zc (SV = 70mL baseline)
    SV = 70.0
    Q0 = (SV / Ts) * (np.pi / 2.0)
    C = Q0 / A
    R = tau / C
    Zc = B / Q0
    
    return float(R), float(C), float(Zc), float(np.sqrt(res.fun))


class DigitalTwinSimulator:
    def __init__(self, age, gender, height, weight, gluc, init_r, init_c, init_zc, init_hr, init_sv, active_conditions=None):
        self.age = age
        self.gender = gender # 1 = female, 2 = male
        self.height = height
        self.weight = weight
        self.gluc = gluc # 1, 2, 3
        
        self.current_params = {
            'R': init_r,
            'C': init_c,
            'Zc': init_zc,
            'HR': init_hr,
            'SV': init_sv
        }
        self.active_conditions = set(active_conditions) if active_conditions else set()
        self.genetic_baseline = self._reverse_engineer_genetics()
        self.history = []
        self.total_days = 0

    def _reverse_engineer_genetics(self):
        base = self.current_params.copy()
        
        # Backtrack the current condition changes to estimate genetic baseline parameters
        if 'sigara' in self.active_conditions:
            base['R'] /= 1.20
            base['C'] /= 0.85
            base['Zc'] /= 1.05
            
        if 'sedanter' in self.active_conditions:
            base['SV'] /= 0.90
            base['HR'] /= 1.10
            base['R'] /= 1.05
            
        if 'alkol' in self.active_conditions:
            base['HR'] /= 1.05
            
        return base

    def _get_target_state(self):
        target = self.genetic_baseline.copy()
        
        # Apply combined multipliers for all active conditions
        if 'sigara' in self.active_conditions:
            target['R'] *= 1.25
            target['C'] *= 0.80
            target['Zc'] *= 1.10
            
        if 'sedanter' in self.active_conditions:
            target['R'] *= 1.05
            target['HR'] *= 1.10
            target['SV'] *= 0.90
            
        if 'spor' in self.active_conditions:
            target['R'] *= 0.85
            target['C'] *= 1.15
            target['HR'] *= 0.85
            target['SV'] *= 1.20
            
        if 'alkol' in self.active_conditions:
            target['HR'] *= 1.05
            
        if 'grip' in self.active_conditions:
            target['R'] *= 0.70
            target['HR'] *= 1.40
            
        return target

    def step(self, action, cardio_model, cardio_scaler):
        # 1-Yıl Sınır Kontrolü (Max 365 Gün Ufku)
        if self.total_days >= 365:
            # Simülasyon 1 yılı aşamaz
            sbp, dbp = self._calc_bp()
            current_age = self.age + (self.total_days / 365.25)
            bmi_val = self.weight / ((self.height / 100) ** 2)
            pp_val = sbp - dbp
            cardio_feats = pd.DataFrame([[
                current_age, self.gender, self.height, self.weight, sbp, dbp,
                1 if 'sigara' in self.active_conditions else 0,
                1 if 'alkol' in self.active_conditions else 0,
                self.gluc, bmi_val, pp_val
            ]], columns=['age_years', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo', 'smoke', 'alco', 'gluc', 'BMI', 'PP'])
            cardio_scaled = cardio_scaler.transform(cardio_feats)
            cardio_risk = float(cardio_model.predict_proba(cardio_scaled)[0][1])
            return {
                'day': self.total_days, 'R': self.current_params['R'], 'C': self.current_params['C'],
                'Zc': self.current_params['Zc'], 'HR': self.current_params['HR'], 'SV': self.current_params['SV'],
                'SBP': sbp, 'DBP': dbp, 'cardio_risk': cardio_risk
            }

        # Update conditions based on action
        if action == "sigara_ic":
            self.active_conditions.add("sigara")
        elif action == "sigara_birak":
            self.active_conditions.discard("sigara")
        elif action == "spor_yap":
            self.active_conditions.add("spor")
            self.active_conditions.discard("sedanter")
        elif action == "sporu_birak":
            self.active_conditions.discard("spor")
            self.active_conditions.add("sedanter")
        elif action == "alkol_ic":
            self.active_conditions.add("alkol")
        elif action == "alkol_birak":
            self.active_conditions.discard("alkol")
        elif action == "grip_ol":
            self.active_conditions.add("grip")
        elif action == "grip_iyiles":
            self.active_conditions.discard("grip")

        # Determine transition rate (Framingham hazard ratio -> R,C recovery rate bridge assumption)
        rates = {
            'sigara_ic': 0.005,
            'sigara_birak': 0.01,
            'spor_yap': 0.008,
            'sporu_birak': 0.015,
            'grip_ol': 0.40,
            'grip_iyiles': 0.20
        }
        rate = rates.get(action, 0.02)
        
        target_state = self._get_target_state()
        self.total_days += 1
        
        # 1-Year Vascular Aging decay
        self.current_params['C'] *= np.exp(-0.008 / 365.0)
        
        # Simple aging simulation
        current_age = self.age + (self.total_days / 365.25)
        
        # Exponential smoothing parameter change
        for param in ['R', 'C', 'Zc', 'HR', 'SV']:
            curr = self.current_params[param]
            targ = target_state[param]
            self.current_params[param] = curr + rate * (targ - curr)
        
        # Calculate resulting blood pressure
        sbp, dbp = self._calc_bp()
        
        # Features for Cardio model:
        # ['age_years', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo', 'smoke', 'alco', 'gluc', 'BMI', 'PP']
        smoke_val = 1 if 'sigara' in self.active_conditions else 0
        alco_val = 1 if 'alkol' in self.active_conditions else 0
        bmi_val = self.weight / ((self.height / 100) ** 2)
        pp_val = sbp - dbp
        
        cardio_feats = pd.DataFrame([[
            current_age,
            self.gender,
            self.height,
            self.weight,
            sbp,
            dbp,
            smoke_val,
            alco_val,
            self.gluc,
            bmi_val,
            pp_val
        ]], columns=['age_years', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo', 'smoke', 'alco', 'gluc', 'BMI', 'PP'])
        
        cardio_scaled = cardio_scaler.transform(cardio_feats)
        cardio_risk = float(cardio_model.predict_proba(cardio_scaled)[0][1])
        
        return {
            'day': self.total_days,
            'R': self.current_params['R'],
            'C': self.current_params['C'],
            'Zc': self.current_params['Zc'],
            'HR': self.current_params['HR'],
            'SV': self.current_params['SV'],
            'SBP': sbp,
            'DBP': dbp,
            'cardio_risk': cardio_risk
        }

    def _calc_bp(self):
        co = (self.current_params['SV'] * self.current_params['HR']) / 60.0
        mean_p = self.current_params['R'] * co
        pulse_p = self.current_params['SV'] / (self.current_params['C'] + 1e-9)
        
        sbp = mean_p + (pulse_p * 2.0 / 3.0)
        dbp = mean_p - (pulse_p / 3.0)
        return sbp, dbp


# --- API ROUTES ---
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    if not models_loaded:
        return jsonify({'error': 'Models not loaded on server.'}), 500
        
    data = request.json
    try:
        # Extract inputs
        age = float(data.get('age', 45))
        gender = int(data.get('gender', 1)) # 1=Female, 2=Male
        height = float(data.get('height', 170))
        weight = float(data.get('weight', 70))
        sbp = float(data.get('sbp', 120))
        dbp = float(data.get('dbp', 80))
        hr = float(data.get('hr', 75))
        sv = float(data.get('sv', 70))
        smoke = int(data.get('smoke', 0))
        alco = int(data.get('alco', 0))
        gluc = int(data.get('gluc', 1)) # 1, 2, 3
        
        # Calculate derived metrics
        bmi = weight / ((height / 100) ** 2)
        pp = sbp - dbp
        
        # 1. Windkessel Predictions (R, C, Zc)
        wk_feats_dict = compute_windkessel_features(hr, sv, sbp, dbp)
        wk_feats_df = pd.DataFrame([wk_feats_dict])
        
        wk_scaled = wk_scaler.transform(wk_feats_df)
        wk_pred = wk_model.predict(wk_scaled)[0]
        
        r_pred = float(wk_pred[0])
        c_pred = float(wk_pred[1])
        zc_pred = float(wk_pred[2])
        
        # 2. Cardio Risk Prediction
        cardio_cols = ['age_years', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo', 'smoke', 'alco', 'gluc', 'BMI', 'PP']
        cardio_feats_df = pd.DataFrame([[
            age, gender, height, weight, sbp, dbp, smoke, alco, gluc, bmi, pp
        ]], columns=cardio_cols)
        
        cardio_scaled = cardio_scaler.transform(cardio_feats_df)
        cardio_prob = float(cardio_model.predict_proba(cardio_scaled)[0][1])
        
        # Compute clinical status
        status_r = 'Normal'
        if r_pred > 1.8:
            status_r = 'Yüksek (Vazokonstrüksiyon - Damar Daralması)'
        elif r_pred < 0.8:
            status_r = 'Düşük (Vazodilatasyon - Damar Genişlemesi)'
            
        status_c = 'Esnek (Normal)'
        if c_pred < 1.0:
            status_c = 'Sertleşmiş (Düşük Damar Elastikiyeti)'
            
        return jsonify({
            'R': r_pred,
            'C': c_pred,
            'Zc': zc_pred,
            'cardio_risk': cardio_prob * 100,
            'status_R': status_r,
            'status_C': status_c,
            'BMI': bmi,
            'PP': pp
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/simulate', methods=['POST'])
def simulate():
    if not models_loaded:
        return jsonify({'error': 'Models not loaded on server.'}), 500
        
    data = request.json
    try:
        # Base Patient Info
        age = float(data.get('age', 45))
        gender = int(data.get('gender', 1))
        height = float(data.get('height', 170))
        weight = float(data.get('weight', 70))
        gluc = int(data.get('gluc', 1))
        
        # Initial Windkessel Parameters
        init_r = float(data.get('R'))
        init_c = float(data.get('C'))
        init_zc = float(data.get('Zc'))
        init_hr = float(data.get('HR'))
        init_sv = float(data.get('SV'))
        
        # Initial Active Conditions
        active_conditions = data.get('active_conditions', [])
        
        # Create Simulator
        sim = DigitalTwinSimulator(
            age=age, gender=gender, height=height, weight=weight, gluc=gluc,
            init_r=init_r, init_c=init_c, init_zc=init_zc, init_hr=init_hr, init_sv=init_sv,
            active_conditions=active_conditions
        )
        
        # Run phases
        phases = data.get('phases', [])
        history = []
        
        # Save baseline day 0
        sbp0, dbp0 = sim._calc_bp()
        bmi0 = weight / ((height / 100) ** 2)
        pp0 = sbp0 - dbp0
        cardio_cols = ['age_years', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo', 'smoke', 'alco', 'gluc', 'BMI', 'PP']
        cardio_feats0 = pd.DataFrame([[
            age, gender, height, weight, sbp0, dbp0, 
            1 if 'sigara' in sim.active_conditions else 0,
            1 if 'alkol' in sim.active_conditions else 0,
            gluc, bmi0, pp0
        ]], columns=cardio_cols)
        c_scaled0 = cardio_scaler.transform(cardio_feats0)
        c_risk0 = float(cardio_model.predict_proba(c_scaled0)[0][1])
        
        history.append({
            'day': 0,
            'scenario': 'Başlangıç',
            'R': init_r,
            'C': init_c,
            'Zc': init_zc,
            'HR': init_hr,
            'SV': init_sv,
            'SBP': sbp0,
            'DBP': dbp0,
            'cardio_risk': c_risk0 * 100
        })
        
        for phase in phases:
            days = int(phase.get('days', 30))
            action = phase.get('action', 'stabil')
            desc = phase.get('description', 'Mevcut Durum')
            
            for _ in range(days):
                res = sim.step(action, cardio_model, cardio_scaler)
                res['scenario'] = desc
                res['cardio_risk'] *= 100 # percentage
                
                # Downsample to save payload: save every 3rd day, or if it is the last day of phase
                if res['day'] % 3 == 0 or _ == days - 1:
                    history.append(res)
                    
        return jsonify({
            'history': history,
            'final_conditions': list(sim.active_conditions)
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 400

@app.route('/api/calibrate-ppg', methods=['POST'])
def calibrate_ppg():
    data = request.json
    try:
        subject_idx = int(data.get('subject_idx', 0))
        calib_duration = float(data.get('calib_duration', 10.0))
        
        if subject_idx < 0 or subject_idx >= 3000:
            return jsonify({'error': 'Subject index must be between 0 and 2999.'}), 400
            
        mat_path = MIMIC_MAT_PATH
        fs = 125.0

        if not os.path.exists(mat_path):
            return jsonify({'error': f'MIMIC .mat file not found at {mat_path}. '
                                     'Set MIMIC_MAT_PATH (see data/README.md).'}), 400

        with h5py.File(mat_path, 'r') as f:
            part3 = f['Part_3']
            ref = part3[subject_idx, 0]
            record = np.array(f[ref])
            
        ppg = record[:, 0]
        abp = record[:, 1]
        
        best_lag, corr_val = find_alignment_lag(ppg, abp, fs)
        
        if best_lag > 0:
            aligned_ppg = ppg[best_lag:]
            aligned_abp = abp[:-best_lag]
        elif best_lag < 0:
            aligned_ppg = ppg[:best_lag]
            aligned_abp = abp[-best_lag:]
        else:
            aligned_ppg = ppg
            aligned_abp = abp
            
        n_calib = int(calib_duration * fs)
        ppg_calib = aligned_ppg[:n_calib]
        abp_calib = aligned_abp[:n_calib]
        
        abp_peaks, _ = sig.find_peaks(abp_calib, distance=50, height=80)
        abp_troughs, _ = sig.find_peaks(-abp_calib, distance=50, height=-80)
        
        ppg_peaks, _ = sig.find_peaks(ppg_calib, distance=50)
        ppg_troughs, _ = sig.find_peaks(-ppg_calib, distance=50)
        
        avg_abp_sbp = float(np.mean(abp_calib[abp_peaks])) if len(abp_peaks) > 0 else float(np.max(abp_calib))
        avg_abp_dbp = float(np.mean(abp_calib[abp_troughs])) if len(abp_troughs) > 0 else float(np.min(abp_calib))
        
        avg_ppg_peaks = float(np.mean(ppg_calib[ppg_peaks])) if len(ppg_peaks) > 0 else float(np.max(ppg_calib))
        avg_ppg_troughs = float(np.mean(ppg_calib[ppg_troughs])) if len(ppg_troughs) > 0 else float(np.min(ppg_calib))
        
        if corr_val > 0:
            alpha = (avg_abp_sbp - avg_abp_dbp) / (avg_ppg_peaks - avg_ppg_troughs + 1e-9)
            beta = avg_abp_sbp - alpha * avg_ppg_peaks
        else:
            alpha = (avg_abp_sbp - avg_abp_dbp) / (avg_ppg_troughs - avg_ppg_peaks + 1e-9)
            beta = avg_abp_sbp - alpha * avg_ppg_troughs
            
        est_abp = alpha * aligned_ppg + beta
        
        calib_rmse = float(np.sqrt(np.mean((abp_calib - est_abp[:n_calib])**2)))
        test_rmse = float(np.sqrt(np.mean((aligned_abp[n_calib:] - est_abp[n_calib:])**2)))
        
        onsets, _ = sig.find_peaks(-aligned_abp, distance=50, prominence=5.0)
        valid_onsets = [o for o in onsets if 10 < o < len(aligned_abp) - 150]
        
        beats = []
        for i in range(len(valid_onsets) - 1):
            idx_start = valid_onsets[i]
            idx_end = valid_onsets[i+1]
            onset_time = idx_start / fs
            
            p_real = aligned_abp[idx_start:idx_end]
            p_est = est_abp[idx_start:idx_end]
            
            n_samples = len(p_real)
            t_beat = np.arange(n_samples) / fs
            T_beat = t_beat[-1]
            
            if T_beat < 0.4 or T_beat > 1.5:
                continue
                
            is_calib = onset_time < calib_duration
            
            real_sbp = float(np.max(p_real))
            real_dbp = float(np.min(p_real))
            est_sbp = float(np.max(p_est))
            est_dbp = float(np.min(p_est))
            
            rmse_wave = float(np.sqrt(np.mean((p_real - p_est)**2)))
            
            r_real, c_real, zc_real, _ = fit_3wk_beat(t_beat, p_real)
            r_est, c_est, zc_est, _ = fit_3wk_beat(t_beat, p_est)
            
            beats.append({
                'time': float(onset_time),
                'is_calib': bool(is_calib),
                'real_sbp': real_sbp,
                'real_dbp': real_dbp,
                'est_sbp': est_sbp,
                'est_dbp': est_dbp,
                'sbp_err': est_sbp - real_sbp,
                'dbp_err': est_dbp - real_dbp,
                'rmse': rmse_wave,
                'r_real': r_real,
                'c_real': c_real,
                'r_est': r_est,
                'c_est': c_est
            })
            
        downsample_factor = 3
        time_vector = (np.arange(len(aligned_abp)) / fs)[::downsample_factor]
        real_abp_ds = aligned_abp[::downsample_factor]
        est_abp_ds = est_abp[::downsample_factor]
        aligned_ppg_ds = ((aligned_ppg - np.mean(aligned_ppg)) / (np.std(aligned_ppg) + 1e-9))[::downsample_factor]
        
        if corr_val < 0:
            aligned_ppg_ds = -aligned_ppg_ds
            
        max_plot_len = int(30 * fs / downsample_factor)
        
        time_series = {
            'time': time_vector[:max_plot_len].tolist(),
            'real_abp': real_abp_ds[:max_plot_len].tolist(),
            'est_abp': est_abp_ds[:max_plot_len].tolist(),
            'aligned_ppg': aligned_ppg_ds[:max_plot_len].tolist()
        }
        
        return jsonify({
            'lag_samples': int(best_lag),
            'lag_seconds': float(best_lag / fs),
            'corr_val': float(corr_val),
            'polarity': 'Direct' if corr_val > 0 else 'Inverted',
            'alpha': float(alpha),
            'beta': float(beta),
            'calib_rmse': calib_rmse,
            'test_rmse': test_rmse,
            'beats': beats,
            'time_series': time_series
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
