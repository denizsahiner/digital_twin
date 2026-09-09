import h5py
import numpy as np
import scipy.signal as sig
import scipy.optimize as opt
import time

import os as _os
mat_path = _os.environ.get(
    'MIMIC_MAT_PATH',
    _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), 'data', 'mimic', 'Part_3.mat'),
)
fs = 125.0

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
    # Bound parameters
    init_3wk = [1.2, 100.0, 0.3 * T_beat, 5.0]
    bounds_3wk = [(0.1, 5.0), (1.0, 1500.0), (0.05, 0.7 * T_beat), (0.0, 60.0)]
    res = opt.minimize(loss_3wk, init_3wk, args=(t, p_wave), bounds=bounds_3wk, method='L-BFGS-B')
    tau, A, Ts, B = res.x
    
    # We can approximate R and C from these parameters:
    # tau = R * C. We can estimate C from stroke volume (which we assume as ~70 mL for normalization)
    # C = SV / pulse_pressure
    # R = tau / C
    # Here, we can just return tau, A (Q0/C), Ts, B (Zc*Q0) as our fitted physical parameters.
    # To convert A and B to R and C:
    # If we assume average ejection flow rate Q_mean = SV / Ts, and peak flow Q0 = Q_mean * pi/2 (for half-sine)
    # Q0 = (SV / Ts) * (pi / 2)
    # Then:
    # C = Q0 / A = (SV * pi) / (2 * Ts * A)
    # R = tau / C
    # Zc = B / Q0
    # Let's use this biophysical mapping! It is extremely elegant.
    SV = 70.0 # nominal ml
    Q0 = (SV / Ts) * (np.pi / 2.0)
    C = Q0 / A
    R = tau / C
    Zc = B / Q0
    
    return R, C, Zc, np.sqrt(res.fun)

def run_test():
    start_time = time.time()
    
    with h5py.File(mat_path, 'r') as f:
        part3 = f['Part_3']
        ref = part3[0, 0]
        data = np.array(f[ref])
        
        ppg = data[:, 0]
        abp = data[:, 1]
        
        # Align
        best_lag, corr_val = find_alignment_lag(ppg, abp)
        
        if best_lag > 0:
            aligned_ppg = ppg[best_lag:]
            aligned_abp = abp[:-best_lag]
        elif best_lag < 0:
            aligned_ppg = ppg[:best_lag]
            aligned_abp = abp[-best_lag:]
        else:
            aligned_ppg = ppg
            aligned_abp = abp
            
        n_calib = int(10 * fs)
        ppg_calib = aligned_ppg[:n_calib]
        abp_calib = aligned_abp[:n_calib]
        
        # Detect peaks and troughs for calibration
        abp_peaks, _ = sig.find_peaks(abp_calib, distance=50, height=80)
        abp_troughs, _ = sig.find_peaks(-abp_calib, distance=50, height=-80)
        
        ppg_peaks, _ = sig.find_peaks(ppg_calib, distance=50)
        ppg_troughs, _ = sig.find_peaks(-ppg_calib, distance=50)
        
        avg_abp_sbp = np.mean(abp_calib[abp_peaks])
        avg_abp_dbp = np.mean(abp_calib[abp_troughs])
        
        avg_ppg_peaks = np.mean(ppg_calib[ppg_peaks])
        avg_ppg_troughs = np.mean(ppg_calib[ppg_troughs])
        
        # Calibrate
        if corr_val > 0:
            alpha = (avg_abp_sbp - avg_abp_dbp) / (avg_ppg_peaks - avg_ppg_troughs)
            beta = avg_abp_sbp - alpha * avg_ppg_peaks
        else:
            alpha = (avg_abp_sbp - avg_abp_dbp) / (avg_ppg_troughs - avg_ppg_peaks)
            beta = avg_abp_sbp - alpha * avg_ppg_troughs
            
        est_abp = alpha * aligned_ppg + beta
        
        # Beat detection on aligned ABP
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
                
            is_calib = onset_time < 10.0
            
            # SBP and DBP
            real_sbp = np.max(p_real)
            real_dbp = np.min(p_real)
            est_sbp = np.max(p_est)
            est_dbp = np.min(p_est)
            
            rmse_wave = np.sqrt(np.mean((p_real - p_est)**2))
            
            # Fit 3WK on real
            r_real, c_real, zc_real, fit_err_real = fit_3wk_beat(t_beat, p_real)
            
            # Fit 3WK on est
            r_est, c_est, zc_est, fit_err_est = fit_3wk_beat(t_beat, p_est)
            
            beats.append({
                'time': onset_time,
                'is_calib': is_calib,
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
            
        print(f"Processed {len(beats)} beats in {time.time() - start_time:.3f} seconds.")
        print(f"Sample beat 0 (Calib): time={beats[0]['time']:.2f}s, real_sbp={beats[0]['real_sbp']:.2f}, est_sbp={beats[0]['est_sbp']:.2f}, R_real={beats[0]['r_real']:.3f}, R_est={beats[0]['r_est']:.3f}")
        print(f"Sample beat -1 (Test): time={beats[-1]['time']:.2f}s, real_sbp={beats[-1]['real_sbp']:.2f}, est_sbp={beats[-1]['est_sbp']:.2f}, R_real={beats[-1]['r_real']:.3f}, R_est={beats[-1]['r_est']:.3f}")

if __name__ == '__main__':
    run_test()
