import h5py
import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt

import os as _os
mat_path = _os.environ.get(
    'MIMIC_MAT_PATH',
    _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), 'data', 'mimic', 'Part_3.mat'),
)
fs = 125.0

def find_alignment_lag(ppg, abp, fs=125.0):
    ppg_std = (ppg - np.mean(ppg)) / (np.std(ppg) + 1e-9)
    abp_std = (abp - np.mean(abp)) / (np.std(abp) + 1e-9)
    
    max_lag_samples = int(0.5 * fs) # 0.5s window
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
    # Find the lag with the absolute maximum correlation
    best_idx = np.argmax(np.abs(corrs))
    best_lag = lags[best_idx]
    best_corr = corrs[best_idx]
    return int(best_lag), best_corr

def test():
    with h5py.File(mat_path, 'r') as f:
        part3 = f['Part_3']
        ref = part3[0, 0]
        data = np.array(f[ref])
        
        ppg = data[:, 0]
        abp = data[:, 1]
        
        best_lag, corr_val = find_alignment_lag(ppg, abp)
        print(f"Best lag: {best_lag} samples ({best_lag / fs:.3f} s), Corr value: {corr_val:.3f}")
        
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
        
        # Detect peaks/troughs
        abp_peaks, _ = sig.find_peaks(abp_calib, distance=50, height=80)
        abp_troughs, _ = sig.find_peaks(-abp_calib, distance=50, height=-80)
        
        ppg_peaks, _ = sig.find_peaks(ppg_calib, distance=50)
        ppg_troughs, _ = sig.find_peaks(-ppg_calib, distance=50)
        
        avg_abp_sbp = np.mean(abp_calib[abp_peaks])
        avg_abp_dbp = np.mean(abp_calib[abp_troughs])
        
        avg_ppg_peaks = np.mean(ppg_calib[ppg_peaks])
        avg_ppg_troughs = np.mean(ppg_calib[ppg_troughs])
        
        print(f"ABP Calib SBP: {avg_abp_sbp:.2f}, DBP: {avg_abp_dbp:.2f}")
        print(f"PPG Calib Peaks: {avg_ppg_peaks:.3f}, Troughs: {avg_ppg_troughs:.3f}")
        
        # Determine alpha and beta based on correlation sign
        if corr_val > 0:
            # Positive correlation: PPG peaks -> ABP peaks, PPG troughs -> ABP troughs
            alpha = (avg_abp_sbp - avg_abp_dbp) / (avg_ppg_peaks - avg_ppg_troughs)
            beta = avg_abp_sbp - alpha * avg_ppg_peaks
            print("Detected POSITIVE correlation polarity.")
        else:
            # Negative correlation (PPG inverted): PPG troughs -> ABP peaks, PPG peaks -> ABP troughs
            alpha = (avg_abp_sbp - avg_abp_dbp) / (avg_ppg_troughs - avg_ppg_peaks)
            beta = avg_abp_sbp - alpha * avg_ppg_troughs
            print("Detected NEGATIVE correlation polarity.")
            
        print(f"Calibration coefficients: alpha={alpha:.3f}, beta={beta:.3f}")
        
        # Apply calibration
        est_abp = alpha * aligned_ppg + beta
        
        calib_rmse = np.sqrt(np.mean((abp_calib - est_abp[:n_calib])**2))
        test_rmse = np.sqrt(np.mean((aligned_abp[n_calib:] - est_abp[n_calib:])**2))
        
        print(f"Calibration Window (0-10s) RMSE: {calib_rmse:.3f} mmHg")
        print(f"Testing Window (10s+) RMSE: {test_rmse:.3f} mmHg")
        
        t = np.arange(len(aligned_abp)) / fs
        plt.figure(figsize=(12, 6))
        plt.plot(t[:2500], aligned_abp[:2500], 'k-', label='Real ABP', lw=2)
        plt.plot(t[:2500], est_abp[:2500], 'r--', label='PPG-Estimated ABP', lw=1.5)
        plt.axvline(x=10.0, color='blue', linestyle=':', label='Calibration Boundary')
        plt.xlabel('Time (s)')
        plt.ylabel('Pressure (mmHg)')
        plt.title('Self-Correcting PPG to ABP Calibration')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('calibration_test_selfcorrecting.png')
        print("Saved test plot to calibration_test_selfcorrecting.png")

if __name__ == '__main__':
    test()
