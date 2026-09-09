import h5py
import numpy as np
import scipy.optimize as opt
import scipy.signal as sig
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import pearsonr

import os as _os
mat_path = _os.environ.get(
    'MIMIC_MAT_PATH',
    _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), 'data', 'mimic', 'Part_3.mat'),
)
fs = 125.0  # Sampling frequency

def sim_2wk(t, p_start, tau, A, Ts):
    omega = np.pi / Ts
    p_sim = np.zeros_like(t)
    systole_mask = t < Ts
    diastole_mask = ~systole_mask
    
    t_sys = t[systole_mask]
    p_sim[systole_mask] = p_start * np.exp(-t_sys/tau) + \
        (A / ((1.0/tau)**2 + omega**2)) * ((1.0/tau)*np.sin(omega*t_sys) - omega*np.cos(omega*t_sys) + omega*np.exp(-t_sys/tau))
    
    p_Ts = p_start * np.exp(-Ts/tau) + \
        (A * omega / ((1.0/tau)**2 + omega**2)) * (np.exp(-Ts/tau) + 1.0)
    
    t_dia = t[diastole_mask]
    p_sim[diastole_mask] = p_Ts * np.exp(-(t_dia - Ts)/tau)
    
    return p_sim

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

def loss_2wk(params, t, p_real):
    tau, A, Ts = params
    p_start = p_real[0]
    p_sim = sim_2wk(t, p_start, tau, A, Ts)
    return np.mean((p_real - p_sim)**2)

def loss_3wk(params, t, p_real):
    tau, A, Ts, B = params
    p_start = p_real[0]
    p_sim = sim_3wk(t, p_start, tau, A, Ts, B)
    return np.mean((p_real - p_sim)**2)

def main():
    # We will sample 5 subjects (cells)
    subject_indices = [0, 50, 100, 150, 200]
    beats_per_subject = 50
    
    results = []
    
    # To plot a sample waveform comparison, we will save the raw signals from the first subject
    sample_real_w = []
    sample_sim_2wk_w = []
    sample_sim_3wk_w = []
    
    with h5py.File(mat_path, 'r') as f:
        part3 = f['Part_3']
        
        for sub_id, sub_idx in enumerate(subject_indices):
            print(f"Processing Subject {sub_idx}...")
            ref = part3[sub_idx, 0]
            data = np.array(f[ref])
            abp = data[:, 1]
            
            # Detect beat onsets
            onsets, _ = sig.find_peaks(-abp, distance=50, prominence=5.0)
            
            # Filter out onsets that are too close to the edges
            valid_onsets = [o for o in onsets if 50 < o < len(abp) - 150]
            
            count = 0
            for i in range(len(valid_onsets) - 1):
                if count >= beats_per_subject:
                    break
                
                idx_start = valid_onsets[i]
                idx_end = valid_onsets[i+1]
                p_real = abp[idx_start:idx_end]
                n_samples = len(p_real)
                t = np.arange(n_samples) / fs
                T_beat = t[-1]
                
                # Check for physiological beat duration (e.g. 0.4s to 1.5s)
                if T_beat < 0.4 or T_beat > 1.5:
                    continue
                
                p_start = p_real[0]
                real_sbp = np.max(p_real)
                real_dbp = np.min(p_real)
                
                # Fit 2WK
                init_2wk = [1.2, 100.0, 0.3 * T_beat]
                bounds_2wk = [(0.1, 5.0), (1.0, 1500.0), (0.05, 0.7 * T_beat)]
                res_2wk = opt.minimize(loss_2wk, init_2wk, args=(t, p_real), bounds=bounds_2wk, method='L-BFGS-B')
                tau_2wk, A_2wk, Ts_2wk = res_2wk.x
                p_sim_2wk = sim_2wk(t, p_start, tau_2wk, A_2wk, Ts_2wk)
                
                # Fit 3WK
                init_3wk = [1.2, 100.0, 0.3 * T_beat, 5.0]
                bounds_3wk = [(0.1, 5.0), (1.0, 1500.0), (0.05, 0.7 * T_beat), (0.0, 60.0)]
                res_3wk = opt.minimize(loss_3wk, init_3wk, args=(t, p_real), bounds=bounds_3wk, method='L-BFGS-B')
                tau_3wk, A_3wk, Ts_3wk, B_3wk = res_3wk.x
                p_sim_3wk = sim_3wk(t, p_start, tau_3wk, A_3wk, Ts_3wk, B_3wk)
                
                # Evaluate metrics
                # 2WK
                rmse_2wk = np.sqrt(np.mean((p_real - p_sim_2wk)**2))
                mae_2wk = np.mean(np.abs(p_real - p_sim_2wk))
                sim_sbp_2wk = np.max(p_sim_2wk)
                sim_dbp_2wk = np.min(p_sim_2wk)
                sbp_err_2wk = sim_sbp_2wk - real_sbp
                dbp_err_2wk = sim_dbp_2wk - real_dbp
                r2_2wk = 1 - (np.sum((p_real - p_sim_2wk)**2) / np.sum((p_real - np.mean(p_real))**2))
                
                # 3WK
                rmse_3wk = np.sqrt(np.mean((p_real - p_sim_3wk)**2))
                mae_3wk = np.mean(np.abs(p_real - p_sim_3wk))
                sim_sbp_3wk = np.max(p_sim_3wk)
                sim_dbp_3wk = np.min(p_sim_3wk)
                sbp_err_3wk = sim_sbp_3wk - real_sbp
                dbp_err_3wk = sim_dbp_3wk - real_dbp
                r2_3wk = 1 - (np.sum((p_real - p_sim_3wk)**2) / np.sum((p_real - np.mean(p_real))**2))
                
                results.append({
                    'subject': sub_idx,
                    'beat_idx': count,
                    'duration': T_beat,
                    'real_sbp': real_sbp,
                    'real_dbp': real_dbp,
                    # 2WK metrics
                    'rmse_2wk': rmse_2wk,
                    'mae_2wk': mae_2wk,
                    'sbp_err_2wk': sbp_err_2wk,
                    'dbp_err_2wk': dbp_err_2wk,
                    'r2_2wk': r2_2wk,
                    'tau_2wk': tau_2wk,
                    'Ts_2wk': Ts_2wk,
                    # 3WK metrics
                    'rmse_3wk': rmse_3wk,
                    'mae_3wk': mae_3wk,
                    'sbp_err_3wk': sbp_err_3wk,
                    'dbp_err_3wk': dbp_err_3wk,
                    'r2_3wk': r2_3wk,
                    'tau_3wk': tau_3wk,
                    'Ts_3wk': Ts_3wk,
                    'B_3wk': B_3wk
                })
                
                # Collect sample waveform from subject 0 (first 8 beats)
                if sub_idx == 0 and count < 8:
                    sample_real_w.extend(p_real)
                    sample_sim_2wk_w.extend(p_sim_2wk)
                    sample_sim_3wk_w.extend(p_sim_3wk)
                
                count += 1

    df = pd.DataFrame(results)
    
    # Save statistics
    print("\n" + "="*50)
    print("STATISTICAL RESULTS OVER {} BEATS:".format(len(df)))
    print("="*50)
    for model in ['2wk', '3wk']:
        print(f"\nModel: {model.upper()}")
        print(f"  RMSE: {df['rmse_' + model].mean():.3f} +/- {df['rmse_' + model].std():.3f} mmHg")
        print(f"  MAE : {df['mae_' + model].mean():.3f} +/- {df['mae_' + model].std():.3f} mmHg")
        print(f"  R^2 : {df['r2_' + model].mean():.3f} +/- {df['r2_' + model].std():.3f}")
        print(f"  SBP Error: {df['sbp_err_' + model].mean():.3f} +/- {df['sbp_err_' + model].std():.3f} mmHg")
        print(f"  DBP Error: {df['dbp_err_' + model].mean():.3f} +/- {df['dbp_err_' + model].std():.3f} mmHg")
        
    # Generate Plots
    
    # Plot 1: Waveform comparison
    plt.figure(figsize=(12, 5))
    t_w = np.arange(len(sample_real_w)) / fs
    plt.plot(t_w, sample_real_w, 'k-', label='Real ABP', lw=2)
    plt.plot(t_w, sample_sim_2wk_w, 'r--', label='2WK (R-C)', lw=1.5)
    plt.plot(t_w, sample_sim_3wk_w, 'b:', label='3WK (R-C-Zc)', lw=1.5)
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Pressure (mmHg)', fontsize=12)
    plt.title('Arterial Blood Pressure (ABP) Waveform Reconstruction: Windkessel vs. Real', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('waveform_comparison.png', dpi=300)
    plt.close()
    
    # Plot 2: Scatter plot of SBP & DBP
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    # SBP 2WK
    axes[0, 0].scatter(df['real_sbp'], df['real_sbp'] + df['sbp_err_2wk'], color='red', alpha=0.6, edgecolors='k')
    axes[0, 0].plot([df['real_sbp'].min(), df['real_sbp'].max()], [df['real_sbp'].min(), df['real_sbp'].max()], 'k--', lw=2)
    axes[0, 0].set_title('2WK: Systolic Blood Pressure (SBP)', fontsize=12)
    axes[0, 0].set_xlabel('Real SBP (mmHg)')
    axes[0, 0].set_ylabel('Simulated SBP (mmHg)')
    axes[0, 0].grid(True, alpha=0.3)
    
    # SBP 3WK
    axes[0, 1].scatter(df['real_sbp'], df['real_sbp'] + df['sbp_err_3wk'], color='blue', alpha=0.6, edgecolors='k')
    axes[0, 1].plot([df['real_sbp'].min(), df['real_sbp'].max()], [df['real_sbp'].min(), df['real_sbp'].max()], 'k--', lw=2)
    axes[0, 1].set_title('3WK: Systolic Blood Pressure (SBP)', fontsize=12)
    axes[0, 1].set_xlabel('Real SBP (mmHg)')
    axes[0, 1].set_ylabel('Simulated SBP (mmHg)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # DBP 2WK
    axes[1, 0].scatter(df['real_dbp'], df['real_dbp'] + df['dbp_err_2wk'], color='red', alpha=0.6, edgecolors='k')
    axes[1, 0].plot([df['real_dbp'].min(), df['real_dbp'].max()], [df['real_dbp'].min(), df['real_dbp'].max()], 'k--', lw=2)
    axes[1, 0].set_title('2WK: Diastolic Blood Pressure (DBP)', fontsize=12)
    axes[1, 0].set_xlabel('Real DBP (mmHg)')
    axes[1, 0].set_ylabel('Simulated DBP (mmHg)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # DBP 3WK
    axes[1, 1].scatter(df['real_dbp'], df['real_dbp'] + df['dbp_err_3wk'], color='blue', alpha=0.6, edgecolors='k')
    axes[1, 1].plot([df['real_dbp'].min(), df['real_dbp'].max()], [df['real_dbp'].min(), df['real_dbp'].max()], 'k--', lw=2)
    axes[1, 1].set_title('3WK: Diastolic Blood Pressure (DBP)', fontsize=12)
    axes[1, 1].set_xlabel('Real DBP (mmHg)')
    axes[1, 1].set_ylabel('Simulated DBP (mmHg)')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle('Systolic & Diastolic BP Tracking: Real vs. Simulated', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('sbp_dbp_scatter.png', dpi=300)
    plt.close()
    
    # Plot 3: RMSE distribution histogram
    plt.figure(figsize=(10, 5))
    plt.hist(df['rmse_2wk'], bins=20, alpha=0.5, color='red', label='2WK (R-C)', edgecolor='k')
    plt.hist(df['rmse_3wk'], bins=20, alpha=0.5, color='blue', label='3WK (R-C-Zc)', edgecolor='k')
    plt.axvline(df['rmse_2wk'].mean(), color='red', linestyle='dashed', linewidth=2, label=f'2WK Mean: {df["rmse_2wk"].mean():.2f}')
    plt.axvline(df['rmse_3wk'].mean(), color='blue', linestyle='dashed', linewidth=2, label=f'3WK Mean: {df["rmse_3wk"].mean():.2f}')
    plt.xlabel('RMSE (mmHg)', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.title('Distribution of Reconstruction RMSE across 250 Beats', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('rmse_distribution.png', dpi=300)
    plt.close()
    
    print("\nSaved all plots:")
    print("  - waveform_comparison.png")
    print("  - sbp_dbp_scatter.png")
    print("  - rmse_distribution.png")
    
    # Save the dataframe to csv for easy review or reference
    df.to_csv('windkessel_validation_results.csv', index=False)
    print("Saved detailed results to windkessel_validation_results.csv")

if __name__ == '__main__':
    main()
