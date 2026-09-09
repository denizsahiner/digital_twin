import h5py
import numpy as np
import scipy.optimize as opt
import scipy.signal as sig
import matplotlib.pyplot as plt

# Load Part_3 mat file
import os as _os
mat_path = _os.environ.get(
    'MIMIC_MAT_PATH',
    _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), 'data', 'mimic', 'Part_3.mat'),
)
fs = 125.0  # Sampling frequency of MIMIC-II dataset is 125 Hz

def get_abp_signal():
    with h5py.File(mat_path, 'r') as f:
        part3 = f['Part_3']
        ref = part3[0, 0] # load the first record/cell
        data = np.array(f[ref])
        # Signal 0: PPG, Signal 1: ABP, Signal 2: ECG
        abp = data[:, 1]
        return abp

def detect_beat_onsets(abp):
    # Find local minima as beat onsets
    # We can use scipy.signal.find_peaks on negative ABP.
    # To avoid noise, we can apply a low-pass filter or use height/distance parameters.
    # A typical beat is at least 0.4s (50 samples at 125Hz) and at most 1.5s (187 samples)
    minima, _ = sig.find_peaks(-abp, distance=50, prominence=5.0)
    return minima

def sim_2wk(t, p_start, tau, A, Ts):
    """
    t: array of time points within the beat (starting from 0)
    p_start: initial pressure P(0)
    tau: RC time constant
    A: Q0 / C
    Ts: systolic ejection time
    """
    omega = np.pi / Ts
    p_sim = np.zeros_like(t)
    
    systole_mask = t < Ts
    diastole_mask = ~systole_mask
    
    # Systole analytical solution
    t_sys = t[systole_mask]
    p_sim[systole_mask] = p_start * np.exp(-t_sys/tau) + \
        (A / ((1.0/tau)**2 + omega**2)) * ((1.0/tau)*np.sin(omega*t_sys) - omega*np.cos(omega*t_sys) + omega*np.exp(-t_sys/tau))
    
    # Pressure at the end of systole
    p_Ts = p_start * np.exp(-Ts/tau) + \
        (A * omega / ((1.0/tau)**2 + omega**2)) * (np.exp(-Ts/tau) + 1.0)
    
    # Diastole analytical solution
    t_dia = t[diastole_mask]
    p_sim[diastole_mask] = p_Ts * np.exp(-(t_dia - Ts)/tau)
    
    return p_sim

def sim_3wk(t, p_start, tau, A, Ts, B):
    """
    t: array of time points within the beat
    p_start: initial pressure P(0)
    tau: RC time constant
    A: Q0 / C
    Ts: systolic ejection time
    B: Zc * Q0
    """
    omega = np.pi / Ts
    p_sim = np.zeros_like(t)
    
    systole_mask = t < Ts
    diastole_mask = ~systole_mask
    
    # P_w is the pressure across C (satisfies 2WK DE)
    # The initial condition for P_w is also p_start since Q(0) = 0
    t_sys = t[systole_mask]
    p_w_sys = p_start * np.exp(-t_sys/tau) + \
        (A / ((1.0/tau)**2 + omega**2)) * ((1.0/tau)*np.sin(omega*t_sys) - omega*np.cos(omega*t_sys) + omega*np.exp(-t_sys/tau))
    
    # Total pressure during systole: P = P_w + Zc * Q
    # Q(t) = Q0 * sin(omega * t), so Zc * Q = B * sin(omega * t)
    p_sim[systole_mask] = p_w_sys + B * np.sin(omega * t_sys)
    
    # Pressure at end of systole (Q(Ts) = 0, so P(Ts) = P_w(Ts))
    p_w_Ts = p_start * np.exp(-Ts/tau) + \
        (A * omega / ((1.0/tau)**2 + omega**2)) * (np.exp(-Ts/tau) + 1.0)
    
    # Diastole: Q = 0, so P = P_w
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
    abp = get_abp_signal()
    onsets = detect_beat_onsets(abp)
    print(f"Detected {len(onsets)} beats in the record of length {len(abp)}.")
    
    # We will fit the first 10 beats to see the performance
    num_beats_to_fit = min(10, len(onsets) - 1)
    
    plt.figure(figsize=(12, 8))
    
    all_real = []
    all_sim_2wk = []
    all_sim_3wk = []
    
    for i in range(num_beats_to_fit):
        idx_start = onsets[i]
        idx_end = onsets[i+1]
        
        p_real = abp[idx_start:idx_end]
        n_samples = len(p_real)
        t = np.arange(n_samples) / fs
        T_beat = t[-1]
        
        # Initial guess and bounds for 2WK
        # tau = RC, A = Q0/C, Ts
        init_2wk = [1.2, 100.0, 0.3 * T_beat]
        bounds_2wk = [(0.1, 5.0), (1.0, 1000.0), (0.05, 0.7 * T_beat)]
        
        res_2wk = opt.minimize(loss_2wk, init_2wk, args=(t, p_real), bounds=bounds_2wk, method='L-BFGS-B')
        tau_2wk, A_2wk, Ts_2wk = res_2wk.x
        p_sim_2wk = sim_2wk(t, p_real[0], tau_2wk, A_2wk, Ts_2wk)
        
        # Initial guess and bounds for 3WK
        init_3wk = [1.2, 100.0, 0.3 * T_beat, 5.0]
        bounds_3wk = [(0.1, 5.0), (1.0, 1000.0), (0.05, 0.7 * T_beat), (0.0, 50.0)]
        
        res_3wk = opt.minimize(loss_3wk, init_3wk, args=(t, p_real), bounds=bounds_3wk, method='L-BFGS-B')
        tau_3wk, A_3wk, Ts_3wk, B_3wk = res_3wk.x
        p_sim_3wk = sim_3wk(t, p_real[0], tau_3wk, A_3wk, Ts_3wk, B_3wk)
        
        all_real.extend(p_real)
        all_sim_2wk.extend(p_sim_2wk)
        all_sim_3wk.extend(p_sim_3wk)
        
        print(f"Beat {i+1}: Duration={T_beat:.3f}s, Samples={n_samples}")
        print(f"  2WK Fit: tau={tau_2wk:.3f}s, A={A_2wk:.2f}, Ts={Ts_2wk:.3f}s, RMSE={np.sqrt(res_2wk.fun):.3f} mmHg")
        print(f"  3WK Fit: tau={tau_3wk:.3f}s, A={A_3wk:.2f}, Ts={Ts_3wk:.3f}s, B={B_3wk:.2f}, RMSE={np.sqrt(res_3wk.fun):.3f} mmHg")

    # Plot comparisons
    all_real = np.array(all_real)
    all_sim_2wk = np.array(all_sim_2wk)
    all_sim_3wk = np.array(all_sim_3wk)
    t_all = np.arange(len(all_real)) / fs
    
    plt.plot(t_all, all_real, 'k-', label='Real ABP', lw=2)
    plt.plot(t_all, all_sim_2wk, 'r--', label='2-element Windkessel (2WK)', lw=1.5)
    plt.plot(t_all, all_sim_3wk, 'b:', label='3-element Windkessel (3WK)', lw=1.5)
    plt.xlabel('Time (s)')
    plt.ylabel('Pressure (mmHg)')
    plt.title('Windkessel Model Fitting & Forward Simulation vs. Real ABP')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('windkessel_fit_test.png', dpi=300)
    print("Saved comparison plot to windkessel_fit_test.png")

if __name__ == '__main__':
    main()
