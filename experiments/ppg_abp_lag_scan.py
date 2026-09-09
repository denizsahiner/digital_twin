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

def test():
    with h5py.File(mat_path, 'r') as f:
        part3 = f['Part_3']
        ref = part3[0, 0]
        data = np.array(f[ref])
        
        ppg = data[:1250, 0]
        abp = data[:1250, 1]
        
        ppg_std = (ppg - np.mean(ppg)) / np.std(ppg)
        abp_std = (abp - np.mean(abp)) / np.std(abp)
        
        # Test lags from -100 to 100 samples
        lags = np.arange(-100, 101)
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
            
            n = min(len(p), len(a))
            corrs.append(np.mean(p[:n] * a[:n]))
            
        corrs = np.array(corrs)
        
        # Print top 5 local minima and maxima
        min_indices = sig.find_peaks(-corrs)[0]
        max_indices = sig.find_peaks(corrs)[0]
        
        print("Local Minima (Negative Correlation Peaks):")
        for idx in min_indices:
            print(f"  Lag: {lags[idx]} samples ({lags[idx]/fs:.3f} s), Corr: {corrs[idx]:.3f}")
            
        print("Local Maxima (Positive Correlation Peaks):")
        for idx in max_indices:
            print(f"  Lag: {lags[idx]} samples ({lags[idx]/fs:.3f} s), Corr: {corrs[idx]:.3f}")
            
if __name__ == '__main__':
    test()
