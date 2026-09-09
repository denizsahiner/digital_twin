import h5py
import numpy as np

import os as _os
mat_path = _os.environ.get(
    'MIMIC_MAT_PATH',
    _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), 'data', 'mimic', 'Part_3.mat'),
)
print("Dereferencing and checking columns...")
try:
    with h5py.File(mat_path, 'r') as f:
        part3 = f['Part_3']
        ref = part3[0, 0]
        data = np.array(f[ref]) # shape is (5000, 3) or (3, 5000)? Let's check.
        print(f"Data shape: {data.shape}")
        
        # If shape is (N, 3), we have 3 columns. If it's (3, N), we have 3 rows.
        # Let's print stats for each dimension
        if data.shape[0] == 3 or data.shape[1] == 3:
            # Let's find which dimension has size 3
            dim_3 = 0 if data.shape[0] == 3 else 1
            n_signals = 3
            for i in range(n_signals):
                if dim_3 == 0:
                    sig = data[i, :]
                else:
                    sig = data[:, i]
                print(f"Signal {i}: shape={sig.shape}, min={np.min(sig):.3f}, max={np.max(sig):.3f}, mean={np.mean(sig):.3f}, std={np.std(sig):.3f}")
except Exception as e:
    print("Error:", e)
