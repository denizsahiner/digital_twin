import h5py
import numpy as np

import os as _os
mat_path = _os.environ.get(
    'MIMIC_MAT_PATH',
    _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), 'data', 'mimic', 'Part_3.mat'),
)
print("Loading v7.3 mat file using h5py...")
try:
    with h5py.File(mat_path, 'r') as f:
        print("Keys in mat file:", list(f.keys()))
        for key in f.keys():
            val = f[key]
            print(f"Key: {key}, Type: {type(val)}, Shape: {val.shape if hasattr(val, 'shape') else 'N/A'}")
            # If it's a dataset, print some info about it
            if isinstance(val, h5py.Dataset):
                print(f"  Dataset dtype: {val.dtype}")
                # Print a small subset or info
                if len(val.shape) > 0:
                    print(f"  First few elements: {val[0] if val.shape[0] > 0 else 'empty'}")
except Exception as e:
    print("Error:", e)
