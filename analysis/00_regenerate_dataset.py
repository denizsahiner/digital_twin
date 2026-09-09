"""
00 - Regenerate the Windkessel ML training dataset.

The training script `models/windkessel/train_windkessel_xgboost.py` expects
`data/realistic_windkessel_dataset.csv`, which is not committed to the repo.
This script reproduces it EXACTLY from the project's own generator
(`data_pipeline/generators/generate_dataset_3.py`): 3-element Windkessel ODE
forward simulation, seed=42, 60k physiologically-filtered samples, realistic
measurement noise on the observed vitals.

This only writes data/realistic_windkessel_dataset.csv if it is missing
(use --force to overwrite).
"""
import argparse
import os
import numpy as np
import pandas as pd
from scipy.integrate import odeint

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "data", "realistic_windkessel_dataset.csv")

NUM_SAMPLES = 60000
DT = 0.005
NUM_BEATS = 10


def generate_inflow(t, hr, sv, systole_duration):
    cycle_duration = 60.0 / hr
    local_t = t % cycle_duration
    if local_t < systole_duration:
        q_max = (sv * np.pi) / (2.0 * systole_duration)
        return q_max * np.sin(np.pi * local_t / systole_duration)
    return 0.0


def wk3_derivative(p, t, r_total, c_total, z_c, hr, sv, systole_duration):
    r_peripheral = r_total - z_c
    if r_peripheral <= 1e-6:
        r_peripheral = 1e-6
    q_in = generate_inflow(t, hr, sv, systole_duration)
    return (q_in / c_total) - (p / (r_peripheral * c_total))


def build(num_samples=NUM_SAMPLES):
    np.random.seed(42)
    data = []
    done = 0
    while len(data) < num_samples:
        HR = np.random.randint(50, 130)
        SV = np.random.uniform(40.0, 120.0)
        R_true = np.random.uniform(0.6, 2.5)
        target_tau = np.random.uniform(1.0, 2.2)
        C_true = target_tau / R_true
        Zc_true = np.random.uniform(0.03, 0.08) * R_true

        cycle_len = 60.0 / HR
        systole_duration = np.clip(0.35 * np.sqrt(cycle_len), 0.18, 0.45)

        t_eval = np.arange(0.0, NUM_BEATS * cycle_len, DT)
        CO = SV * HR / 60.0
        p0 = [R_true * CO]
        p_vals = odeint(
            wk3_derivative, y0=p0, t=t_eval,
            args=(R_true, C_true, Zc_true, HR, SV, systole_duration),
        ).flatten()

        samples_per_beat = int(cycle_len / DT)
        last_cycle = p_vals[-samples_per_beat:]
        real_sys = np.max(last_cycle)
        real_dia = np.min(last_cycle)
        real_map = np.mean(last_cycle)

        obs_sys = real_sys + np.random.normal(0, 3.0)
        obs_dia = real_dia + np.random.normal(0, 2.0)
        obs_map = real_map + np.random.normal(0, 1.5)

        if (80 < obs_sys < 210) and (40 < obs_dia < 130) and (obs_sys - obs_dia > 20):
            obs_pp = obs_sys - obs_dia
            data.append([
                R_true, C_true, Zc_true,
                HR, SV, obs_sys, obs_dia, obs_map, obs_pp, systole_duration,
            ])
            done += 1
            if done % 5000 == 0:
                print(f"  {done}/{num_samples}")

    cols = ["R_True", "C_True", "Zc_True",
            "HR", "SV", "Sys_Obs", "Dia_Obs", "MAP_Obs", "PP_Obs", "Sys_Duration"]
    return pd.DataFrame(data, columns=cols)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--n", type=int, default=NUM_SAMPLES)
    args = ap.parse_args()

    if os.path.exists(OUT) and not args.force:
        print(f"exists, skipping: {OUT}")
    else:
        print(f"generating {args.n} samples ...")
        df = build(args.n)
        df.to_csv(OUT, index=False)
        print(f"wrote {OUT}  shape={df.shape}")
        print(df.describe().round(3).to_string())
