# Cardiovascular Digital Twin

A physics-informed digital twin of the arterial system. It estimates non-invasive
haemodynamic parameters from cuff / wearable vitals, projects how they drift under
lifestyle changes over a 1-year horizon, and serves it all through a small Flask web app.

Two ML models sit on top of a **3-element Windkessel** (`R`–`C`–`Zc`) ODE model of arterial pressure:

| Model | Task | Inputs | Outputs |
|---|---|---|---|
| **Windkessel regressor** (`models/windkessel/`) | XGBoost multi-output regression | HR, SV, SBP, DBP, MAP, PP (+ physics-derived features) | `R` (systemic vascular resistance), `C` (arterial compliance), `Zc` (characteristic impedance) |
| **Cardio risk classifier** (`models/cardio/`) | XGBoost binary classifier | age, sex, height, weight, SBP, DBP, smoking, alcohol, glucose, BMI, PP | 10-year cardiovascular risk probability |

A PPG→ABP self-calibration endpoint reconstructs an arterial waveform from a photoplethysmogram
and fits the 3-element Windkessel per beat (`app.py` → `/api/calibrate-ppg`).

## Results (Windkessel regressor)

5-fold cross-validation on 60k synthetic ODE samples:

| Target | MAE | R² |
|---|---|---|
| `R` | 0.021 mmHg·s/mL | **0.997** |
| `C` | 0.094 mL/mmHg | **0.93** |
| `Zc` | 0.018 mmHg·s/mL | 0.57 — not identifiable from cuff-only vitals |

An ablation study shows the ML cuts prediction error **−67 % (R)** and **−22 % (C)** versus a
pure physics-analytic baseline. Full analysis: [`docs/ml_deep_dive_report.md`](docs/ml_deep_dive_report.md).

## Repository layout

```
app.py                     Flask app (predict / simulate / calibrate-PPG)   :5001
web/                       templates + static assets
models/
  cardio/                  train_cardio.py  -> cardio_model_nolabs.pkl
  windkessel/              train_windkessel_xgboost.py -> realistic_model.pkl
                           predict_mimic_r_c_z.py  (apply model to MIMIC cohort)
data_pipeline/
  extract_cardio.py        Kaggle cardio dataset -> feature table
  extract_mimic.py         MIMIC-IV demo -> merged vitals/labs/habits table
  generators/              3-element Windkessel ODE synthetic-data generators
simulation/
  time_simulator.py        standalone lifestyle -> R/C/Zc projection prototype
  windkessel_fit.py        fit 2WK / 3WK to real ABP beats
  windkessel_validation.py 2WK vs 3WK reconstruction error over many beats
analysis/                  ML deep-dive: CV, SHAP, ablation, error analysis (see analysis/README.md)
experiments/               throwaway probes: PPG/ABP lag, calibration, .mat inspection
docs/                      reports + figures
data/                      git-ignored — see data/README.md for how to populate
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Datasets and trained `.pkl` files are **not** in the repo (third-party data / regenerable
artifacts). See [`data/README.md`](data/README.md).

## Reproduce the pipeline

```bash
# 1. synthetic Windkessel training data (3-element ODE, seed 42, 60k samples, ~4 min)
python data_pipeline/generators/generate_dataset_3.py

# 2. train the Windkessel R/C/Zc regressor
python models/windkessel/train_windkessel_xgboost.py

# 3. cardio risk model (needs data/cardio_train.csv from Kaggle)
python data_pipeline/extract_cardio.py
python models/cardio/train_cardio.py

# 4. run the deep-dive analysis (writes analysis/outputs/)
python analysis/01_cross_validation.py
python analysis/04_ablation_study.py
```

## Run the web app

```bash
python app.py            # http://localhost:5001
```

The `/api/calibrate-ppg` route additionally needs a MIMIC cuffless-BP `.mat` file:

```bash
export MIMIC_MAT_PATH=/abs/path/to/Part_3.mat
```

## Modelling assumptions & limits

- **Synthetic training data.** The Windkessel regressor is trained on ODE-simulated
  beats with added measurement noise — no real patient labels. Cross-validation
  variance is near zero because the data is homogeneous, not because generalization
  is proven.
- **`Zc` is not identifiable** from cuff SBP/DBP/MAP (R² ≈ 0.57 across physics, ML and
  hybrid). Reliable `Zc` needs pulse-wave-contour / PPG features.
- **1-year horizon.** Lifestyle projections are capped at 365 days; the transition
  rates are heuristic bridge assumptions from population-level hazard ratios, not
  individually validated.
- **Cardio dataset bias.** The Kaggle cardio data under-represents smoking/alcohol
  risk; see `docs/project_evaluation_report.md`.

## Data sources

- Kaggle — Cardiovascular Disease dataset
- Kaggle — Cuff-Less Blood Pressure Estimation (`Part_3.mat`)
- PhysioNet — MIMIC-IV Clinical Database Demo

Links and exact placement in [`data/README.md`](data/README.md).
