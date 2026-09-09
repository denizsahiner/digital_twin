"""Shared helpers for the ML deep-dive analysis.

Reproduces the feature engineering and model configuration from
`models/windkessel/train_windkessel_xgboost.py` so every analysis script
(CV, SHAP, ablation, error analysis) uses an identical pipeline.
"""
import os
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.multioutput import MultiOutputRegressor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA_PATH = os.path.join(ROOT, "data", "realistic_windkessel_dataset.csv")
OUT_DIR = os.path.join(HERE, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

TARGETS = ["R_True", "C_True", "Zc_True"]

# raw measured vitals available to a smartwatch / cuff monitor
RAW_FEATURES = ["HR", "SV", "Sys_Obs", "Dia_Obs", "MAP_Obs", "PP_Obs"]

# physics-derived features already used by the production model
PHYSICS_FEATURES = ["Shape_Index", "Stiffness_Obs", "R_Obs", "C_Physics"]

# exact feature list of the production model realistic_model.pkl
PROD_FEATURES = RAW_FEATURES + PHYSICS_FEATURES


def load_raw():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"{DATA_PATH} missing - run analysis/00_regenerate_dataset.py first"
        )
    return pd.read_csv(DATA_PATH)


def add_physics_features(df):
    """The 4 physics-derived features from the production training script."""
    df = df.copy()
    df["Stiffness_Obs"] = df["PP_Obs"] / df["SV"]
    df["R_Obs"] = df["MAP_Obs"] / ((df["SV"] * df["HR"]) / 60.0)
    df["Shape_Index"] = (df["MAP_Obs"] - df["Dia_Obs"]) / df["PP_Obs"]
    cycle_time = 60.0 / df["HR"]
    dia_duration = cycle_time - df["Sys_Duration"]
    df["C_Physics"] = dia_duration / (
        df["R_Obs"] * np.log(df["Sys_Obs"] / df["Dia_Obs"]) + 1e-6
    )
    return df


def add_engineered_features(df):
    """New candidate features tested in 02_feature_engineering.py.

    All are derivable from the same measured vitals (HR, SV, SBP, DBP, MAP, PP,
    systolic duration) - nothing that needs an invasive catheter.
    """
    df = df.copy()
    cycle_time = 60.0 / df["HR"]
    dia_duration = cycle_time - df["Sys_Duration"]
    co = df["SV"] * df["HR"] / 60.0

    # 1. Mean systolic ejection rate  ~ dP/dt proxy: pulse pressure built per
    #    second of ejection. Higher -> stiffer aorta / higher Zc.
    df["Ejection_Rate"] = df["PP_Obs"] / df["Sys_Duration"]

    # 2. Diastolic decay rate proxy: fraction of pulse pressure lost per second
    #    of diastole. Governed by the RC time constant -> informs C.
    df["Dia_Decay_Rate"] = (df["Sys_Obs"] - df["Dia_Obs"]) / dia_duration

    # 3. Windkessel time constant tau estimated from the diastolic pressure
    #    ratio decay:  tau = t_dia / ln(Psys/Pdia). Direct handle on R*C.
    df["Tau_Est"] = dia_duration / np.log(df["Sys_Obs"] / df["Dia_Obs"] + 1e-9)

    # 4. Systolic fraction of the cardiac cycle (duty cycle). Shorter systole at
    #    a given HR shifts the pressure morphology.
    df["Systolic_Fraction"] = df["Sys_Duration"] / cycle_time

    # 5. Form factor: how far MAP sits between DBP and SBP. Pure waveform shape,
    #    sensitive to wave reflection / Zc.
    df["Form_Factor"] = (df["MAP_Obs"] - df["Dia_Obs"]) / (
        df["Sys_Obs"] - df["Dia_Obs"]
    )

    # 6. Peak systolic flow / pulse pressure ~ input impedance proxy for Zc.
    q_max = (df["SV"] * np.pi) / (2.0 * df["Sys_Duration"])
    df["Impedance_Proxy"] = df["PP_Obs"] / q_max

    # 7. Cardiac output normalised by MAP == 1/R_Obs, but kept as CO for the
    #    model to combine non-linearly with the rest.
    df["CO"] = co

    return df


ENGINEERED_FEATURES = [
    "Ejection_Rate",
    "Dia_Decay_Rate",
    "Tau_Est",
    "Systolic_Fraction",
    "Form_Factor",
    "Impedance_Proxy",
    "CO",
]


def make_model():
    """Identical hyper-parameters to the production training script."""
    return MultiOutputRegressor(
        xgb.XGBRegressor(
            n_estimators=1000,
            learning_rate=0.02,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=-1,
            random_state=42,
        )
    )


def make_single_model():
    return xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.02,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=-1,
        random_state=42,
    )


def metrics(y_true, y_pred):
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": r2_score(y_true, y_pred),
    }
