"""
02 - Feature engineering review + impact test.

Current model inputs (from models/windkessel/train_windkessel_xgboost.py):
  raw measured vitals : HR, SV, Sys_Obs, Dia_Obs, MAP_Obs, PP_Obs
  physics-derived      : Shape_Index, Stiffness_Obs, R_Obs, C_Physics

We add 7 new clinically-motivated derived features (see common.add_engineered_features)
and measure their effect with 3-fold CV on three configurations:
  A  raw only
  B  raw + physics  (== production model)
  C  raw + physics + engineered
Plus a leave-one-in test: raw+physics + each engineered feature alone.
"""
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

import common as C

N_SPLITS = 3


def cv_eval(df, feats):
    X = df[feats].to_numpy()
    Y = df[C.TARGETS].to_numpy()
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    acc = {t: {"MAE": [], "R2": []} for t in C.TARGETS}
    for tr, te in kf.split(X):
        sc = StandardScaler().fit(X[tr])
        model = C.make_model()
        model.fit(sc.transform(X[tr]), Y[tr])
        pred = model.predict(sc.transform(X[te]))
        for i, t in enumerate(C.TARGETS):
            m = C.metrics(Y[te, i], pred[:, i])
            acc[t]["MAE"].append(m["MAE"])
            acc[t]["R2"].append(m["R2"])
    return {
        t: {
            "MAE": float(np.mean(acc[t]["MAE"])),
            "MAE_std": float(np.std(acc[t]["MAE"])),
            "R2": float(np.mean(acc[t]["R2"])),
            "R2_std": float(np.std(acc[t]["R2"])),
        }
        for t in C.TARGETS
    }


def fmt(name, r):
    return " | ".join(
        f"{t} MAE {r[t]['MAE']:.4f} R2 {r[t]['R2']:.4f}" for t in C.TARGETS
    )


def main():
    df = C.add_engineered_features(C.add_physics_features(C.load_raw()))

    print("RAW features    :", C.RAW_FEATURES)
    print("PHYSICS features :", C.PHYSICS_FEATURES)
    print("NEW engineered   :", C.ENGINEERED_FEATURES)
    print(f"\n{N_SPLITS}-fold CV, XGBoost (production hyper-params)\n")

    configs = {
        "A_raw_only": C.RAW_FEATURES,
        "B_raw_physics(production)": C.PROD_FEATURES,
        "C_raw_physics_engineered": C.PROD_FEATURES + C.ENGINEERED_FEATURES,
    }
    results = {}
    for name, feats in configs.items():
        r = cv_eval(df, feats)
        results[name] = r
        print(f"[{name}]  ({len(feats)} feats)")
        for t in C.TARGETS:
            print(
                f"   {t:<8} MAE {r[t]['MAE']:.4f} +/- {r[t]['MAE_std']:.4f}"
                f"   R2 {r[t]['R2']:.4f} +/- {r[t]['R2_std']:.4f}"
            )

    # leave-one-in: production set + a single engineered feature
    print("\n--- production + ONE engineered feature (delta vs production) ---")
    base = results["B_raw_physics(production)"]
    loi = {}
    for f in C.ENGINEERED_FEATURES:
        r = cv_eval(df, C.PROD_FEATURES + [f])
        loi[f] = r
        deltas = " ".join(
            f"{t} dMAE {r[t]['MAE']-base[t]['MAE']:+.4f} dR2 {r[t]['R2']-base[t]['R2']:+.4f}"
            for t in C.TARGETS
        )
        print(f"  +{f:<18} {deltas}")

    # full engineered set delta
    print("\n--- FULL engineered set vs production (delta) ---")
    full = results["C_raw_physics_engineered"]
    for t in C.TARGETS:
        print(
            f"  {t:<8} dMAE {full[t]['MAE']-base[t]['MAE']:+.4f}"
            f"   dR2 {full[t]['R2']-base[t]['R2']:+.4f}"
        )

    with open(f"{C.OUT_DIR}/02_feature_engineering.json", "w") as fh:
        json.dump({"configs": results, "leave_one_in": loi}, fh, indent=2)
    print(f"\nwrote {C.OUT_DIR}/02_feature_engineering.json")


if __name__ == "__main__":
    main()
