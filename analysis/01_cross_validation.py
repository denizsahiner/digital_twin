"""
01 - Real 5-fold cross-validation for the Windkessel R/C/Zc model.

The production script does a single 80/20 split. Here we run KFold(5, shuffled)
with the scaler re-fit inside every fold (no leakage) and report the per-fold
distribution (mean +/- std) of MAE, RMSE and R2 for each target parameter.
"""
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

import common as C

N_SPLITS = 5


def main():
    df = C.add_physics_features(C.load_raw())
    X = df[C.PROD_FEATURES].to_numpy()
    Y = df[C.TARGETS].to_numpy()
    print(f"dataset: {X.shape[0]} rows, {X.shape[1]} features -> {C.TARGETS}")

    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    rows = []
    for fold, (tr, te) in enumerate(kf.split(X), 1):
        sc = StandardScaler().fit(X[tr])
        model = C.make_model()
        model.fit(sc.transform(X[tr]), Y[tr])
        pred = model.predict(sc.transform(X[te]))
        for i, t in enumerate(C.TARGETS):
            m = C.metrics(Y[te, i], pred[:, i])
            m.update(fold=fold, target=t)
            rows.append(m)
        print(f"  fold {fold} done")

    res = pd.DataFrame(rows)
    res.to_csv(f"{C.OUT_DIR}/01_cv_per_fold.csv", index=False)

    summary = (
        res.groupby("target")[["MAE", "RMSE", "R2"]]
        .agg(["mean", "std"])
        .reindex(C.TARGETS)
    )
    print("\n=== 5-FOLD CV SUMMARY (mean +/- std across folds) ===")
    lines = []
    for t in C.TARGETS:
        s = summary.loc[t]
        line = (
            f"{t:<8} | MAE {s[('MAE','mean')]:.4f} +/- {s[('MAE','std')]:.4f} "
            f"| RMSE {s[('RMSE','mean')]:.4f} +/- {s[('RMSE','std')]:.4f} "
            f"| R2 {s[('R2','mean')]:.4f} +/- {s[('R2','std')]:.4f}"
        )
        print(line)
        lines.append(line)

    out = {
        "n_splits": N_SPLITS,
        "n_rows": int(X.shape[0]),
        "features": C.PROD_FEATURES,
        "per_target": {
            t: {
                "MAE_mean": float(summary.loc[t, ("MAE", "mean")]),
                "MAE_std": float(summary.loc[t, ("MAE", "std")]),
                "RMSE_mean": float(summary.loc[t, ("RMSE", "mean")]),
                "RMSE_std": float(summary.loc[t, ("RMSE", "std")]),
                "R2_mean": float(summary.loc[t, ("R2", "mean")]),
                "R2_std": float(summary.loc[t, ("R2", "std")]),
            }
            for t in C.TARGETS
        },
        "text": lines,
    }
    with open(f"{C.OUT_DIR}/01_cv_summary.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {C.OUT_DIR}/01_cv_per_fold.csv and 01_cv_summary.json")


if __name__ == "__main__":
    main()
