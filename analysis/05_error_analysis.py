"""
05 - Error / residual analysis for the production hybrid model.

Train on 80%, analyse residuals on the 20% hold-out:
  - residual distribution + bias per target
  - the 20 worst-predicted test samples per target (dumped to csv)
  - error stratified by physiological bins: HR, MAP, PP, SV, R magnitude, tau=R*C
    -> is error concentrated in a sub-group or spread evenly?
  - calibration scatter (pred vs true) coloured by abs error

The synthetic cohort has no age / disease label, so stratification is by
vital-sign range only (documented limitation).
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import common as C


def strat_table(df, err_col, by, bins):
    cats = pd.cut(df[by], bins=bins)
    g = df.groupby(cats, observed=True)[err_col].agg(["mean", "std", "count"])
    return g


def main():
    df = C.add_physics_features(C.load_raw())
    df["tau"] = df["R_True"] * df["C_True"]
    feats = C.PROD_FEATURES

    tr_idx, te_idx = train_test_split(
        np.arange(len(df)), test_size=0.2, random_state=42
    )
    tr, te = df.iloc[tr_idx], df.iloc[te_idx].copy()

    sc = StandardScaler().fit(tr[feats])
    model = C.make_model()
    model.fit(sc.transform(tr[feats]), tr[C.TARGETS].to_numpy())
    pred = model.predict(sc.transform(te[feats]))

    summary = {}
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for i, t in enumerate(C.TARGETS):
        res = pred[:, i] - te[t].to_numpy()          # signed residual
        ae = np.abs(res)
        te[f"res_{t}"] = res
        te[f"ae_{t}"] = ae
        summary[t] = {
            "residual_mean_bias": float(res.mean()),
            "residual_std": float(res.std()),
            "MAE": float(ae.mean()),
            "p90_abs_err": float(np.percentile(ae, 90)),
            "p99_abs_err": float(np.percentile(ae, 99)),
            "max_abs_err": float(ae.max()),
        }
        print(f"\n=== {t} ===")
        for k, v in summary[t].items():
            print(f"   {k:<20} {v:.5f}")

        # worst 20
        worst = te.assign(abs_err=ae).nlargest(20, "abs_err")[
            ["HR", "SV", "Sys_Obs", "Dia_Obs", "MAP_Obs", "PP_Obs",
             "R_True", "C_True", "Zc_True", "tau", "abs_err"]
        ]
        worst.to_csv(f"{C.OUT_DIR}/05_worst20_{t}.csv", index=False)

        # stratified error
        print(f"   -- MAE of {t} by physiological bin --")
        strata = {}
        for by, bins in [
            ("HR", [50, 60, 70, 80, 90, 100, 130]),
            ("MAP_Obs", [50, 80, 95, 110, 125, 175]),
            ("PP_Obs", [15, 30, 40, 50, 65, 125]),
            ("SV", [40, 60, 80, 100, 120]),
            ("R_True", [0.6, 1.0, 1.4, 1.8, 2.5]),
            ("tau", [0.3, 1.0, 1.4, 1.8, 4.0]),
        ]:
            g = strat_table(te, f"ae_{t}", by, bins)
            strata[by] = {str(k): {"mean": float(v["mean"]), "count": int(v["count"])}
                          for k, v in g.iterrows()}
            spread = g["mean"].max() / g["mean"].min() if g["mean"].min() > 0 else float("nan")
            print(f"      {by:<9} bin-MAE range {g['mean'].min():.4f}..{g['mean'].max():.4f}"
                  f"  (x{spread:.2f})")
        summary[t]["strata"] = strata

        # plots
        ax = axes[0, i]
        scx = ax.scatter(te[t], pred[:, i], c=ae, cmap="viridis", s=6, alpha=0.5)
        lo, hi = te[t].min(), te[t].max()
        ax.plot([lo, hi], [lo, hi], "r--", lw=1)
        ax.set_title(f"{t}: pred vs true")
        ax.set_xlabel("true")
        ax.set_ylabel("pred")
        fig.colorbar(scx, ax=ax, label="|err|")

        ax2 = axes[1, i]
        ax2.hist(res, bins=60, color="#55A868")
        ax2.axvline(0, color="k", lw=1)
        ax2.axvline(res.mean(), color="r", ls="--", lw=1, label=f"bias {res.mean():+.4f}")
        ax2.set_title(f"{t}: residual (pred-true)")
        ax2.legend()

    plt.tight_layout()
    plt.savefig(f"{C.OUT_DIR}/05_error_analysis.png", dpi=130)
    plt.close()

    # correlation of abs error with vitals (are errors random or structured?)
    corr = {}
    for t in C.TARGETS:
        c = te[[f"ae_{t}", "HR", "SV", "MAP_Obs", "PP_Obs", "R_True", "C_True",
                "Zc_True", "tau"]].corr()[f"ae_{t}"].drop(f"ae_{t}")
        corr[t] = {k: float(v) for k, v in c.items()}
        print(f"\ncorr(|err_{t}|, vitals): " +
              ", ".join(f"{k} {v:+.3f}" for k, v in corr[t].items()))
    summary["abs_err_vs_vital_correlation"] = corr

    with open(f"{C.OUT_DIR}/05_error_analysis.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nwrote {C.OUT_DIR}/05_error_analysis.json + 05_error_analysis.png + 05_worst20_*.csv")


if __name__ == "__main__":
    main()
