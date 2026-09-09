"""
04 - Ablation study: how much does the ML actually add over pure physics?

Same held-out test set (80/20, seed 42) for every approach.

  (a) physics-only   closed-form analytic estimates, NO learning
        R  = MAP / CO                                  (CO = SV*HR/60)
        C  = t_dia / (R_obs * ln(SBP/DBP))             (log-decay)
        Zc = 0.055 * R_obs                             (mid of the 3-8% prior;
                                                        physics has no closed form)
  (b) ML-only        XGBoost on raw measured vitals only
        HR, SV, SBP, DBP, MAP, PP
  (c) hybrid         XGBoost on raw vitals + the 4 physics-derived features
                     (== production realistic_model.pkl)
  (c2) hybrid-residual  physics gives R,C ; XGBoost learns the residual
                        (true - physics) from raw vitals, Zc falls back to ML

Reports MAE / RMSE / R2 per target and the marginal deltas.
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import common as C


def physics_predict(df):
    co = df["SV"] * df["HR"] / 60.0
    r = df["MAP_Obs"] / co
    cycle = 60.0 / df["HR"]
    t_dia = cycle - df["Sys_Duration"]
    c = t_dia / (r * np.log(df["Sys_Obs"] / df["Dia_Obs"]) + 1e-6)
    zc = 0.055 * r
    return np.column_stack([r, c, zc])


def fit_ml(feats, Xtr, Xte, Ytr):
    sc = StandardScaler().fit(Xtr[feats])
    model = C.make_model()
    model.fit(sc.transform(Xtr[feats]), Ytr.to_numpy())
    return model.predict(sc.transform(Xte[feats]))


def report(name, y_true, y_pred, store):
    print(f"\n[{name}]")
    store[name] = {}
    for i, t in enumerate(C.TARGETS):
        m = C.metrics(y_true[:, i], y_pred[:, i])
        store[name][t] = m
        print(f"   {t:<8} MAE {m['MAE']:.4f}  RMSE {m['RMSE']:.4f}  R2 {m['R2']:.4f}")


def main():
    df = C.add_physics_features(C.load_raw())
    tr, te = train_test_split(df, test_size=0.2, random_state=42)
    Ytr = tr[C.TARGETS]
    Yte = te[C.TARGETS].to_numpy()

    store = {}

    # (a) physics only
    report("a_physics_only", Yte, physics_predict(te), store)

    # (b) ML only - raw vitals
    pred_b = fit_ml(C.RAW_FEATURES, tr, te, Ytr)
    report("b_ml_only_raw", Yte, pred_b, store)

    # (c) hybrid == production
    pred_c = fit_ml(C.PROD_FEATURES, tr, te, Ytr)
    report("c_hybrid_production", Yte, pred_c, store)

    # (c2) hybrid residual: ML corrects the physics estimate
    phys_tr = physics_predict(tr)
    phys_te = physics_predict(te)
    resid_tr = Ytr.to_numpy() - phys_tr
    sc = StandardScaler().fit(tr[C.RAW_FEATURES])
    mres = C.make_model()
    mres.fit(sc.transform(tr[C.RAW_FEATURES]), resid_tr)
    pred_c2 = phys_te + mres.predict(sc.transform(te[C.RAW_FEATURES]))
    report("c2_hybrid_residual", Yte, pred_c2, store)

    # marginal gains (MAE reduction, R2 gain)
    def delta(a, b):
        return {
            t: {
                "dMAE": store[b][t]["MAE"] - store[a][t]["MAE"],
                "dR2": store[b][t]["R2"] - store[a][t]["R2"],
            }
            for t in C.TARGETS
        }

    gains = {
        "ml_only_vs_physics": delta("a_physics_only", "b_ml_only_raw"),
        "hybrid_vs_ml_only": delta("b_ml_only_raw", "c_hybrid_production"),
        "hybrid_vs_physics": delta("a_physics_only", "c_hybrid_production"),
        "residual_vs_physics": delta("a_physics_only", "c2_hybrid_residual"),
    }
    print("\n=== MARGINAL GAINS (negative dMAE = better, positive dR2 = better) ===")
    for k, d in gains.items():
        print(f" {k}")
        for t in C.TARGETS:
            print(f"    {t:<8} dMAE {d[t]['dMAE']:+.4f}   dR2 {d[t]['dR2']:+.4f}")

    # plot: R2 grouped bar
    approaches = ["a_physics_only", "b_ml_only_raw", "c_hybrid_production", "c2_hybrid_residual"]
    labels = ["physics-only", "ML-only", "hybrid (prod)", "hybrid-residual"]
    x = np.arange(len(C.TARGETS))
    w = 0.2
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    for j, a in enumerate(approaches):
        ax1.bar(x + (j - 1.5) * w, [store[a][t]["R2"] for t in C.TARGETS], w, label=labels[j])
        ax2.bar(x + (j - 1.5) * w, [store[a][t]["MAE"] for t in C.TARGETS], w, label=labels[j])
    for ax, ttl, yl in [(ax1, "R2 by approach", "R2"), (ax2, "MAE by approach", "MAE")]:
        ax.set_xticks(x)
        ax.set_xticklabels(C.TARGETS)
        ax.set_title(ttl)
        ax.set_ylabel(yl)
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{C.OUT_DIR}/04_ablation.png", dpi=130)
    plt.close()

    with open(f"{C.OUT_DIR}/04_ablation.json", "w") as f:
        json.dump({"metrics": store, "gains": gains}, f, indent=2)
    print(f"\nwrote {C.OUT_DIR}/04_ablation.json + 04_ablation.png")


if __name__ == "__main__":
    main()
