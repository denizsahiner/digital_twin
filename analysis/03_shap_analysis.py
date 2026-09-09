"""
03 - SHAP explainability for the Windkessel R/C/Zc model.

Trains one XGBoost regressor per target on the production feature set, then uses
shap.TreeExplainer to rank feature contributions. Outputs:
  - top-5 features per target (mean |SHAP|)  -> console + json
  - one SHAP summary (beeswarm) plot per target -> outputs/03_shap_<target>.png
  - combined bar plot                           -> outputs/03_shap_bar_all.png
"""
import json
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import common as C

warnings.filterwarnings("ignore")


def main():
    df = C.add_physics_features(C.load_raw())
    feats = C.PROD_FEATURES
    X = df[feats]
    Y = df[C.TARGETS]

    Xtr, Xte, Ytr, Yte = train_test_split(X, Y, test_size=0.2, random_state=42)
    sc = StandardScaler().fit(Xtr)
    Xtr_s = sc.transform(Xtr)
    Xte_s = sc.transform(Xte)

    # background / explain sample for speed
    rng = np.random.default_rng(42)
    idx = rng.choice(len(Xte_s), size=min(4000, len(Xte_s)), replace=False)
    Xexp = Xte_s[idx]

    top5 = {}
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    for k, t in enumerate(C.TARGETS):
        model = C.make_single_model()
        model.fit(Xtr_s, Ytr[t].to_numpy())

        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(Xexp)
        mean_abs = np.abs(sv).mean(axis=0)
        order = np.argsort(mean_abs)[::-1]
        ranked = [(feats[i], float(mean_abs[i])) for i in order]
        top5[t] = ranked[:5]
        print(f"\n=== {t}: top-5 features by mean|SHAP| ===")
        for name, val in ranked[:5]:
            print(f"   {name:<16} {val:.5f}")

        # beeswarm
        plt.sca(axes[k])
        shap.summary_plot(
            sv, features=Xexp, feature_names=feats, show=False, plot_size=None
        )
        axes[k].set_title(f"SHAP summary - {t}")

    plt.tight_layout()
    plt.savefig(f"{C.OUT_DIR}/03_shap_summary_all.png", dpi=130)
    plt.close()

    # combined bar
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for k, t in enumerate(C.TARGETS):
        names = [n for n, _ in top5[t]][::-1]
        vals = [v for _, v in top5[t]][::-1]
        axes[k].barh(names, vals, color="#4C72B0")
        axes[k].set_title(f"{t}: top-5 mean|SHAP|")
        axes[k].set_xlabel("mean |SHAP value|")
    plt.tight_layout()
    plt.savefig(f"{C.OUT_DIR}/03_shap_bar_all.png", dpi=130)
    plt.close()

    with open(f"{C.OUT_DIR}/03_shap_top5.json", "w") as f:
        json.dump(top5, f, indent=2)
    print(f"\nwrote {C.OUT_DIR}/03_shap_top5.json + 03_shap_summary_all.png + 03_shap_bar_all.png")


if __name__ == "__main__":
    main()
