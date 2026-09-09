# `analysis/` — ML deep-dive

Post-hoc analysis of the Windkessel `R / C / Zc` XGBoost regressor: real
cross-validation, feature-engineering ablation, SHAP explainability, a
physics-vs-ML ablation study, and residual/error analysis.

Full write-up with numbers and figures: [`docs/ml_deep_dive_report.md`](../docs/ml_deep_dive_report.md).

## Run

```bash
python analysis/00_regenerate_dataset.py      # writes data/realistic_windkessel_dataset.csv (~4 min, seed 42, 60k)
python analysis/01_cross_validation.py         # 5-fold CV, MAE/RMSE/R2 mean +/- std
python analysis/02_feature_engineering.py      # raw vs +physics vs +engineered features
python analysis/03_shap_analysis.py            # TreeExplainer, top-5 features per target
python analysis/04_ablation_study.py           # physics-only / ML-only / hybrid
python analysis/05_error_analysis.py           # residuals, worst cases, error by vital strata
```

Raw outputs (JSON / CSV / PNG) land in `analysis/outputs/` (git-ignored).
`common.py` holds the shared feature engineering and the model factory, kept
identical to `models/windkessel/train_windkessel_xgboost.py`.

## Headline results (5-fold CV, 60k synthetic samples)

| Target | MAE | R² |
|---|---|---|
| `R` (systemic vascular resistance) | 0.021 | 0.997 |
| `C` (arterial compliance) | 0.094 | 0.93 |
| `Zc` (characteristic impedance) | 0.018 | 0.57 — not identifiable from cuff vitals |

Ablation: ML cuts MAE **−67 % (R)** / **−22 % (C)** vs the physics-analytic
baseline; injecting the physics-derived features on top of raw vitals adds
almost nothing (ΔR² ≤ 0.002).
