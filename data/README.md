# `data/`

All data files are **git-ignored** (raw datasets are third-party / not redistributable).
Put the files below here to reproduce the pipeline.

| File | Source | Used by |
|---|---|---|
| `cardio_train.csv` | Kaggle — [Cardiovascular Disease dataset](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset) (`;`-separated) | `data_pipeline/extract_cardio.py` |
| `cardio_train.sample.csv` | **committed** — first 100 rows of the above, for smoke-testing the pipeline without the full download | `data_pipeline/extract_cardio.py` |
| `mimic-iv-clinical-database-demo-2.2/` | PhysioNet — [MIMIC-IV Clinical Database Demo](https://physionet.org/content/mimic-iv-demo/) | `data_pipeline/extract_mimic.py` |
| `mimic/Part_3.mat` | Kaggle — [Cuff-Less Blood Pressure Estimation](https://www.kaggle.com/datasets/mkachuee/BloodPressureDataset) (`Part_3.mat`) | `app.py` calibrate-PPG endpoint, `simulation/windkessel_*.py`, `experiments/*` |

### Derived files (created by the pipeline, also ignored)

```
data/cardio_train_feature.csv                    <- python data_pipeline/extract_cardio.py
data/mimic_all_filtered.csv                      <- python data_pipeline/extract_mimic.py
data/realistic_windkessel_dataset.csv            <- python data_pipeline/generators/generate_dataset_3.py
                                                    (or: python analysis/00_regenerate_dataset.py)
data/mimic_hemodynamic_predictions_filtered.csv  <- python models/windkessel/predict_mimic_r_c_z.py
```

### Path overrides

The MIMIC inputs can live anywhere via environment variables:

```bash
export MIMIC_MAT_PATH=/abs/path/to/Part_3.mat
export MIMIC_IV_DEMO_DIR=/abs/path/to/mimic-iv-clinical-database-demo-2.2
```
