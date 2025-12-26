# renthop-lightautoml-vs-custom

**Kaggle competition:** Two Sigma Connect: Rental Listing Inquiries (RentHop)

This repository contains a mini-research project for the **LightAutoML (LAMA)** course assignment:

1) **LAMA baseline** (≥2 different configurations, pick the best)  
2) **Alternative solution without LAMA** (several pipelines + feature engineering + hyperparameter tuning)

---

## Project structure

```
.
├── data/
│   ├── raw/                 # Kaggle downloaded files: train.json, test.json, sample_submission.csv
│   └── processed/           # Cached feature tables
├── notebooks/
│   ├── 00_setup_and_data.ipynb
│   ├── 01_eda.ipynb
│   ├── 02_features.ipynb
│   ├── 03_lama_baselines.ipynb
│   ├── 04_custom_models.ipynb
│   └── 05_ensemble_and_submit.ipynb
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── io.py
│   ├── features/
│   │   ├── __init__.py
│   │   ├── build.py
│   │   └── text.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── cv.py
│   │   ├── lama.py
│   │   └── custom.py
│   └── utils/
│       ├── __init__.py
│       ├── logging.py
│       └── seed.py
├── artifacts/
│   ├── models/
│   ├── oof/
│   └── submissions/
├── requirements.txt
└── .gitignore
```

---

## Quickstart

### 1) Install deps

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

### 2) Download Kaggle data

Option A (recommended): use the **Kaggle CLI** (see `notebooks/00_setup_and_data.ipynb`).

You should end with:

```
data/raw/train.json
data/raw/test.json
data/raw/sample_submission.csv
```

### 3) Run notebooks in order

1. `00_setup_and_data.ipynb` — environment & data sanity checks  
2. `01_eda.ipynb` — target & feature EDA (plots, tables, hypotheses)  
3. `02_features.ipynb` — feature engineering + cached dataset  
4. `03_lama_baselines.ipynb` — **two LAMA baselines** + comparison  
5. `04_custom_models.ipynb` — **non-LAMA pipelines** + Optuna tuning  
6. `05_ensemble_and_submit.ipynb` — blending + final submission

---

## Validation strategy (leakage-aware)

We use **time-aware holdout** as the primary validation:  
- sort by `created` timestamp  
- train on earlier listings, validate on later listings  

This mimics a real-world “future listings” scenario and prevents accidental temporal leakage.
We additionally provide a **StratifiedKFold** CV utility for robustness checks.

---

## Notes

- All notebooks are written to be **reproducible**, with fixed seeds and artifacts saved to `./artifacts/`.
- The custom solution includes multiple pipelines (tabular GBDT, TF‑IDF+linear, tuned model, and optional blend).
