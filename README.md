# Bot Fraud Detection in Mobile Advertising

A modular implementation of the two detectors from the thesis *Bot Fraud Detection
in Mobile Advertising*, built on the real **TalkingData AdTracking** dataset:

* **Algorithm 1 — Supervised:** feature-engineered **XGBoost** with class-imbalance
  handling, isotonic probability calibration, SHAP explanations, and an
  FPR-targeted operating threshold.
* **Algorithm 2 — Unsupervised:** a deep **autoencoder + Isolation Forest** that
  learns "normal" traffic, fuses reconstruction error with a latent-space anomaly
  score, and re-calibrates to track concept drift.

Both detectors share one 38-feature, 4-view representation (temporal, network/IP,
device, conversion) and are compared on a baseline test set and across eight
attack scenarios.

---

## 1. Repository layout

```
botfraud-detection/
├── config.py                  # all paths + hyperparameters (report Tables 3 & 4)
├── requirements.txt
├── data/                      # <-- put TalkingData train.csv here (you download it)
├── src/
│   ├── data_loader.py         # read real CSVs, chronological train/val/test split
│   ├── feature_engineering.py # the shared 38-feature, 4-view pipeline
│   ├── algorithm1_xgboost.py  # supervised detector
│   ├── algorithm2_autoencoder.py  # unsupervised detector (PyTorch AE + iForest)
│   ├── scenarios.py           # 8 curated attack scenarios + training-set builder
│   ├── evaluation.py          # metrics, ROC/PR/confusion plots, tables
│   └── utils.py               # seeding, timing, normalisation, thresholding
├── scripts/
│   ├── run_all.py             # full pipeline end to end
│   ├── run_algorithm1.py      # train + evaluate Algorithm 1 only
│   └── run_algorithm2.py      # train + evaluate Algorithm 2 only
├── tests/
│   └── smoke_test.py          # generates a tiny schema-matching CSV; verifies the pipeline
├── figures/                   # generated plots
└── results/                   # generated metric tables (csv)
```

---

## 2. Setup

```bash
python -m venv .venv && source .venv/bin/activate      # optional
pip install -r requirements.txt
```

Tested with Python 3.12. PyTorch runs on CPU or GPU automatically; a GPU (e.g.
your RTX-class card) speeds up Algorithm 2 but is not required.

---

## 3. Get the data

This project reads the **real** TalkingData corpus; it is not bundled (≈7 GB,
~184 M rows).

1. Download from Kaggle: *TalkingData AdTracking Fraud Detection Challenge*.
2. Place `train.csv` at: `botfraud-detection/data/train.csv`.

Raw schema: `ip, app, device, os, channel, click_time, attributed_time, is_attributed`.

By default the loader draws a reproducible ~5 M-row sample across the file
(`config.DATA.sample_rows`) so it runs on a normal machine. Set it to `None` to
use the whole corpus.

---

## 4. Run

```bash
python scripts/run_all.py          # data -> features -> both detectors -> evaluation
python scripts/run_algorithm1.py   # supervised only
python scripts/run_algorithm2.py   # unsupervised only
```

Outputs: metric tables in `results/`, plots in `figures/`
(`roc_baseline.png`, `pr_baseline.png`, `cm_algo1.png`, `cm_algo2.png`,
`feat_importance.png`).

**Verify the install without the download:**

```bash
python tests/smoke_test.py
```

This writes a tiny CSV with the real TalkingData schema (and deliberately
learnable signal) and runs the full pipeline. It is a correctness check only,
not a results reproduction.

---

