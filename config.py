"""
Central configuration for the Bot Fraud Detection project.

All tunable knobs live here so the pipeline is reproducible from a single place.
Hyperparameters mirror the mid-term report (Table 3 for XGBoost, Table 4 for the
autoencoder + Isolation Forest).
"""
from __future__ import annotations
import os
from dataclasses import dataclass, field

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")          # put TalkingData train.csv here
ARTIFACT_DIR = os.path.join(PROJECT_ROOT, "artifacts")  # cached models / features
FIGURE_DIR = os.path.join(PROJECT_ROOT, "figures")      # generated plots
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")     # metric tables (csv)

for _d in (ARTIFACT_DIR, FIGURE_DIR, RESULTS_DIR):
    os.makedirs(_d, exist_ok=True)

# Expected raw file (download from Kaggle: TalkingData AdTracking Fraud Detection)
TALKINGDATA_TRAIN = os.path.join(DATA_DIR, "train.csv")

# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
RANDOM_SEED = 42

# --------------------------------------------------------------------------- #
# Data loading / sampling
# --------------------------------------------------------------------------- #
@dataclass
class DataConfig:
    # The full corpus is ~184M rows. For development use a sample; set to None
    # to read everything (needs lots of RAM). The report used ~5M stratified rows.
    sample_rows: int | None = 5_000_000
    # Read the file in chunks of this size when sampling, to bound memory.
    chunksize: int = 2_000_000
    # Temporal split: TalkingData spans 4 calendar days (2017-11-06 .. 2017-11-09).
    # First 3 days -> train/val, last day -> test  (report Section 2.1).
    test_day: int = 9          # day-of-month used as the held-out test day
    val_fraction: float = 0.15  # fraction of the train portion held out for validation
    # Raw dtypes keep memory low on the big CSV.
    dtypes: dict = field(default_factory=lambda: {
        "ip": "uint32", "app": "uint16", "device": "uint16",
        "os": "uint16", "channel": "uint16", "is_attributed": "uint8",
    })


# --------------------------------------------------------------------------- #
# Feature engineering
# --------------------------------------------------------------------------- #
@dataclass
class FeatureConfig:
    # Window (seconds) used for "clicks per IP in window" / burst features.
    click_window_seconds: int = 3600       # 1 hour
    burst_window_seconds: int = 60         # 1 minute
    # Laplace smoothing for conversion-rate features.
    conversion_smoothing: float = 1.0
    # A click is "datacenter/proxy"-flagged if its IP fans out to more than this
    # many distinct device/os combos (proxy heuristic; real IPs are anonymised,
    # so this is a stand-in for an external reputation feed -- see README).
    datacenter_fanout_threshold: int = 50


# --------------------------------------------------------------------------- #
# Algorithm 1 -- XGBoost  (report Table 3)
# --------------------------------------------------------------------------- #
@dataclass
class XGBConfig:
    objective: str = "binary:logistic"
    eval_metric: str = "auc"
    max_depth: int = 8
    learning_rate: float = 0.05
    n_estimators: int = 600
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_lambda: float = 1.0
    min_child_weight: int = 5
    early_stopping_rounds: int = 40
    # scale_pos_weight is computed at fit time as n_neg / n_pos.
    tree_method: str = "hist"
    n_jobs: int = -1
    # Probability calibration method: "isotonic" (report) or "sigmoid".
    calibration: str = "isotonic"


# --------------------------------------------------------------------------- #
# Algorithm 2 -- Autoencoder + Isolation Forest  (report Table 4)
# --------------------------------------------------------------------------- #
@dataclass
class AEConfig:
    # 38 -> 24 -> 12 -> 8 -> 12 -> 24 -> 38
    encoder_layers: tuple = (24, 12)
    latent_dim: int = 8
    activation: str = "relu"
    optimizer: str = "adam"
    learning_rate: float = 1e-3
    epochs: int = 50
    batch_size: int = 256
    # Reconstruction-error decision threshold (report tau = 0.028). If None, it is
    # chosen automatically from the validation reconstruction-error distribution.
    recon_threshold: float | None = 0.028
    # Isolation Forest on the latent code.
    iforest_trees: int = 200
    iforest_subsample: int = 256
    # Fused score:  s = alpha * norm(recon_err) + (1 - alpha) * iforest_score
    fusion_alpha: float = 0.6
    # Drift re-calibration: refresh threshold / refit iForest every K batches.
    recalibration_period: int = 200


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
@dataclass
class EvalConfig:
    # Target false-positive rate used to pick the operating threshold
    # (deployment priority: do not block genuine users -- report Section 2.4).
    target_fpr: float = 0.02
    # Curated scenario set (report Table 1): 8 scenarios, ~50k rows, ~30% fraud.
    scenario_size: int = 50_000
    scenario_fraud_rate: float = 0.30


# Singletons imported elsewhere.
DATA = DataConfig()
FEATURES = FeatureConfig()
XGB = XGBConfig()
AE = AEConfig()
EVAL = EvalConfig()

# The canonical 38-feature schema is defined in feature_engineering.py
N_FEATURES = 38
