"""Train + evaluate ONLY Algorithm 1 (supervised XGBoost).
Usage: python scripts/run_algorithm1.py"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.utils import set_seed
from src.data_loader import load_talkingdata
from src.feature_engineering import FeatureEngineer, FEATURE_NAMES
from src.algorithm1_xgboost import XGBoostDetector
from src import evaluation as ev
from src import scenarios as sc


def main():
    set_seed(config.RANDOM_SEED)
    split = load_talkingdata(); print(split.summary())
    fe = FeatureEngineer().fit(split.train)
    tr_frame, ytr = sc.build_training_set(split.train, size=min(120_000, 4 * len(split.train)))
    va_frame, yva = sc.build_training_set(split.val, size=min(30_000, 4 * len(split.val)),
                                          seed=config.RANDOM_SEED + 1)
    det = XGBoostDetector().fit(fe.transform(tr_frame).values, ytr,
                                fe.transform(va_frame).values, yva,
                                feature_names=FEATURE_NAMES)
    base = sc.build_scenario("normal", split.test)
    Xb = fe.transform(base.frame).values
    s = det.predict_fraud_proba(Xb); p = (s >= det.threshold).astype(int)
    m = ev.compute_metrics(base.y_fraud, p, s, ev.measure_throughput(det, Xb))
    print("Algorithm 1 metrics (baseline):", m.as_row())
    print("Top features:", list(det.feature_importances().items())[:8])


if __name__ == "__main__":
    main()
