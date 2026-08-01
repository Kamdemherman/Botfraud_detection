"""Train + evaluate ONLY Algorithm 2 (unsupervised AE + iForest).
Usage: python scripts/run_algorithm2.py"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.utils import set_seed
from src.data_loader import load_talkingdata
from src.feature_engineering import FeatureEngineer
from src.algorithm2_autoencoder import AutoEncoderDetector
from src import evaluation as ev
from src import scenarios as sc


def main():
    set_seed(config.RANDOM_SEED)
    split = load_talkingdata(); print(split.summary())
    fe = FeatureEngineer().fit(split.train)
    tr_frame, ytr = sc.build_training_set(split.train, size=min(120_000, 4 * len(split.train)))
    va_frame, yva = sc.build_training_set(split.val, size=min(30_000, 4 * len(split.val)),
                                          seed=config.RANDOM_SEED + 1)
    Xtr = fe.transform(tr_frame).values
    det = AutoEncoderDetector().fit(Xtr[ytr == 0],
                                    X_val=fe.transform(va_frame).values, y_val_fraud=yva)
    base = sc.build_scenario("normal", split.test)
    Xb = fe.transform(base.frame).values
    s = det.predict_fraud_proba(Xb); p = (s >= det.threshold).astype(int)
    m = ev.compute_metrics(base.y_fraud, p, s, ev.measure_throughput(det, Xb))
    print("Algorithm 2 metrics (baseline):", m.as_row())
    print("Final reconstruction loss:", round(det.train_history[-1], 4))


if __name__ == "__main__":
    main()
