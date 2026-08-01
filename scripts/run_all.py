"""
End-to-end run: data -> features -> both detectors -> evaluation + scenarios.

Flow (kept coherent so both detectors and the evaluation share ONE fraud
definition -- real background = legitimate, injected attack patterns = fraud):

  1. Load real TalkingData and split chronologically.
  2. Fit feature engineering (conversion priors + scaler) on the TRAIN split only.
  3. Build a supervised TRAIN mix (train period) and VAL mix (val period).
  4. Algorithm 1: supervised XGBoost on explicit fraud labels.
     Algorithm 2: unsupervised AE + iForest on presumed-normal background traffic.
  5. Baseline evaluation on a 'normal' scenario drawn from the TEST period.
  6. Robustness sweep across all 8 scenarios drawn from the TEST period.

Usage (from the repo root, with data/train.csv in place):
    python scripts/run_all.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import config
from src.utils import set_seed
from src.data_loader import load_talkingdata
from src.feature_engineering import FeatureEngineer, FEATURE_NAMES
from src.algorithm1_xgboost import XGBoostDetector
from src.algorithm2_autoencoder import AutoEncoderDetector
from src import evaluation as ev
from src import scenarios as sc


def main():
    set_seed(config.RANDOM_SEED)
    print("=" * 70)
    print("Bot Fraud Detection -- full pipeline")
    print("=" * 70)

    # 1) Data ---------------------------------------------------------------
    split = load_talkingdata()
    print(split.summary())

    # 2) Feature engineering fit on TRAIN only -----------------------------
    fe = FeatureEngineer().fit(split.train)

    # 3) Supervised train / val mixes (explicit fraud labels) --------------
    tr_frame, ytr = sc.build_training_set(split.train, size=min(120_000, 4 * len(split.train)))
    va_frame, yva = sc.build_training_set(split.val, size=min(30_000, 4 * len(split.val)),
                                          seed=config.RANDOM_SEED + 1)
    Xtr = fe.transform(tr_frame).values
    Xva = fe.transform(va_frame).values
    print(f"\nTrain mix {Xtr.shape} (fraud={ytr.mean():.2%}) | "
          f"Val mix {Xva.shape} (fraud={yva.mean():.2%})")

    # 4a) Algorithm 1 -- supervised XGBoost --------------------------------
    print("\n[Algorithm 1] training supervised XGBoost ...")
    a1 = XGBoostDetector().fit(Xtr, ytr, Xva, yva, feature_names=FEATURE_NAMES)
    print(f"  best_iteration = {a1.best_iteration}, threshold = {a1.threshold:.4f}")

    # 4b) Algorithm 2 -- unsupervised AE + iForest on NORMAL background ----
    print("\n[Algorithm 2] training unsupervised autoencoder + Isolation Forest ...")
    X_normal = Xtr[ytr == 0]
    a2 = AutoEncoderDetector().fit(X_normal, X_val=Xva, y_val_fraud=yva)
    print(f"  final recon loss = {a2.train_history[-1]:.4f}, "
          f"threshold = {a2.threshold:.4f}")

    # 5) Baseline evaluation (a 'normal' scenario from the TEST period) -----
    print("\n[Eval] baseline (normal scenario, test period) ...")
    base = sc.build_scenario("normal", split.test)
    Xb = fe.transform(base.frame).values
    yb = base.y_fraud
    s1 = a1.predict_fraud_proba(Xb); p1 = (s1 >= a1.threshold).astype(int)
    s2 = a2.predict_fraud_proba(Xb); p2 = (s2 >= a2.threshold).astype(int)
    m1 = ev.compute_metrics(yb, p1, s1, ev.measure_throughput(a1, Xb))
    m2 = ev.compute_metrics(yb, p2, s2, ev.measure_throughput(a2, Xb))
    print("  Algorithm 1:", m1.as_row())
    print("  Algorithm 2:", m2.as_row())

    ev.plot_roc({"Algorithm 1": (yb, s1), "Algorithm 2": (yb, s2)},
                os.path.join(config.FIGURE_DIR, "roc_baseline.png"))
    ev.plot_pr({"Algorithm 1": (yb, s1), "Algorithm 2": (yb, s2)},
               os.path.join(config.FIGURE_DIR, "pr_baseline.png"))
    ev.plot_confusion(yb, p1, os.path.join(config.FIGURE_DIR, "cm_algo1.png"),
                      "Algorithm 1 -- confusion (baseline)")
    ev.plot_confusion(yb, p2, os.path.join(config.FIGURE_DIR, "cm_algo2.png"),
                      "Algorithm 2 -- confusion (baseline)")
    ev.plot_feature_importance(a1.feature_importances(),
                               os.path.join(config.FIGURE_DIR, "feat_importance.png"))

    # 6) Scenario robustness sweep -----------------------------------------
    print("\n[Eval] scenario sweep (8 scenarios, test period) ...")
    rows = []
    for scen in sc.build_all(split.test):
        Xs = fe.transform(scen.frame).values
        ys = scen.y_fraud
        sa = a1.predict_fraud_proba(Xs); pa = (sa >= a1.threshold).astype(int)
        sb = a2.predict_fraud_proba(Xs); pb = (sb >= a2.threshold).astype(int)
        ma = ev.compute_metrics(ys, pa, sa)
        mb = ev.compute_metrics(ys, pb, sb)
        rows.append({"scenario": scen.name,
                     "P_A1": round(ma.precision, 3), "R_A1": round(ma.recall, 3),
                     "F1_A1": round(ma.f1, 3), "AUC_A1": round(ma.roc_auc, 3),
                     "P_A2": round(mb.precision, 3), "R_A2": round(mb.recall, 3),
                     "F1_A2": round(mb.f1, 3), "AUC_A2": round(mb.roc_auc, 3)})
        print(f"  {scen.name:16s} F1: A1={ma.f1:.3f}  A2={mb.f1:.3f}")

    table = ev.save_table(rows, os.path.join(config.RESULTS_DIR, "scenario_summary.csv"))
    ev.save_table([{"detector": "Algorithm 1 (XGBoost)", **m1.as_row()},
                   {"detector": "Algorithm 2 (AE+iForest)", **m2.as_row()}],
                  os.path.join(config.RESULTS_DIR, "baseline_metrics.csv"))

    print("\nScenario summary table:")
    print(table.to_string(index=False))
    print(f"\nArtifacts written to:\n  {config.FIGURE_DIR}\n  {config.RESULTS_DIR}")
    print("Done.")


if __name__ == "__main__":
    main()
