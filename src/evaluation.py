"""
Evaluation harness (report Section 2.4).

Six metrics: precision, recall, F1, ROC-AUC, false-positive rate, throughput.
Plus ROC / precision-recall curves and confusion matrices, and a scenario-by-
scenario comparison table for the two detectors.
"""
from __future__ import annotations
import os
import time
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix,
                             roc_curve, precision_recall_curve)

import config


@dataclass
class Metrics:
    precision: float
    recall: float
    f1: float
    roc_auc: float
    fpr: float
    throughput: float  # records / second

    def as_row(self) -> dict:
        return {k: round(v, 4) for k, v in asdict(self).items()}


def compute_metrics(y_true, y_pred, scores, throughput=float("nan")) -> Metrics:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr = fp / max(fp + tn, 1)
    try:
        auc = roc_auc_score(y_true, scores)
    except ValueError:
        auc = float("nan")
    return Metrics(
        precision=precision_score(y_true, y_pred, zero_division=0),
        recall=recall_score(y_true, y_pred, zero_division=0),
        f1=f1_score(y_true, y_pred, zero_division=0),
        roc_auc=auc, fpr=fpr, throughput=throughput,
    )


def measure_throughput(detector, X, repeats: int = 1) -> float:
    """Records per second for batched scoring (excludes training)."""
    n = len(X)
    start = time.perf_counter()
    for _ in range(repeats):
        detector.predict_fraud_proba(X)
    elapsed = time.perf_counter() - start
    return (n * repeats) / max(elapsed, 1e-9)


# ------------------------------- plots ---------------------------------- #
def plot_roc(curves: dict, path: str, title="ROC curves (baseline test set)"):
    plt.figure(figsize=(6, 5))
    for name, (y, s) in curves.items():
        fpr, tpr, _ = roc_curve(y, s)
        auc = roc_auc_score(y, s)
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], "--", color="grey", label="Random")
    plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
    plt.title(title); plt.legend(loc="lower right"); plt.tight_layout()
    plt.savefig(path, dpi=130); plt.close()


def plot_pr(curves: dict, path: str, title="Precision-Recall (baseline test set)"):
    plt.figure(figsize=(6, 5))
    for name, (y, s) in curves.items():
        prec, rec, _ = precision_recall_curve(y, s)
        plt.plot(rec, prec, label=name)
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title(title); plt.legend(loc="lower left"); plt.tight_layout()
    plt.savefig(path, dpi=130); plt.close()


def plot_confusion(y_true, y_pred, path, title="Confusion matrix"):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    plt.figure(figsize=(4.5, 4))
    plt.imshow(cm, cmap="Blues")
    plt.xticks([0, 1], ["Pred Legit", "Pred Fraud"])
    plt.yticks([0, 1], ["True Legit", "True Fraud"])
    total = cm.sum()
    for i in range(2):
        for j in range(2):
            plt.text(j, i, f"{cm[i, j]:,}\n({100*cm[i, j]/total:.1f}%)",
                     ha="center", va="center")
    plt.title(title); plt.tight_layout(); plt.savefig(path, dpi=130); plt.close()


def plot_feature_importance(importances: dict, path: str, top_k: int = 12):
    items = list(importances.items())[:top_k][::-1]
    names = [k for k, _ in items]; vals = [v for _, v in items]
    plt.figure(figsize=(7, 5))
    plt.barh(names, vals)
    plt.xlabel("Relative importance (XGBoost gain)")
    plt.title("Algorithm 1 -- top feature importances")
    plt.tight_layout(); plt.savefig(path, dpi=130); plt.close()


def save_table(rows: list[dict], path: str) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    return df
