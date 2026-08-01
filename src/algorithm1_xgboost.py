"""
Algorithm 1 -- Supervised feature-engineered XGBoost detector
(report Section 2.2, Figure 3, Table 3).

The model is trained on an EXPLICIT fraud label (positive class = fraud), built
from real background traffic labelled legitimate plus injected attack patterns
labelled fraud -- the report's "attribution flag augmented by the curated scenario
labels". This keeps the supervised notion of fraud identical to the one used at
evaluation time, and avoids the trivial label leakage that arises if you instead
predict the raw conversion flag.

  * scale_pos_weight = n_neg / n_pos handles class imbalance (report eq. 2.9).
  * Probabilities are isotonically calibrated on the validation slice.
  * SHAP gives per-decision explanations (report 'Feature importance').
  * The operating threshold targets a fixed false-positive rate, reflecting the
    priority of not blocking genuine users.
"""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier

import config
from src.utils import threshold_for_fpr


def _calibrate(base_estimator, X_val, y_val, method):
    """Isotonic/sigmoid calibration that works across sklearn versions."""
    try:  # sklearn >= 1.6 deprecates cv='prefit' in favour of FrozenEstimator
        from sklearn.frozen import FrozenEstimator
        cal = CalibratedClassifierCV(FrozenEstimator(base_estimator), method=method)
    except Exception:
        cal = CalibratedClassifierCV(base_estimator, method=method, cv="prefit")
    cal.fit(X_val, y_val)
    return cal


@dataclass
class XGBoostDetector:
    cfg: config.XGBConfig = None
    model: XGBClassifier = None
    calibrator: object = None
    threshold: float = 0.5            # threshold on the fraud probability
    feature_names: list = None

    def __post_init__(self):
        if self.cfg is None:
            self.cfg = config.XGB

    # ------------------------------- fit -------------------------------- #
    def fit(self, X_train, y_fraud_train, X_val, y_fraud_val, feature_names=None):
        """``y_fraud`` = 1 for fraud, 0 for legitimate."""
        self.feature_names = list(feature_names) if feature_names is not None else None
        y_fraud_train = np.asarray(y_fraud_train).astype(int)
        y_fraud_val = np.asarray(y_fraud_val).astype(int)
        n_pos = int((y_fraud_train == 1).sum())
        n_neg = int((y_fraud_train == 0).sum())
        spw = max(n_neg / max(n_pos, 1), 1e-3)

        self.model = XGBClassifier(
            objective=self.cfg.objective,
            eval_metric=self.cfg.eval_metric,
            max_depth=self.cfg.max_depth,
            learning_rate=self.cfg.learning_rate,
            n_estimators=self.cfg.n_estimators,
            subsample=self.cfg.subsample,
            colsample_bytree=self.cfg.colsample_bytree,
            reg_lambda=self.cfg.reg_lambda,
            min_child_weight=self.cfg.min_child_weight,
            scale_pos_weight=spw,
            tree_method=self.cfg.tree_method,
            n_jobs=self.cfg.n_jobs,
            random_state=config.RANDOM_SEED,
            early_stopping_rounds=self.cfg.early_stopping_rounds,
        )
        self.model.fit(X_train, y_fraud_train,
                       eval_set=[(X_val, y_fraud_val)], verbose=False)

        # Calibrate P(fraud) on validation, then choose the operating threshold.
        self.calibrator = _calibrate(self.model, X_val, y_fraud_val,
                                     self.cfg.calibration)
        scores_val = self.predict_fraud_proba(X_val)
        self.threshold = threshold_for_fpr(y_fraud_val, scores_val,
                                           config.EVAL.target_fpr)
        return self

    # --------------------------- inference ------------------------------ #
    def predict_fraud_proba(self, X) -> np.ndarray:
        """Calibrated P(fraud)."""
        return self.calibrator.predict_proba(X)[:, 1]

    def predict_fraud(self, X) -> np.ndarray:
        return (self.predict_fraud_proba(X) >= self.threshold).astype(int)

    @property
    def best_iteration(self):
        return getattr(self.model, "best_iteration", None)

    # --------------------------- explain -------------------------------- #
    def feature_importances(self) -> dict:
        imp = self.model.feature_importances_
        names = self.feature_names or [f"f{i}" for i in range(len(imp))]
        return dict(sorted(zip(names, imp), key=lambda kv: kv[1], reverse=True))

    def shap_values(self, X, max_rows: int = 2000):
        """SHAP per-feature contributions (report SHAP attribution layer)."""
        import shap
        Xs = X[:max_rows]
        explainer = shap.TreeExplainer(self.model)
        return explainer.shap_values(Xs), Xs
