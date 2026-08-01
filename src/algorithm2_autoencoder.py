"""
Algorithm 2 -- Unsupervised autoencoder + Isolation Forest detector
(report Section 2.3, Figure 7, Table 4).

Pipeline:
  1. A symmetric deep autoencoder (38-24-12-8-12-24-38, ReLU) is trained with Adam
     on NORMAL (converting) traffic only -- it never sees fraud labels.
  2. At scoring time each event yields a reconstruction error e = ||x - x_hat||^2
     and a latent code z = f(x).
  3. An Isolation Forest scores z (anomalies isolate with short path lengths).
  4. The two signals are min-max normalised and fused:
         s = alpha * norm(e) + (1 - alpha) * iforest_anomaly      (report eq. 14)
  5. An adaptive threshold is periodically re-calibrated from recent low-score
     (presumed normal) traffic so the detector tracks concept drift.

Fraud score = the fused anomaly score (higher = more anomalous = more fraud-like).
"""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import IsolationForest

import config
from src.utils import minmax_normalise, threshold_for_fpr


# ----------------------------- the network ------------------------------ #
class _AutoEncoder(nn.Module):
    def __init__(self, n_features: int, enc_layers, latent_dim: int):
        super().__init__()
        dims = [n_features, *enc_layers, latent_dim]
        enc = []
        for a, b in zip(dims[:-1], dims[1:]):
            enc += [nn.Linear(a, b), nn.ReLU()]
        self.encoder = nn.Sequential(*enc[:-1])  # no ReLU on the latent code
        dec = []
        rdims = dims[::-1]
        for a, b in zip(rdims[:-1], rdims[1:]):
            dec += [nn.Linear(a, b), nn.ReLU()]
        self.decoder = nn.Sequential(*dec[:-1])  # no ReLU on the output layer

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z


@dataclass
class AutoEncoderDetector:
    cfg: config.AEConfig = None
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    model: _AutoEncoder = None
    iforest: IsolationForest = None
    threshold: float = None
    _recon_lo: float = 0.0
    _recon_hi: float = 1.0
    _ifa_lo: float = 0.0
    _ifa_hi: float = 1.0
    _batches_seen: int = 0

    def __post_init__(self):
        if self.cfg is None:
            self.cfg = config.AE

    # ------------------------------- fit -------------------------------- #
    def fit(self, X_normal: np.ndarray, X_val=None, y_val_fraud=None):
        """Train on NORMAL traffic only. Optionally use a labelled validation set
        to choose the fused-score operating threshold for a target FPR."""
        X_normal = np.asarray(X_normal, dtype=np.float32)
        torch.manual_seed(config.RANDOM_SEED)
        self.model = _AutoEncoder(X_normal.shape[1], self.cfg.encoder_layers,
                                  self.cfg.latent_dim).to(self.device)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.cfg.learning_rate)
        loss_fn = nn.MSELoss()

        data = torch.tensor(X_normal, device=self.device)
        n = len(data)
        history = []
        self.model.train()
        for epoch in range(self.cfg.epochs):
            perm = torch.randperm(n, device=self.device)
            epoch_loss = 0.0
            for i in range(0, n, self.cfg.batch_size):
                idx = perm[i:i + self.cfg.batch_size]
                xb = data[idx]
                opt.zero_grad()
                xhat, _ = self.model(xb)
                loss = loss_fn(xhat, xb)
                loss.backward()
                opt.step()
                epoch_loss += loss.item() * len(xb)
            history.append(epoch_loss / n)

        # Fit Isolation Forest on the latent codes of normal traffic.
        z_norm = self._encode(X_normal)
        self.iforest = IsolationForest(
            n_estimators=self.cfg.iforest_trees,
            max_samples=min(self.cfg.iforest_subsample, len(z_norm)),
            random_state=config.RANDOM_SEED,
        ).fit(z_norm)

        # Establish normalisation bounds from normal traffic.
        recon = self._recon_error(X_normal)
        ifa = -self.iforest.decision_function(z_norm)  # higher = more anomalous
        self._recon_lo, self._recon_hi = float(recon.min()), float(np.quantile(recon, 0.999))
        self._ifa_lo, self._ifa_hi = float(ifa.min()), float(ifa.max())

        # Choose the fused-score threshold.
        if X_val is not None and y_val_fraud is not None:
            s_val = self.score(np.asarray(X_val, dtype=np.float32))
            self.threshold = threshold_for_fpr(y_val_fraud, s_val, config.EVAL.target_fpr)
        else:
            s_norm = self._fuse(recon, ifa)
            self.threshold = float(np.quantile(s_norm, 1.0 - config.EVAL.target_fpr))
        self.train_history = history
        return self

    # --------------------------- internals ------------------------------ #
    def _encode(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            x = torch.tensor(np.asarray(X, dtype=np.float32), device=self.device)
            _, z = self.model(x)
        return z.cpu().numpy()

    def _recon_error(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            x = torch.tensor(np.asarray(X, dtype=np.float32), device=self.device)
            xhat, _ = self.model(x)
            err = ((xhat - x) ** 2).mean(dim=1)
        return err.cpu().numpy()

    def _fuse(self, recon: np.ndarray, ifa: np.ndarray) -> np.ndarray:
        nr = minmax_normalise(recon, self._recon_lo, self._recon_hi)
        ni = minmax_normalise(ifa, self._ifa_lo, self._ifa_hi)
        a = self.cfg.fusion_alpha
        return a * nr + (1.0 - a) * ni

    # --------------------------- inference ------------------------------ #
    def score(self, X: np.ndarray) -> np.ndarray:
        """Fused anomaly score (= fraud score)."""
        X = np.asarray(X, dtype=np.float32)
        recon = self._recon_error(X)
        z = self._encode(X)
        ifa = -self.iforest.decision_function(z)
        return self._fuse(recon, ifa)

    def predict_fraud_proba(self, X) -> np.ndarray:
        return self.score(X)

    def predict_fraud(self, X) -> np.ndarray:
        if self.threshold is None:
            raise RuntimeError("Detector not fitted / threshold unset.")
        return (self.score(X) >= self.threshold).astype(int)

    def reconstruction_error(self, X) -> np.ndarray:
        return self._recon_error(np.asarray(X, dtype=np.float32))

    # ---------------------- drift re-calibration ------------------------ #
    def maybe_recalibrate(self, X_recent: np.ndarray) -> bool:
        """Call once per processed batch. Every ``recalibration_period`` batches it
        refits the Isolation Forest on recent presumed-normal latent codes and
        refreshes the threshold (report 'Adaptive thresholding and drift
        re-calibration'). Returns True when a recalibration happened."""
        self._batches_seen += 1
        if self._batches_seen % self.cfg.recalibration_period != 0:
            return False
        s = self.score(X_recent)
        normal = X_recent[s < self.threshold]            # keep presumed-normal events
        if len(normal) >= self.cfg.iforest_subsample:
            z = self._encode(normal)
            self.iforest = IsolationForest(
                n_estimators=self.cfg.iforest_trees,
                max_samples=min(self.cfg.iforest_subsample, len(z)),
                random_state=config.RANDOM_SEED,
            ).fit(z)
            ifa = -self.iforest.decision_function(z)
            self._ifa_lo, self._ifa_hi = float(ifa.min()), float(ifa.max())
            self.threshold = float(np.quantile(self.score(normal),
                                               1.0 - config.EVAL.target_fpr))
        return True
