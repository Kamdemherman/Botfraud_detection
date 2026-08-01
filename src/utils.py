"""Small shared helpers: seeding, timing, score normalisation, thresholding."""
from __future__ import annotations
import os
import random
import time
from contextlib import contextmanager

import numpy as np


def set_seed(seed: int) -> None:
    """Fix all RNGs we touch so every reported number is reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(False)  # keep cuDNN fast; seeds suffice
    except Exception:
        pass


@contextmanager
def timer(name: str = "block"):
    """Context manager that prints wall-clock time -- used for throughput stats."""
    start = time.perf_counter()
    yield (lambda: time.perf_counter() - start)
    elapsed = time.perf_counter() - start
    print(f"[timer] {name}: {elapsed:.3f}s")


def minmax_normalise(x: np.ndarray, lo: float | None = None,
                     hi: float | None = None) -> np.ndarray:
    """Scale an array to [0, 1]. Used to put recon-error and iForest scores on a
    common footing before fusion (report eq. 14)."""
    x = np.asarray(x, dtype=np.float64)
    lo = np.nanmin(x) if lo is None else lo
    hi = np.nanmax(x) if hi is None else hi
    if hi - lo < 1e-12:
        return np.zeros_like(x)
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0)


def threshold_for_fpr(y_true: np.ndarray, scores: np.ndarray,
                      target_fpr: float) -> float:
    """Pick the score threshold whose false-positive rate is closest to (but not
    above) ``target_fpr``. Reflects the deployment priority of not blocking real
    users (report Section 2.4 'Operating point')."""
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=np.float64)
    neg_scores = np.sort(scores[y_true == 0])
    if len(neg_scores) == 0:
        return float(scores.max())
    # FPR <= target  <=>  threshold at the (1 - target) quantile of negatives.
    idx = int(np.ceil((1.0 - target_fpr) * len(neg_scores))) - 1
    idx = min(max(idx, 0), len(neg_scores) - 1)
    return float(neg_scores[idx]) + 1e-9
