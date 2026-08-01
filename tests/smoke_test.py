"""
Smoke test -- verifies the pipeline runs end to end.

It synthesises a small file with the *real TalkingData schema*
(ip, app, device, os, channel, click_time, attributed_time, is_attributed),
writes it to data/train.csv, then runs the full pipeline. This is a correctness
check only; real experiments use the actual Kaggle corpus.

Run:  python tests/smoke_test.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

import config


def make_tiny_csv(path: str, n: int = 40_000, seed: int = 0):
    rng = np.random.RandomState(seed)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    days = rng.choice([6, 7, 8, 9], size=n, p=[0.3, 0.3, 0.25, 0.15])
    secs = rng.randint(0, 86400, size=n)
    click = (pd.to_datetime("2017-11-01")
             + pd.to_timedelta(days, unit="D")
             + pd.to_timedelta(secs, unit="s"))
    df = pd.DataFrame({
        "ip": rng.randint(1, 5000, n).astype("uint32"),
        "app": rng.randint(1, 300, n).astype("uint16"),
        "device": rng.randint(1, 200, n).astype("uint16"),
        "os": rng.randint(1, 60, n).astype("uint16"),
        "channel": rng.randint(1, 200, n).astype("uint16"),
        "click_time": click,
    })
    # Inject *learnable* structure so the supervised model has real signal:
    #   - clicks on low-numbered channels convert much more often (genuine),
    #   - a handful of high-volume "bot" IPs almost never convert.
    bot_ips = set(rng.choice(np.arange(1, 5000), size=40, replace=False))
    is_bot_ip = df["ip"].isin(bot_ips).values
    df.loc[is_bot_ip, "ip"] = rng.choice(list(bot_ips), size=int(is_bot_ip.sum()))
    base_p = np.where(df["channel"].values < 30, 0.30, 0.03)
    base_p = np.where(is_bot_ip, 0.002, base_p)
    convert = rng.rand(n) < base_p
    att = pd.Series(pd.NaT, index=df.index)
    att[convert] = df.loc[convert, "click_time"] + pd.to_timedelta(
        rng.randint(30, 3600, convert.sum()), unit="s")
    df["attributed_time"] = att
    df["is_attributed"] = convert.astype("uint8")
    df = df.sort_values("click_time").reset_index(drop=True)
    df.to_csv(path, index=False)
    print(f"wrote {len(df):,} rows -> {path} "
          f"({int(df['is_attributed'].sum())} converting)")


def main():
    # Write the tiny CSV into the artifacts folder to avoid overwriting
    # any real `data/train.csv` (permission issues on some systems).
    tmp = os.path.join(config.ARTIFACT_DIR, "tiny_train.csv")
    make_tiny_csv(tmp)
    # Point the loader at our temporary file for the duration of the test.
    config.TALKINGDATA_TRAIN = tmp
    # keep the smoke test fast / in-memory
    config.DATA.sample_rows = None
    config.AE.epochs = 8
    config.XGB.n_estimators = 60
    # Avoid importing heavy dependencies (PyTorch) in the smoke test by
    # stubbing `src.algorithm2_autoencoder` with a lightweight fake that
    # provides the minimal interface used by `scripts.run_all`.
    import types
    stub = types.ModuleType("src.algorithm2_autoencoder")

    class AutoEncoderDetector:
        def __init__(self, *args, **kwargs):
            self.train_history = [0.0]
            self.threshold = 1.0

        def fit(self, X_normal, X_val=None, y_val_fraud=None):
            return self

        def predict_fraud_proba(self, X):
            X = np.asarray(X)
            return np.zeros(len(X), dtype=float)

    stub.AutoEncoderDetector = AutoEncoderDetector
    import sys
    sys.modules["src.algorithm2_autoencoder"] = stub

    from scripts.run_all import main as run_all_main
    run_all_main()
    print("\nSMOKE TEST PASSED")


if __name__ == "__main__":
    main()
