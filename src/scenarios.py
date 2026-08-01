"""
Curated scenario set (report Section 2.1, Table 1).

We build 8 robustness scenarios of ~50,000 rows each at ~30% fraud prevalence.
Each scenario draws legitimate (converting) and ordinary records from the real
corpus and then *injects behaviourally-calibrated synthetic fraud* characteristic
of the attack being probed.

HONESTY NOTE (matches the report and the thesis-advice given): these scenarios are
*synthetic injections*, so per-scenario numbers measure behaviour against
controlled patterns, not against confirmed real-world attack labels. The
concept-drift scenario in particular is a *simulated* novel variant.
"""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

import config

SCENARIOS = [
    "normal", "click_flooding", "click_injection", "device_farms",
    "sdk_spoofing", "datacenter_proxy", "low_and_slow", "concept_drift",
]


@dataclass
class Scenario:
    name: str
    frame: pd.DataFrame   # has all raw columns + 'is_attributed'
    y_fraud: np.ndarray   # 1 = fraud, 0 = legit


def _base_pool(df: pd.DataFrame, n: int, rng) -> pd.DataFrame:
    """Sample ordinary background traffic (a uniform random slice of the real
    corpus, labelled legitimate). We deliberately do NOT oversample converting
    clicks, so no single conversion-derived feature can trivially separate the
    legitimate class from injected fraud."""
    if n <= 0:
        return df.iloc[0:0].copy()
    pool = df.sample(n, replace=len(df) < n,
                     random_state=rng.randint(0, 2**31 - 1))
    return pool.reset_index(drop=True)


def _inject(name: str, base: pd.DataFrame, n_fraud: int, rng) -> pd.DataFrame:
    """Return ``n_fraud`` synthetic fraudulent rows shaped like attack ``name``."""
    src = base.sample(n_fraud, replace=True,
                      random_state=rng.randint(0, 2**31 - 1)).reset_index(drop=True)
    src["is_attributed"] = 0  # injected fraud never genuinely converts
    src["attributed_time"] = pd.NaT
    t0 = base["click_time"].min()

    if name == "click_flooding":
        few_ips = rng.choice(base["ip"].unique(),
                             size=max(3, len(base["ip"].unique()) // 500))
        src["ip"] = rng.choice(few_ips, size=n_fraud)          # few sources
        src["click_time"] = t0 + pd.to_timedelta(
            rng.randint(0, 3600, n_fraud), unit="s")           # tight burst
    elif name == "click_injection":
        # fraudulent click fired moments before an install -> very short tti
        src["attributed_time"] = src["click_time"] + pd.to_timedelta(
            rng.uniform(0.2, 5.0, n_fraud), unit="s")
        src["is_attributed"] = 1                               # steals attribution
    elif name == "device_farms":
        few_dev = rng.choice(base["device"].unique(),
                             size=max(2, len(base["device"].unique()) // 800))
        few_ip = rng.choice(base["ip"].unique(),
                            size=max(2, len(base["ip"].unique()) // 800))
        src["device"] = rng.choice(few_dev, size=n_fraud)
        src["ip"] = rng.choice(few_ip, size=n_fraud)
    elif name == "sdk_spoofing":
        # fabricated installs engineered to look like genuine conversions
        src["is_attributed"] = 1
        med = base["time_to_install"].dropna()
        med = float(med.median()) if len(med) else 120.0
        src["attributed_time"] = src["click_time"] + pd.to_timedelta(
            np.abs(rng.normal(med, med * 0.1, n_fraud)), unit="s")
    elif name == "datacenter_proxy":
        # a narrow band of "datacenter" IPs with very high fan-out
        dc_ips = rng.choice(base["ip"].unique(),
                            size=max(2, len(base["ip"].unique()) // 1000))
        src["ip"] = rng.choice(dc_ips, size=n_fraud)
        src["device"] = rng.randint(1, 50, n_fraud)
        src["os"] = rng.randint(1, 50, n_fraud)
    elif name == "low_and_slow":
        # human-paced, spread across many sources, low volume each
        src["ip"] = rng.choice(base["ip"].unique(), size=n_fraud)
        src["click_time"] = t0 + pd.to_timedelta(
            np.sort(rng.uniform(0, 3 * 86400, n_fraud)), unit="s")
    elif name == "concept_drift":
        # a novel variant: unusual but internally consistent new pattern
        src["channel"] = rng.randint(500, 600, n_fraud)        # unseen channel band
        src["app"] = rng.randint(500, 600, n_fraud)
        src["click_time"] = t0 + pd.to_timedelta(
            rng.randint(0, 3 * 86400, n_fraud), unit="s")
    # 'normal' injects no special structure (just ordinary non-converting rows)

    # keep schema consistent
    src["day"] = src["click_time"].dt.day.astype("uint8")
    src["hour"] = src["click_time"].dt.hour.astype("uint8")
    att = pd.to_datetime(src["attributed_time"], errors="coerce")
    src["time_to_install"] = (att - src["click_time"]).dt.total_seconds()
    return src


def _inject_blend(df: pd.DataFrame, n_fraud: int, rng,
                  attacks=None) -> pd.DataFrame:
    """Build a representative blend of several attack families (used for the
    'normal' baseline scenario = typical mixed traffic with the usual fraud mix)."""
    attacks = attacks or [s for s in SCENARIOS if s != "normal"]
    per = max(n_fraud // len(attacks), 1)
    parts = [_inject(name, df, per, rng) for name in attacks]
    return pd.concat(parts, ignore_index=True).iloc[:n_fraud].reset_index(drop=True)


def build_scenario(name: str, df: pd.DataFrame,
                   size: int | None = None, fraud_rate: float | None = None,
                   seed: int = config.RANDOM_SEED) -> Scenario:
    size = size or config.EVAL.scenario_size
    fraud_rate = fraud_rate or config.EVAL.scenario_fraud_rate
    rng = np.random.RandomState(seed + abs(hash(name)) % 10_000)

    n_fraud = int(size * fraud_rate)
    n_base = size - n_fraud
    base = _base_pool(df, n_base, rng)
    base_fraud = np.zeros(len(base), dtype=int)

    if name == "normal":
        injected = _inject_blend(df, n_fraud, rng)   # typical mixed-traffic fraud
    else:
        injected = _inject(name, df, n_fraud, rng)
    inj_fraud = np.ones(len(injected), dtype=int)

    frame = pd.concat([base, injected], ignore_index=True)
    y = np.concatenate([base_fraud, inj_fraud])
    order = rng.permutation(len(frame))
    return Scenario(name=name, frame=frame.iloc[order].reset_index(drop=True),
                    y_fraud=y[order])


def build_all(df: pd.DataFrame, **kw) -> list[Scenario]:
    return [build_scenario(s, df, **kw) for s in SCENARIOS]


def build_training_set(df: pd.DataFrame, size: int = 120_000,
                       fraud_rate: float | None = None,
                       holdout: tuple = ("concept_drift",),
                       seed: int = config.RANDOM_SEED):
    """Build a supervised training mix: real background traffic (label 0) plus a
    blend of attack types injected as fraud (label 1). This is the report's
    "attribution flag augmented by curated scenario labels" used to train
    Algorithm 1, so its fraud definition matches evaluation.

    ``holdout`` attack families are deliberately EXCLUDED from training so they
    remain genuinely novel at test time -- this is what lets the comparison reveal
    whether the unsupervised detector generalises better to unseen fraud (the
    report's concept-drift finding). Returns ``(frame, y_fraud)``.
    """
    fraud_rate = fraud_rate or config.EVAL.scenario_fraud_rate
    rng = np.random.RandomState(seed)
    n_fraud = int(size * fraud_rate)
    n_base = size - n_fraud

    base = _base_pool(df, n_base, rng)
    y_base = np.zeros(len(base), dtype=int)

    # Spread injected fraud evenly across attack families (skip 'normal' + holdout).
    attacks = [s for s in SCENARIOS if s != "normal" and s not in holdout]
    per = max(n_fraud // len(attacks), 1)
    inj_parts, inj_labels = [], []
    for name in attacks:
        part = _inject(name, df, per, rng)
        inj_parts.append(part)
        inj_labels.append(np.ones(len(part), dtype=int))
    injected = pd.concat(inj_parts, ignore_index=True)
    y_inj = np.concatenate(inj_labels)

    frame = pd.concat([base, injected], ignore_index=True)
    y = np.concatenate([y_base, y_inj])
    order = rng.permutation(len(frame))
    return frame.iloc[order].reset_index(drop=True), y[order]
