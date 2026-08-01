"""
Multi-view feature engineering -- the shared 38-feature representation that feeds
*both* detectors (report Section 2.1, Table 2).

Four views:
    Temporal      (11)  timing behaviour: hour, inter-click gaps, bursts, time-to-install
    Network / IP  (12)  source reputation & fan-out: per-IP counts, entropy, cardinality
    Device         (8)  device population: per-device counts, repeat ratios
    Conversion     (7)  downstream outcome: smoothed conversion rates, timing anomaly
                        --------
                          38

Leakage discipline (report Section 4 'Preventing temporal leakage'):
    * Conversion-rate features are *fit on the training split only* and then mapped
      onto validation / test. They are the leakage-sensitive features.
    * Count / cardinality / entropy features are observational (computable from the
      traffic available at scoring time) and are computed within whatever frame is
      being transformed.
    * A single StandardScaler is fit on train features and reused everywhere.
"""
from __future__ import annotations
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import config

# Canonical, ordered feature names (length must equal config.N_FEATURES = 38).
TEMPORAL_FEATURES = [
    "hour", "hour_sin", "hour_cos",
    "inter_click_delta", "ip_gap_mean", "ip_gap_std",
    "burst_count", "clicks_last_hour_ip",
    "time_to_install", "has_install", "tti_is_short",
]
NETWORK_FEATURES = [
    "ip_click_count", "ip_app_click_count", "ip_channel_nunique",
    "ip_app_nunique", "ip_device_nunique", "ip_os_nunique",
    "ip_device_os_cardinality", "ip_channel_entropy", "ip_app_entropy",
    "clicks_per_ip_per_hour", "datacenter_flag", "ip_hour_nunique",
]
DEVICE_FEATURES = [
    "device", "os", "device_os_combo", "device_click_count",
    "device_app_nunique", "device_ip_nunique", "repeat_device_ratio",
    "device_channel_nunique",
]
CONVERSION_FEATURES = [
    "conv_rate_by_ip", "conv_rate_by_app", "conv_rate_by_channel",
    "conv_rate_by_device", "app_click_count", "channel_click_count",
    "attribution_window_anomaly",
]
ALL_FEATURE_NAMES = (TEMPORAL_FEATURES + NETWORK_FEATURES
                     + DEVICE_FEATURES + CONVERSION_FEATURES)
assert len(ALL_FEATURE_NAMES) == config.N_FEATURES, len(ALL_FEATURE_NAMES)

# --------------------------------------------------------------------------- #
# Note on post-attribution features (see README "Labelling & leakage").
# These four are derived from ``attributed_time``. They are SAFE to use here
# because the supervised target is an explicit, attribution-INDEPENDENT fraud
# label (real background labelled legitimate + injected attack patterns labelled
# fraud), not the raw conversion flag. If you ever switch the target back to the
# raw ``is_attributed`` proxy, set EXCLUDE_POST_ATTRIBUTION=True to avoid leakage,
# because under that proxy ``has_install`` would copy the label into the inputs.
# --------------------------------------------------------------------------- #
POST_ATTRIBUTION_FEATURES = [
    "time_to_install", "has_install", "tti_is_short", "attribution_window_anomaly",
]
EXCLUDE_POST_ATTRIBUTION = False
FEATURE_NAMES = ([f for f in ALL_FEATURE_NAMES if f not in POST_ATTRIBUTION_FEATURES]
                 if EXCLUDE_POST_ATTRIBUTION else list(ALL_FEATURE_NAMES))
MODEL_FEATURE_COUNT = len(FEATURE_NAMES)

FEATURE_VIEWS = {
    "temporal": [f for f in TEMPORAL_FEATURES if f in FEATURE_NAMES],
    "network": NETWORK_FEATURES,
    "device": DEVICE_FEATURES,
    "conversion": [f for f in CONVERSION_FEATURES if f in FEATURE_NAMES],
}


def _entropy(counts: np.ndarray) -> float:
    p = counts / counts.sum()
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


@dataclass
class FeatureEngineer:
    """Fit conversion priors + scaler on train; transform any frame to 38 features."""
    cfg: config.FeatureConfig = field(default_factory=lambda: config.FEATURES)
    _conv_priors: dict = field(default_factory=dict)
    _global_conv: float = 0.0
    _device_os_map: dict = field(default_factory=dict)
    _app_tti_median: dict = field(default_factory=dict)
    _scaler: StandardScaler | None = None

    # ----------------------------- fit ---------------------------------- #
    def fit(self, train: pd.DataFrame) -> "FeatureEngineer":
        s = self.cfg.conversion_smoothing
        self._global_conv = float(train["is_attributed"].mean())

        # Smoothed conversion priors per key (fit on TRAIN ONLY -> no leakage).
        for key in ("ip", "app", "channel", "device"):
            grp = train.groupby(key)["is_attributed"].agg(["sum", "count"])
            rate = (grp["sum"] + s * self._global_conv) / (grp["count"] + s)
            self._conv_priors[key] = rate.to_dict()

        # device/os combo encoding learned on train; unseen -> -1.
        combo = (train["device"].astype(np.int64) * 1000
                 + train["os"].astype(np.int64))
        cats = pd.unique(combo)
        self._device_os_map = {c: i for i, c in enumerate(cats)}

        # Per-app median time-to-install (for the attribution-window anomaly).
        tti = train.loc[train["is_attributed"] == 1].groupby("app")["time_to_install"]
        self._app_tti_median = tti.median().to_dict()

        # Fit the scaler on the engineered train matrix.
        raw = self._build_raw(train)
        self._scaler = StandardScaler().fit(raw.values)
        return self

    # --------------------------- transform ------------------------------ #
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self._scaler is None:
            raise RuntimeError("FeatureEngineer.transform called before fit().")
        raw = self._build_raw(df)
        scaled = self._scaler.transform(raw.values)
        return pd.DataFrame(scaled, columns=FEATURE_NAMES, index=df.index)

    def fit_transform(self, train: pd.DataFrame) -> pd.DataFrame:
        return self.fit(train).transform(train)

    # --------------------- raw (unscaled) features ---------------------- #
    def _build_raw(self, df: pd.DataFrame) -> pd.DataFrame:
        f = pd.DataFrame(index=df.index)
        win = self.cfg.click_window_seconds

        # ---- Temporal ------------------------------------------------- #
        f["hour"] = df["hour"].astype(np.float64)
        f["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
        f["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)

        order = df.sort_values(["ip", "click_time"]).index
        gap = (df.loc[order]
                 .groupby("ip")["click_time"].diff().dt.total_seconds())
        gap = gap.reindex(df.index)
        f["inter_click_delta"] = gap.fillna(gap.median() if gap.notna().any() else 0.0)
        ipgap = f.groupby(df["ip"])["inter_click_delta"]
        f["ip_gap_mean"] = ipgap.transform("mean")
        f["ip_gap_std"] = ipgap.transform("std").fillna(0.0)

        # bursts: clicks by same ip within +/- burst window (approx via 1-min bucket)
        bucket = (df["click_time"].astype("int64") // 10**9
                  // self.cfg.burst_window_seconds)
        f["burst_count"] = df.groupby([df["ip"], bucket])["ip"].transform("count")
        hour_bucket = df["click_time"].dt.floor("h")
        f["clicks_last_hour_ip"] = df.groupby([df["ip"], hour_bucket])["ip"].transform("count")

        tti = df["time_to_install"]
        f["time_to_install"] = tti.fillna(-1.0)
        f["has_install"] = tti.notna().astype(np.float64)
        f["tti_is_short"] = ((tti.notna()) & (tti < 60)).astype(np.float64)

        # ---- Network / IP --------------------------------------------- #
        g_ip = df.groupby("ip")
        f["ip_click_count"] = g_ip["ip"].transform("count")
        f["ip_app_click_count"] = df.groupby(["ip", "app"])["ip"].transform("count")
        f["ip_channel_nunique"] = g_ip["channel"].transform("nunique")
        f["ip_app_nunique"] = g_ip["app"].transform("nunique")
        f["ip_device_nunique"] = g_ip["device"].transform("nunique")
        f["ip_os_nunique"] = g_ip["os"].transform("nunique")
        combo = df["device"].astype(np.int64) * 1000 + df["os"].astype(np.int64)
        f["ip_device_os_cardinality"] = combo.groupby(df["ip"]).transform("nunique")

        # entropy of channel / app distribution per IP
        ch_ent = g_ip["channel"].apply(lambda s: _entropy(s.value_counts().values))
        ap_ent = g_ip["app"].apply(lambda s: _entropy(s.value_counts().values))
        f["ip_channel_entropy"] = df["ip"].map(ch_ent).astype(np.float64)
        f["ip_app_entropy"] = df["ip"].map(ap_ent).astype(np.float64)

        f["clicks_per_ip_per_hour"] = df.groupby([df["ip"], df["hour"]])["ip"].transform("count")
        f["ip_hour_nunique"] = g_ip["hour"].transform("nunique")
        f["datacenter_flag"] = (f["ip_device_os_cardinality"]
                                > self.cfg.datacenter_fanout_threshold).astype(np.float64)

        # ---- Device --------------------------------------------------- #
        f["device"] = df["device"].astype(np.float64)
        f["os"] = df["os"].astype(np.float64)
        f["device_os_combo"] = combo.map(self._device_os_map).fillna(-1).astype(np.float64)
        g_dev = df.groupby("device")
        f["device_click_count"] = g_dev["device"].transform("count")
        f["device_app_nunique"] = g_dev["app"].transform("nunique")
        f["device_ip_nunique"] = g_dev["ip"].transform("nunique")
        f["device_channel_nunique"] = g_dev["channel"].transform("nunique")
        f["repeat_device_ratio"] = (f["device_click_count"]
                                    / f["ip_click_count"].clip(lower=1))

        # ---- Conversion (train-fit priors) ---------------------------- #
        for key, col in (("ip", "conv_rate_by_ip"), ("app", "conv_rate_by_app"),
                         ("channel", "conv_rate_by_channel"),
                         ("device", "conv_rate_by_device")):
            f[col] = df[key].map(self._conv_priors[key]).fillna(self._global_conv)
        f["app_click_count"] = df.groupby("app")["app"].transform("count")
        f["channel_click_count"] = df.groupby("channel")["channel"].transform("count")
        app_med = df["app"].map(self._app_tti_median)
        anom = (df["time_to_install"] - app_med).abs()
        f["attribution_window_anomaly"] = anom.fillna(0.0)

        # Order, clean, return.
        f = f[FEATURE_NAMES].replace([np.inf, -np.inf], 0.0).fillna(0.0)
        return f.astype(np.float64)


def fraud_label(df: pd.DataFrame) -> np.ndarray:
    """Fraud = non-converting click (report labelling). 1 = fraud, 0 = legitimate."""
    return (df["is_attributed"].values == 0).astype(int)
