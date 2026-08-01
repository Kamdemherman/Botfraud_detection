"""
Load the real TalkingData AdTracking corpus and build a *strictly chronological*
train / validation / test split (report Section 2.1).

Download (Kaggle): "TalkingData AdTracking Fraud Detection Challenge".
Place ``train.csv`` at  botfraud-detection/data/train.csv  (path in config.py).

Raw schema (6 fields + attribution):
    ip, app, device, os, channel, click_time, attributed_time, is_attributed

Labelling note (IMPORTANT -- see README):
    The corpus only ships ``is_attributed`` (did the click convert to an install).
    Following the report, a *converting* click is treated as legitimate and a
    *non-converting* click as the (bot) fraud class. We model the rare converting
    signal directly and expose the fraud score as its complement; this reproduces
    the benchmark AUC while keeping the (acknowledged) proxy-label limitation
    explicit.
"""
from __future__ import annotations
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

import config
import tempfile
import shutil


@dataclass
class SplitData:
    """Holds the three chronological partitions plus the raw frames."""
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame

    def summary(self) -> str:
        def _line(name, df):
            conv = int(df["is_attributed"].sum())
            return (f"  {name:5s}: {len(df):>10,} rows | "
                    f"converting(legit)={conv:>8,} "
                    f"({100*conv/max(len(df),1):.3f}%)")
        return "\n".join([
            "Chronological split:",
            _line("train", self.train),
            _line("val", self.val),
            _line("test", self.test),
        ])


def _read_sampled(path: str, cfg: config.DataConfig) -> pd.DataFrame:
    """Read the (huge) CSV. If ``sample_rows`` is set, draw an approximately
    uniform sample across chunks so we never hold the whole file in memory."""
    parse_cols = ["click_time", "attributed_time"]
    # When sampling, avoid letting pandas parse dates for every chunk which
    # can be slow and memory-heavy due to internal caching. Read raw strings
    # for chunks and convert only the sampled rows. If no sampling is used,
    # fall back to the default behavior and let pandas parse dates for the
    # whole file.
    if cfg.sample_rows is None:
        read_kwargs = dict(dtype=cfg.dtypes, parse_dates=parse_cols)
        return pd.read_csv(path, **read_kwargs)
    else:
        # read as raw dtypes (dates as strings) for faster chunked sampling
        read_kwargs = dict(dtype=cfg.dtypes)

    frames, collected = [], 0
    # First, a cheap row count so we can size the per-chunk sampling fraction.
    total = sum(1 for _ in open(path, "rb")) - 1  # minus header
    frac = min(1.0, cfg.sample_rows / max(total, 1))
    rng = np.random.RandomState(config.RANDOM_SEED)

    # Cap the per-chunk row count to avoid pandas allocating very large
    # intermediate arrays (systems with limited RAM can fail when chunks are
    # too large). Choose a conservative cap while still reading in chunks.
    chunk_cap = 200_000
    read_chunksize = min(cfg.chunksize, chunk_cap)

    # To avoid holding all sampled frames in memory (which `pd.concat` may
    # allocate large temporaries for), write sampled rows to a temporary CSV
    # on disk as we go, then return the path to that CSV for downstream
    # streaming processing.
    tmp_sample_fd, tmp_sample_path = tempfile.mkstemp(prefix="sampled_", suffix=".csv", dir=config.ARTIFACT_DIR)
    os.close(tmp_sample_fd)
    header_written = False

    for chunk in pd.read_csv(path, chunksize=read_chunksize, **read_kwargs):
        take = chunk.sample(frac=frac, random_state=rng.randint(0, 2**31 - 1))
        # Convert sampled rows' date columns to ISO strings for safe round-trip.
        if "click_time" in take.columns:
            take["click_time"] = pd.to_datetime(take["click_time"], errors="coerce")
        if "attributed_time" in take.columns:
            take["attributed_time"] = pd.to_datetime(take["attributed_time"], errors="coerce")

        # Append to temp CSV.
        take.to_csv(tmp_sample_path, mode="a", index=False, header=not header_written)
        header_written = True
        collected += len(take)
        if collected >= cfg.sample_rows:
            break

    # If we wrote more than requested due to chunk sampling, trim by rewriting
    # the first N rows to a final temp file.
    if collected > cfg.sample_rows:
        final_fd, final_path = tempfile.mkstemp(prefix="sampled_trimmed_", suffix=".csv", dir=config.ARTIFACT_DIR)
        os.close(final_fd)
        # Read only the needed number of rows and write them out.
        with pd.read_csv(tmp_sample_path, chunksize=read_chunksize) as reader:
            written = 0
            header_out = False
            for sub in reader:
                take = sub if (written + len(sub) <= cfg.sample_rows) else sub.iloc[: (cfg.sample_rows - written)]
                take.to_csv(final_path, mode="a", index=False, header=not header_out)
                header_out = True
                written += len(take)
                if written >= cfg.sample_rows:
                    break
        # Remove the larger temporary file and use the trimmed one.
        try:
            os.remove(tmp_sample_path)
        except OSError:
            pass
        tmp_sample_path = final_path

    # Return the path to the sampled CSV for streaming downstream.
    return tmp_sample_path


def load_talkingdata(path: str | None = None,
                     cfg: config.DataConfig | None = None) -> SplitData:
    """Public entry point. Returns a :class:`SplitData` with chronological splits."""
    path = path or config.TALKINGDATA_TRAIN
    cfg = cfg or config.DATA
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"TalkingData file not found at '{path}'.\n"
            f"Download 'train.csv' from the Kaggle TalkingData AdTracking "
            f"competition and place it there (or pass an explicit path)."
        )

    df = _read_sampled(path, cfg)

    # If `_read_sampled` returned a path (disk-backed sample) or a DataFrame,
    # stream that sampled CSV into per-day temporary files so we never hold the
    # full sampled frame in memory while sorting per-day. If a DataFrame was
    # returned, write it to a temporary CSV and reuse the same streaming logic.
    sampled_csv = None
    if isinstance(df, str) and os.path.exists(df):
        sampled_csv = df
    elif isinstance(df, pd.DataFrame):
        fd, tmp_path = tempfile.mkstemp(prefix="sampled_df_", suffix=".csv", dir=config.ARTIFACT_DIR)
        os.close(fd)
        # Ensure date-like columns serialize safely for later parsing.
        tmp = df.copy()
        if "click_time" in tmp.columns:
            tmp["click_time"] = pd.to_datetime(tmp["click_time"]).dt.strftime("%Y-%m-%d %H:%M:%S")
        if "attributed_time" in tmp.columns:
            tmp["attributed_time"] = pd.to_datetime(tmp["attributed_time"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")
        tmp.to_csv(tmp_path, index=False)
        del tmp
        # Free the original DataFrame to reduce peak memory.
        try:
            del df
        except Exception:
            pass
        sampled_csv = tmp_path

    if sampled_csv is not None and os.path.exists(sampled_csv):
        day_tmp_files: dict[int, str] = {}
        chunk_cap = 200_000
        read_chunksize = min(cfg.chunksize, chunk_cap)

        for chunk in pd.read_csv(sampled_csv, chunksize=read_chunksize, dtype=cfg.dtypes, parse_dates=["click_time", "attributed_time"]):
            # derived cols
            chunk["day"] = chunk["click_time"].dt.day.astype("uint8")
            chunk["hour"] = chunk["click_time"].dt.hour.astype("uint8")
            if "attributed_time" in chunk.columns:
                att = pd.to_datetime(chunk["attributed_time"], errors="coerce")
                chunk["time_to_install"] = (att - chunk["click_time"]).dt.total_seconds()
            else:
                chunk["time_to_install"] = np.nan

            for d, sub in chunk.groupby("day"):
                if d not in day_tmp_files:
                    fd, p = tempfile.mkstemp(prefix=f"day_{d}_", suffix=".csv", dir=config.ARTIFACT_DIR)
                    os.close(fd)
                    day_tmp_files[d] = p
                    sub.to_csv(p, index=False, header=True)
                else:
                    sub.to_csv(day_tmp_files[d], mode="a", index=False, header=False)

        # Build the concatenated, chronologically sorted DataFrame by reading
        # each per-day file (these are much smaller) and sorting within day.
        frames_sorted = []
        for d in sorted(day_tmp_files.keys()):
            part = pd.read_csv(day_tmp_files[d], dtype=cfg.dtypes, parse_dates=["click_time", "attributed_time"]) 
            part["day"] = part["click_time"].dt.day.astype("uint8")
            part["hour"] = part["click_time"].dt.hour.astype("uint8")
            if "attributed_time" in part.columns:
                att = pd.to_datetime(part["attributed_time"], errors="coerce")
                part["time_to_install"] = (att - part["click_time"]).dt.total_seconds()
            else:
                part["time_to_install"] = np.nan
            part = part.sort_values("click_time", kind="mergesort")
            frames_sorted.append(part)

        # Clean up temp files
        try:
            os.remove(sampled_csv)
        except OSError:
            pass
        for p in day_tmp_files.values():
            try:
                os.remove(p)
            except OSError:
                pass

        df = pd.concat(frames_sorted, ignore_index=True)
        df.index = pd.RangeIndex(start=0, stop=len(df))

    # Derived raw columns used by feature engineering.
    df["click_time"] = pd.to_datetime(df["click_time"])
    df["day"] = df["click_time"].dt.day.astype("uint8")
    df["hour"] = df["click_time"].dt.hour.astype("uint8")
    # time-to-install (seconds); NaN when the click never converted.
    if "attributed_time" in df.columns:
        att = pd.to_datetime(df["attributed_time"], errors="coerce")
        df["time_to_install"] = (att - df["click_time"]).dt.total_seconds()
    else:
        df["time_to_install"] = np.nan

    # Sort chronologically. Avoid sorting the whole large frame at once which
    # can be memory-heavy and may hit pandas internal bugs. The dataset spans
    # only a few distinct days, so sort within each `day` and concatenate the
    # per-day frames (stable merge sort) to produce a fully chronological
    # ordering while keeping memory pressure low.
    frames_sorted = []
    for d in sorted(df["day"].unique()):
        sub = df[df["day"] == d].sort_values("click_time", kind="mergesort")
        frames_sorted.append(sub)
    df = pd.concat(frames_sorted, ignore_index=True)
    # Ensure a compact 0..N-1 index without extra copies.
    df.index = pd.RangeIndex(start=0, stop=len(df))

    # Strict chronological split: last day = test, earlier days = train/val.
    test_mask = df["day"] == cfg.test_day
    test = df[test_mask].reset_index(drop=True)
    trainval = df[~test_mask].reset_index(drop=True)

    # Validation = the most recent slice of the train portion (still in the past
    # relative to test), so no future information leaks backwards.
    n_val = int(len(trainval) * cfg.val_fraction)
    train = trainval.iloc[: len(trainval) - n_val].reset_index(drop=True)
    val = trainval.iloc[len(trainval) - n_val:].reset_index(drop=True)

    return SplitData(train=train, val=val, test=test)
