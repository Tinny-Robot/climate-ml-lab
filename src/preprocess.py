"""
Preprocessing pipeline for Ogun weather data.
- Load data/raw/ogun_weather.csv
- Convert date to datetime index
- Interpolate missing values
- Create lag-1 features for all numeric variables
- Add time features: month, day, weekday
- Save to data/processed/clean_weather.csv
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import pandas as pd

from .utils import PROCESSED_DIR, RAW_DIR, get_logger

LOGGER = get_logger(__name__)


def load_raw(path: Optional[Path] = None) -> pd.DataFrame:
    if path is None:
        path = RAW_DIR / "ogun_weather.csv"
    df = pd.read_csv(path)
    if "date" not in df.columns:
        raise ValueError("Expected 'date' column in raw data")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.set_index("date").sort_index()
    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Interpolate missing values column-wise (time-based)
    df = df.asfreq("1D")  # ensure daily continuity
    df = df.interpolate(method="time", limit_direction="both")

    # Create lag-1 features
    for col in df.columns:
        df[f"{col}_lag1"] = df[col].shift(1)

    # Time features
    df["month"] = df.index.month
    df["day"] = df.index.day
    df["weekday"] = df.index.weekday  # Monday=0

    # Drop initial row with NaNs due to lag
    df = df.dropna()
    return df


def save_processed(df: pd.DataFrame, out_path: Optional[Path] = None) -> Path:
    if out_path is None:
        out_path = PROCESSED_DIR / "clean_weather.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=True)
    LOGGER.info("Saved processed data to %s (%d rows, %d cols)", out_path, len(df), df.shape[1])
    return out_path


def main() -> int:
    try:
        raw_df = load_raw()
        proc_df = preprocess(raw_df)
        save_processed(proc_df)
        return 0
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Preprocessing failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
