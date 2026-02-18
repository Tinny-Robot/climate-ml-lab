"""
Data loader for Open-Meteo Historical Weather API for Abeokuta, Ogun State, Nigeria.

This module fetches hourly weather data and aggregates it to daily resolution
for the following variables:
- temperature_2m       -> daily mean temperature (C)
- relative_humidity_2m -> daily mean humidity (%)
- wind_speed_10m       -> daily mean wind speed (m/s)
- precipitation        -> daily total precipitation (mm)
- surface_pressure     -> daily mean surface pressure (hPa)

Output CSV: data/raw/ogun_weather.csv
"""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

from .utils import RAW_DIR, get_logger

LOGGER = get_logger(__name__)
OPEN_METEO_BASE = "https://archive-api.open-meteo.com/v1/era5"


def fetch_hourly(
    latitude: float = 7.16,
    longitude: float = 3.35,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    timezone: str = "Africa/Lagos",
) -> pd.DataFrame:
    """Fetch hourly data from Open-Meteo Archive API and return a DataFrame.

    If start_date or end_date is None, fetches from 2015-01-01 to today.
    """
    if start_date is None:
        start_date = "2015-01-01"
    if end_date is None:
        end_date = dt.date.today().isoformat()

    hourly_vars = [
        "temperature_2m",
        "relative_humidity_2m",
        "wind_speed_10m",
        "precipitation",
        "surface_pressure",
    ]

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join(hourly_vars),
        "timezone": timezone,
    }

    LOGGER.info("Requesting hourly data from Open-Meteo: %s", params)

    try:
        resp = requests.get(OPEN_METEO_BASE, params=params, timeout=60)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as exc:
        LOGGER.exception("Failed to fetch data: %s", exc)
        raise

    if "hourly" not in data or "time" not in data["hourly"]:
        LOGGER.error("Unexpected response format: missing 'hourly.time'")
        raise ValueError("Unexpected response format from Open-Meteo")

    hourly = data["hourly"]
    df = pd.DataFrame(hourly)
    # Rename time column and parse
    df = df.rename(columns={"time": "datetime"})
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True).dt.tz_convert(timezone)
    df = df.set_index("datetime").sort_index()

    # Ensure numeric types
    for col in [c for c in df.columns if c != "datetime"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    LOGGER.info("Fetched %d hourly rows spanning %s to %s", len(df), df.index.min(), df.index.max())
    return df


def aggregate_daily(df_hourly: pd.DataFrame) -> pd.DataFrame:
    """Aggregate hourly weather to daily metrics.

    - temperature_2m: daily mean
    - relative_humidity_2m: daily mean
    - wind_speed_10m: daily mean
    - precipitation: daily sum
    - surface_pressure: convert Pa to hPa if needed, daily mean
    """
    df = df_hourly.copy()

    # Open-Meteo surface_pressure in Pa; convert to hPa
    if "surface_pressure" in df.columns:
        df["surface_pressure_hpa"] = df["surface_pressure"] / 100.0
    else:
        df["surface_pressure_hpa"] = pd.NA

    daily = pd.DataFrame()
    daily["temperature_c"] = df["temperature_2m"].resample("1D").mean()
    daily["humidity_pct"] = df["relative_humidity_2m"].resample("1D").mean()
    daily["wind_speed_mps"] = df["wind_speed_10m"].resample("1D").mean()
    daily["precipitation_mm"] = df["precipitation"].resample("1D").sum()
    daily["surface_pressure_hpa"] = df["surface_pressure_hpa"].resample("1D").mean()

    daily.index.name = "date"

    # Remove days where all values are NaN
    daily = daily.dropna(how="all")
    return daily.reset_index()


def save_daily_csv(daily_df: pd.DataFrame, out_path: Optional[Path] = None) -> Path:
    """Save daily dataframe to CSV under data/raw/ by default."""
    if out_path is None:
        out_path = RAW_DIR / "ogun_weather.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    daily_df.to_csv(out_path, index=False)
    LOGGER.info("Saved daily data to %s (%d rows)", out_path, len(daily_df))
    return out_path


def main(argv: Optional[list[str]] = None) -> int:
    """CLI entry point to fetch and save daily weather data.

    Usage (optional args):
      python -m src.data_loader [start_date] [end_date]
    Dates in YYYY-MM-DD.
    """
    try:
        start = argv[0] if argv and len(argv) >= 1 else None
        end = argv[1] if argv and len(argv) >= 2 else None
        hourly = fetch_hourly(start_date=start, end_date=end)
        daily = aggregate_daily(hourly)
        save_daily_csv(daily)
        return 0
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Data loading failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
