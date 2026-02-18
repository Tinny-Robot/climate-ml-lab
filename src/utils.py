"""
Utility helpers for the Climate ML Lab project.
- Logging configuration
- Project paths
- Reproducibility utilities
"""
from __future__ import annotations

import logging
import os
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np


# ---------- Paths ----------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
VISUALS_DIR = PROJECT_ROOT / "visuals"
MODELS_DIR = PROJECT_ROOT / "models"
SAVED_MODELS_DIR = MODELS_DIR / "saved_models"
REPORTS_DIR = PROJECT_ROOT / "reports"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
LOGS_DIR = PROJECT_ROOT / "logs"

for d in [DATA_DIR, RAW_DIR, PROCESSED_DIR, VISUALS_DIR, MODELS_DIR, SAVED_MODELS_DIR, REPORTS_DIR, NOTEBOOKS_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ---------- Logging ----------

def get_logger(name: str = "climate_ml_lab", level: int = logging.INFO) -> logging.Logger:
    """Create and configure a module-level logger.

    Logs to console and to logs/climate_ml_lab.log
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # already configured

    logger.setLevel(level)

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(LOGS_DIR / "climate_ml_lab.log")
    fh.setLevel(level)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    logger.propagate = False
    return logger


# ---------- Reproducibility ----------

@dataclass
class RandomState:
    seed: int = 42

    def set(self) -> None:
        os.environ["PYTHONHASHSEED"] = str(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)


__all__ = [
    "PROJECT_ROOT",
    "DATA_DIR",
    "RAW_DIR",
    "PROCESSED_DIR",
    "VISUALS_DIR",
    "MODELS_DIR",
    "SAVED_MODELS_DIR",
    "REPORTS_DIR",
    "NOTEBOOKS_DIR",
    "LOGS_DIR",
    "get_logger",
    "RandomState",
]
