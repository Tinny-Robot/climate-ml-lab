"""
Visualization helpers for saving plots consistently to the visuals/ folder.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt

from .utils import VISUALS_DIR


def save_current_fig(filename: str, subdir: Optional[str] = None, dpi: int = 150, tight: bool = True) -> Path:
    """Save the current matplotlib figure to visuals/.

    Args:
        filename: e.g., "temp_trend.png"
        subdir: optional subdirectory inside visuals
        dpi: image resolution
        tight: apply tight_layout before saving
    Returns:
        Path to the saved file
    """
    out_dir = VISUALS_DIR if not subdir else VISUALS_DIR / subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename
    if tight:
        plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    return path


__all__ = ["save_current_fig"]
