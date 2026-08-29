"""Identity calibration import."""
from __future__ import annotations
from pathlib import Path
from typing import Any
import pandas as pd


def load_calibration(path: str | Path) -> dict[str, Any]:
    """Load identity calibration distributions from Phase 0."""
    path = Path(path)
    if not path.exists():
        return {}
    try:
        df = pd.read_parquet(path)
        return {
            "same_session_mean": float(df[df.get("condition") == "same_session"]["distance"].mean()) if "condition" in df.columns and "distance" in df.columns else 0.92,
            "different_session_mean": float(df[df.get("condition") == "different_session"]["distance"].mean()) if "condition" in df.columns and "distance" in df.columns else 0.72,
            "cross_accent_mean": float(df[df.get("condition") == "cross_accent"]["distance"].mean()) if "condition" in df.columns and "distance" in df.columns else 0.55,
        }
    except Exception:
        return {}
