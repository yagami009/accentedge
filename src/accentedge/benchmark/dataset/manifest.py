"""Dataset manifest loading and validation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from loguru import logger

from accentedge.benchmark.schemas import DatasetItem


def load_manifest(path: str | Path) -> list[DatasetItem]:
    """Load benchmark manifest from Parquet file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")
    
    df = pd.read_parquet(str(path))
    items = []
    for _, row in df.iterrows():
        row_dict = {k: v for k, v in row.to_dict().items() if pd.notna(v)}
        items.append(DatasetItem(**row_dict))
    
    logger.info(f"Loaded {len(items)} items from {path}")
    return items


def save_manifest(items: list[DatasetItem], path: str | Path) -> None:
    """Save manifest to Parquet."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    rows = [item.model_dump() for item in items]
    df = pd.DataFrame(rows)
    df.to_parquet(str(path), index=False)
    logger.info(f"Saved {len(items)} items to {path}")


def load_manifest_dataframe(path: str | Path) -> pd.DataFrame:
    """Load manifest as DataFrame."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")
    return pd.read_parquet(str(path))
