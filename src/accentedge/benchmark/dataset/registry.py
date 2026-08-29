"""Dataset registry for managing benchmark datasets."""

from __future__ import annotations

from pathlib import Path

from loguru import logger

from accentedge.benchmark.dataset.manifest import load_manifest
from accentedge.benchmark.dataset.validator import validate_dataset


class DatasetRegistry:
    """Registry for benchmark datasets."""

    def __init__(self, data_root: str | Path = "data/"):
        self.data_root = Path(data_root)
        self._datasets: dict[str, list] = {}

    def load(self, name: str, manifest_path: str | Path | None = None) -> list:
        """Load a dataset by name."""
        if name in self._datasets:
            return self._datasets[name]
        
        if manifest_path is None:
            manifest_path = self.data_root / "manifests" / f"{name}.parquet"
        
        items = load_manifest(manifest_path)
        self._datasets[name] = items
        logger.info(f"Registered dataset '{name}': {len(items)} items")
        return items

    def validate(self, name: str) -> "DatasetValidationResult":
        """Validate a loaded dataset."""
        if name not in self._datasets:
            raise KeyError(f"Dataset '{name}' not loaded")
        return validate_dataset(self._datasets[name])

    def get(self, name: str) -> list:
        """Get loaded dataset items."""
        if name not in self._datasets:
            raise KeyError(f"Dataset '{name}' not loaded. Call load() first.")
        return self._datasets[name]
