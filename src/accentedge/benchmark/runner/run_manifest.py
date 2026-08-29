"""Run manifest creation and hashing."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from ..schemas import RunManifest


def compute_config_hash(config_path: Path) -> str:
    """SHA-256 hash of config file contents."""
    return hashlib.sha256(config_path.read_bytes()).hexdigest()


def compute_dataset_hash(manifest_path: Path) -> str:
    """SHA-256 hash of dataset manifest."""
    return hashlib.sha256(manifest_path.read_bytes()).hexdigest()


def create_run_manifest(
    candidate_name: str,
    candidate_hash: str,
    config_hash: str,
    split: str,
    condition: str,
    dataset_hash: str,
    conversion_strength: float | None = None,
) -> RunManifest:
    """Create a run manifest from current execution context."""
    return RunManifest(
        run_id=f"{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}_{candidate_name}",
        benchmark_version="1.0.0",
        dataset_hash=dataset_hash,
        split=split,
        candidate_name=candidate_name,
        candidate_version="unknown",
        candidate_hash=candidate_hash,
        git_commit=_get_git_commit(),
        python_version=_get_python_version(),
        timestamp=datetime.now(timezone.utc),
        condition=condition,
        conversion_strength=conversion_strength,
    )


def manifest_to_dict(manifest: RunManifest) -> dict:
    """Convert RunManifest to JSON-serializable dict."""
    return manifest.model_dump(mode="json")


def _get_git_commit() -> str | None:
    try:
        import subprocess
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            return result.stdout.strip()[:12]
    except Exception:
        pass
    return None


def _get_python_version() -> str:
    import sys
    return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
