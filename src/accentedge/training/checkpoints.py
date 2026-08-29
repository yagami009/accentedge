"""Checkpoint provenance manifest — every checkpoint has a sidecar .json."""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _git_commit() -> str | None:
    """Return the current HEAD commit hash, or None if not a git repo."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent.parent.parent.parent,
            stderr=subprocess.DEVNULL,
        ).decode().strip()[:12]
    except Exception:
        return None


def _config_hash(obj: Any) -> str:
    """Stable hash of any serialisable config object."""
    import pickle

    return hashlib.sha256(pickle.dumps(obj, protocol=5)).hexdigest()[:16]


@dataclass
class CheckpointManifest:
    """Provenance record for a single checkpoint."""

    checkpoint_id: str = ""
    architecture_id: str = ""
    version: str = "0.0.0"
    git_commit: str | None = None
    config_hash: str = ""
    training_manifest_hash: str = ""
    training_data_lineage_hash: str = ""
    parent_checkpoint_ids: list[str] = field(default_factory=list)
    pretrained_weight_sources: list[str] = field(default_factory=list)
    licenses: list[str] = field(default_factory=list)
    commercial_use_status: str = "UNKNOWN"
    seed: int = 42
    training_steps: int = 0
    training_hours: float = 0.0
    optimizer: str = ""
    scheduler: str = ""
    hardware: str = ""
    wall_clock_seconds: float = 0.0
    best_validation_metric: float | None = None
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


def save_checkpoint_manifest(
    model: Any,
    config: Any,
    training_info: dict[str, Any],
    path: str | Path,
) -> Path:
    """Write a ``<path>.json`` sidecar manifest alongside a checkpoint file.

    Parameters
    ----------
    model:
        The trained model (used to derive ``architecture_id`` and weights hash).
    config:
        The configuration object used for this run.
    training_info:
        Arbitrary dict with at least ``seed``, ``optimizer``, ``scheduler``,
        ``training_steps``, ``wall_clock_seconds``, ``hardware``.
    path:
        Base path for the checkpoint (e.g. ``ckpt.pt``).  Manifest is saved
        to ``ckpt.pt.json``.
    """
    base = Path(path)
    base.parent.mkdir(parents=True, exist_ok=True)

    ckpt_id = base.stem

    # Derive weights hash from the state dict if available
    weights_hash = ""
    if hasattr(model, "state_dict"):
        import pickle

        try:
            state = {
                k: v.detach().cpu().numpy().tobytes()
                for k, v in model.state_dict().items()
            }
            weights_hash = hashlib.sha256(
                pickle.dumps(state, protocol=5)
            ).hexdigest()[:16]
        except Exception:
            pass

    manifest = CheckpointManifest(
        checkpoint_id=ckpt_id,
        architecture_id=getattr(model, "architecture_id", training_info.get("architecture_id", "")),
        version=training_info.get("version", "0.0.0"),
        git_commit=_git_commit(),
        config_hash=_config_hash(config),
        training_manifest_hash=_config_hash(training_info),
        training_data_lineage_hash=training_info.get("data_lineage_hash", ""),
        parent_checkpoint_ids=training_info.get("parent_checkpoint_ids", []),
        pretrained_weight_sources=training_info.get("pretrained_weight_sources", []),
        licenses=training_info.get("licenses", ["Proprietary"]),
        commercial_use_status=training_info.get(
            "commercial_use_status", getattr(model, "commercial_use_status", "UNKNOWN")
        ),
        seed=training_info.get("seed", 42),
        training_steps=training_info.get("training_steps", 0),
        training_hours=training_info.get("training_hours", 0.0),
        optimizer=training_info.get("optimizer", ""),
        scheduler=training_info.get("scheduler", ""),
        hardware=training_info.get("hardware", ""),
        wall_clock_seconds=training_info.get("wall_clock_seconds", 0.0),
        best_validation_metric=training_info.get("best_validation_metric"),
        timestamp=datetime.now(timezone.utc).isoformat(),
    )

    manifest_path = Path(str(base) + ".json")
    manifest_path.write_text(json.dumps(asdict(manifest), indent=2, default=str))
    return manifest_path


def load_checkpoint_manifest(
    path: str | Path,
) -> CheckpointManifest:
    """Read a sidecar manifest created by :func:`save_checkpoint_manifest`."""
    manifest_path = Path(str(path) + ".json")
    data = json.loads(manifest_path.read_text())
    return CheckpointManifest(**data)
