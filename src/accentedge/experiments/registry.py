"""Experiment record and registry."""

from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path
from typing import Literal


class ExperimentRecord:
    def __init__(
        self,
        experiment_id: str,
        architecture: str,
        architecture_version: str,
        config_hash: str,
        benchmark_version: str,
        dev_split_hash: str,
        seed: int = 42,
        training_checkpoint: str | None = None,
        training_data_hash: str | None = None,
        hardware: dict | None = None,
        status: Literal["running", "completed", "failed"] = "running",
        created_at: datetime | None = None,
        completed_at: datetime | None = None,
    ) -> None:
        self.experiment_id = experiment_id
        self.architecture = architecture
        self.architecture_version = architecture_version
        self.config_hash = config_hash
        self.training_checkpoint = training_checkpoint
        self.training_data_hash = training_data_hash
        self.benchmark_version = benchmark_version
        self.dev_split_hash = dev_split_hash
        self.hardware = hardware or {}
        self.seed = seed
        self.status = status
        self.created_at = created_at or datetime.utcnow()
        self.completed_at = completed_at


class ExperimentRegistry:
    def __init__(self, path: str | Path = "experiments/registry.json") -> None:
        self.path = Path(path)
        self._records: dict[str, ExperimentRecord] = {}
        if self.path.exists():
            import json

            raw = json.loads(self.path.read_text(encoding="utf-8"))
            for entry in raw:
                rec = ExperimentRecord(**entry)
                self._records[rec.experiment_id] = rec

    def register(self, record: ExperimentRecord) -> None:
        self._records[record.experiment_id] = record
        self._save()

    def get(self, experiment_id: str) -> ExperimentRecord:
        if experiment_id not in self._records:
            raise KeyError(f"Experiment {experiment_id!r} not found")
        return self._records[experiment_id]

    def list_by_architecture(self, arch_id: str) -> list[ExperimentRecord]:
        return [r for r in self._records.values() if r.architecture == arch_id]

    def list_completed(self) -> list[ExperimentRecord]:
        return [r for r in self._records.values() if r.status == "completed"]

    def _save(self) -> None:
        import json

        data = [
            {
                "experiment_id": r.experiment_id,
                "architecture": r.architecture,
                "architecture_version": r.architecture_version,
                "config_hash": r.config_hash,
                "training_checkpoint": r.training_checkpoint,
                "training_data_hash": r.training_data_hash,
                "benchmark_version": r.benchmark_version,
                "dev_split_hash": r.dev_split_hash,
                "hardware": r.hardware,
                "seed": r.seed,
                "status": r.status,
                "created_at": r.created_at.isoformat(),
                "completed_at": r.completed_at.isoformat() if r.completed_at else None,
            }
            for r in self._records.values()
        ]
        self.path.write_text(json.dumps(data, indent=2), encoding="utf-8")
