"""Data schemas for training items and manifests."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field


class TrainingItem(BaseModel):
    item_id: str
    speaker_id: str
    dataset_id: str
    dataset_version: str
    split: Literal["train", "validation"]
    source_audio_path: Path
    source_audio_sha256: str
    source_duration_ms: float
    source_sample_rate: int
    target_audio_path: Path | None = None
    target_audio_sha256: str | None = None
    target_generator_id: str | None = None
    target_generator_version: str | None = None
    target_admission_status: Literal["accepted", "rejected", "pending"] | None = None
    transcript_path: Path | None = None
    alignment_path: Path | None = None
    feature_annotation_path: Path | None = None
    license_id: str
    commercial_use_status: Literal[
        "RESEARCH_ONLY", "COMMERCIAL_ELIGIBLE", "UNKNOWN"
    ] = "UNKNOWN"
    lineage_status: Literal["clean", "contaminated", "unknown"] = "unknown"


def propagate_commercial_status(items: list[TrainingItem]) -> str:
    if not items:
        return "UNKNOWN"
    statuses = {item.commercial_use_status for item in items}
    if "RESEARCH_ONLY" in statuses:
        return "RESEARCH_ONLY"
    if "UNKNOWN" in statuses:
        return "UNKNOWN"
    return "COMMERCIAL_ELIGIBLE"


class TrainingManifest(BaseModel):
    items: list[TrainingItem]
    version: str = "1.0.0"
    created_at: datetime = Field(default_factory=datetime.utcnow)
    manifest_hash: str | None = None

    def compute_hash(self) -> str:
        import hashlib

        payload = (
            "|".join(sorted(item.item_id for item in self.items)) + "|" + self.version
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @classmethod
    def from_file(cls, path: str | Path) -> "TrainingManifest":
        import json

        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.model_validate(raw)

    def save(self, path: str | Path) -> None:
        import json

        self.manifest_hash = self.compute_hash()
        Path(path).write_text(
            json.dumps(self.model_dump(mode="json"), indent=2, default=str),
            encoding="utf-8",
        )

    def validate_speaker_splits(self) -> list[str]:
        train_speakers: set[str] = set()
        val_speakers: set[str] = set()
        warnings: list[str] = []
        for item in self.items:
            if item.split == "train":
                train_speakers.add(item.speaker_id)
            elif item.split == "validation":
                val_speakers.add(item.speaker_id)
        overlap = train_speakers & val_speakers
        if overlap:
            warnings.append(f"Speaker leakage between train and validation: {overlap}")
        return warnings
