"""Manifest loading and validation."""

from pathlib import Path
from typing import Literal

from accentedge.data.schema import TrainingItem, TrainingManifest


class ValidationResult:
    def __init__(self) -> None:
        self.valid: bool = True
        self.issues: list[str] = []
        self.speaker_overlap: set[str] = set()

    def add_issue(self, issue: str) -> None:
        self.valid = False
        self.issues.append(issue)

    def add_overlap(self, speakers: set[str]) -> None:
        self.speaker_overlap |= speakers
        if speakers:
            self.valid = False
            self.issues.append(
                f"Speaker leakage detected: {sorted(speakers)}"
            )


def validate_training_manifest(
    manifest: TrainingManifest,
    train_speakers: set[str],
    val_speakers: set[str],
) -> ValidationResult:
    result = ValidationResult()
    if not manifest.items:
        result.add_issue("Manifest is empty")
    seen_ids: set[str] = set()
    for item in manifest.items:
        if item.item_id in seen_ids:
            result.add_issue(f"Duplicate item_id: {item.item_id}")
        seen_ids.add(item.item_id)
        if item.split == "train":
            train_speakers.add(item.speaker_id)
        elif item.split == "validation":
            val_speakers.add(item.speaker_id)
    overlap = train_speakers & val_speakers
    if overlap:
        result.add_overlap(overlap)
    return result
