"""Probe calibration metadata."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any


@dataclass
class ProbeCalibration:
    probe_name: str
    reference_corpus_version: str
    embedding_model: str
    layer: int
    centroid_version: str
    feature_definition: str
    noise_floor: float
    known_limitations: list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "probe_name": self.probe_name,
            "reference_corpus_version": self.reference_corpus_version,
            "embedding_model": self.embedding_model,
            "layer": self.layer,
            "centroid_version": self.centroid_version,
            "feature_definition": self.feature_definition,
            "noise_floor": self.noise_floor,
            "known_limitations": self.known_limitations or [],
        }
