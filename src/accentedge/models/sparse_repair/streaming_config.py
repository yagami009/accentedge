"""Streaming configuration for SparseRepair Candidate."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SparseRepairConfig:
    """Configuration for sparse-repair streaming behavior."""

    detection_threshold: float = 0.5
    min_repair_duration_ms: int = 50
    fade_samples: int = 256
    conversion_strength: float = 0.7
    sr: int = 16000
    feature_dim: int = 2
    hidden_dim: int = 8
    frame_ms: float = 10.0

    @property
    def frame_samples(self) -> int:
        """Number of audio samples per frame."""
        return max(1, int(self.sr * self.frame_ms / 1000))

    @property
    def min_repair_samples(self) -> int:
        """Minimum repair duration in samples."""
        return max(1, int(self.sr * self.min_repair_duration_ms / 1000))

    def as_dict(self) -> dict[str, object]:
        return {
            "detection_threshold": self.detection_threshold,
            "min_repair_duration_ms": self.min_repair_duration_ms,
            "fade_samples": self.fade_samples,
            "conversion_strength": self.conversion_strength,
            "sr": self.sr,
            "feature_dim": self.feature_dim,
            "hidden_dim": self.hidden_dim,
            "frame_ms": self.frame_ms,
        }
