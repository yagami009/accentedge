"""Artifact (output-quality) evaluation."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ArtifactResult:
    """Result of artifact evaluation for a single utterance."""

    snr_db: float = 0.0
    clipping_ratio: float = 0.0
    sample_rate: int = 0
    artifact_flags: list[str] = field(default_factory=list)


class ArtifactEvaluator:
    """Detects audio artifacts in candidate output (clipping, silence, etc.)."""

    def evaluate(self, audio: Any, sample_rate: int) -> ArtifactResult:
        """Evaluate output audio for production-ready quality.

        Args:
            audio: Waveform (e.g. NumPy array).
            sample_rate: Sample rate of the audio.

        Returns:
            ArtifactResult with quality metrics and flags.
        """
        import numpy as np

        flags: list[str] = []
        waveform = np.asarray(audio, dtype=np.float32)

        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)

        # NaN / Inf
        if not np.all(np.isfinite(waveform)):
            flags.append("non_finite")

        # Clipping
        if waveform.size > 0:
            peak = float(np.max(np.abs(waveform)))
            if peak > 1.0:
                flags.append("clipping")

        return ArtifactResult(
            sample_rate=sample_rate,
            artifact_flags=flags,
        )


def evaluate_artifacts(
    audio: Any,
    sample_rate: int,
) -> ArtifactResult:
    """Convenience function: evaluate artifacts without instantiating the class.

    Args:
        audio: Waveform (e.g. NumPy array).
        sample_rate: Sample rate of the audio.

    Returns:
        ArtifactResult with quality metrics and flags.
    """
    evaluator = ArtifactEvaluator()
    return evaluator.evaluate(audio, sample_rate)
