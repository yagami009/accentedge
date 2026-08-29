"""Candidate adapter base classes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class CandidateMetadata:
    """Metadata about a candidate model."""
    name: str
    version: str
    description: str
    target_accent: str
    supports_conversion_strength: bool = False
    artifact_hash: str | None = None
    configuration: dict[str, Any] = field(default_factory=dict)


@dataclass
class CandidateOutput:
    """Output from a candidate model."""
    audio: np.ndarray
    sample_rate: int
    metadata: dict[str, Any] = field(default_factory=dict)


class CandidateAdapter(ABC):
    """Abstract base class for candidate speech transformation adapters."""

    @property
    @abstractmethod
    def metadata(self) -> CandidateMetadata:
        """Return candidate metadata."""
        ...

    @abstractmethod
    def process(
        self,
        audio: np.ndarray,
        sample_rate: int,
        context: BenchmarkContext,
    ) -> CandidateOutput:
        """Transform input audio."""
        ...

    def prepare(self) -> None:
        """Optional one-time setup."""
        pass

    def close(self) -> None:
        """Optional cleanup."""
        pass


@dataclass
class BenchmarkContext:
    """Context for a single benchmark item."""
    target_accent: str
    conversion_strength: float | None = None
    utterance_id: str | None = None
    speaker_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
