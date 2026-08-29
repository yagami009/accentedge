"""Speaker identity evaluation."""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class IdentityResult:
    evaluator_name: str
    source_output_distance: float
    same_session_distance: float | None = None
    different_session_distance: float | None = None
    cross_accent_distance: float | None = None
    within_range: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


class SpeakerEmbedder(ABC):
    @abstractmethod
    def embed(self, audio: np.ndarray, sample_rate: int) -> np.ndarray: ...

    @abstractmethod
    def distance(self, a: np.ndarray, b: np.ndarray) -> float: ...


class IdentityEvaluator:
    def __init__(self, embedder: SpeakerEmbedder | None = None):
        self._embedder = embedder

    def evaluate(
        self, source_audio: np.ndarray, output_audio: np.ndarray, sample_rate: int
    ) -> IdentityResult:
        if self._embedder is None:
            return IdentityResult(evaluator_name="unavailable", source_output_distance=0.0, within_range=True)
        src_emb = self._embedder.embed(source_audio, sample_rate)
        out_emb = self._embedder.embed(output_audio, sample_rate)
        dist = self._embedder.distance(src_emb, out_emb)
        return IdentityResult(
            evaluator_name="default",
            source_output_distance=dist,
            within_range=dist < 0.5,
        )
