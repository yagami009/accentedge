"""Passthrough adapter — returns input unchanged."""

from __future__ import annotations

import numpy as np

from .base import BenchmarkContext, CandidateAdapter, CandidateMetadata, CandidateOutput


class PassthroughAdapter(CandidateAdapter):
    """Passthrough baseline — returns a copy of input audio."""

    def __init__(self) -> None:
        self._meta = CandidateMetadata(
            name="passthrough",
            version="1.0.0",
            description="Passthrough baseline (no transformation)",
            target_accent="source",
        )

    @property
    def metadata(self) -> CandidateMetadata:
        return self._meta

    def process(
        self,
        audio: np.ndarray,
        sample_rate: int,
        context: BenchmarkContext,
    ) -> CandidateOutput:
        return CandidateOutput(
            audio=audio.copy(),
            sample_rate=sample_rate,
            metadata={"source_id": context.utterance_id},
        )
