"""RTF profiling for streaming candidates."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from accentedge.models.interfaces import (
    StreamingCandidate,
    StreamingSession,
)
from accentedge.streaming.chunker import Chunker


@dataclass
class RTFMeasurement:
    audio_ms: float = 0.0
    compute_ms: float = 0.0

    @property
    def rtf(self) -> float:
        if self.audio_ms <= 0:
            return 0.0
        return self.compute_ms / self.audio_ms


def measure_rtf(
    candidate: StreamingCandidate,
    session: StreamingSession,
    audio_chunks: list[np.ndarray],
    sample_rate: int,
) -> list[RTFMeasurement]:
    measurements: list[RTFMeasurement] = []
    for chunk in audio_chunks:
        t0 = time.perf_counter()
        candidate.process_chunk(session, chunk, sample_rate)
        t1 = time.perf_counter()
        audio_ms = len(chunk) / sample_rate * 1000.0
        compute_ms = (t1 - t0) * 1000.0
        measurements.append(RTFMeasurement(audio_ms=audio_ms, compute_ms=compute_ms))
    return measurements
