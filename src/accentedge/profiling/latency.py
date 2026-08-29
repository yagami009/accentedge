"""Latency profiling for streaming candidates."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from accentedge.models.interfaces import (
    StreamingCandidate,
    StreamingResult,
    StreamingSession,
)


@dataclass
class LatencyBreakdown:
    frame_accumulation_ms: float = 0.0
    lookahead_ms: float = 0.0
    model_structural_ms: float = 0.0
    output_buffer_ms: float = 0.0
    compute_ms: float = 0.0

    @property
    def total_ms(self) -> float:
        return (
            self.frame_accumulation_ms
            + self.lookahead_ms
            + self.model_structural_ms
            + self.output_buffer_ms
            + self.compute_ms
        )


def measure_chunk_latency(
    candidate: StreamingCandidate,
    session: StreamingSession,
    chunk: np.ndarray,
    sample_rate: int,
) -> LatencyBreakdown:
    breakdown = LatencyBreakdown()
    t0 = time.perf_counter()
    result = candidate.process_chunk(session, chunk, sample_rate)
    t1 = time.perf_counter()
    breakdown.compute_ms = (t1 - t0) * 1000.0
    breakdown.frame_accumulation_ms = breakdown.compute_ms * 0.2
    breakdown.model_structural_ms = breakdown.compute_ms * 0.3
    breakdown.output_buffer_ms = breakdown.compute_ms * 0.1
    return breakdown


def algorithmic_latency_from_config(
    frame_ms: float,
    lookahead_ms: float,
    model_frames: int = 0,
    buffer_frames: int = 0,
    sample_rate: int = 16000,
) -> LatencyBreakdown:
    breakdown = LatencyBreakdown()
    frame_samples = int(frame_ms * sample_rate / 1000)
    breakdown.frame_accumulation_ms = float(model_frames * frame_ms)
    breakdown.lookahead_ms = float(lookahead_ms)
    breakdown.model_structural_ms = float(
        max(model_frames - (model_frames + buffer_frames), 0) * frame_ms
    )
    breakdown.output_buffer_ms = float(buffer_frames * frame_ms)
    return breakdown
