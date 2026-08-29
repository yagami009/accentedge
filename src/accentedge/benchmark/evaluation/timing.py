"""Timing evaluation."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class TimingResult:
    duration_ratio: float
    duration_delta_ms: float
    within_bounds: bool
    metadata: dict[str, Any] = field(default_factory=dict)


class TimingEvaluator:
    def __init__(self, tolerance_ms: float = 50.0):
        self.tolerance_ms = tolerance_ms

    def evaluate(
        self, source_audio: np.ndarray, output_audio: np.ndarray, source_sr: int, output_sr: int
    ) -> TimingResult:
        source_dur = len(source_audio) / source_sr * 1000.0
        output_dur = len(output_audio) / output_sr * 1000.0
        ratio = output_dur / source_dur if source_dur > 0 else 1.0
        delta = output_dur - source_dur
        return TimingResult(
            duration_ratio=ratio,
            duration_delta_ms=delta,
            within_bounds=abs(delta) <= self.tolerance_ms,
        )
