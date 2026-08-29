"""Chunk-boundary artifact measurement."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import math

import numpy as np


@dataclass
class Discontinuity:
    sample_index: int
    amplitude_jump_db: float
    phase_jump: float


@dataclass
class ContinuityResult:
    discontinuities: list[Discontinuity] = field(default_factory=list)
    max_amplitude_jump_db: float = 0.0
    max_phase_jump: float = 0.0


def measure_chunk_boundary_artifacts(
    output_audio: np.ndarray, chunk_size_samples: int
) -> ContinuityResult:
    result = ContinuityResult()
    if output_audio.size < 2 or chunk_size_samples <= 0:
        return result
    boundaries = list(range(chunk_size_samples, len(output_audio), chunk_size_samples))
    for b in boundaries:
        if b <= 0 or b >= len(output_audio):
            continue
        left = float(output_audio[b - 1])
        right = float(output_audio[b])
        jump = abs(right - left)
        if left == 0 and right == 0:
            db_jump = 0.0
        elif left == 0:
            db_jump = float("inf")
        else:
            ratio = (right - left) / abs(left)
            db_jump = 20.0 * math.log10(abs(ratio)) if ratio > 0 else float("inf")
        result.discontinuities.append(
            Discontinuity(sample_index=b, amplitude_jump_db=db_jump, phase_jump=0.0)
        )
        if db_jump != float("inf"):
            result.max_amplitude_jump_db = max(
                result.max_amplitude_jump_db, db_jump
            )
    return result
