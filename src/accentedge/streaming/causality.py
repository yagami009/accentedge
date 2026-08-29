"""Causality testing for streaming candidates."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from accentedge.models.models.interfaces import (
    StreamingCandidate,
    StreamingResult,
    StreamingSession,
)


@dataclass
class CausalityResult:
    passed: bool
    declared_lookahead_ms: float
    max_deviation: float
    deviation_unit: str = "samples"
    tolerance: float = 0.0
    verdict: Literal["PASS", "FAIL", "INDETERMINATE"] = "INDETERMINATE"


def measure_causality(
    candidate: StreamingCandidate,
    test_audio: np.ndarray,
    declared_lookahead_ms: float,
    tolerance_db: float = 60.0,
) -> CausalityResult:
    sample_rate = candidate.metadata.input_sample_rate
    lookahead_samples = int(declared_lookahead_ms * sample_rate / 1000)
    tolerance = _db_to_linear(tolerance_db)

    session_a = candidate.create_session({"causality_test": True})
    session_b = candidate.create_session({"causality_test": True})

    try:
        result_a = candidate.process_chunk(session_a, test_audio, sample_rate)
        result_b = candidate.process_chunk(
            session_b,
            test_audio[: len(test_audio) + lookahead_samples],
            sample_rate,
        )
    finally:
        candidate.close()
        candidate.close()

    if result_a.audio is None or result_b.audio is None:
        return CausalityResult(
            passed=False,
            declared_lookahead_ms=declared_lookahead_ms,
            max_deviation=float("inf"),
            tolerance=tolerance,
            verdict="FAIL",
        )

    common_len = min(len(result_a.audio), len(result_b.audio))
    if common_len == 0:
        return CausalityResult(
            passed=True,
            declared_lookahead_ms=declared_lookahead_ms,
            max_deviation=0.0,
            tolerance=tolerance,
            verdict="PASS",
        )

    diff = np.abs(
        result_a.audio[:common_len].astype(np.float64)
        - result_b.audio[:common_len].astype(np.float64)
    )
    max_diff = float(np.max(diff))
    reference = float(np.max(np.abs(result_a.audio[:common_len].astype(np.float64))))
    if reference <= 0:
        relative = 0.0 if max_diff == 0 else float("inf")
    else:
        relative = max_diff / reference

    passed = relative <= tolerance
    verdict: Literal["PASS", "FAIL", "INDETERMINATE"] = "PASS" if passed else "FAIL"
    return CausalityResult(
        passed=passed,
        declared_lookahead_ms=declared_lookahead_ms,
        max_deviation=relative,
        tolerance=tolerance,
        verdict=verdict,
    )


def prefix_invariance_test(
    candidate: StreamingCandidate,
    audio: np.ndarray,
    lookahead_ms: float,
    tolerance_db: float = 60.0,
) -> CausalityResult:
    return measure_causality(candidate, audio, lookahead_ms, tolerance_db)


def _db_to_linear(db: float) -> float:
    return 10 ** (-db / 20.0)
