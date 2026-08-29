"""State growth measurement for long-running sessions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from accentedge.models.models.interfaces import StreamingCandidate, StreamingSession


@dataclass
class StateGrowthResult:
    grew: bool = False
    sizes: list[tuple[int, int]] = field(default_factory=list)
    growth_rate: float = 0.0
    verdict: Literal["BOUNDED", "LINEAR_GROWTH", "UNKNOWN"] = "UNKNOWN"


async def measure_state_growth(
    candidate_factory,
    duration_seconds: int = 1800,
    sample_rate: int = 16000,
) -> StateGrowthResult:
    result = StateGrowthResult()
    total_samples = duration_seconds * sample_rate
    checkpoints = {
        60 * sample_rate,
        10 * 60 * sample_rate,
        30 * 60 * sample_rate,
    }
    candidate = candidate_factory()
    candidate.prepare("cpu", "fp32")
    session = candidate.create_session({"state_growth_test": True})
    try:
        chunk_len = int(0.1 * sample_rate)
        produced = 0
        while produced < total_samples:
            chunk = np.zeros(chunk_len, dtype=np.float32)
            candidate.process_chunk(session, chunk, sample_rate)
            produced += chunk_len
            if produced in checkpoints or produced >= total_samples:
                size = session.state_size_bytes()
                result.sizes.append((produced, size))
    finally:
        candidate.close()
    if len(result.sizes) < 2:
        result.verdict = "UNKNOWN"
        return result
    times = [s[0] / float(sample_rate) for s in result.sizes]
    sizes = [s[1] for s in result.sizes]
    dt = times[-1] - times[0]
    ds = sizes[-1] - sizes[0]
    result.growth_rate = float(ds / dt) if dt > 0 else 0.0
    result.grew = result.growth_rate > 0
    # LINEAR_GROWTH threshold: > 1000 bytes per second
    if result.growth_rate > 1000.0:
        result.verdict = "LINEAR_GROWTH"
    else:
        result.verdict = "BOUNDED"
    return result
