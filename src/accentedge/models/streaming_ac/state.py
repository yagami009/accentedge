"""Session state management for Streaming AC candidate."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class StreamingACSession:
    """Session state for Streaming AC candidate."""

    session_id: str
    encoder_cache: np.ndarray | None = None
    speaker_state: dict = field(default_factory=lambda: {
        "embedding": None,
        "confidence": 0.0,
        "updated_at_sample": 0,
    })
    decoder_state: dict = field(default_factory=lambda: {
        "buffer": [],
        "step": 0,
    })
    timeline: list[tuple[int, int, int, int]] = field(default_factory=list)
    step: int = 0

    def state_size_bytes(self) -> int:
        total = 0
        if self.encoder_cache is not None and isinstance(self.encoder_cache, np.ndarray):
            total += self.encoder_cache.nbytes
        for key in ["embedding"]:
            val = self.speaker_state.get(key)
            if isinstance(val, np.ndarray):
                total += val.nbytes
        for item in self.decoder_state.get("buffer", []):
            if isinstance(item, np.ndarray):
                total += item.nbytes
        return total
