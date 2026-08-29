"""Streaming session state management."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np


class StreamingSessionState:
    def __init__(
        self,
        session_id: str,
        candidate_id: str,
        encoder_cache: list[np.ndarray] | None = None,
        speaker_state: dict | None = None,
        decoder_state: dict | None = None,
        timeline: list[tuple[int, int, int]] | None = None,
        samples_processed: int = 0,
    ) -> None:
        self.session_id = session_id
        self.candidate_id = candidate_id
        self.encoder_cache = encoder_cache or []
        self.speaker_state = speaker_state or {}
        self.decoder_state = decoder_state or {}
        self.timeline = timeline or []
        self.samples_processed = samples_processed

    def state_size_bytes(self) -> int:
        total = 0
        for arr in self.encoder_cache:
            if isinstance(arr, np.ndarray):
                total += arr.nbytes
        for v in self.speaker_state.values():
            if isinstance(v, np.ndarray):
                total += v.nbytes
        for v in self.decoder_state.values():
            if isinstance(v, np.ndarray):
                total += v.nbytes
        return total
