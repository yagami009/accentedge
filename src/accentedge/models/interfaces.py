"""Candidate model interfaces and protocol."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Protocol

import numpy as np
from pydantic import BaseModel


class CandidateMetadata(BaseModel):
    architecture_id: str = ""
    version: str = "0.0.0"
    input_sample_rate: int = 16000
    frame_ms: float = 10.0
    preferred_chunk_ms: int = 80
    required_lookahead_ms: int = 0
    left_context_ms: int = 0
    supports_conversion_strength: bool = False
    supports_target_accent: bool = False
    requires_reference_speaker: bool = False
    uses_text_at_inference: bool = False
    parameter_count: int | None = None
    commercial_use_status: str = "UNKNOWN"


class StreamingSession:
    def __init__(
        self,
        session_id: str,
        created_at: datetime | None = None,
        state: dict[str, Any] | None = None,
        samples_processed: int = 0,
    ) -> None:
        self.session_id = session_id
        self.created_at = created_at or datetime.utcnow()
        self.state = state or {}
        self.samples_processed = samples_processed

    def state_size_bytes(self) -> int:
        total = 0
        for v in self.state.values():
            if isinstance(v, np.ndarray):
                total += v.nbytes
            elif isinstance(v, (bytes, bytearray)):
                total += len(v)
            elif isinstance(v, (list, tuple)):
                for item in v:
                    if isinstance(item, np.ndarray):
                        total += item.nbytes
        return total


class StreamingResult:
    def __init__(
        self,
        audio: np.ndarray,
        sample_rate: int,
        input_start_sample: int,
        input_end_sample: int,
        output_start_sample: int,
        output_end_sample: int,
        algorithmic_delay_samples: int = 0,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.audio = audio
        self.sample_rate = sample_rate
        self.input_start_sample = input_start_sample
        self.input_end_sample = input_end_sample
        self.output_start_sample = output_start_sample
        self.output_end_sample = output_end_sample
        self.algorithmic_delay_samples = algorithmic_delay_samples
        self.metadata = metadata or {}


class StreamingCandidate(Protocol):
    metadata: CandidateMetadata

    def prepare(self, device: str, precision: str) -> None: ...

    def create_session(self, config: dict[str, Any]) -> StreamingSession: ...

    def process_chunk(
        self,
        session: StreamingSession,
        audio_chunk: np.ndarray,
        sample_rate: int,
    ) -> StreamingResult: ...

    def flush(self, session: StreamingSession) -> list[StreamingResult]: ...

    def reset(self, session: StreamingSession) -> None: ...

    def close(self) -> None: ...
