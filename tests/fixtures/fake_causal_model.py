"""Fake causal model for testing — deterministic, past-only."""

from __future__ import annotations

from datetime import datetime

import numpy as np

from accentedge_lab.models.interfaces import (
    CandidateMetadata,
    StreamingCandidate,
    StreamingResult,
    StreamingSession,
)


class FakeCausalModel:
    metadata = CandidateMetadata(
        architecture_id="fake_causal",
        version="0.0.1",
        input_sample_rate=16000,
        frame_ms=10.0,
        preferred_chunk_ms=80,
        required_lookahead_ms=0,
        left_context_ms=0,
    )

    def prepare(self, device: str, precision: str) -> None:
        pass

    def create_session(self, config: dict) -> StreamingSession:
        return StreamingSession(
            session_id="causal_session",
            state={"step": 0},
        )

    def process_chunk(
        self,
        session: StreamingSession,
        audio_chunk: np.ndarray,
        sample_rate: int,
    ) -> StreamingResult:
        # No state growth — bounded behavior
        session.state["step"] = session.state.get("step", 0) + 1
        session.samples_processed += len(audio_chunk)
        return StreamingResult(
            audio=audio_chunk.copy(),
            sample_rate=sample_rate,
            input_start_sample=session.samples_processed - len(audio_chunk),
            input_end_sample=session.samples_processed,
            output_start_sample=session.samples_processed - len(audio_chunk),
            output_end_sample=session.samples_processed,
        )

    def flush(self, session: StreamingSession) -> list[StreamingResult]:
        return []

    def reset(self, session: StreamingSession) -> None:
        session.state["step"] = 0
        session.samples_processed = 0

    def close(self) -> None:
        pass
