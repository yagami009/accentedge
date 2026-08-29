"""Fake model with unbounded state growth for state-growth tests."""

from __future__ import annotations

import numpy as np

from accentedge_lab.models.interfaces import (
    CandidateMetadata,
    StreamingCandidate,
    StreamingResult,
    StreamingSession,
)


class FakeGrowingStateModel:
    metadata = CandidateMetadata(
        architecture_id="fake_growing",
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
            session_id="growing_session",
            state={"encoder_cache": []},
        )

    def process_chunk(
        self,
        session: StreamingSession,
        audio_chunk: np.ndarray,
        sample_rate: int,
    ) -> StreamingResult:
        # Append EVERY frame => unbounded growth
        session.state.setdefault("encoder_cache", []).append(audio_chunk.copy())
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
        session.state["encoder_cache"] = []
        session.samples_processed = 0

    def close(self) -> None:
        pass
