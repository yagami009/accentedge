"""Tests for candidate interfaces."""

from __future__ import annotations

import numpy as np
from datetime import datetime

from accentedge_lab.models.interfaces import (
    CandidateMetadata,
    StreamingResult,
    StreamingSession,
)
from tests.fixtures.fake_causal_model import FakeCausalModel


class TestCandidateMetadata:
    def test_defaults(self) -> None:
        m = CandidateMetadata()
        assert m.input_sample_rate == 16000
        assert m.requires_reference_speaker is False

    def test_finalist_requirements(self) -> None:
        m = CandidateMetadata(requires_reference_speaker=True)
        assert m.requires_reference_speaker is True


class TestStreamingSession:
    def test_creation(self) -> None:
        sess = StreamingSession(session_id="s1")
        assert sess.session_id == "s1"
        assert sess.state_size_bytes() == 0

    def test_state_size(self) -> None:
        sess = StreamingSession(
            session_id="s2",
            state={"arr": np.zeros(100, dtype=np.float32)},
        )
        assert sess.state_size_bytes() == 400


class TestStreamingResult:
    def test_fields(self) -> None:
        r = StreamingResult(
            audio=np.zeros(10, dtype=np.float32),
            sample_rate=16000,
            input_start_sample=0,
            input_end_sample=10,
            output_start_sample=0,
            output_end_sample=10,
        )
        assert r.input_end_sample == 10
        assert r.metadata == {}

    def test_fake_model(self) -> None:
        model = FakeCausalModel()
        model.prepare("cpu", "fp32")
        session = model.create_session({})
        result = model.process_chunk(session, np.ones(1600, dtype=np.float32), 16000)
        assert result.audio.size == 1600
        model.close()
