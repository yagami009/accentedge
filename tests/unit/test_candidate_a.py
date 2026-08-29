"""Tests for Streaming AC Candidate A."""

from __future__ import annotations

import platform

import numpy as np
import pytest

from accentedge_lab.models.streaming_ac import StreamingACCandidate, StreamingACConfig
from accentedge_lab.models.interfaces import StreamingResult
from accentedge_lab.models.streaming_ac.low_lookahead import (
    LowLookaheadContentProsodyEncoder,
)
from accentedge_lab.models.streaming_ac.paper_style import (
    ContentProsodyEncoder as PaperStyleContentProsodyEncoder,
)


def _make_chunk(samples: int = 1600, dim: int = 80) -> np.ndarray:
    # Returns 2-D frame features for the 80-D encoder path.
    t = max(1, samples // dim)
    return np.ones((t, dim), dtype=np.float32)


class TestMetadata:
    def test_metadata(self) -> None:
        candidate = StreamingACCandidate()
        assert candidate.metadata.architecture_id == "streaming_ac"
        assert candidate.metadata.input_sample_rate == 16000
        assert candidate.metadata.frame_ms == 20.0
        assert candidate.metadata.preferred_chunk_ms == 80
        assert candidate.metadata.required_lookahead_ms == 640
        assert candidate.metadata.left_context_ms == 0
        assert candidate.metadata.supports_conversion_strength is True
        assert candidate.metadata.supports_target_accent is True
        assert candidate.metadata.requires_reference_speaker is False
        assert candidate.metadata.uses_text_at_inference is False


class TestPrepare:
    def test_prepare(self) -> None:
        candidate = StreamingACCandidate()
        candidate.prepare("cpu", "fp32")
        assert candidate is not None

    @pytest.mark.skipif(
        platform.system() != "Darwin" or platform.machine() != "arm64",
        reason="MPS only on Apple Silicon",
    )
    def test_prepare_mps(self) -> None:
        candidate = StreamingACCandidate()
        candidate.prepare("mps", "fp16")
        assert candidate is not None


class TestCreateSession:
    def test_create_session(self) -> None:
        candidate = StreamingACCandidate()
        session = candidate.create_session({"session_id": "s1"})
        assert session.session_id == "s1"
        assert "ac" in session.state


class TestProcessChunk:
    def test_process_chunk(self) -> None:
        candidate = StreamingACCandidate()
        candidate.prepare("cpu", "fp32")
        session = candidate.create_session({"session_id": "c1"})
        chunk = _make_chunk()
        result = candidate.process_chunk(session, chunk, 16000)
        assert isinstance(result, StreamingResult)
        assert result.audio is not None
        assert result.output_end_sample >= result.output_start_sample

    def test_streaming_state_isolation(self) -> None:
        candidate = StreamingACCandidate()
        candidate.prepare("cpu", "fp32")
        session_a = candidate.create_session({"session_id": "a"})
        session_b = candidate.create_session({"session_id": "b"})
        chunk = _make_chunk()
        result_a = candidate.process_chunk(session_a, chunk, 16000)
        result_b = candidate.process_chunk(session_b, chunk, 16000)
        assert result_a.output_start_sample == result_b.output_start_sample
        assert result_a.input_end_sample == chunk.shape[0]
        assert result_b.input_end_sample == chunk.shape[0]

    def test_reset(self) -> None:
        candidate = StreamingACCandidate()
        candidate.prepare("cpu", "fp32")
        session = candidate.create_session({"session_id": "r1"})
        chunk = _make_chunk()
        candidate.process_chunk(session, chunk, 16000)
        candidate.reset(session)
        assert session.state["ac"].step == 0


class TestPaperStyleConfig:
    def test_paper_style_config(self) -> None:
        config = StreamingACConfig(mode="paper_style")
        assert config.right_context_ms == 640
        assert config.hidden_dim == 256
        assert config.num_layers == 4
        candidate = StreamingACCandidate(config=config)
        assert candidate.metadata.required_lookahead_ms == 640
        assert isinstance(candidate.encoder, PaperStyleContentProsodyEncoder)


class TestLowLookaheadConfig:
    def test_low_lookahead_config(self) -> None:
        config = StreamingACConfig(mode="low_lookahead")
        assert config.right_context_ms == 0
        assert config.hidden_dim == 128
        assert config.num_layers == 2
        candidate = StreamingACCandidate(config=config)
        assert candidate.config.right_context_ms == 0
        assert isinstance(candidate.encoder, LowLookaheadContentProsodyEncoder)

    def test_low_lookahead_configurable(self) -> None:
        config = StreamingACConfig(
            mode="low_lookahead",
            right_context_ms=200,
            hidden_dim=96,
            num_layers=3,
        )
        candidate = StreamingACCandidate(config=config)
        assert candidate.config.right_context_ms == 200
        assert candidate.config.hidden_dim == 96
        assert candidate.config.num_layers == 3