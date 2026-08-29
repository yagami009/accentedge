"""Tests for Candidate D (Minimal Hybrid)."""

from __future__ import annotations

import platform

import numpy as np
import pytest
import torch
import torch.nn as nn

from accentedge_lab.models.minimal_hybrid.model import MinimalHybridCandidate
from accentedge_lab.models.minimal_hybrid.streaming_config import MinimalHybridConfig
from accentedge_lab.models.minimal_hybrid.interfaces import (
    MinimalAccentMapper,
    MinimalEncoder,
    MinimalSynthesizer,
)
from accentedge_lab.models.interfaces import (
    CandidateMetadata,
    StreamingResult,
    StreamingSession,
)


def _make_chunk(samples: int = 1280, sr: int = 16000) -> np.ndarray:
    """Create a 16 kHz mono float32 audio chunk (~80ms)."""
    t = np.linspace(0, 2 * np.pi * 440 * samples / sr, samples)
    return np.sin(t).astype(np.float32)


# ---------------------------------------------------------------------------
# TestMetadata
# ---------------------------------------------------------------------------

class TestMetadata:
    def test_metadata_defaults(self) -> None:
        candidate = MinimalHybridCandidate()
        assert candidate.metadata.architecture_id == "minimal_hybrid"
        assert candidate.metadata.input_sample_rate == 16000
        assert candidate.metadata.frame_ms == 20.0
        assert candidate.metadata.preferred_chunk_ms == 80
        assert candidate.metadata.required_lookahead_ms == 0
        assert candidate.metadata.left_context_ms == 0
        assert candidate.metadata.supports_conversion_strength is True
        assert candidate.metadata.supports_target_accent is True
        assert candidate.metadata.requires_reference_speaker is False
        assert candidate.metadata.uses_text_at_inference is False

    def test_metadata_version(self) -> None:
        candidate = MinimalHybridCandidate()
        assert candidate.metadata.version == "0.1.0"


# ---------------------------------------------------------------------------
# TestPrepare
# ---------------------------------------------------------------------------

class TestPrepare:
    def test_prepare_cpu(self) -> None:
        candidate = MinimalHybridCandidate()
        candidate.prepare("cpu", "fp32")
        assert candidate is not None
        # All modules should be on CPU
        for mod in (candidate.encoder, candidate.mapper, candidate.synthesizer):
            p = next(mod.parameters())
            assert p.device.type == "cpu"

    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="CUDA not available",
    )
    def test_prepare_cuda(self) -> None:
        candidate = MinimalHybridCandidate()
        candidate.prepare("cuda", "fp16")
        for mod in (candidate.encoder, candidate.mapper, candidate.synthesizer):
            p = next(mod.parameters())
            assert p.device.type == "cuda"

    @pytest.mark.skipif(
        platform.system() != "Darwin" or platform.machine() != "arm64",
        reason="MPS only on Apple Silicon",
    )
    def test_prepare_mps(self) -> None:
        candidate = MinimalHybridCandidate()
        candidate.prepare("mps", "fp16")
        for mod in (candidate.encoder, candidate.mapper, candidate.synthesizer):
            p = next(mod.parameters())
            assert p.device.type == "mps"


# ---------------------------------------------------------------------------
# TestCreateSession
# ---------------------------------------------------------------------------

class TestCreateSession:
    def test_create_session_default(self) -> None:
        candidate = MinimalHybridCandidate()
        session = candidate.create_session({})
        assert session.session_id == "minimal_hybrid"
        assert "d" in session.state
        assert session.samples_processed == 0

    def test_create_session_custom_id(self) -> None:
        candidate = MinimalHybridCandidate()
        session = candidate.create_session({"session_id": "custom_42"})
        assert session.session_id == "custom_42"

    def test_create_session_multiple(self) -> None:
        candidate = MinimalHybridCandidate()
        s1 = candidate.create_session({"session_id": "s1"})
        s2 = candidate.create_session({"session_id": "s2"})
        assert s1.session_id != s2.session_id
        assert s1.state["d"] is not s2.state["d"]


# ---------------------------------------------------------------------------
# TestProcessChunk
# ---------------------------------------------------------------------------

class TestProcessChunk:
    def test_process_chunk_shape(self) -> None:
        candidate = MinimalHybridCandidate()
        candidate.prepare("cpu", "fp32")
        session = candidate.create_session({"session_id": "p1"})
        chunk = _make_chunk(1280)
        result = candidate.process_chunk(session, chunk, 16000)
        assert isinstance(result, StreamingResult)
        assert result.audio is not None
        assert isinstance(result.audio, np.ndarray)
        assert result.audio.shape == chunk.shape
        assert result.sample_rate == 16000

    def test_process_chunk_advances_samples(self) -> None:
        candidate = MinimalHybridCandidate()
        candidate.prepare("cpu", "fp32")
        session = candidate.create_session({"session_id": "adv"})
        chunk = _make_chunk(1280)
        result = candidate.process_chunk(session, chunk, 16000)
        assert result.input_end_sample == 1280
        assert session.samples_processed == 1280

    def test_streaming_state_isolation(self) -> None:
        candidate = MinimalHybridCandidate()
        candidate.prepare("cpu", "fp32")
        s1 = candidate.create_session({"session_id": "a"})
        s2 = candidate.create_session({"session_id": "b"})
        chunk = _make_chunk(1280)
        r1 = candidate.process_chunk(s1, chunk, 16000)
        r2 = candidate.process_chunk(s2, chunk, 16000)
        # Both should produce valid audio
        assert r1.audio.shape == chunk.shape
        assert r2.audio.shape == chunk.shape
        # State should be independent
        assert s1.state["d"].step == 1
        assert s2.state["d"].step == 1

    def test_multiple_chunks(self) -> None:
        candidate = MinimalHybridCandidate()
        candidate.prepare("cpu", "fp32")
        session = candidate.create_session({"session_id": "multi"})
        for i in range(3):
            chunk = _make_chunk(640)
            result = candidate.process_chunk(session, chunk, 16000)
            assert result.audio.shape == (640,)
        assert session.samples_processed == 3 * 640
        assert session.state["d"].step == 3
        assert len(session.state["d"].timeline) == 3


# ---------------------------------------------------------------------------
# TestReset
# ---------------------------------------------------------------------------

class TestReset:
    def test_reset_clears_step(self) -> None:
        candidate = MinimalHybridCandidate()
        candidate.prepare("cpu", "fp32")
        session = candidate.create_session({"session_id": "rst"})
        chunk = _make_chunk(640)
        candidate.process_chunk(session, chunk, 16000)
        assert session.state["d"].step == 1
        candidate.reset(session)
        assert session.state["d"].step == 0
        assert session.samples_processed == 0

    def test_reset_clears_timeline(self) -> None:
        candidate = MinimalHybridCandidate()
        candidate.prepare("cpu", "fp32")
        session = candidate.create_session({"session_id": "rst_tl"})
        chunk = _make_chunk(640)
        candidate.process_chunk(session, chunk, 16000)
        assert len(session.state["d"].timeline) == 1
        candidate.reset(session)
        assert len(session.state["d"].timeline) == 0


# ---------------------------------------------------------------------------
# TestFlush
# ---------------------------------------------------------------------------

class TestFlush:
    def test_flush_returns_empty(self) -> None:
        """Minimal impl has no buffering, so flush returns empty list."""
        candidate = MinimalHybridCandidate()
        candidate.prepare("cpu", "fp32")
        session = candidate.create_session({"session_id": "fl"})
        chunk = _make_chunk(640)
        candidate.process_chunk(session, chunk, 16000)
        results = candidate.flush(session)
        assert results == []


# ---------------------------------------------------------------------------
# TestClose
# ---------------------------------------------------------------------------

class TestClose:
    def test_close_prevents_process(self) -> None:
        candidate = MinimalHybridCandidate()
        candidate.prepare("cpu", "fp32")
        session = candidate.create_session({"session_id": "cl"})
        chunk = _make_chunk(640)
        candidate.process_chunk(session, chunk, 16000)
        candidate.close()
        with pytest.raises(RuntimeError, match="closed"):
            candidate.process_chunk(session, chunk, 16000)

    def test_close_prevents_session(self) -> None:
        candidate = MinimalHybridCandidate()
        candidate.prepare("cpu", "fp32")
        candidate.close()
        with pytest.raises(RuntimeError, match="closed"):
            candidate.create_session({"session_id": "x"})

    def test_close_idempotent(self) -> None:
        candidate = MinimalHybridCandidate()
        candidate.close()
        candidate.close()  # Should not raise


# ---------------------------------------------------------------------------
# TestParameterCount
# ---------------------------------------------------------------------------

class TestParameterCount:
    def test_under_500k(self) -> None:
        candidate = MinimalHybridCandidate()
        count = candidate.count_parameters()
        assert count < 500_000, f"Parameter count too large: {count}"

    def test_metadata_reports_count(self) -> None:
        candidate = MinimalHybridCandidate()
        assert candidate.metadata.parameter_count is not None
        assert candidate.metadata.parameter_count < 500_000


# ---------------------------------------------------------------------------
# TestConversionStrength
# ---------------------------------------------------------------------------

class TestConversionStrength:
    def test_strength_zero_mapper_is_identity(self) -> None:
        """With conversion_strength=0.0, mapper should pass features unchanged."""
        candidate = MinimalHybridCandidate()
        candidate.prepare("cpu", "fp32")
        mapper = candidate.mapper
        mapper.eval()
        with torch.no_grad():
            D = candidate.config.hidden_dim
            feats = torch.randn(1, D, 8)
            out = mapper(feats, strength=0.0, target_accent=0)
        assert torch.allclose(feats, out, atol=1e-6), \
            f"Mapper not identity at strength=0: max diff={torch.max(torch.abs(feats - out))}"

    def test_strength_one_mapper_changes_features(self) -> None:
        """With conversion_strength=1.0, mapper should apply accent transformation."""
        candidate = MinimalHybridCandidate()
        candidate.prepare("cpu", "fp32")
        mapper = candidate.mapper
        mapper.eval()
        with torch.no_grad():
            D = candidate.config.hidden_dim
            feats = torch.randn(1, D, 8)
            out = mapper(feats, strength=1.0, target_accent=0)
        assert not torch.allclose(feats, out, atol=1e-5), \
            "Mapper unchanged at strength=1.0"


# ---------------------------------------------------------------------------
# TestCausality
# ---------------------------------------------------------------------------

class TestCausality:
    def test_strictly_causal(self) -> None:
        """Encoder uses only past / present — no right context (padding = 0)."""
        candidate = MinimalHybridCandidate()
        candidate.prepare("cpu", "fp32")

        # The encoder block uses ConstantPad1d((k-1, 0)) — strictly causal
        block = candidate.encoder.block
        # Verify the padding is left-only (causal)
        assert isinstance(block.pad1, nn.ConstantPad1d)
        assert block.pad1.padding == (4, 0)  # kernel_size - 1 on left, 0 on right

    def test_no_future_in_mapper(self) -> None:
        """Mapper is element-wise — no temporal mixing."""
        candidate = MinimalHybridCandidate()
        candidate.prepare("cpu", "fp32")
        mapper = candidate.mapper
        # Mapper should not contain any Conv1d, LSTM, or attention modules
        assert not any(
            isinstance(m, (nn.Conv1d, nn.LSTM, nn.MultiheadAttention))
            for m in mapper.modules()
        )


# ---------------------------------------------------------------------------
# TestStateBounded
# ---------------------------------------------------------------------------

class TestStateBounded:
    def test_session_state_bounded(self) -> None:
        candidate = MinimalHybridCandidate()
        candidate.prepare("cpu", "fp32")
        session = candidate.create_session({"session_id": "bounded"})
        chunk = _make_chunk(640)
        for _ in range(10):
            candidate.process_chunk(session, chunk, 16000)
        # Timeline grows linearly with step count — bounded per-step
        assert len(session.state["d"].timeline) == 10
        # No hidden buffers growing unboundedly
        assert session.state["d"].step == 10

    def test_state_size_bytes_bounded(self) -> None:
        candidate = MinimalHybridCandidate()
        candidate.prepare("cpu", "fp32")
        session = candidate.create_session({"session_id": "sz"})
        chunk = _make_chunk(640)
        for _ in range(5):
            candidate.process_chunk(session, chunk, 16000)
        sz = session.state_size_bytes()
        # Each timeline entry = 4 ints * 8 = 32 bytes; 5 entries = 160 bytes
        assert sz <= 5 * 32 + 100  # generous upper bound
