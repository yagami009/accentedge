"""Tests for accentedge_benchmark.candidates — adapter framework."""

from __future__ import annotations

import numpy as np
import pytest

from accentedge_benchmark.candidates.base import (
    CandidateAdapter,
    BenchmarkContext,
    CandidateMetadata,
    CandidateOutput,
)
from accentedge_benchmark.candidates.passthrough import PassthroughAdapter
from accentedge_benchmark.candidates.file_output import FileOutputAdapter
from accentedge_benchmark.candidates.registry import register, get, available


# ── PassthroughAdapter ───────────────────────────────────────────────────────


class TestPassthroughAdapter:
    def test_returns_identical_audio(self):
        adapter = PassthroughAdapter()
        audio = np.sin(np.linspace(0, 1, 16000)).astype(np.float32)
        ctx = BenchmarkContext(target_accent="en-US")
        out = adapter.process(audio, 16000, ctx)
        np.testing.assert_array_equal(out.audio, audio)

    def test_preserves_sample_rate(self):
        adapter = PassthroughAdapter()
        audio = np.zeros(100, dtype=np.float32)
        out = adapter.process(audio, 8000, BenchmarkContext(target_accent="en-US"))
        assert out.sample_rate == 8000

    def test_metadata(self):
        adapter = PassthroughAdapter()
        meta = adapter.metadata
        assert meta.name == "passthrough"
        assert meta.version == "1.0.0"
        assert meta.target_accent == "source"

    def test_process_includes_utterance_id_in_metadata(self):
        adapter = PassthroughAdapter()
        audio = np.zeros(100, dtype=np.float32)
        ctx = BenchmarkContext(target_accent="source", utterance_id="utt_001")
        out = adapter.process(audio, 16000, ctx)
        assert out.metadata.get("source_id") == "utt_001"

    def test_process_returns_copy_not_view(self):
        """Modifying output audio must not affect input."""
        adapter = PassthroughAdapter()
        audio = np.ones(100, dtype=np.float32)
        ctx = BenchmarkContext(target_accent="source")
        out = adapter.process(audio, 16000, ctx)
        out.audio[0] = 999.0
        assert audio[0] == 1.0


# ── FileOutputAdapter ─────────────────────────────────────────────────���──────


class TestFileOutputAdapter:
    def test_reads_wav_file(self, tmp_path):
        sr = 16000
        expected = (0.5 * np.sin(2 * np.pi * 440 * np.linspace(0, 1, sr))).astype(np.float32)
        fpath = tmp_path / "utt_001.wav"
        import soundfile as sf
        sf.write(str(fpath), expected, sr, subtype="PCM_16")

        adapter = FileOutputAdapter(tmp_path)
        ctx = BenchmarkContext(target_accent="source", utterance_id="utt_001")
        out = adapter.process(np.zeros(100, dtype=np.float32), sr, ctx)
        assert out.sample_rate == sr
        np.testing.assert_array_almost_equal(out.audio, expected, decimal=4)

    def test_missing_file_raises(self, tmp_path):
        adapter = FileOutputAdapter(tmp_path)
        ctx = BenchmarkContext(target_accent="source", utterance_id="nonexistent")
        with pytest.raises(FileNotFoundError):
            adapter.process(np.zeros(100, dtype=np.float32), 16000, ctx)

    def test_utterance_id_required(self, tmp_path):
        adapter = FileOutputAdapter(tmp_path)
        ctx = BenchmarkContext(target_accent="source", utterance_id=None)
        with pytest.raises(ValueError):
            adapter.process(np.zeros(100, dtype=np.float32), 16000, ctx)

    def test_metadata_includes_source_path(self, tmp_path):
        sr = 16000
        expected = np.zeros(100, dtype=np.float32)
        fpath = tmp_path / "utt_abc.wav"
        import soundfile as sf
        sf.write(str(fpath), expected, sr, subtype="PCM_16")

        adapter = FileOutputAdapter(tmp_path)
        ctx = BenchmarkContext(target_accent="source", utterance_id="utt_abc")
        out = adapter.process(np.zeros(100, dtype=np.float32), sr, ctx)
        assert "source_path" in out.metadata


# ── CandidateRegistry ────────────────────────────────────────────────────────


class TestCandidateRegistry:
    def test_available_contains_passthrough(self):
        names = available()
        assert "passthrough" in names

    def test_get_passthrough(self):
        cls = get("passthrough")
        assert cls is PassthroughAdapter

    def test_get_unknown_raises(self):
        with pytest.raises(KeyError, match="Unknown candidate"):
            get("nonexistent_candidate")

    def test_register_new_adapter(self):
        class DummyAdapter(CandidateAdapter):
            @property
            def metadata(self):
                return CandidateMetadata(name="dummy", version="0.0.0", target_accent="x")

            def process(self, audio, sample_rate, context):
                return CandidateOutput(audio=audio, sample_rate=sample_rate)

        register("dummy_test", DummyAdapter)
        assert "dummy_test" in available()
        cls = get("dummy_test")
        assert cls is DummyAdapter

    def test_register_overwrites_existing(self):
        class OverrideAdapter(CandidateAdapter):
            @property
            def metadata(self):
                return CandidateMetadata(name="override", version="0.0.0", target_accent="x")

            def process(self, audio, sample_rate, context):
                return CandidateOutput(audio=audio, sample_rate=sample_rate)

        register("override_test", OverrideAdapter)
        assert get("override_test") is OverrideAdapter


# ── BenchmarkContext ─────────────────────────────────────────────────────────


class TestBenchmarkContext:
    def test_defaults(self):
        ctx = BenchmarkContext(target_accent="en-US-neutral")
        assert ctx.target_accent == "en-US-neutral"
        assert ctx.conversion_strength is None
        assert ctx.utterance_id is None
        assert ctx.speaker_id is None
        assert ctx.metadata == {}

    def test_custom(self):
        ctx = BenchmarkContext(
            target_accent="en-IN",
            conversion_strength=0.75,
            utterance_id="utt_001",
            speaker_id="spk_001",
            metadata={"note": "test"},
        )
        assert ctx.target_accent == "en-IN"
        assert ctx.conversion_strength == 0.75
        assert ctx.utterance_id == "utt_001"
        assert ctx.speaker_id == "spk_001"
        assert ctx.metadata["note"] == "test"

    def test_metadata_independent(self):
        ctx1 = BenchmarkContext(target_accent="en-US")
        ctx2 = BenchmarkContext(target_accent="en-US")
        ctx1.metadata["key"] = "value"
        # metadata should be independent by default (factory default)
        assert "key" not in ctx2.metadata
