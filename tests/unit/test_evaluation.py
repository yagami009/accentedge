"""Tests for accentedge_benchmark.evaluation — evaluation framework."""

from __future__ import annotations

import numpy as np
import pytest

from accentedge_benchmark.schemas import CriticalEntity, EntityType, PronunciationToken, SourceStatus
from accentedge_benchmark.evaluation.content import ContentEvaluator
from accentedge_benchmark.evaluation.entities import EntityEvaluator
from accentedge_benchmark.evaluation.artifacts import ArtifactEvaluator, evaluate_artifacts
from accentedge_benchmark.evaluation.timing import TimingEvaluator
from accentedge_benchmark.evaluation.pronunciation import PronunciationEvaluator
from accentedge_benchmark.evaluation.naturalness import NaturalnessEvaluator


# ── ContentEvaluator ─────────────────────────────────────────────────────────


class TestContentEvaluator:
    def test_no_asr_returns_zero_wer(self, tmp_path):
        """Without an ASR backend, WER and CER should be 0.0."""
        import soundfile as sf
        sr = 16000
        audio = np.zeros(sr, dtype=np.float32)
        fpath = tmp_path / "test.wav"
        sf.write(str(fpath), audio, sr)

        evaluator = ContentEvaluator()
        result = evaluator.evaluate(str(fpath), "hello world")
        assert result.wer == 0.0
        assert result.cer == 0.0

    def test_with_mock_asr(self, tmp_path):
        """With a mock ASR backend, the evaluator uses the transcription."""
        import soundfile as sf
        sr = 16000
        audio = np.zeros(sr, dtype=np.float32)
        fpath = tmp_path / "test.wav"
        sf.write(str(fpath), audio, sr)

        class MockASR:
            def transcribe(self, path):
                return "hello world"

        evaluator = ContentEvaluator(asr_backend=MockASR())
        result = evaluator.evaluate(str(fpath), "hello world")
        assert result.wer == 0.0
        assert result.recognized_text == "hello world"

    def test_with_mock_asr_mismatch(self, tmp_path):
        import soundfile as sf
        sr = 16000
        audio = np.zeros(sr, dtype=np.float32)
        fpath = tmp_path / "test.wav"
        sf.write(str(fpath), audio, sr)

        class BadASR:
            def transcribe(self, path):
                return "completely different text"

        evaluator = ContentEvaluator(asr_backend=BadASR())
        result = evaluator.evaluate(str(fpath), "hello world")
        assert result.wer > 0.0
        assert result.recognized_text == "completely different text"

    def test_word_count(self, tmp_path):
        import soundfile as sf
        sr = 16000
        audio = np.zeros(sr, dtype=np.float32)
        fpath = tmp_path / "test.wav"
        sf.write(str(fpath), audio, sr)

        evaluator = ContentEvaluator()
        result = evaluator.evaluate(str(fpath), "hello world test")
        assert result.word_count == 3


# ── EntityEvaluator ──────────────────────────────────────────────────────────


class TestEntityEvaluator:
    def test_all_entities_found(self):
        evaluator = EntityEvaluator()
        entities = [
            CriticalEntity(
                entity_id="e1",
                utterance_id="u1",
                entity_type=EntityType.MONEY,
                surface="$50.00",
                normalized="USD:50.00",
                start_char=0,
                end_char=6,
            ),
        ]
        result = evaluator.evaluate(entities, "pay USD:50.00 now")
        assert result.entities_evaluated == 1
        assert result.entities_correct == 1
        assert result.entity_rate == 1.0

    def test_entity_not_found(self):
        evaluator = EntityEvaluator()
        entities = [
            CriticalEntity(
                entity_id="e1",
                utterance_id="u1",
                entity_type=EntityType.MONEY,
                surface="$50",
                normalized="USD:50.00",
                start_char=0,
                end_char=3,
            ),
        ]
        result = evaluator.evaluate(entities, "pay nothing")
        assert result.entities_correct == 0
        assert result.entity_rate == 0.0

    def test_empty_entities(self):
        evaluator = EntityEvaluator()
        result = evaluator.evaluate([], "hello world")
        assert result.entities_evaluated == 0
        assert result.entity_rate == 0.0

    def test_by_type_breakdown(self):
        evaluator = EntityEvaluator()
        entities = [
            CriticalEntity(
                entity_id="e1",
                utterance_id="u1",
                entity_type=EntityType.MONEY,
                surface="$50",
                normalized="USD:50.00",
                start_char=0,
                end_char=3,
            ),
            CriticalEntity(
                entity_id="e2",
                utterance_id="u1",
                entity_type=EntityType.DATE,
                surface="January 5",
                normalized="01/05",
                start_char=10,
                end_char=20,
            ),
        ]
        result = evaluator.evaluate(entities, "pay $50.00 on january 5")
        assert "MONEY" in result.by_type
        assert "DATE" in result.by_type
        assert result.by_type["MONEY"]["rate"] == 1.0
        assert result.by_type["DATE"]["rate"] == 1.0


# ── ArtifactEvaluator ────────────────────────────────────────────────────────


class TestArtifactEvaluator:
    def test_clean_audio_no_flags(self):
        audio = np.zeros(16000, dtype=np.float32)
        evaluator = ArtifactEvaluator()
        result = evaluator.evaluate(audio, 16000)
        assert result.sample_rate == 16000
        assert result.artifact_flags == []

    def test_nan_detected(self):
        audio = np.full(100, 0.5, dtype=np.float32)
        audio[50] = np.nan
        evaluator = ArtifactEvaluator()
        result = evaluator.evaluate(audio, 16000)
        assert "non_finite" in result.artifact_flags

    def test_inf_detected(self):
        audio = np.full(100, 0.5, dtype=np.float32)
        audio[50] = np.inf
        evaluator = ArtifactEvaluator()
        result = evaluator.evaluate(audio, 16000)
        assert "non_finite" in result.artifact_flags

    def test_clipping_detected(self):
        audio = np.ones(100, dtype=np.float32) * 2.0
        evaluator = ArtifactEvaluator()
        result = evaluator.evaluate(audio, 16000)
        assert "clipping" in result.artifact_flags

    def test_no_clipping_when_below_threshold(self):
        audio = np.ones(100, dtype=np.float32) * 0.5
        evaluator = ArtifactEvaluator()
        result = evaluator.evaluate(audio, 16000)
        assert "clipping" not in result.artifact_flags

    def test_convenience_function(self):
        audio = np.zeros(100, dtype=np.float32)
        result = evaluate_artifacts(audio, 16000)
        assert isinstance(result, type(ArtifactEvaluator().evaluate(audio, 16000)))


# ── TimingEvaluator ──────────────────────────────────────────────────────────


class TestTimingEvaluator:
    def test_same_duration_ratio_one(self):
        evaluator = TimingEvaluator()
        audio = np.zeros(16000, dtype=np.float32)
        result = evaluator.evaluate(audio, audio, 16000, 16000)
        assert result.duration_ratio == pytest.approx(1.0)
        assert result.duration_delta_ms == pytest.approx(0.0)
        assert result.within_bounds is True

    def test_longer_output_ratio_greater_than_one(self):
        evaluator = TimingEvaluator()
        src = np.zeros(8000, dtype=np.float32)
        out = np.zeros(16000, dtype=np.float32)
        result = evaluator.evaluate(src, out, 8000, 8000)
        assert result.duration_ratio == pytest.approx(2.0)

    def test_outside_bounds(self):
        evaluator = TimingEvaluator(tolerance_ms=10.0)
        src = np.zeros(8000, dtype=np.float32)
        # Output is 100ms longer than source at 8kHz = 800 samples longer = 100ms
        out = np.zeros(8800, dtype=np.float32)
        result = evaluator.evaluate(src, out, 8000, 8000)
        assert result.within_bounds is False

    def test_within_bounds(self):
        evaluator = TimingEvaluator(tolerance_ms=100.0)
        src = np.zeros(8000, dtype=np.float32)
        out = np.zeros(8800, dtype=np.float32)
        result = evaluator.evaluate(src, out, 8000, 8000)
        assert result.within_bounds is True


# ── PronunciationEvaluator ───────────────────────────────────────────────────


class TestPronunciationEvaluator:
    def test_counts_correction(self):
        evaluator = PronunciationEvaluator()
        source = [
            PronunciationToken(
                token_id="t1", utterance_id="u1", word="hello",
                feature="h-vowel", phone_label="HH AH L OW",
                start_ms=0.0, end_ms=500.0,
                source_status=SourceStatus.DEVIANT,
            ),
        ]
        output = [
            PronunciationToken(
                token_id="t1", utterance_id="u1", word="hello",
                feature="h-vowel", phone_label="HH AH L OW",
                start_ms=0.0, end_ms=500.0,
                source_status=SourceStatus.ALREADY_TARGET,
            ),
        ]
        results = evaluator.evaluate(source, output)
        assert len(results) == 1
        assert results[0].eligible_correction == 1
        assert results[0].corrected == 1
        assert results[0].correction_rate == pytest.approx(1.0)

    def test_counts_damage(self):
        evaluator = PronunciationEvaluator()
        source = [
            PronunciationToken(
                token_id="t1", utterance_id="u1", word="hello",
                feature="h-vowel", phone_label="HH AH L OW",
                start_ms=0.0, end_ms=500.0,
                source_status=SourceStatus.ALREADY_TARGET,
            ),
        ]
        output = [
            PronunciationToken(
                token_id="t1", utterance_id="u1", word="hello",
                feature="h-vowel", phone_label="HH AH L OW",
                start_ms=0.0, end_ms=500.0,
                source_status=SourceStatus.DEVIANT,
            ),
        ]
        results = evaluator.evaluate(source, output)
        assert results[0].eligible_damage == 1
        assert results[0].damaged == 1
        assert results[0].damage_rate == pytest.approx(1.0)

    def test_counts_ambiguous(self):
        evaluator = PronunciationEvaluator()
        source = [
            PronunciationToken(
                token_id="t1", utterance_id="u1", word="hello",
                feature="h-vowel", phone_label="HH AH L OW",
                start_ms=0.0, end_ms=500.0,
                source_status=SourceStatus.AMBIGUOUS,
            ),
        ]
        output = []
        results = evaluator.evaluate(source, output)
        assert results[0].ambiguous == 1


# ── NaturalnessEvaluator ─────────────────────────────────────────────────────


class TestNaturalnessEvaluator:
    def test_unavailable_without_model(self):
        """Without torch/transformers installed, model stays None."""
        evaluator = NaturalnessEvaluator()
        result = evaluator.evaluate("/nonexistent/path.wav")
        assert result.evaluator_name == "unavailable"
        assert result.predicted_mos is None
        assert result.is_auto is False

    def test_result_defaults(self):
        from accentedge_benchmark.evaluation.naturalness import NaturalnessResult
        r = NaturalnessResult()
        assert r.predicted_mos is None
        assert r.human_mos is None
        assert r.evaluator_name == "unavailable"
        assert r.is_auto is False
