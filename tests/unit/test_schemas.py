
"""Tests for accentedge_benchmark.schemas - Pydantic v2 models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from accentedge_benchmark.schemas import (
    Partition, Family, EntityType, SourceStatus, AlignmentSource,
    DegradationCondition, ErrorCategory,
    DatasetItem, CriticalEntity, PronunciationToken,
    CandidateMetadata, BenchmarkContext,
    RunManifest, MetricResult, FailureRecord,
)


class TestPartition:
    def test_values(self):
        assert Partition.DEV == "dev"
        assert Partition.LOCKED_TEST == "locked_test"
        assert Partition.CALIBRATION == "calibration"

    def test_from_string(self):
        assert Partition("dev") == Partition.DEV


class TestFamily:
    def test_values(self):
        assert Family.BPO_SCRIPTED == "bpo_scripted"
        assert Family.CRITICAL_ENTITY == "critical_entity"
        assert Family.PRONUNCIATION_CONTRAST == "pronunciation_contrast"
        assert Family.ALREADY_TARGET == "already_target"
        assert Family.BPO_SPONTANEOUS == "bpo_spontaneous"
        assert Family.GENERAL_SPONTANEOUS == "general_spontaneous"


class TestEntityType:
    def test_common_values(self):
        assert EntityType.MONEY == "MONEY"
        assert EntityType.DATE == "DATE"
        assert EntityType.NUMBER == "NUMBER"


class TestSourceStatus:
    def test_values(self):
        assert SourceStatus.ALREADY_TARGET == "ALREADY_TARGET"
        assert SourceStatus.DEVIANT == "DEVIANT"
        assert SourceStatus.AMBIGUOUS == "AMBIGUOUS"

    def test_count(self):
        assert len(SourceStatus) == 3


class TestDegradationCondition:
    def test_values(self):
        assert DegradationCondition.CLEAN == "clean"
        assert DegradationCondition.NB == "nb"
        assert DegradationCondition.NOISY == "noisy"
        assert DegradationCondition.NB_NOISY == "nb_noisy"

    def test_count(self):
        assert len(DegradationCondition) == 4


class TestErrorCategory:
    def test_count(self):
        assert len(ErrorCategory) == 10

    def test_has_common_values(self):
        assert ErrorCategory.INPUT_AUDIO_ERROR.value == "INPUT_AUDIO_ERROR"
        assert ErrorCategory.CANDIDATE_ERROR.value == "CANDIDATE_ERROR"
        assert ErrorCategory.METRIC_ERROR.value == "METRIC_ERROR"


# ── DatasetItem ─────────────────────────────────────────────────────────────

class TestDatasetItem:
    def test_minimal_valid(self):
        item = DatasetItem(
            utterance_id="utt_001",
            speaker_id="spk_001",
            partition=Partition.DEV,
            family=Family.BPO_SCRIPTED,
            canonical_path="/data/spk_001/utt_001.wav",
            sample_rate=16000,
            duration_ms=2500.0,
            transcript_verbatim="hello world",
            transcript_normalized="hello world",
            audio_sha256="a" * 64,
        )
        assert item.utterance_id == "utt_001"
        assert item.speaker_id == "spk_001"
        assert item.partition == Partition.DEV
        assert item.family == Family.BPO_SCRIPTED
        assert item.bpo_experience is False
        assert item.transcript_verbatim == "hello world"
        assert item.transcript_normalized == "hello world"
        assert item.audio_sha256 == "a" * 64

    def test_invalid_utterance_id_rejected(self):
        with pytest.raises(ValidationError):
            DatasetItem(
                utterance_id="INVALID!",
                speaker_id="spk_001",
                partition=Partition.DEV,
                family=Family.BPO_SCRIPTED,
                canonical_path="/data/utt.wav",
                sample_rate=16000,
                duration_ms=1000.0,
                transcript_verbatim="hi",
                transcript_normalized="hi",
                audio_sha256="a" * 64,
            )

    def test_invalid_sample_rate_rejected(self):
        with pytest.raises(ValidationError):
            DatasetItem(
                utterance_id="utt_001",
                speaker_id="spk_001",
                partition=Partition.DEV,
                family=Family.BPO_SCRIPTED,
                canonical_path="/data/utt.wav",
                sample_rate=123,
                duration_ms=1000.0,
                transcript_verbatim="hi",
                transcript_normalized="hi",
                audio_sha256="a" * 64,
            )

    def test_duration_must_be_positive(self):
        with pytest.raises(ValidationError):
            DatasetItem(
                utterance_id="utt_001",
                speaker_id="spk_001",
                partition=Partition.DEV,
                family=Family.BPO_SCRIPTED,
                canonical_path="/data/utt.wav",
                sample_rate=16000,
                duration_ms=0.0,
                transcript_verbatim="hi",
                transcript_normalized="hi",
                audio_sha256="a" * 64,
            )

    def test_with_all_optional_fields(self):
        item = DatasetItem(
            utterance_id="utt_001",
            speaker_id="spk_001",
            partition=Partition.CALIBRATION,
            family=Family.CRITICAL_ENTITY,
            canonical_path="/data/utt.wav",
            sample_rate=48000,
            duration_ms=5000.0,
            transcript_verbatim="text",
            transcript_normalized="text",
            audio_sha256="b" * 64,
        )
        assert item.partition == Partition.CALIBRATION
        assert item.family == Family.CRITICAL_ENTITY
        assert item.sample_rate == 48000


# ── CriticalEntity ──────────────────────��───────────────────────────────────

class TestCriticalEntity:
    def test_valid(self):
        entity = CriticalEntity(
            entity_id="e1", utterance_id="u1",
            entity_type=EntityType.MONEY,
            surface="$50", normalized="USD:50.00",
            start_char=0, end_char=3,
        )
        assert entity.entity_id == "e1"
        assert entity.entity_type == EntityType.MONEY
        assert entity.mandatory_exact_preservation is True

    def test_end_before_start_raises(self):
        with pytest.raises(ValidationError, match="end_char must be >= start_char"):
            CriticalEntity(
                entity_id="e1", utterance_id="u1",
                entity_type=EntityType.MONEY,
                surface="$50", normalized="USD:50.00",
                start_char=10, end_char=5,
            )

    def test_negative_start_raises(self):
        with pytest.raises(ValidationError, match="greater than or equal to 0"):
            CriticalEntity(
                entity_id="e1", utterance_id="u1",
                entity_type=EntityType.MONEY,
                surface="$50", normalized="USD:50.00",
                start_char=-1, end_char=3,
            )

    def test_span_within_transcript(self):
        entity = CriticalEntity(
            entity_id="e1", utterance_id="u1",
            entity_type=EntityType.MONEY,
            surface="$50", normalized="USD:50.00",
            start_char=0, end_char=3,
        )
        assert entity.start_char <= entity.end_char <= len("$50")


# ── PronunciationToken ─────────────────────────────────────────────────────

class TestPronunciationToken:
    def test_deviant_status(self):
        tok = PronunciationToken(
            token_id="t1", utterance_id="u1", word="hello",
            feature="h-vowel", phone_label="HH AH L OW",
            start_ms=0.0, end_ms=500.0,
            source_status=SourceStatus.DEVIANT,
        )
        assert tok.source_status == SourceStatus.DEVIANT

    def test_already_target_status(self):
        tok = PronunciationToken(
            token_id="t1", utterance_id="u1", word="hello",
            feature="h-vowel", phone_label="HH AH L OW",
            start_ms=0.0, end_ms=500.0,
            source_status=SourceStatus.ALREADY_TARGET,
        )
        assert tok.source_status == SourceStatus.ALREADY_TARGET

    def test_ambiguous_status(self):
        tok = PronunciationToken(
            token_id="t1", utterance_id="u1", word="hello",
            feature="h-vowel", phone_label="HH AH L OW",
            start_ms=0.0, end_ms=500.0,
            source_status=SourceStatus.AMBIGUOUS,
        )
        assert tok.source_status == SourceStatus.AMBIGUOUS

    def test_end_before_start_raises(self):
        with pytest.raises(ValidationError, match="end_ms must be >= start_ms"):
            PronunciationToken(
                token_id="t1", utterance_id="u1", word="hello",
                feature="h-vowel", phone_label="HH AH L OW",
                start_ms=500.0, end_ms=100.0,
                source_status=SourceStatus.DEVIANT,
            )

    def test_negative_start_raises(self):
        with pytest.raises(ValidationError, match="greater than or equal to 0"):
            PronunciationToken(
                token_id="t1", utterance_id="u1", word="hello",
                feature="h-vowel", phone_label="HH AH L OW",
                start_ms=-1.0, end_ms=500.0,
                source_status=SourceStatus.DEVIANT,
            )


# ─�� RunManifest ─────────────────────────────────────────────────────────────

class TestRunManifest:
    def test_creation(self):
        manifest = RunManifest(
            run_id="run_001",
            candidate_name="cand_a",
            candidate_hash="abc",
            config_hash="cfg",
            split="dev",
            condition="clean",
            dataset_hash="ds_hash",
            benchmark_version="1.0.0",
        )
        assert manifest.candidate_name == "cand_a"
        assert manifest.split == "dev"
        assert manifest.condition == "clean"
        assert manifest.benchmark_version == "1.0.0"


# ── MetricResult ────────────────────────────────────────────────────────────

class TestMetricResult:
    def test_creation(self):
        result = MetricResult(
            metric_name="wer",
            value=0.15,
            count=15,
            total=100,
        )
        assert result.metric_name == "wer"
        assert result.value == 0.15
        assert result.count == 15
        assert result.total == 100
        assert result.rate == 0.15

    def test_rate_none_when_total_zero(self):
        result = MetricResult(
            metric_name="wer", value=0.0, count=0, total=0,
        )
        assert result.rate is None

    def test_rate_computed_when_total_positive(self):
        result = MetricResult(
            metric_name="entity_rate", value=8.0, count=8, total=10,
        )
        assert result.rate == 0.8


# ── FailureRecord ───────────────────────────────────────────────────────────

class TestFailureRecord:
    def test_creation(self):
        rec = FailureRecord(
            utterance_id="utt_001",
            candidate_name="cand_a",
            error_category=ErrorCategory.RUNTIME_ERROR,
            error_message="test error",
            stack_trace="Traceback...",
            run_id="run_001",
        )
        assert rec.utterance_id == "utt_001"
        assert rec.candidate_name == "cand_a"
        assert rec.error_category == ErrorCategory.RUNTIME_ERROR
        assert rec.error_message == "test error"
        assert rec.run_id == "run_001"

    def test_default_timestamp(self):
        rec = FailureRecord(
            utterance_id="u1",
            candidate_name="c",
            error_category=ErrorCategory.TIMEOUT,
            error_message="timeout",
        )
        assert rec.timestamp.tzinfo is not None

    def test_defaults(self):
        rec = FailureRecord(
            utterance_id="u1",
            candidate_name="c",
            error_category=ErrorCategory.INPUT_AUDIO_ERROR,
            error_message="not found",
        )
        assert rec.stack_trace is None
        assert rec.run_id is None
        assert rec.metadata == {}
