"""Core schemas for AccentEdge BPO Benchmark v1."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, model_validator


class Partition(str, Enum):
    DEV = "dev"
    LOCKED_TEST = "locked_test"
    CALIBRATION = "calibration"


class Family(str, Enum):
    BPO_SCRIPTED = "bpo_scripted"
    CRITICAL_ENTITY = "critical_entity"
    PRONUNCIATION_CONTRAST = "pronunciation_contrast"
    ALREADY_TARGET = "already_target"
    BPO_SPONTANEOUS = "bpo_spontaneous"
    GENERAL_SPONTANEOUS = "general_spontaneous"


class EntityType(str, Enum):
    NUMBER = "NUMBER"
    MONEY = "MONEY"
    DATE = "DATE"
    TIME = "TIME"
    PERSON_NAME = "PERSON_NAME"
    ADDRESS = "ADDRESS"
    POSTCODE = "POSTCODE"
    ACCOUNT_ID = "ACCOUNT_ID"
    ALPHANUMERIC = "ALPHANUMERIC"
    EMAIL = "EMAIL"
    PHONE_NUMBER = "PHONE_NUMBER"


class SourceStatus(str, Enum):
    ALREADY_TARGET = "ALREADY_TARGET"
    DEVIANT = "DEVIANT"
    AMBIGUOUS = "AMBIGUOUS"


class AlignmentSource(str, Enum):
    AUTO = "AUTO"
    HUMAN_CORRECTED = "HUMAN_CORRECTED"


class DegradationCondition(str, Enum):
    CLEAN = "clean"
    NB = "nb"
    NOISY = "noisy"
    NB_NOISY = "nb_noisy"


class ErrorCategory(str, Enum):
    INPUT_AUDIO_ERROR = "INPUT_AUDIO_ERROR"
    CANDIDATE_ERROR = "CANDIDATE_ERROR"
    INVALID_OUTPUT = "INVALID_OUTPUT"
    ALIGNMENT_ERROR = "ALIGNMENT_ERROR"
    ASR_ERROR = "ASR_ERROR"
    SPEAKER_EVAL_ERROR = "SPEAKER_EVAL_ERROR"
    PRONUNCIATION_PROBE_ERROR = "PRONUNCIATION_PROBE_ERROR"
    METRIC_ERROR = "METRIC_ERROR"
    RUNTIME_ERROR = "RUNTIME_ERROR"
    TIMEOUT = "TIMEOUT"


class DatasetItem(BaseModel):
    utterance_id: str = Field(..., pattern=r"^[a-z0-9_]+$")
    speaker_id: str
    partition: Partition
    family: Family
    prompt_id: str | None = None
    raw_path: str | None = None
    canonical_path: str
    sample_rate: int = Field(default=16000, ge=8000, le=48000)
    duration_ms: float = Field(default=0.0, gt=0)
    accent_strength: float = Field(default=0.0, ge=0.0, le=1.0)
    bpo_experience: bool = False
    transcript_verbatim: str = Field(...)
    transcript_normalized: str = Field(...)
    audio_sha256: str = Field(...)
    annotation_version: str = "1.0.0"
    has_critical_entities: bool = False
    has_target_features: bool = False
    l1_category: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _auto_flags(self):
        if self.family == Family.CRITICAL_ENTITY:
            self.has_critical_entities = True
        if self.family == Family.PRONUNCIATION_CONTRAST:
            self.has_target_features = True
        return self


class CriticalEntity(BaseModel):
    entity_id: str
    utterance_id: str
    entity_type: EntityType
    surface: str
    normalized: str
    start_char: int = Field(..., ge=0)
    end_char: int = Field(..., ge=0)
    mandatory_exact_preservation: bool = True

    @model_validator(mode="after")
    def _validate_span(self):
        if self.end_char < self.start_char:
            raise ValueError("end_char must be >= start_char")
        return self


class PronunciationToken(BaseModel):
    token_id: str
    utterance_id: str
    word: str
    feature: str
    phone_label: str
    start_ms: float = Field(..., ge=0)
    end_ms: float = Field(...)
    source_status: SourceStatus
    annotator_confidence: float | None = Field(None, ge=0.0, le=1.0)
    annotation_version: str = "1.0.0"
    notes: str | None = None

    @model_validator(mode="after")
    def _validate_times(self):
        if self.end_ms < self.start_ms:
            raise ValueError("end_ms must be >= start_ms")
        return self


class CandidateMetadata(BaseModel):
    name: str
    version: str = "1.0.0"
    description: str = ""
    target_accent: str = "en-US-neutral"
    supports_conversion_strength: bool = False
    artifact_hash: str | None = None
    configuration: dict[str, Any] = Field(default_factory=dict)


class BenchmarkContext(BaseModel):
    target_accent: str = "en-US-neutral"
    conversion_strength: float | None = None
    utterance_id: str | None = None
    speaker_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class CandidateOutput(BaseModel):
    audio: Any = Field(...)
    sample_rate: int = 16000
    metadata: dict[str, Any] = Field(default_factory=dict)


class RunManifest(BaseModel):
    run_id: str
    benchmark_version: str = "1.0.0"
    dataset_hash: str = ""
    split: str = "dev"
    candidate_name: str = ""
    candidate_version: str = "unknown"
    candidate_hash: str = ""
    config_hash: str = ""
    git_commit: str | None = None
    python_version: str = ""
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    condition: str = "clean"
    conversion_strength: float | None = None
    evaluator_versions: dict[str, str] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class MetricResult(BaseModel):
    metric_name: str
    value: float | None = None
    count: int = 0
    total: int = 0
    rate: float | None = None
    confidence_interval: tuple[float, float] | None = None
    slice_key: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _compute_rate(self):
        if self.rate is None and self.total > 0:
            self.rate = self.count / self.total
        return self


class FailureRecord(BaseModel):
    utterance_id: str
    candidate_name: str
    error_category: ErrorCategory
    error_message: str
    stack_trace: str | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    run_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


__all__ = [
    "Partition",
    "Family",
    "EntityType",
    "SourceStatus",
    "AlignmentSource",
    "DegradationCondition",
    "ErrorCategory",
    "DatasetItem",
    "CriticalEntity",
    "PronunciationToken",
    "CandidateMetadata",
    "BenchmarkContext",
    "CandidateOutput",
    "RunManifest",
    "MetricResult",
    "FailureRecord",
]
