"""Per-item failure isolation for benchmark runs."""

from __future__ import annotations

import json
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np


class ErrorCategory(str, Enum):
    INPUT_AUDIO_ERROR = "INPUT_AUDIO_ERROR"
    CANDIDATE_ERROR = "CANDIDATE_ERROR"
    INVALID_OUTPUT = "INVALID_OUTPUT"
    ALIGNMENT_ERROR = "ALIGNMENT_ERROR"
    ASR_ERROR = "ASR_ERROR"
    SPEAKER_EVAL_ERROR = "SPEAKER_EVAL_ERROR"
    PRONUNCIATION_PROBE_ERROR = "PRONUNCIATION_PROBE_ERROR"
    METRIC_ERROR = "METRIC_ERROR"


@dataclass
class FailureRecord:
    """Record of a single item-level failure."""
    utterance_id: str
    candidate_name: str
    error_category: ErrorCategory
    error_message: str
    stack_trace: str | None = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    run_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def classify_error(exc: BaseException, context: str) -> ErrorCategory:
    """Classify an exception into an error category."""
    msg = str(exc).lower()
    if "audio" in context.lower() or "load" in msg or "file" in msg:
        return ErrorCategory.INPUT_AUDIO_ERROR
    if "candidate" in context.lower() or "process" in msg:
        return ErrorCategory.CANDIDATE_ERROR
    if "invalid" in msg or "nan" in msg or "inf" in msg or "clip" in msg:
        return ErrorCategory.INVALID_OUTPUT
    if "align" in msg:
        return ErrorCategory.ALIGNMENT_ERROR
    if "asr" in msg or "transcri" in msg:
        return ErrorCategory.ASR_ERROR
    if "speaker" in msg or "identity" in msg:
        return ErrorCategory.SPEAKER_EVAL_ERROR
    if "probe" in msg or "pronunc" in msg:
        return ErrorCategory.PRONUNCIATION_PROBE_ERROR
    if isinstance(exc, ValueError):
        return ErrorCategory.CANDIDATE_ERROR
    return ErrorCategory.METRIC_ERROR


def record_failure(
    utterance_id: str,
    candidate_name: str,
    exc: BaseException,
    context: str = "",
    run_id: str | None = None,
) -> FailureRecord:
    """Create a FailureRecord from an exception."""
    category = classify_error(exc, context)
    tb_str = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    return FailureRecord(
        utterance_id=utterance_id,
        candidate_name=candidate_name,
        error_category=category,
        error_message=str(exc),
        stack_trace=tb_str,
        run_id=run_id,
    )


class FailureCollector:
    """Collects failures during a benchmark run."""

    def __init__(self):
        self._failures: list[FailureRecord] = []

    def record(self, failure: FailureRecord) -> None:
        self._failures.append(failure)

    @property
    def count(self) -> int:
        return len(self._failures)

    @property
    def failures(self) -> list[FailureRecord]:
        return list(self._failures)

    def to_jsonl(self, path: Path) -> None:
        """Write failures to a JSONL file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            for failure in self._failures:
                record = failure.__dict__.copy()
                record["timestamp"] = failure.timestamp.isoformat()
                record["error_category"] = failure.error_category.value
                f.write(json.dumps(record) + "\n")

    def by_category(self) -> dict[str, int]:
        """Count failures by category."""
        counts: dict[str, int] = {}
        for f in self._failures:
            cat = f.error_category.value
            counts[cat] = counts.get(cat, 0) + 1
        return counts
