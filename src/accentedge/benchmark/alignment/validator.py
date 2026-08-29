"""Alignment validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..schemas import AlignmentSource


@dataclass
class AlignmentValidationResult:
    valid: bool
    issues: list[str] = None

    def __post_init__(self):
        if self.issues is None:
            self.issues = []


def validate_alignment(
    alignment: dict[str, list[dict[str, Any]]],
    audio_duration_ms: float,
) -> AlignmentValidationResult:
    """Validate that alignment bounds are within audio duration."""
    issues: list[str] = []
    duration_s = audio_duration_ms / 1000.0

    for tier_name, intervals in alignment.items():
        for iv in intervals:
            start = float(iv.get("start", 0))
            end = float(iv.get("end", 0))
            if start < 0:
                issues.append(f"Tier {tier_name}: negative start time {start}")
            if end > duration_s:
                issues.append(f"Tier {tier_name}: end {end} exceeds audio duration {duration_s}")
            if end < start:
                issues.append(f"Tier {tier_name}: end {end} < start {start}")

    return AlignmentValidationResult(valid=len(issues) == 0, issues=issues)
