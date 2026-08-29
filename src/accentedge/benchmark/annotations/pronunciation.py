"""Pronunciation token annotation handling."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
from ..schemas import PronunciationToken, SourceStatus

@dataclass
class CorrectionResult:
    token: PronunciationToken
    corrected: bool = False
    damaged: bool = False
    off_target: bool = False
    distance_to_target: float = 0.0
    details: dict[str, Any] = field(default_factory=dict)

def compute_correction_damage(
    tokens: list[PronunciationToken],
    results: list[PronunciationToken],
) -> tuple[int, int, int]:
    corrected = sum(1 for t, r in zip(tokens, results)
                    if t.source_status == SourceStatus.DEVIANT and r.source_status == SourceStatus.ALREADY_TARGET)
    damaged = sum(1 for t, r in zip(tokens, results)
                  if t.source_status == SourceStatus.ALREADY_TARGET and r.source_status == SourceStatus.DEVIANT)
    ambiguous = sum(1 for t in tokens if t.source_status == SourceStatus.AMBIGUOUS)
    return corrected, damaged, ambiguous
