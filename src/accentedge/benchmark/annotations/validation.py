"""Annotation validation."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
from ..schemas import CriticalEntity, PronunciationToken

@dataclass
class AnnotationValidationResult:
    valid: bool
    issues: list[str] = None
    def __post_init__(self):
        if self.issues is None:
            self.issues = []

def validate_entities(entities: list[CriticalEntity], transcript: str) -> AnnotationValidationResult:
    issues = []
    for e in entities:
        if e.start_char >= len(transcript) or e.end_char > len(transcript):
            issues.append(f'{e.entity_id}: span outside transcript')
        if e.end_char < e.start_char:
            issues.append(f'{e.entity_id}: end before start')
    return AnnotationValidationResult(valid=len(issues) == 0, issues=issues)

def validate_pronunciation_tokens(
    tokens: list[PronunciationToken], 
    audio_duration_ms: float
) -> AnnotationValidationResult:
    issues = []
    for t in tokens:
        if t.start_ms < 0:
            issues.append(f'{t.token_id}: negative start time')
        if t.end_ms > audio_duration_ms:
            issues.append(f'{t.token_id}: end beyond audio duration')
        if t.end_ms < t.start_ms:
            issues.append(f'{t.token_id}: end before start')
    return AnnotationValidationResult(valid=len(issues) == 0, issues=issues)
