"""Critical entity annotation handling."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from ..schemas import CriticalEntity, EntityType, SourceStatus, PronunciationToken


@dataclass
class EntityMatch:
    """A matched entity in a transcript."""
    entity: CriticalEntity
    recognized: str
    correct: bool
    match_type: str
    confidence: float | None = None


def normalize_money(text: str) -> str:
    """Normalize money expressions to USD:N format."""
    text = re.sub(r"\$(\d+(?:\.\d{2})?)", r"USD:\1", text)
    text = re.sub(r"(\d+)\s*dollars?", r"USD:\1.00", text, flags=re.IGNORECASE)
    text = re.sub(r"(\d+)\s*cents?", r"USD:0.\1", text, flags=re.IGNORECASE)
    return text


def normalize_date(text: str) -> str:
    """Normalize date expressions."""
    months = {
        "january": "01", "february": "02", "march": "03", "april": "04",
        "may": "05", "june": "06", "july": "07", "august": "08",
        "september": "09", "october": "10", "november": "11", "december": "12",
    }
    pattern = re.compile(
        r"(" + "|".join(months.keys()) + r")\s+(\d+)",
        re.IGNORECASE,
    )
    def replace(m):
        month_name = m.group(1).lower()
        day = m.group(2)
        return f"{months.get(month_name, '00')}/{day}"
    return pattern.sub(replace, text)


def normalize_entity(entity: CriticalEntity) -> str:
    """Normalize an entity's surface form for comparison."""
    if entity.entity_type in (EntityType.MONEY, EntityType.DATE):
        return entity.normalized
    if entity.entity_type in (EntityType.PERSON_NAME, EntityType.ALPHANUMERIC):
        return entity.surface.strip()
    return entity.surface.lower().strip()


def _normalize_for_matching(text: str, entity_type: EntityType) -> str:
    """Apply entity-type-specific normalization to text for matching."""
    if entity_type == EntityType.MONEY:
        return normalize_money(text)
    if entity_type == EntityType.DATE:
        return normalize_date(text)
    if entity_type in (EntityType.PERSON_NAME, EntityType.ALPHANUMERIC):
        return text.strip()
    return text.lower().strip()


def match_entities(
    reference_entities: list[CriticalEntity],
    recognized_text: str,
) -> list[EntityMatch]:
    """Match reference entities against recognized transcript."""
    results = []

    for entity in reference_entities:
        expected = _normalize_for_matching(entity.surface, entity.entity_type)
        normalized_text = _normalize_for_matching(recognized_text, entity.entity_type)
        found = expected in normalized_text
        results.append(EntityMatch(
            entity=entity,
            recognized=recognized_text,
            correct=found,
            match_type="exact" if found else "missing",
        ))

    return results


def compute_entity_error_rate(matches: list[EntityMatch]) -> dict[str, Any]:
    """Compute entity-level accuracy metrics."""
    total = len(matches)
    if total == 0:
        return {"accuracy": 1.0, "error_rate": 0.0, "total": 0, "correct": 0}
    correct = sum(1 for m in matches if m.correct)
    return {
        "accuracy": correct / total,
        "error_rate": (total - correct) / total,
        "total": total,
        "correct": correct,
        "by_type": _group_by_type(matches),
    }


def _group_by_type(matches: list[EntityMatch]) -> dict[str, dict[str, int]]:
    """Group matches by entity type."""
    result: dict[str, dict[str, int]] = {}
    for m in matches:
        t = m.entity.entity_type.value
        if t not in result:
            result[t] = {"total": 0, "correct": 0}
        result[t]["total"] += 1
        if m.correct:
            result[t]["correct"] += 1
    return result


def compute_correction_damage(
    source: list[PronunciationToken],
    output: list[PronunciationToken],
) -> tuple[int, int, int]:
    """Count corrected, damaged, and ambiguous tokens.

    A token is *corrected* when its source status is DEVIANT and the output
    status is ALREADY_TARGET.
    A token is *damaged* when its source status is ALREADY_TARGET and the
    output status is DEVIANT.
    A token is *ambiguous* when its source status is AMBIGUOUS.
    """
    corrected = sum(
        1 for t, r in zip(source, output)
        if t.source_status == SourceStatus.DEVIANT
        and r.source_status == SourceStatus.ALREADY_TARGET
    )
    damaged = sum(
        1 for t, r in zip(source, output)
        if t.source_status == SourceStatus.ALREADY_TARGET
        and r.source_status == SourceStatus.DEVIANT
    )
    ambiguous = sum(
        1 for t in source
        if t.source_status == SourceStatus.AMBIGUOUS
    )
    return corrected, damaged, ambiguous
