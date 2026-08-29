"""Critical entity evaluation."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..annotations.entities import match_entities, normalize_entity
from ..schemas import CriticalEntity, EntityType


@dataclass
class EntityResult:
    """Result of critical-entity evaluation for a single utterance."""

    entities_evaluated: int = 0
    entities_correct: int = 0
    entity_rate: float = 0.0
    by_type: dict[str, dict[str, Any]] = field(default_factory=dict)
    matches: list[Any] = field(default_factory=list)


class EntityEvaluator:
    """Evaluates preservation of critical entities in recognized transcripts."""

    def evaluate(
        self,
        entities: list[CriticalEntity],
        recognized_text: str,
    ) -> EntityResult:
        """Evaluate entity accuracy for a single utterance.

        Args:
            entities: List of critical entities extracted from the reference.
            recognized_text: The recognized/candidate transcript text.

        Returns:
            EntityResult with per-entity accuracy and per-type breakdown.
        """
        matches = match_entities(entities, recognized_text)
        correct = sum(1 for m in matches if m.correct)

        by_type: dict[str, dict[str, Any]] = {}
        for m in matches:
            et = m.entity.entity_type.value
            if et not in by_type:
                by_type[et] = {"correct": 0, "total": 0}
            by_type[et]["total"] += 1
            if m.correct:
                by_type[et]["correct"] += 1

        for et in by_type:
            total = by_type[et]["total"]
            by_type[et]["rate"] = by_type[et]["correct"] / total if total > 0 else 0.0

        return EntityResult(
            entities_evaluated=len(matches),
            entities_correct=correct,
            entity_rate=correct / len(matches) if matches else 0.0,
            by_type=by_type,
            matches=matches,
        )
