"""
Per-token realization labeling for Phase 0 Gate -1A.

Implements the TGFP v2 labeling workflow:
  ALREADY-TARGET — realization is acceptably compatible with target
  DEVIANT        — differs meaningfully on evaluated dimension
  AMBIGUOUS      — annotator cannot confidently decide

Also provides adjudication, correction-rate, and damage-rate computation.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import numpy as np

from accentedge.phase0.annotations import (
    Alignment,
    AnnotationDB,
    TokenAnnotation,
    TokenLabel,
    LABEL_NAMES,
    Utterance,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Labeling workflow
# ---------------------------------------------------------------------------

@dataclass
class LabelingSession:
    """Manages the labeling workflow for one utterance.

    Attributes:
        utterance_id: the utterance being labeled
        annotator_id: who is labeling
        db: the AnnotationDB to read/write
        reference_alignment: canonical target alignment (for damage/correction)
        target_dimensions: ordered list of target pronunciation dimensions
    """
    utterance_id: str
    annotator_id: str
    db: AnnotationDB
    reference_alignment: Optional[Alignment] = None
    target_dimensions: list[str] = field(default_factory=list)

    # internal scratch
    _label_map: dict[str, TokenLabel] = field(default_factory=dict, repr=False)
    _confidence_map: dict[str, float] = field(default_factory=dict, repr=False)
    _notes_map: dict[str, str] = field(default_factory=dict, repr=False)

    def __post_init__(self):
        utterance = self.db.get_utterance(self.utterance_id)
        if utterance is None:
            raise ValueError(
                f"Utterance {self.utterance_id} not found in database"
            )

    # ------------------------------------------------------------------
    # Labeling
    # ------------------------------------------------------------------

    def apply_labels(
        self,
        token_labels: dict[str, TokenLabel],
        confidences: Optional[dict[str, float]] = None,
        notes: Optional[dict[str, str]] = None,
    ) -> list[TokenAnnotation]:
        """Assign labels to tokens for this utterance.

        Updates the AnnotationDB in place (creating a versioned update).

        Args:
            token_labels: mapping of token_id -> TokenLabel
            confidences: optional mapping of token_id -> annotator confidence
            notes: optional mapping of token_id -> note text

        Returns:
            List of updated TokenAnnotation objects.
        """
        utterance = self.db.get_utterance(self.utterance_id)
        assert utterance is not None

        updated: list[TokenAnnotation] = []
        for token in utterance.tokens:
            if token.token_id not in token_labels:
                continue

            old_status = token.status
            token.status = token_labels[token.token_id]
            token.annotator_confidence = confidences.get(
                token.token_id, token.annotator_confidence
            ) if confidences else token.annotator_confidence
            token.notes = notes.get(token.token_id, token.notes) if notes else token.notes

            self._label_map[token.token_id] = token.status
            self._confidence_map[token.token_id] = token.annotator_confidence
            self._notes_map[token.token_id] = token.notes

            logger.debug(
                "Token '%s' (%s): %s -> %s by %s",
                token.word,
                token.token_id,
                LABEL_NAMES[old_status],
                LABEL_NAMES[token.status],
                self.annotator_id,
            )
            updated.append(token)

        # Record a versioned update
        self.db.update_utterance(
            self.utterance_id,
            tokens=utterance.tokens,
        )

        logger.info(
            "Labeled %d tokens for utterance %s by annotator %s",
            len(updated), self.utterance_id, self.annotator_id,
        )
        return updated

    def label_by_word(
        self,
        word_label_map: dict[str, TokenLabel],
        confidence: float = 1.0,
    ) -> list[TokenAnnotation]:
        """Convenience: label all occurrences of a word with the same label.

        Args:
            word_label_map: word string -> TokenLabel
            confidence: annotator confidence for all labeled tokens

        Returns:
            Updated tokens.
        """
        utterance = self.db.get_utterance(self.utterance_id)
        assert utterance is not None

        token_labels: dict[str, TokenLabel] = {}
        for token in utterance.tokens:
            if token.word in word_label_map:
                token_labels[token.token_id] = word_label_map[token.word]

        return self.apply_labels(
            token_labels,
            confidences={tid: confidence for tid in token_labels},
        )

    # ------------------------------------------------------------------
    # Adjudication
    # ------------------------------------------------------------------

    def adjudicate(
        self,
        ambiguous_token_ids: list[str],
        final_label: TokenLabel,
        adjudicator_id: str = "adjudicator",
    ) -> list[TokenAnnotation]:
        """Resolve AMBIGUOUS cases via a second (adjudicator) pass.

        Overwrites the token status with *final_label* and records the
        adjudicator in the token notes.

        Args:
            ambiguous_token_ids: list of token_ids to adjudicate
            final_label: the resolved label
            adjudicator_id: who performed adjudication

        Returns:
            Updated TokenAnnotation objects.
        """
        utterance = self.db.get_utterance(self.utterance_id)
        assert utterance is not None

        notes_map = {}
        for token in utterance.tokens:
            if token.token_id in ambiguous_token_ids:
                token.status = final_label
                note = (
                    f"adjudicated by {adjudicator_id}: "
                    f"{LABEL_NAMES[token.status]}"
                )
                token.notes = f"{token.notes}; {note}".strip("; ")
                notes_map[token.token_id] = token.notes
                logger.info(
                    "Adjudicated token '%s' (%s) -> %s",
                    token.word, token.token_id, LABEL_NAMES[final_label],
                )

        self.db.update_utterance(
            self.utterance_id,
            tokens=utterance.tokens,
        )
        return [t for t in utterance.tokens if t.token_id in ambiguous_token_ids]


# ---------------------------------------------------------------------------
# Correction / damage rate computation
# ---------------------------------------------------------------------------

@dataclass
class CorrectionDamageReport:
    """Report of correction and damage rates for a corpus."""
    correction_rate: float
    damage_rate: float
    corrected_count: int
    total_deviant: int
    damaged_count: int
    total_already_target: int
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "correction_rate": self.correction_rate,
            "damage_rate": self.damage_rate,
            "corrected_count": self.corrected_count,
            "total_deviant": self.total_deviant,
            "damaged_count": self.damaged_count,
            "total_already_target": self.total_already_target,
            "details": self.details,
        }


def _token_is_corrected(
    token: TokenAnnotation,
    reference_phones: Optional[list[str]] = None,
    target_dimension: Optional[str] = None,
) -> bool:
    """Heuristic to decide if a DEVIANT token has been corrected.

    Checks the token notes for the 'corrected' marker. Optionally
    checks whether the observed realization matches the reference target
    for the given dimension.

    Args:
        token: token to check
        reference_phones: optional list of target phones for this token
        target_dimension: optional linguistic dimension for filtering
    """
    if "corrected" not in token.notes.lower():
        return False
    if target_dimension and token.target_dimension != target_dimension:
        return False
    return True


def _token_is_damaged(
    token: TokenAnnotation,
    reference_phones: Optional[list[str]] = None,
    target_dimension: Optional[str] = None,
) -> bool:
    """Heuristic to decide if an ALREADY-TARGET token has been damaged.

    Args:
        token: token to check
        reference_phones: optional target phones (future: compare phones)
        target_dimension: optional dimension filter
    """
    if "damaged" not in token.notes.lower():
        return False
    if target_dimension and token.target_dimension != target_dimension:
        return False
    return True


def compute_correction_rate(
    db: AnnotationDB,
    target_dimension: Optional[str] = None,
) -> CorrectionDamageReport:
    """Compute correction rate per TGFP v2.

    Correction rate = deviant tokens moved toward target / all deviant tokens

    Args:
        db: AnnotationDB to evaluate
        target_dimension: if given, filter to this dimension only

    Returns:
        CorrectionDamageReport
    """
    deviant = db.deviant_tokens()
    if target_dimension:
        deviant = [t for t in deviant if t.target_dimension == target_dimension]

    total = len(deviant)
    corrected = sum(1 for t in deviant if _token_is_corrected(t))
    rate = corrected / total if total > 0 else 0.0

    return CorrectionDamageReport(
        correction_rate=rate,
        damage_rate=0.0,
        corrected_count=corrected,
        total_deviant=total,
        damaged_count=0,
        total_already_target=0,
    )


def compute_damage_rate(
    db: AnnotationDB,
    target_dimension: Optional[str] = None,
) -> CorrectionDamageReport:
    """Compute damage rate per TGFP v2.

    Damage rate = already-correct tokens made worse / all already-correct tokens

    Args:
        db: AnnotationDB to evaluate
        target_dimension: if given, filter to this dimension only

    Returns:
        CorrectionDamageReport
    """
    at_tokens = db.already_target_tokens()
    if target_dimension:
        at_tokens = [t for t in at_tokens if t.target_dimension == target_dimension]

    total = len(at_tokens)
    damaged = sum(1 for t in at_tokens if _token_is_damaged(t))
    rate = damaged / total if total > 0 else 0.0

    return CorrectionDamageReport(
        correction_rate=0.0,
        damage_rate=rate,
        corrected_count=0,
        total_deviant=0,
        damaged_count=damaged,
        total_already_target=total,
    )


def compute_correction_damage_report(
    db: AnnotationDB,
    target_dimension: Optional[str] = None,
) -> CorrectionDamageReport:
    """Compute both correction and damage rates in one pass.

    Per TGFP v2:
      Correction rate = deviant tokens moved toward target / all deviant tokens
      Damage rate     = already-correct tokens made worse / all already-correct tokens

    The "reference" for knowing what counts as correction vs damage is the
    token's target_dimension and notes field (annotator tags "corrected" /
    "damaged"). In a more advanced version, reference_phones from the
    canonical target alignment would be compared against observed phones.

    Args:
        db: AnnotationDB to evaluate
        target_dimension: optional filter on target dimension

    Returns:
        CorrectionDamageReport with both rates.
    """
    deviant = db.deviant_tokens()
    at_tokens = db.already_target_tokens()

    if target_dimension:
        deviant = [t for t in deviant if t.target_dimension == target_dimension]
        at_tokens = [t for t in at_tokens if t.target_dimension == target_dimension]

    corrected = sum(1 for t in deviant if _token_is_corrected(t))
    damaged = sum(1 for t in at_tokens if _token_is_damaged(t))

    correction_rate = corrected / len(deviant) if deviant else 0.0
    damage_rate = damaged / len(at_tokens) if at_tokens else 0.0

    return CorrectionDamageReport(
        correction_rate=correction_rate,
        damage_rate=damage_rate,
        corrected_count=corrected,
        total_deviant=len(deviant),
        damaged_count=damaged,
        total_already_target=len(at_tokens),
        details={
            "target_dimension_filter": target_dimension,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
    )
