"""
Data structures for Phase 0 annotations and utterance management.

Defines:
- TokenLabel: per-token realization labels (ALREADY_TARGET, DEVIANT, AMBIGUOUS)
- TokenAnnotation: annotation for a single token
- Alignment: phone-level timing information
- Utterance: source recording with full annotation
- AnnotationDB: collection of utterances for the experiment
- AnnotationVersion: immutable snapshot of an annotation state for history
"""

import copy
import json
import logging
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import IntEnum
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class TokenLabel(IntEnum):
    """Per-token pronunciation status."""
    ALREADY_TARGET = 0    # Realization is already acceptably target-like
    DEVIANT = 1           # Differs meaningfully from target
    AMBIGUOUS = 2         # Annotator cannot confidently decide


LABEL_NAMES = {
    TokenLabel.ALREADY_TARGET: "ALREADY-TARGET",
    TokenLabel.DEVIANT: "DEVIANT",
    TokenLabel.AMBIGUOUS: "AMBIGUOUS",
}


@dataclass
class TokenAnnotation:
    """Annotation for a single word/token in an utterance."""
    word: str
    word_start_ms: float
    word_end_ms: float
    canonical_phone: str
    observed_realization: str
    phone_start_ms: float
    phone_end_ms: float
    status: TokenLabel
    target_dimension: str = ""  # e.g. "RHO", "TH", "FLAP"
    annotator_confidence: float = 1.0
    notes: str = ""
    token_id: str = ""

    def __post_init__(self):
        if not self.token_id:
            self.token_id = str(uuid.uuid4())[:8]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["status"] = LABEL_NAMES[self.status]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "TokenAnnotation":
        d = d.copy()
        status_str = d["status"]
        # Accept both "ALREADY-TARGET" (serialized) and "ALREADY_TARGET" (raw)
        if status_str in LABEL_NAMES.values():
            for key, val in LABEL_NAMES.items():
                if val == status_str:
                    d["status"] = key
                    break
        else:
            d["status"] = TokenLabel[status_str]
        return cls(**d)


@dataclass
class Alignment:
    """Phone-level alignment for an utterance."""
    utterance_id: str
    phones: list[tuple[str, float, float]]  # (phone, start_ms, end_ms)
    words: list[tuple[str, float, float]]   # (word, start_ms, end_ms)
    source: str = "manual"  # "manual", "forced_align", "corrected"
    corrected: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Alignment":
        return cls(**d)


@dataclass
class Utterance:
    """A single source recording with full annotation."""
    utterance_id: str
    speaker_id: str
    text: str
    audio_path: str
    set_type: str  # "A" realistic BPO, "B" contrast-dense, "C" spontaneous, "D" already-target-dense
    duration_seconds: float
    sample_rate: int = 22050
    alignment: Optional[Alignment] = None
    tokens: list[TokenAnnotation] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        if self.alignment:
            d["alignment"] = self.alignment.to_dict()
        d["tokens"] = [t.to_dict() for t in self.tokens]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Utterance":
        d = d.copy()
        if d.get("alignment"):
            d["alignment"] = Alignment.from_dict(d["alignment"])
        d["tokens"] = [TokenAnnotation.from_dict(t) for t in d.get("tokens", [])]
        return cls(**d)


@dataclass
class AnnotationVersion:
    """Immutable snapshot of an utterance's annotation state at a point in time."""
    version_id: str
    utterance_id: str
    timestamp: str  # ISO-8601 UTC
    annotation_data: dict
    change_summary: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "AnnotationVersion":
        return cls(**d)


class AnnotationDB:
    """Collection of annotated utterances for Phase 0.

    Supports CRUD, versioning, disagreement tracking, and JSON interchange.
    """

    def __init__(self):
        self.utterances: dict[str, Utterance] = {}
        self.speakers: set[str] = set()
        self._versions: list[AnnotationVersion] = []
        self._meta: dict = {"created_at": self._now()}

    # ------------------------------------------------------------------
    # Versioning helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def _record_version(
        self,
        utterance_id: str,
        change_summary: str,
        annotation_data: dict,
    ) -> None:
        v = AnnotationVersion(
            version_id=str(uuid.uuid4())[:12],
            utterance_id=utterance_id,
            timestamp=self._now(),
            annotation_data=annotation_data,
            change_summary=change_summary,
        )
        self._versions.append(v)

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------
    def add_utterance(self, utterance: Utterance) -> None:
        """Add a new utterance (creates initial version)."""
        self.utterances[utterance.utterance_id] = utterance
        self.speakers.add(utterance.speaker_id)
        self._record_version(
            utterance.utterance_id,
            "initial_add",
            utterance.to_dict(),
        )
        logger.info(
            "Added utterance %s (speaker %s, set %s)",
            utterance.utterance_id, utterance.speaker_id, utterance.set_type,
        )

    def get_utterance(self, utterance_id: str) -> Optional[Utterance]:
        """Retrieve an utterance by ID."""
        return self.utterances.get(utterance_id)

    def update_utterance(
        self,
        utterance_id: str,
        **updates,
    ) -> Optional[Utterance]:
        """Update fields of an existing utterance. Records a new version.

        Accepted keyword args: alignment, tokens, text, notes, etc.
        Only updates fields that are explicitly passed.
        """
        utterance = self.utterances.get(utterance_id)
        if utterance is None:
            logger.warning("update_utterance: %s not found", utterance_id)
            return None

        for key, value in updates.items():
            if not hasattr(utterance, key):
                logger.warning(
                    "update_utterance: unknown field '%s' for %s",
                    key, utterance_id,
                )
                continue
            setattr(utterance, key, value)

        self._record_version(
            utterance_id,
            f"update: {', '.join(sorted(updates.keys()))}",
            utterance.to_dict(),
        )
        logger.debug("Updated utterance %s", utterance_id)
        return utterance

    def add(self, utterance: Utterance) -> None:
        """Alias for add_utterance (backwards compat)."""
        self.add_utterance(utterance)

    def get(self, utterance_id: str) -> Optional[Utterance]:
        """Alias for get_utterance (backwards compat)."""
        return self.get_utterance(utterance_id)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------
    def by_speaker(self, speaker_id: str) -> list[Utterance]:
        """Get all utterances for a speaker."""
        return [u for u in self.utterances.values() if u.speaker_id == speaker_id]

    def by_set(self, set_type: str) -> list[Utterance]:
        """Get all utterances of a given set type."""
        return [u for u in self.utterances.values() if u.set_type == set_type]

    def deviant_tokens(self) -> list[TokenAnnotation]:
        """Get all tokens labeled DEVIANT across all utterances."""
        return [
            t for u in self.utterances.values()
            for t in u.tokens
            if t.status == TokenLabel.DEVIANT
        ]

    def already_target_tokens(self) -> list[TokenAnnotation]:
        """Get all tokens labeled ALREADY-TARGET."""
        return [
            t for u in self.utterances.values()
            for t in u.tokens
            if t.status == TokenLabel.ALREADY_TARGET
        ]

    def ambiguous_tokens(self) -> list[TokenAnnotation]:
        """Get all tokens labeled AMBIGUOUS."""
        return [
            t for u in self.utterances.values()
            for t in u.tokens
            if t.status == TokenLabel.AMBIGUOUS
        ]

    # ------------------------------------------------------------------
    # Disagreement tracking
    # ------------------------------------------------------------------
    def compute_disagreement(
        self,
        annotator_a_labels: dict[str, TokenLabel],
        annotator_b_labels: dict[str, TokenLabel],
    ) -> dict:
        """Compare two annotators' token labels.

        Args:
            annotator_a_labels: mapping of token_id -> TokenLabel
            annotator_b_labels: mapping of token_id -> TokenLabel

        Returns:
            dict with:
              - agreement_rate: float
              - disagreement_count: int
              - total_compared: int
              - disagreements: list of {token_id, label_a, label_b}
              - confusion_matrix: {label_a: {label_b: count}}
        """
        all_token_ids = set(annotator_a_labels) & set(annotator_b_labels)
        total = len(all_token_ids)
        if total == 0:
            return {
                "agreement_rate": 0.0,
                "disagreement_count": 0,
                "total_compared": 0,
                "disagreements": [],
                "confusion_matrix": {},
            }

        disagreements = []
        confusion: dict[str, dict[str, int]] = {}

        for tid in all_token_ids:
            la = annotator_a_labels[tid]
            lb = annotator_b_labels[tid]
            if la != lb:
                a_str = LABEL_NAMES[la]
                b_str = LABEL_NAMES[lb]
                disagreements.append({
                    "token_id": tid,
                    "label_a": a_str,
                    "label_b": b_str,
                })
                confusion.setdefault(a_str, {})
                confusion[a_str][b_str] = confusion[a_str].get(b_str, 0) + 1

        agree_count = total - len(disagreements)
        return {
            "agreement_rate": agree_count / total,
            "disagreement_count": len(disagreements),
            "total_compared": total,
            "disagreements": disagreements,
            "confusion_matrix": confusion,
        }

    # ------------------------------------------------------------------
    # Correction & damage rates
    # ------------------------------------------------------------------
    def correction_rate(self) -> tuple[float, int, int]:
        """
        Calculate correction rate over DEVIANT tokens.
        A token is "corrected" if its notes field contains 'corrected'.

        Returns:
            (rate, corrected_count, total_deviant_count)
        """
        deviant = self.deviant_tokens()
        if not deviant:
            return 0.0, 0, 0
        corrected = sum(1 for t in deviant if "corrected" in t.notes.lower())
        return corrected / len(deviant), corrected, len(deviant)

    def damage_rate(self) -> tuple[float, int, int]:
        """
        Calculate damage rate over ALREADY-TARGET tokens.
        A token is "damaged" if its notes field contains 'damaged'.

        Returns:
            (rate, damaged_count, total_already_target_count)
        """
        at = self.already_target_tokens()
        if not at:
            return 0.0, 0, 0
        damaged = sum(1 for t in at if "damaged" in t.notes.lower())
        return damaged / len(at), damaged, len(at)

    # ------------------------------------------------------------------
    # Version history
    # ------------------------------------------------------------------
    def version_history(self, utterance_id: str) -> list[dict]:
        """Return version history for a single utterance."""
        return [
            v.to_dict() for v in self._versions
            if v.utterance_id == utterance_id
        ]

    def all_versions(self) -> list[dict]:
        """Return all versions across all utterances."""
        return [v.to_dict() for v in self._versions]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: Path) -> None:
        """Save database to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "_meta": self._meta,
            "speakers": sorted(self.speakers),
            "utterances": [u.to_dict() for u in self.utterances.values()],
            "versions": [v.to_dict() for v in self._versions],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(
            "AnnotationDB saved: %s (%d utterances, %d versions)",
            path, len(self.utterances), len(self._versions),
        )

    @classmethod
    def load(cls, path: Path) -> "AnnotationDB":
        """Load database from JSON."""
        with open(path) as f:
            data = json.load(f)
        db = cls()
        db._meta = data.get("_meta", db._meta)
        db.speakers = set(data.get("speakers", []))
        for u_data in data["utterances"]:
            db.add(Utterance.from_dict(u_data))
        # Reconstruct versions but skip duplicate add versions from load
        db._versions = []
        for v_data in data.get("versions", []):
            db._versions.append(AnnotationVersion.from_dict(v_data))
        return db

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------
    @property
    def speaker_count(self) -> int:
        return len(self.speakers)

    @property
    def utterance_count(self) -> int:
        return len(self.utterances)
