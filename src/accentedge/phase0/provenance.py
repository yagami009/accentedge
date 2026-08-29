"""
Provenance tracking for Phase 0 experiments.

Every generated target must carry enough metadata to answer:
"Exactly how was this WAV created?"

Extended with ProvenanceChain for full lineage tracking,
chain verification, and provenance comparison utilities.
"""

import hashlib
import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class ProvenanceRecord:
    """Complete provenance for a generated audio file."""
    experiment_id: str
    utterance_id: str
    speaker_id: str
    strategy: str           # "strategy_a", "strategy_b", "strategy_c"
    conversion_strength: float
    source_path: str
    source_hash: str
    output_path: str
    output_hash: str
    config: dict = field(default_factory=dict)
    software_version: str = "0.1.0-phase0"
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    notes: str = ""

    def save(self, path: Path) -> None:
        """Save provenance as JSON alongside the audio file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)
        logger.info(f"Provenance saved: {path}")

    @classmethod
    def load(cls, path: Path) -> "ProvenanceRecord":
        """Load provenance from JSON."""
        with open(path) as f:
            data = json.load(f)
        return cls(**data)

    def to_dict(self) -> dict:
        """Serialize provenance to a dictionary."""
        return asdict(self)


def compute_audio_hash(path: Path) -> str:
    """SHA-256 hash of audio file bytes."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def create_experiment_id() -> str:
    """Generate a unique experiment ID."""
    return f"exp_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"


@dataclass
class ProvenanceChain:
    """
    Full lineage from source recording → gold (natural cross-accent) → generated target.

    Every generated target must have a chain that answers:
    - What source was this derived from?
    - What gold utterance defined the target pronunciation?
    - What strategy, strength, and config produced the output?
    """

    experiment_id: str
    utterance_id: str
    speaker_id: str
    chain_type: str  # "source→target" or "source→gold→target"

    source_record: Optional[ProvenanceRecord] = None
    gold_record: Optional[ProvenanceRecord] = None
    target_record: Optional[ProvenanceRecord] = None

    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = {
            "experiment_id": self.experiment_id,
            "utterance_id": self.utterance_id,
            "speaker_id": self.speaker_id,
            "chain_type": self.chain_type,
            "source_record": self.source_record.to_dict() if self.source_record else None,
            "gold_record": self.gold_record.to_dict() if self.gold_record else None,
            "target_record": self.target_record.to_dict() if self.target_record else None,
            "metadata": self.metadata,
        }
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ProvenanceChain":
        d = d.copy()
        if d.get("source_record"):
            d["source_record"] = ProvenanceRecord(**d["source_record"])
        if d.get("gold_record"):
            d["gold_record"] = ProvenanceRecord(**d["gold_record"])
        if d.get("target_record"):
            d["target_record"] = ProvenanceRecord(**d["target_record"])
        return cls(**d)

    def save(self, path) -> None:
        """Save chain as JSON."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"ProvenanceChain saved: {p}")


def verify_chain(chain: ProvenanceChain) -> list[str]:
    """
    Verify that a provenance chain is complete and internally consistent.

    Checks:
    - experiment_id matches across all records
    - speaker_id matches across all records
    - utterance_id matches across all records
    - source_hash on target_record matches the actual source file (if source_record present)
    - Hashes are non-empty strings
    - Chain type matches which records are present

    Returns list of error strings. Empty list = valid.
    """
    errors = []

    ids = {
        "experiment_id": chain.experiment_id,
        "utterance_id": chain.utterance_id,
        "speaker_id": chain.speaker_id,
    }

    records = {
        "source": chain.source_record,
        "gold": chain.gold_record,
        "target": chain.target_record,
    }

    # Check IDs match across records
    for role, rec in records.items():
        if rec is None:
            continue
        for key, expected in ids.items():
            actual = getattr(rec, key, None)
            if actual != expected:
                errors.append(
                    f"{role}_record {key} mismatch: expected '{expected}', got '{actual}'"
                )

    # Check source hash matches target's claimed source hash
    if chain.source_record and chain.target_record:
        if chain.source_record.source_hash != chain.target_record.source_hash:
            errors.append(
                f"source_hash mismatch: source has '{chain.source_record.source_hash}', "
                f"target claims '{chain.target_record.source_hash}'"
            )

    # Check hashes are non-empty
    for role, rec in records.items():
        if rec is None:
            continue
        if not rec.source_hash:
            errors.append(f"{role}_record has empty source_hash")
        if not rec.output_hash:
            errors.append(f"{role}_record has empty output_hash")

    # Check chain type consistency
    if chain.chain_type == "source→target":
        if not chain.source_record or not chain.target_record:
            errors.append(
                "chain_type is 'source→target' but missing source or target record"
            )
    elif chain.chain_type == "source→gold→target":
        if not chain.source_record or not chain.gold_record or not chain.target_record:
            errors.append(
                "chain_type is 'source→gold→target' but missing required record"
            )

    return errors


def provenance_diff(a: ProvenanceRecord, b: ProvenanceRecord) -> dict:
    """
    Compare two provenance records and report what changed.

    Returns a dict with:
    - "unchanged": list of fields that are identical
    - "changed": list of fields that differ, with old/new values
    """
    a_dict = asdict(a)
    b_dict = asdict(b)

    changed = []
    unchanged = []

    all_keys = set(a_dict.keys()) | set(b_dict.keys())
    for key in sorted(all_keys):
        av = a_dict.get(key)
        bv = b_dict.get(key)
        if av == bv:
            unchanged.append(key)
        else:
            changed.append({"field": key, "old": str(av), "new": str(bv)})

    return {"unchanged": unchanged, "changed": changed}
