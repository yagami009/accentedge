"""Dataset validation: schemas, hashes, partitions, leakage."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pandas as pd
from loguru import logger

from accentedge.benchmark.schemas import DatasetItem, Partition, CriticalEntity, PronunciationToken
from accentedge.benchmark.audio.hashing import sha256_file
from accentedge.benchmark.audio.validate import validate_audio, AudioValidationResult


class ValidationIssue:
    def __init__(self, severity: str, message: str, item_id: str | None = None):
        self.severity = severity  # "error", "warning", "info"
        self.message = message
        self.item_id = item_id or "global"

    def __str__(self):
        prefix = {"error": "ERROR", "warning": "WARN", "info": "INFO"}.get(self.severity, self.severity.upper())
        return f"[{prefix}] {self.item_id}: {self.message}"


class DatasetValidationResult:
    def __init__(self):
        self.issues: list[ValidationIssue] = []
        self.is_valid: bool = True

    def add_error(self, message: str, item_id: str | None = None):
        self.issues.append(ValidationIssue("error", message, item_id))
        self.is_valid = False

    def add_warning(self, message: str, item_id: str | None = None):
        self.issues.append(ValidationIssue("warning", message, item_id))

    def add_info(self, message: str, item_id: str | None = None):
        self.issues.append(ValidationIssue("info", message, item_id))


def validate_dataset(
    items: list[DatasetItem],
    check_audio: bool = True,
    check_splits: bool = True,
    check_duplicates: bool = True,
) -> DatasetValidationResult:
    """Validate complete dataset integrity.
    
    Checks: schema, audio files, hashes, partition leakage, duplicate utterances.
    """
    result = DatasetValidationResult()
    
    if not items:
        result.add_error("Dataset is empty")
        return result
    
    # Check for duplicate utterance IDs
    utterance_ids = [item.utterance_id for item in items]
    duplicates = [uid for uid, count in Counter(utterance_ids).items() if count > 1]
    if duplicates:
        result.add_error(f"Duplicate utterance IDs: {duplicates}")
    
    # Check partition leakage
    if check_splits:
        speaker_partitions: dict[str, set[str]] = {}
        for item in items:
            if item.speaker_id not in speaker_partitions:
                speaker_partitions[item.speaker_id] = set()
            speaker_partitions[item.speaker_id].add(item.partition.value)
        
        leaked = {spk: parts for spk, parts in speaker_partitions.items() if len(parts) > 1}
        if leaked:
            result.add_error(f"Speaker partition leakage: {leaked}")
    
    # Check duplicate audio hashes across partitions
    if check_duplicates:
        hash_to_items: dict[str, list[str]] = {}
        for item in items:
            if item.audio_sha256:
                if item.audio_sha256 not in hash_to_items:
                    hash_to_items[item.audio_sha256] = []
                hash_to_items[item.audio_sha256].append(item.utterance_id)
        
        dup_hashes = {h: ids for h, ids in hash_to_items.items() if len(ids) > 1}
        if dup_hashes:
            result.add_error(f"Duplicate audio hashes across items: {dup_hashes}")
    
    # Validate audio files exist and match hashes
    if check_audio:
        for item in items:
            if not item.canonical_path:
                result.add_warning(f"No canonical path for {item.utterance_id}", item.utterance_id)
                continue
            
            path = Path(item.canonical_path)
            if not path.exists():
                result.add_error(f"Audio file not found: {path}", item.utterance_id)
                continue
            
            try:
                actual_hash = sha256_file(path)
                if actual_hash != item.audio_sha256:
                    result.add_error(
                        f"Hash mismatch: expected {item.audio_sha256[:12]}..., got {actual_hash[:12]}...",
                        item.utterance_id,
                    )
            except Exception as exc:
                result.add_error(f"Hash check failed: {exc}", item.utterance_id)
    
    logger.info(f"Validation complete: {len(result.issues)} issues, valid={result.is_valid}")
    return result
