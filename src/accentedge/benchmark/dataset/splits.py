"""Speaker-disjoint dataset splitting with stratification."""

from __future__ import annotations

import random
from collections import defaultdict

import numpy as np
from loguru import logger

from accentedge.benchmark.schemas import DatasetItem, Partition


def _stratify(speakers: list[str], metadata: dict[str, dict], factors: list[str], seed: int) -> dict[str, list[str]]:
    """Stratify speakers by metadata factors using deterministic binning."""
    rng = random.Random(seed)
    n = len(speakers)
    buckets: dict[str, list[str]] = defaultdict(list)
    
    for spk in speakers:
        spk_meta = metadata.get(spk, {})
        key_parts = []
        for factor in factors:
            val = spk_meta.get(factor, "unknown")
            key_parts.append(str(val))
        key = "|".join(key_parts)
        buckets[key].append(spk)
    
    # Shuffle within each bucket
    for key in buckets:
        rng.shuffle(buckets[key])
    
    return dict(buckets)


def build_splits(
    items: list[DatasetItem],
    speaker_metadata: dict[str, dict],
    dev_count: int = 24,
    locked_test_count: int = 24,
    seed: int = 42,
    stratification_factors: list[str] | None = None,
    calibration_exclude: bool = True,
) -> tuple[list[DatasetItem], list[DatasetItem]]:
    """Build speaker-disjoint dev/locked_test splits.

    Args:
        items: Full dataset
        speaker_metadata: Dict of speaker_id -> metadata dict
        dev_count: Number of dev speakers
        locked_test_count: Number of locked_test speakers
        seed: Random seed for determinism
        stratification_factors: Metadata keys to stratify by
        calibration_exclude: If True, exclude speakers already in CALIBRATION
            partition from dev/locked_test.

    Returns:
        (dev_items, locked_test_items)

    Raises:
        ValueError: If splits would overlap or not enough speakers
    """
    if stratification_factors is None:
        stratification_factors = ["l1_category", "accent_strength", "bpo_experience"]

    all_speakers = list({item.speaker_id for item in items})

    # Exclude calibration speakers if requested
    if calibration_exclude:
        calibration_speakers = {
            item.speaker_id for item in items if item.partition == Partition.CALIBRATION
        }
        available_speakers = [s for s in all_speakers if s not in calibration_speakers]
    else:
        available_speakers = all_speakers

    total_needed = dev_count + locked_test_count

    if len(available_speakers) < total_needed:
        raise ValueError(
            f"Not enough speakers: have {len(available_speakers)}, need {total_needed}"
        )
    
    # Stratify speakers
    buckets = _stratify(available_speakers, speaker_metadata, stratification_factors, seed)

    # Assign speakers to splits proportionally from each bucket
    rng = random.Random(seed)
    dev_speakers: list[str] = []
    test_speakers: list[str] = []

    bucket_items = sorted(buckets.items(), key=lambda x: rng.random())
    dev_remaining = dev_count
    test_remaining = locked_test_count
    total_bucketed = sum(len(v) for v in buckets.values())

    for key, spk_list in bucket_items:
        if not spk_list:
            continue
        proportion = len(spk_list) / total_bucketed
        n_dev = max(0, min(int(round(proportion * dev_count)), dev_remaining, len(spk_list)))
        n_test = min(len(spk_list) - n_dev, test_remaining)
        rng.shuffle(spk_list)
        dev_speakers.extend(spk_list[:n_dev])
        test_speakers.extend(spk_list[n_dev:n_dev + n_test])
        dev_remaining -= n_dev
        test_remaining -= n_test

    # Fill any shortfall
    remaining = [s for s in available_speakers if s not in dev_speakers and s not in test_speakers]
    rng.shuffle(remaining)
    for spk in remaining:
        if dev_remaining > 0:
            dev_speakers.append(spk)
            dev_remaining -= 1
        elif test_remaining > 0:
            test_speakers.append(spk)
            test_remaining -= 1
    
    # Verify disjoint
    assert len(set(dev_speakers) & set(test_speakers)) == 0, "Split leakage detected"
    
    dev_set = set(dev_speakers)
    test_set = set(test_speakers)
    
    dev_items = [item for item in items if item.speaker_id in dev_set]
    test_items = [item for item in items if item.speaker_id in test_set]
    
    logger.info(f"Splits: {len(dev_speakers)} dev speakers ({len(dev_items)} items), "
                f"{len(test_speakers)} test speakers ({len(test_items)} items)")
    
    return dev_items, test_items


def validate_splits(dev_items: list[DatasetItem], test_items: list[DatasetItem]) -> list[str]:
    """Validate speaker disjointness of splits."""
    issues = []
    dev_speakers = {item.speaker_id for item in dev_items}
    test_speakers = {item.speaker_id for item in test_items}
    
    overlap = dev_speakers & test_speakers
    if overlap:
        issues.append(f"Speaker leakage detected: {overlap}")
    
    return issues
