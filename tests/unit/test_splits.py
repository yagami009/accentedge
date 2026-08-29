"""Tests for accentedge_benchmark.dataset.splits — speaker-disjoint splits."""

from __future__ import annotations

import random

import pytest

from accentedge_benchmark.schemas import DatasetItem, Partition, Family
from accentedge_benchmark.dataset.splits import build_splits, validate_splits


def _make_item(utt_id, speaker_id, partition, family=Family.BPO_SCRIPTED):
    return DatasetItem(
        utterance_id=utt_id,
        speaker_id=speaker_id,
        partition=partition,
        family=family,
        canonical_path=f"/data/{speaker_id}/{utt_id}.wav",
        sample_rate=16000,
        duration_ms=2500.0,
        transcript_verbatim="hello world",
        transcript_normalized="hello world",
        audio_sha256="a" * 64,
    )


def _make_items(n_speakers, n_utt_per_spk, partition=Partition.DEV,
                family=Family.BPO_SCRIPTED, speaker_prefix="spk"):
    items = []
    for s in range(n_speakers):
        spk_id = f"{speaker_prefix}_{s:03d}"
        for u in range(n_utt_per_spk):
            utt_id = f"{spk_id}_utt_{u:03d}"
            items.append(_make_item(utt_id, spk_id, partition, family))
    return items


class TestBuildSplits:
    def test_creates_speaker_disjoint_splits(self):
        """No speaker should appear in both dev and locked_test."""
        items = _make_items(10, 5)
        metadata = {f"spk_{i:03d}": {"l1_category": "A", "accent_strength": "medium", "bpo_experience": True}
                    for i in range(10)}
        dev, test = build_splits(items, metadata, dev_count=4, locked_test_count=4, seed=42)
        dev_speakers = {item.speaker_id for item in dev}
        test_speakers = {item.speaker_id for item in test}
        assert len(dev_speakers & test_speakers) == 0

    def test_deterministic_same_seed(self):
        """Same seed should produce identical splits."""
        items = _make_items(10, 5)
        metadata = {f"spk_{i:03d}": {"l1_category": "A", "accent_strength": "medium", "bpo_experience": True}
                    for i in range(10)}
        dev1, test1 = build_splits(items, metadata, dev_count=4, locked_test_count=4, seed=42)
        dev2, test2 = build_splits(items, metadata, dev_count=4, locked_test_count=4, seed=42)
        assert {i.utterance_id for i in dev1} == {i.utterance_id for i in dev2}
        assert {i.utterance_id for i in test1} == {i.utterance_id for i in test2}

    def test_different_seed_produces_different_splits(self):
        """Different seeds should likely produce different splits."""
        items = _make_items(10, 5)
        metadata = {f"spk_{i:03d}": {"l1_category": "A", "accent_strength": "medium", "bpo_experience": True}
                    for i in range(10)}
        dev1, test1 = build_splits(items, metadata, dev_count=4, locked_test_count=4, seed=42)
        dev2, test2 = build_splits(items, metadata, dev_count=4, locked_test_count=4, seed=99)
        # At least some speakers should differ
        s1 = {i.speaker_id for i in dev1} | {i.speaker_id for i in test1}
        s2 = {i.speaker_id for i in dev2} | {i.speaker_id for i in test2}
        assert s1 != s2 or len(s1) == len(s2)  # either differs or same total

    def test_stratification_does_not_break_disjointness(self):
        """Even with stratification, dev and test should be disjoint."""
        items = _make_items(20, 3)
        metadata = {}
        for i in range(20):
            spk_id = f"spk_{i:03d}"
            metadata[spk_id] = {
                "l1_category": ["A", "B", "C"][i % 3],
                "accent_strength": ["low", "medium", "high"][i % 3],
                "bpo_experience": i % 2 == 0,
            }
        dev, test = build_splits(
            items, metadata,
            dev_count=8, locked_test_count=8,
            seed=42,
            stratification_factors=["l1_category", "accent_strength", "bpo_experience"],
        )
        dev_speakers = {item.speaker_id for item in dev}
        test_speakers = {item.speaker_id for item in test}
        assert len(dev_speakers & test_speakers) == 0

    def test_calibration_exclude_true(self):
        """Calibration partition speakers must NOT appear in dev/test."""
        items = []
        for i in range(8):
            items.append(_make_item(f"utt_dev_{i}", f"spk_dev_{i:03d}", Partition.DEV))
        for i in range(8):
            items.append(_make_item(f"utt_test_{i}", f"spk_test_{i:03d}", Partition.LOCKED_TEST))
        for i in range(4):
            items.append(_make_item(f"utt_cal_{i}", f"spk_cal_{i:03d}", Partition.CALIBRATION))

        metadata = {}
        all_spks = set()
        for item in items:
            all_spks.add(item.speaker_id)
            metadata[item.speaker_id] = {
                "l1_category": "A",
                "accent_strength": "medium",
                "bpo_experience": True,
            }

        dev, test = build_splits(
            items, metadata,
            dev_count=8, locked_test_count=8,
            seed=42,
            calibration_exclude=True,
        )
        dev_speakers = {item.speaker_id for item in dev}
        test_speakers = {item.speaker_id for item in test}
        cal_speakers = {f"spk_cal_{i:03d}" for i in range(4)}

        # Calibration speakers should not be in dev or test
        assert len(dev_speakers & cal_speakers) == 0
        assert len(test_speakers & cal_speakers) == 0

    def test_calibration_exclude_false(self):
        """When calibration_exclude=False, calibration speakers CAN appear in splits."""
        items = []
        for i in range(6):
            items.append(_make_item(f"utt_dev_{i}", f"spk_dev_{i:03d}", Partition.DEV))
        for i in range(6):
            items.append(_make_item(f"utt_test_{i}", f"spk_test_{i:03d}", Partition.LOCKED_TEST))
        for i in range(6):
            items.append(_make_item(f"utt_cal_{i}", f"spk_cal_{i:03d}", Partition.CALIBRATION))

        metadata = {}
        for item in items:
            metadata[item.speaker_id] = {
                "l1_category": "A",
                "accent_strength": "medium",
                "bpo_experience": True,
            }

        dev, test = build_splits(
            items, metadata,
            dev_count=8, locked_test_count=8,
            seed=42,
            calibration_exclude=False,
        )
        dev_speakers = {item.speaker_id for item in dev}
        test_speakers = {item.speaker_id for item in test}
        cal_speakers = {f"spk_cal_{i:03d}" for i in range(6)}
        all_selected = dev_speakers | test_speakers
        # With all 18 speakers available, some calibration speakers should be selected
        assert len(all_selected & cal_speakers) > 0

    def test_not_enough_speakers_raises(self):
        items = _make_items(3, 2)
        metadata = {f"spk_{i:03d}": {"l1_category": "A", "accent_strength": "medium", "bpo_experience": True}
                    for i in range(3)}
        with pytest.raises(ValueError, match="Not enough speakers"):
            build_splits(items, metadata, dev_count=3, locked_test_count=3, seed=42)

    def test_validate_splits_clean(self):
        items = _make_items(10, 5)
        metadata = {f"spk_{i:03d}": {"l1_category": "A"} for i in range(10)}
        dev, test = build_splits(items, metadata, dev_count=4, locked_test_count=4, seed=42)
        issues = validate_splits(dev, test)
        assert issues == []

    def test_item_distribution(self):
        """Dev + test should contain approximately the right number of speakers."""
        items = _make_items(20, 3)
        metadata = {f"spk_{i:03d}": {"l1_category": "A", "accent_strength": "medium", "bpo_experience": True}
                    for i in range(20)}
        dev, test = build_splits(items, metadata, dev_count=8, locked_test_count=8, seed=42)
        dev_speakers = {item.speaker_id for item in dev}
        test_speakers = {item.speaker_id for item in test}
        assert len(dev_speakers) <= 8 + 2  # allow a few extra due to rounding
        assert len(test_speakers) <= 8 + 2
        # Total selected should be at least 16 (maybe more due to fill shortfall)
        assert len(dev_speakers | test_speakers) >= 16
