"""Tests for speaker split validation."""

from __future__ import annotations

import pytest

from accentedge_lab.data.schema import TrainingItem, TrainingManifest
from accentedge_lab.data.validation import validate_training_manifest


def make_item(item_id: str, speaker_id: str, split: str) -> TrainingItem:
    return TrainingItem(
        item_id=item_id,
        speaker_id=speaker_id,
        dataset_id="ds1",
        dataset_version="1.0",
        split=split,
        source_audio_path="/tmp/test.wav",
        source_audio_sha256="abcd" * 8,
        source_duration_ms=1000.0,
        source_sample_rate=16000,
        license_id="mit",
        commercial_use_status="UNKNOWN",
    )


class TestSpeakerSplitValidation:
    def test_overlap_detected(self) -> None:
        manifest = TrainingManifest(
            items=[make_item("a", "spk1", "train"), make_item("b", "spk1", "validation")]
        )
        train = set()
        val = set()
        result = validate_training_manifest(manifest, train, val)
        assert result.valid is False
        assert "spk1" in result.speaker_overlap

    def test_clean_splits(self) -> None:
        manifest = TrainingManifest(
            items=[
                make_item("a", "spk1", "train"),
                make_item("b", "spk2", "validation"),
            ]
        )
        train = set()
        val = set()
        result = validate_training_manifest(manifest, train, val)
        assert result.valid is True
        assert result.speaker_overlap == set()

    def test_leakage_warnings(self) -> None:
        manifest = TrainingManifest(
            items=[
                make_item("a", "spk1", "train"),
                make_item("b", "spk1", "validation"),
            ]
        )
        warnings = manifest.validate_speaker_splits()
        assert len(warnings) > 0
