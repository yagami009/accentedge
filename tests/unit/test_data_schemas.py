"""Tests for data schemas."""

from __future__ import annotations

from pathlib import Path

import pytest

from accentedge_lab.data.schema import (
    TrainingItem,
    TrainingManifest,
    propagate_commercial_status,
)


def make_item(item_id: str, split: str = "train") -> TrainingItem:
    return TrainingItem(
        item_id=item_id,
        speaker_id="spk1",
        dataset_id="ds1",
        dataset_version="1.0",
        split=split,
        source_audio_path=Path("/tmp/test.wav"),
        source_audio_sha256="abcd" * 8,
        source_duration_ms=1000.0,
        source_sample_rate=16000,
        license_id="mit",
        commercial_use_status="UNKNOWN",
    )


class TestTrainingItem:
    def test_creation(self) -> None:
        item = make_item("item1")
        assert item.item_id == "item1"
        assert item.lineage_status == "unknown"

    def test_lineage_status(self) -> None:
        item = make_item("item2")
        item.lineage_status = "clean"
        assert item.lineage_status == "clean"


class TestCommercialPropagation:
    def test_research_only_propagates(self) -> None:
        items = [make_item("a"), make_item("b")]
        items[0].commercial_use_status = "RESEARCH_ONLY"
        assert propagate_commercial_status(items) == "RESEARCH_ONLY"

    def test_unknown_propagates(self) -> None:
        items = [make_item("a"), make_item("b")]
        items[0].commercial_use_status = "UNKNOWN"
        items[1].commercial_use_status = "COMMERCIAL_ELIGIBLE"
        assert propagate_commercial_status(items) == "UNKNOWN"

    def test_all_commercial_eligible(self) -> None:
        items = [make_item("a"), make_item("b")]
        items[0].commercial_use_status = "COMMERCIAL_ELIGIBLE"
        items[1].commercial_use_status = "COMMERCIAL_ELIGIBLE"
        assert propagate_commercial_status(items) == "COMMERCIAL_ELIGIBLE"


class TestTrainingManifest:
    def test_hash(self, tmp_path: Path) -> None:
        manifest = TrainingManifest(items=[make_item("a"), make_item("b")], version="1.0")
        h = manifest.compute_hash()
        assert len(h) == 64

    def test_save_load(self, tmp_path: Path) -> None:
        p = tmp_path / "manifest.json"
        manifest = TrainingManifest(items=[make_item("a")], version="1.0")
        manifest.save(p)
        loaded = TrainingManifest.from_file(p)
        assert loaded.version == "1.0"
        assert loaded.items[0].item_id == "a"
