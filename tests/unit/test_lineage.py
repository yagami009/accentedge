"""Tests for data lineage and commercial propagation."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from accentedge_lab.data.lineage import DataLineage
from accentedge_lab.data.schema import propagate_commercial_status


class TestDataLineage:
    def test_creation(self) -> None:
        dl = DataLineage(
            dataset_id="ds1",
            dataset_version="1.0",
            license_id="mit",
            commercial_use_status="COMMERCIAL_ELIGIBLE",
        )
        assert dl.dataset_id == "ds1"

    def test_round_trip(self) -> None:
        dl = DataLineage(
            dataset_id="ds1",
            dataset_version="1.0",
            license_id="mit",
            commercial_use_status="RESEARCH_ONLY",
            source_url="https://example.com",
            download_timestamp=datetime.now(tz=timezone.utc),
            preprocessing_steps=["normalize"],
        )
        d = dl.to_dict()
        restored = DataLineage.from_dict(d)
        assert restored.dataset_id == "ds1"


class TestCommercialPropagation:
    def test_research_only_wins(self) -> None:
        from accentedge_lab.data.schema import TrainingItem

        items = [
            TrainingItem(
                item_id="a",
                speaker_id="spk1",
                dataset_id="ds1",
                dataset_version="1.0",
                split="train",
                source_audio_path="/tmp/x.wav",
                source_audio_sha256="abcd",
                source_duration_ms=1000.0,
                source_sample_rate=16000,
                license_id="mit",
                commercial_use_status="RESEARCH_ONLY",
            ),
            TrainingItem(
                item_id="b",
                speaker_id="spk2",
                dataset_id="ds1",
                dataset_version="1.0",
                split="train",
                source_audio_path="/tmp/y.wav",
                source_audio_sha256="efgh",
                source_duration_ms=1000.0,
                source_sample_rate=16000,
                license_id="mit",
                commercial_use_status="COMMERCIAL_ELIGIBLE",
            ),
        ]
        assert propagate_commercial_status(items) == "RESEARCH_ONLY"
