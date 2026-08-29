"""Tests for accentedge_benchmark.runner — manifest, resume, failures."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from accentedge_benchmark.schemas import (
    Partition, Family, ErrorCategory,
    DatasetItem, RunManifest, FailureRecord,
)
from accentedge_benchmark.runner.run_manifest import create_run_manifest, manifest_to_dict
from accentedge_benchmark.runner.failures import (
    FailureRecord as RunnerFailureRecord,
    FailureCollector, record_failure, classify_error,
)


# ── Run manifest ─────────────────────────────────────────────────────────────


class TestCreateRunManifest:
    def test_produces_valid_manifest(self):
        ds_hash = hashlib.sha256(b"dataset").hexdigest()
        manifest = create_run_manifest(
            candidate_name="cand_a",
            candidate_hash="abc123",
            config_hash="cfg_hash",
            split="dev",
            condition="clean",
            dataset_hash=ds_hash,
        )
        assert isinstance(manifest, RunManifest)
        assert manifest.candidate_name == "cand_a"
        assert manifest.split == "dev"
        assert manifest.condition == "clean"
        assert manifest.dataset_hash == ds_hash
        assert manifest.benchmark_version == "1.0.0"
        assert manifest.conversion_strength is None

    def test_with_conversion_strength(self):
        ds_hash = hashlib.sha256(b"dataset").hexdigest()
        manifest = create_run_manifest(
            candidate_name="cand_a",
            candidate_hash="abc",
            config_hash="cfg",
            split="locked_test",
            condition="nb",
            dataset_hash=ds_hash,
            conversion_strength=0.75,
        )
        assert manifest.conversion_strength == 0.75

    def test_run_id_contains_candidate_name(self):
        ds_hash = hashlib.sha256(b"dataset").hexdigest()
        manifest = create_run_manifest(
            candidate_name="cand_a",
            candidate_hash="abc",
            config_hash="cfg",
            split="dev",
            condition="clean",
            dataset_hash=ds_hash,
        )
        assert "cand_a" in manifest.run_id
        assert manifest.run_id.endswith("_cand_a")

    def test_to_dict_serializable(self):
        ds_hash = hashlib.sha256(b"dataset").hexdigest()
        manifest = create_run_manifest(
            candidate_name="cand_a",
            candidate_hash="abc",
            config_hash="cfg",
            split="dev",
            condition="clean",
            dataset_hash=ds_hash,
        )
        d = manifest_to_dict(manifest)
        assert isinstance(d, dict)
        assert d["candidate_name"] == "cand_a"

    def test_conditions_valid(self):
        ds_hash = hashlib.sha256(b"dataset").hexdigest()
        for cond in ["clean", "nb", "noisy", "nb_noisy"]:
            m = create_run_manifest(
                candidate_name="cand_a",
                candidate_hash="abc",
                config_hash="cfg",
                split="dev",
                condition=cond,
                dataset_hash=ds_hash,
            )
            assert m.condition == cond


# ── Resume ───────────────────────────────────────────────────────────────────


class TestResume:
    def test_save_and_load_completed(self, tmp_path):
        from accentedge_benchmark.runner.resume import save_completed_item, load_completed_items
        items = [DatasetItem(
            utterance_id="utt_001",
            speaker_id="spk_001",
            partition=Partition.DEV,
            family=Family.BPO_SCRIPTED,
            canonical_path="/data/utt_001.wav",
            sample_rate=16000,
            duration_ms=2500.0,
            transcript_verbatim="hello world",
            transcript_normalized="hello world",
            audio_sha256="a" * 64,
        )]
        save_completed_item(tmp_path, items[0], metadata={})
        loaded = load_completed_items(tmp_path)
        assert len(loaded) == 1
        assert loaded[0].utterance_id == "utt_001"

    def test_empty_directory_returns_empty(self, tmp_path):
        from accentedge_benchmark.runner.resume import load_completed_items
        loaded = load_completed_items(tmp_path)
        assert loaded == []


# ── Failures ─────────────────────────────────────────────────────────��───────


class TestRecordFailure:
    def test_creates_failure_record(self):
        try:
            raise ValueError("test error")
        except ValueError as exc:
            rec = record_failure(
                utterance_id="utt_001",
                candidate_name="cand_a",
                exc=exc,
                context="processing",
                run_id="run_001",
            )
        # The runner's FailureRecord is a dataclass, not the Pydantic schema
        assert isinstance(rec, RunnerFailureRecord)
        assert rec.utterance_id == "utt_001"
        assert rec.candidate_name == "cand_a"
        assert "ValueError" in rec.stack_trace
        assert "test error" in rec.stack_trace

    def test_classify_value_error(self):
        category = classify_error(ValueError("bad input"), "")
        assert category == ErrorCategory.CANDIDATE_ERROR

    def test_failure_collector(self):
        collector = FailureCollector()
        assert collector.count == 0
        collector.record(record_failure(
            utterance_id="utt_001",
            candidate_name="cand_a",
            exc=ValueError("test error"),
        ))
        assert collector.count == 1

    def test_failure_collector_to_jsonl(self, tmp_path):
        collector = FailureCollector()
        collector.record(record_failure(
            utterance_id="utt_001",
            candidate_name="cand_a",
            exc=ValueError("test error"),
        ))
        jsonl_path = tmp_path / "failures.jsonl"
        collector.to_jsonl(jsonl_path)
        assert jsonl_path.exists()
        lines = jsonl_path.read_text().strip().split("\n")
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["utterance_id"] == "utt_001"
        assert "error_category" in record

    def test_failure_collector_by_category(self):
        collector = FailureCollector()
        collector.record(record_failure(
            utterance_id="utt_001",
            candidate_name="cand_a",
            exc=ValueError("test"),
        ))
        counts = collector.by_category()
        assert "CANDIDATE_ERROR" in counts
        assert counts["CANDIDATE_ERROR"] == 1


class TestFailureRecord:
    def test_default_timestamp(self):
        now = datetime.now(timezone.utc)
        rec = RunnerFailureRecord(
            utterance_id="utt_001",
            candidate_name="cand_a",
            error_category=ErrorCategory.METRIC_ERROR,
            error_message="test error",
        )
        assert rec.timestamp is not None
        diff = (datetime.now(timezone.utc) - rec.timestamp).total_seconds()
        assert diff < 5

    def test_schema_failure_record(self):
        from accentedge_benchmark.schemas import FailureRecord as SchemaFR
        rec = SchemaFR(
            utterance_id="utt_001",
            candidate_name="cand_a",
            error_category=ErrorCategory.INPUT_AUDIO_ERROR,
            error_message="bad audio",
        )
        assert rec.utterance_id == "utt_001"
        assert rec.error_category == ErrorCategory.INPUT_AUDIO_ERROR
        assert rec.stack_trace is None
        assert rec.run_id is None
        assert rec.metadata == {}
