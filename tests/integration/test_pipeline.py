"""End-to-end integration test for the AccentEdge BPO Benchmark pipeline."""

from __future__ import annotations

import json
import hashlib
import numpy as np
import soundfile as sf
from pathlib import Path

import pytest

from accentedge_benchmark.schemas import DatasetItem, Partition, Family, BenchmarkContext
from accentedge_benchmark.dataset.splits import build_splits
from accentedge_benchmark.candidates.passthrough import PassthroughAdapter
from accentedge_benchmark.candidates.file_output import FileOutputAdapter
from accentedge_benchmark.runner.run_manifest import create_run_manifest
from accentedge_benchmark.runner.benchmark import BenchmarkRunner
from accentedge_benchmark.reporting.json_report import generate_json_report
from accentedge_benchmark.statistics.bootstrap import speaker_bootstrap


def _create_synthetic_audio(path: Path, sr: int = 16000, duration_s: float = 1.0):
    """Write a synthetic sine-wave WAV file."""
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False)
    waveform = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), waveform, sr, subtype="PCM_16")
    return waveform, sr


def _make_dataset(tmp_path: Path, n_speakers: int = 8, n_utt: int = 4) -> tuple[list[DatasetItem], dict]:
    """Create a synthetic benchmark dataset on disk."""
    items = []
    metadata = {}
    for s in range(n_speakers):
        spk_id = f"spk_{s:03d}"
        metadata[spk_id] = {
            "l1_category": ["A", "B"][s % 2],
            "accent_strength": ["low", "high"][s % 2],
            "bpo_experience": s % 2 == 0,
        }
        for u in range(n_utt):
            utt_id = f"{spk_id}_utt_{u:03d}"
            audio_path = tmp_path / "canonical" / f"{utt_id}.wav"
            waveform, sr = _create_synthetic_audio(audio_path)
            audio_hash = hashlib.sha256(audio_path.read_bytes()).hexdigest()
            items.append(DatasetItem(
                utterance_id=utt_id,
                speaker_id=spk_id,
                partition=Partition.DEV,
                family=Family.BPO_SCRIPTED,
                canonical_path=str(audio_path),
                sample_rate=sr,
                duration_ms=(len(waveform) / sr) * 1000.0,
                transcript_verbatim=f"hello world utterance {u}",
                transcript_normalized=f"hello world utterance {u}",
                audio_sha256=audio_hash,
            ))
    return items, metadata


class TestIntegrationPipeline:
    def test_end_to_end_passthrough(self, tmp_path):
        # 1. Create synthetic dataset
        items, metadata = _make_dataset(tmp_path, n_speakers=8, n_utt=4)
        assert len(items) == 32

        # 2. Build speaker-disjoint splits
        dev_items, test_items = build_splits(
            items, metadata,
            dev_count=4, locked_test_count=4,
            seed=42,
        )
        dev_speakers = {i.speaker_id for i in dev_items}
        test_speakers = {i.speaker_id for i in test_items}
        assert len(dev_speakers & test_speakers) == 0
        assert len(dev_speakers) >= 4
        assert len(test_speakers) >= 4

        # 3. Run passthrough candidate
        adapter = PassthroughAdapter()
        runner = BenchmarkRunner(
            candidate=adapter,
            split="dev",
            condition="clean",
            output_dir=str(tmp_path / "runs"),
        )
        result = runner.run(dev_items)
        assert result["total_items"] == len(dev_items)
        assert result["succeeded"] == result["total_items"]
        assert result["failed"] == 0
        assert len(runner.outputs) == len(dev_items)

    def test_outputs_match_inputs(self, tmp_path):
        """Passthrough candidate should produce audio identical to input."""
        items, metadata = _make_dataset(tmp_path, n_speakers=4, n_utt=2)
        dev_items, _ = build_splits(
            items, metadata,
            dev_count=2, locked_test_count=2,
            seed=42,
        )

        adapter = PassthroughAdapter()
        runner = BenchmarkRunner(
            candidate=adapter,
            split="dev",
            condition="clean",
            output_dir=str(tmp_path / "runs"),
        )
        runner.run(dev_items)

        for item, output in zip(dev_items, runner.outputs):
            # Load the original
            original, sr = sf.read(item.canonical_path, dtype="float32")
            if original.ndim > 1:
                original = original.mean(axis=1)
            np.testing.assert_array_almost_equal(
                output.audio, original.astype(np.float32), decimal=4,
            )
            assert output.sample_rate == sr

    def test_generate_json_report(self, tmp_path):
        items, metadata = _make_dataset(tmp_path, n_speakers=4, n_utt=2)
        dev_items, _ = build_splits(
            items, metadata,
            dev_count=2, locked_test_count=2,
            seed=42,
        )

        adapter = PassthroughAdapter()
        runner = BenchmarkRunner(
            candidate=adapter,
            split="dev",
            condition="clean",
            output_dir=str(tmp_path / "runs"),
        )
        run_result = runner.run(dev_items)

        # Compute a simple summary
        summary = {
            "candidate": run_result["candidate"],
            "total": run_result["total_items"],
            "succeeded": run_result["succeeded"],
            "failed": run_result["failed"],
        }

        ds_hash = hashlib.sha256(b"synthetic_dataset").hexdigest()
        manifest = create_run_manifest(
            candidate_name="passthrough",
            candidate_hash="abc123",
            config_hash="cfg",
            split="dev",
            condition="clean",
            dataset_hash=ds_hash,
        )
        from accentedge_benchmark.runner.run_manifest import manifest_to_dict
        manifest_dict = manifest_to_dict(manifest)

        report_path = tmp_path / "reports" / "report.json"
        generate_json_report(summary, report_path, run_manifest=manifest_dict)
        assert report_path.exists()

        with open(report_path) as f:
            report = json.load(f)
        assert report["benchmark_version"] == "1.0.0"
        assert "run_manifest" in report
        assert report["run_manifest"]["candidate_name"] == "passthrough"
        assert report["summary"]["total"] == len(dev_items)

    def test_file_output_adapter_in_pipeline(self, tmp_path):
        """Pre-generate outputs, then run file_output adapter."""
        items, metadata = _make_dataset(tmp_path, n_speakers=4, n_utt=2)
        dev_items, _ = build_splits(
            items, metadata,
            dev_count=2, locked_test_count=2,
            seed=42,
        )

        # Pre-generate candidate outputs
        output_dir = tmp_path / "candidate_outputs"
        output_dir.mkdir(parents=True, exist_ok=True)
        for item in dev_items:
            audio, sr = sf.read(item.canonical_path, dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            out_path = output_dir / f"{item.utterance_id}.wav"
            sf.write(str(out_path), audio, sr, subtype="PCM_16")

        adapter = FileOutputAdapter(output_dir)
        runner = BenchmarkRunner(
            candidate=adapter,
            split="dev",
            condition="clean",
            output_dir=str(tmp_path / "runs"),
        )
        result = runner.run(dev_items)
        assert result["succeeded"] == len(dev_items)
        assert result["failed"] == 0

    def test_speaker_statistics_after_run(self, tmp_path):
        """Compute speaker-level metrics after a benchmark run."""
        items, metadata = _make_dataset(tmp_path, n_speakers=6, n_utt=3)
        dev_items, _ = build_splits(
            items, metadata,
            dev_count=3, locked_test_count=3,
            seed=42,
        )

        adapter = PassthroughAdapter()
        runner = BenchmarkRunner(
            candidate=adapter,
            split="dev",
            condition="clean",
            output_dir=str(tmp_path / "runs"),
        )
        runner.run(dev_items)

        # Compute per-speaker "duration ratio" as a mock metric
        speaker_metrics = {}
        for item, output in zip(dev_items, runner.outputs):
            src_dur = len(item.utterance_id)  # dummy metric
            out_dur = len(output.audio)
            speaker_metrics[item.speaker_id] = out_dur / src_dur if src_dur > 0 else 1.0

        bootstrap_result = speaker_bootstrap(
            speaker_metrics,
            metric_fn=np.mean,
            n_replicates=500,
            seed=42,
        )
        assert bootstrap_result.ci_lower <= bootstrap_result.point_estimate <= bootstrap_result.ci_upper
        assert bootstrap_result.n_speakers >= 3

    def test_report_contains_expected_fields(self, tmp_path):
        items, metadata = _make_dataset(tmp_path, n_speakers=4, n_utt=2)
        dev_items, _ = build_splits(
            items, metadata,
            dev_count=2, locked_test_count=2,
            seed=42,
        )

        adapter = PassthroughAdapter()
        runner = BenchmarkRunner(
            candidate=adapter,
            split="dev",
            condition="clean",
            output_dir=str(tmp_path / "runs"),
        )
        run_result = runner.run(dev_items)

        summary = {
            "candidate": run_result["candidate"],
            "split": run_result["split"],
            "condition": run_result["condition"],
            "total": run_result["total_items"],
            "succeeded": run_result["succeeded"],
            "failed": run_result["failed"],
        }

        ds_hash = hashlib.sha256(b"synthetic").hexdigest()
        manifest = create_run_manifest(
            candidate_name="passthrough",
            candidate_hash="abc",
            config_hash="cfg",
            split="dev",
            condition="clean",
            dataset_hash=ds_hash,
        )
        from accentedge_benchmark.runner.run_manifest import manifest_to_dict
        report_path = tmp_path / "report.json"
        generate_json_report(summary, report_path, run_manifest=manifest_to_dict(manifest))

        with open(report_path) as f:
            report = json.load(f)

        # Verify all expected top-level fields
        assert "benchmark_version" in report
        assert "run_manifest" in report
        assert "summary" in report
        assert report["benchmark_version"] == "1.0.0"
        assert report["summary"]["candidate"] == "passthrough"
        assert report["summary"]["total"] == len(dev_items)
        assert report["run_manifest"]["condition"] == "clean"
        assert "timestamp" in report["run_manifest"]
