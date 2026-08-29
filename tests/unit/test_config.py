"""Tests for accentedge_benchmark.config — YAML loading and validation."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import yaml

from accentedge_benchmark.config import (
    BenchmarkConfig,
    DatasetConfig,
    SplitsConfig,
    DegradationConfig,
    RunnerConfig,
    BenchmarkSettings,
    load_config,
)


class TestLoadConfig:
    def test_load_valid_yaml(self, tmp_path):
        cfg_data = {
            "benchmark": {"benchmark_name": "Test", "benchmark_version": "1.0.0"},
            "dataset": {},
            "splits": {},
            "degradation": {},
            "runner": {},
        }
        fpath = tmp_path / "config.yaml"
        fpath.write_text(yaml.dump(cfg_data))
        config = load_config(fpath)
        assert isinstance(config, BenchmarkSettings)
        assert config.benchmark.benchmark_name == "Test"

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_config(tmp_path / "nonexistent.yaml")

    def test_invalid_sample_rate_raises(self, tmp_path):
        """BenchmarkConfig canonical_sample_rate must be 8000-48000."""
        cfg_data = {
            "benchmark": {"canonical_sample_rate": 100},
            "dataset": {},
            "splits": {},
            "degradation": {},
            "runner": {},
        }
        fpath = tmp_path / "bad_config.yaml"
        fpath.write_text(yaml.dump(cfg_data))
        with pytest.raises(ValueError, match="Config validation failed"):
            load_config(fpath)

    def test_default_config_values(self, tmp_path):
        cfg_data = {
            "benchmark": {},
            "dataset": {},
            "splits": {},
            "degradation": {},
            "runner": {},
        }
        fpath = tmp_path / "config.yaml"
        fpath.write_text(yaml.dump(cfg_data))
        config = load_config(fpath)
        assert config.benchmark.canonical_sample_rate == 16000
        assert config.dataset.min_speakers == 30
        assert config.splits.dev_count == 24
        assert config.splits.locked_test_count == 24
        assert config.splits.seed == 42
        assert config.splits.calibration_exclude is True
        assert config.degradation.nb_target_sr == 8000
        assert config.runner.max_parallel == 4


class TestConfigValidation:
    def test_canonical_sample_rate_upper_bound(self, tmp_path):
        cfg_data = {
            "benchmark": {"canonical_sample_rate": 48001},
            "dataset": {},
            "splits": {},
            "degradation": {},
            "runner": {},
        }
        fpath = tmp_path / "config.yaml"
        fpath.write_text(yaml.dump(cfg_data))
        with pytest.raises(ValueError):
            load_config(fpath)

    def test_utterances_per_speaker_min(self, tmp_path):
        cfg_data = {
            "benchmark": {},
            "dataset": {"utterances_per_speaker": 0},
            "splits": {},
            "degradation": {},
            "runner": {},
        }
        fpath = tmp_path / "config.yaml"
        fpath.write_text(yaml.dump(cfg_data))
        with pytest.raises(ValueError):
            load_config(fpath)

    def test_dev_count_min(self, tmp_path):
        cfg_data = {
            "benchmark": {},
            "dataset": {},
            "splits": {"dev_count": 0},
            "degradation": {},
            "runner": {},
        }
        fpath = tmp_path / "config.yaml"
        fpath.write_text(yaml.dump(cfg_data))
        with pytest.raises(ValueError):
            load_config(fpath)
