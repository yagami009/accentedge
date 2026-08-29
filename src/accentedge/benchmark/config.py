"""Configuration loading and validation."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field, ValidationError


class BenchmarkConfig(BaseModel):
    benchmark_name: str = "AccentEdge BPO Benchmark"
    benchmark_version: str = "1.0.0"
    canonical_sample_rate: int = Field(default=16000, ge=8000, le=48000)
    canonical_format: str = "wav"
    master_sample_rate: int = Field(default=48000)


class DatasetConfig(BaseModel):
    manifest_path: str = "data/manifests/benchmark.parquet"
    raw_dir: str = "data/raw/"
    derived_dir: str = "data/derived/"
    min_speakers: int = Field(default=30, ge=1)
    preferred_speakers: int = Field(default=48)
    utterances_per_speaker: int = Field(default=28, ge=1)


class SplitsConfig(BaseModel):
    dev_count: int = Field(default=24, ge=1)
    locked_test_count: int = Field(default=24, ge=1)
    seed: int = 42
    calibration_exclude: bool = True
    stratification_factors: list[str] = ["l1_category", "accent_strength", "bpo_experience"]


class DegradationConfig(BaseModel):
    conditions: list[str] = ["clean", "nb", "noisy", "nb_noisy"]
    nb_target_sr: int = 8000
    nb_ulaw: bool = True
    noisy_snr_db: float = 15.0
    deterministic_seed: int = 42


class RunnerConfig(BaseModel):
    output_dir: str = "runs/"
    max_parallel: int = 4
    resume_enabled: bool = True
    locked_test_registry: str = "runs/locked_test_registry.jsonl"


class BenchmarkSettings(BaseModel):
    benchmark: BenchmarkConfig
    dataset: DatasetConfig
    splits: SplitsConfig
    degradation: DegradationConfig
    runner: RunnerConfig


def load_config(path: str | Path) -> BenchmarkSettings:
    """Load and validate benchmark configuration from YAML."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path) as f:
        data = yaml.safe_load(f)
    try:
        return BenchmarkSettings(**data)
    except ValidationError as exc:
        raise ValueError(f"Config validation failed: {exc}") from exc


def get_config() -> BenchmarkSettings:
    """Load default config from configs/benchmark.yaml."""
    return load_config(Path(__file__).parent.parent.parent / "configs" / "benchmark.yaml")
