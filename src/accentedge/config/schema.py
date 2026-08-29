"""Phase 2 configuration schema."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class TrainingConfig(BaseModel):
    max_steps: int = 50000
    max_audio_hours: float = 1000.0
    max_wall_clock_hours: float = 24.0
    batch_size: int = 32
    learning_rate: float = 1e-4
    seed: int = 42
    device: Literal["cpu", "cuda", "mps", "auto"] = "auto"
    precision: Literal["fp32", "fp16", "bf16"] = "fp32"
    gradient_clip_norm: float = 1.0
    checkpoint_every_n_steps: int = 1000
    validation_every_n_steps: int = 500
    identity_pair_ratio: float = 0.25


class StreamingConfig(BaseModel):
    default_chunk_ms: int = 80
    default_lookahead_ms: int = 0
    default_left_context_ms: int = 0
    chunk_sizes_to_sweep: list[int] = Field(default_factory=lambda: [20, 40, 80, 160])
    lookahead_to_sweep: list[int] = Field(
        default_factory=lambda: [0, 20, 40, 80, 160, 320]
    )
    long_session_minutes: int = 30


class HardwareConfig(BaseModel):
    cpu_model: str = "unknown"
    logical_cores: int = 1
    ram_gb: float = 0.0
    gpu_model: str | None = None
    torch_threads: int | None = None
    interop_threads: int = 2


class CandidateConfig(BaseModel):
    architecture_id: str
    enabled: bool = True
    config_path: str | None = None


class Phase2Config(BaseModel):
    phase2_id: str
    phase: Literal["2"] = "2"
    data_root: Path = Field(default=Path("data"))
    experiments_root: Path = Field(default=Path("experiments"))
    benchmark_repo_path: str = "accentedge-benchmark"
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    streaming: StreamingConfig = Field(default_factory=StreamingConfig)
    hardware: HardwareConfig = Field(default_factory=HardwareConfig)
    candidates: list[CandidateConfig] = Field(default_factory=list)
