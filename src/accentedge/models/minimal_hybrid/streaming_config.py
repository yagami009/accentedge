"""Streaming configuration for MinimalHybrid Candidate D."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class MinimalHybridConfig:
    """Configuration for Candidate D MinimalHybrid streaming behavior."""

    frame_ms: float = 20.0
    chunk_ms: int = 80
    hidden_dim: int = 64
    num_accents: int = 5
    sample_rate: int = 16000
    conversion_strength: float = 0.5
    encoder_kernel_size: int = 5
    hop_length: int = 160

    @property
    def frame_samples(self) -> int:
        """Number of audio samples per frame."""
        return int(self.sample_rate * self.frame_ms / 1000)

    def as_dict(self) -> dict[str, object]:
        return {
            "frame_ms": self.frame_ms,
            "chunk_ms": self.chunk_ms,
            "hidden_dim": self.hidden_dim,
            "num_accents": self.num_accents,
            "sample_rate": self.sample_rate,
            "conversion_strength": self.conversion_strength,
            "encoder_kernel_size": self.encoder_kernel_size,
            "hop_length": self.hop_length,
        }
