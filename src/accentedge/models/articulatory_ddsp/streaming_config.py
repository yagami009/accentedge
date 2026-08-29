"""Streaming configuration for Articulatory/DDSP candidate."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ArticulatoryStreamingConfig:
    """Configuration for Candidate B streaming behavior."""

    frame_ms: int = 10
    chunk_ms: int = 40
    harmonic_oscillators: int = 64
    noise_bands: int = 32
    encoder_hidden: int = 128
    accent_dim: int = 32
    conversion_strength: float = 0.5
