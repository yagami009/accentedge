"""Component interfaces for Streaming AC candidate."""

from __future__ import annotations

from typing import Protocol

import torch


class ContentProsodyEncoder(Protocol):
    """Encode raw audio into content/prosody frames."""

    def encode(
        self, audio: torch.Tensor, state: dict | None = None
    ) -> "ContentFrame": ...


class AccentBottleneck(Protocol):
    """Map content frames to accent-conditioned latents."""

    def map(
        self, content: torch.Tensor, speaker: torch.Tensor
    ) -> "AccentLatent": ...


class SpeakerEncoder(Protocol):
    """Extract speaker embedding from an audio chunk."""

    def encode(
        self, audio_chunk: torch.Tensor, state: dict | None = None
    ) -> "SpeakerEmbedding": ...


class CausalOrChunkedSynthesizer(Protocol):
    """Synthesize audio from latent, speaker, f0, and energy."""

    def synthesize(
        self,
        latent: torch.Tensor,
        speaker: torch.Tensor,
        f0: torch.Tensor,
        energy: torch.Tensor,
    ) -> "AudioChunk": ...


class ContentFrame:
    """Container for encoded content/prosody features."""

    def __init__(self, features: torch.Tensor, state: dict | None = None) -> None:
        self.features = features
        self.state = state or {}


class AccentLatent:
    """Container for accent-conditioned latent representation."""

    def __init__(self, latent: torch.Tensor) -> None:
        self.latent = latent


class SpeakerEmbedding:
    """Container for speaker embedding."""

    def __init__(self, embedding: torch.Tensor, confidence: float = 1.0) -> None:
        self.embedding = embedding
        self.confidence = confidence


class AudioChunk:
    """Container for synthesized audio chunk."""

    def __init__(self, audio: torch.Tensor) -> None:
        self.audio = audio

