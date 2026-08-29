"""Streaming configuration for Candidate C (Token Translation)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TokenTranslationConfig:
    """Configuration for token translation candidate streaming behavior.

    Attributes:
        token_rate_hz: Token frames per second (default 50 = 20ms frames).
        token_dim: Dimensionality of soft token embeddings (default 128).
        chunk_ms: Preferred streaming chunk size in milliseconds (default 80).
        lookahead_frames: Number of future tokens the translator may access (default 0 = strict causal).
        translator_layers: Number of layers in the translator network (default 2).
        translator_hidden: Hidden dimension size for the translator (default 256).
        conversion_strength: Default accent conversion strength in [0, 1].
        num_accents: Number of target accents supported by the model.
        speaker_dim: Dimension of speaker conditioning embeddings.
    """

    token_rate_hz: int = 50
    token_dim: int = 128
    chunk_ms: int = 80
    lookahead_frames: int = 0
    translator_layers: int = 2
    translator_hidden: int = 256
    conversion_strength: float = 0.5
    num_accents: int = 5
    accent_dim: int = 32
    speaker_dim: int = 64

    @property
    def frame_ms(self) -> float:
        """Duration of a single token frame in milliseconds."""
        return 1000.0 / self.token_rate_hz

    @property
    def lookahead_ms(self) -> float:
        """Lookahead in milliseconds."""
        return self.lookahead_frames * self.frame_ms

    @property
    def chunk_samples(self) -> int:
        """Number of audio samples per preferred chunk at 16 kHz."""
        return int(self.chunk_ms * 16)

    def as_dict(self) -> dict[str, object]:
        return {
            "token_rate_hz": self.token_rate_hz,
            "token_dim": self.token_dim,
            "chunk_ms": self.chunk_ms,
            "lookahead_frames": self.lookahead_frames,
            "translator_layers": self.translator_layers,
            "translator_hidden": self.translator_hidden,
            "conversion_strength": self.conversion_strength,
            "num_accents": self.num_accents,
            "speaker_dim": self.speaker_dim,
        }
