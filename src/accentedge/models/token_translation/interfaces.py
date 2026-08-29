"""Token translation interfaces for Candidate C.

Hypothesis: mapping structured content representations (phonetic/acoustic
tokens) is easier for accent transformation than direct waveform regression.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import torch


@dataclass
class SpeechToken:
    """Single soft speech token produced by the causal tokenizer.

    Attributes:
        token_id: Integer identifier (optional for continuous embeddings).
        token_embedding: Continuous soft embedding vector (dim=128).
        timestamp_ms: Start time of this token in the source audio.
        duration_ms: Duration this token represents.
        is_speech: Whether this frame contains speech (vs silence/padding).
    """

    token_id: int
    token_embedding: torch.Tensor
    timestamp_ms: float
    duration_ms: float
    is_speech: bool = True


class TokenSequence:
    """Wrapper around a list of SpeechToken instances with batch utilities.

    Provides conversion to/from dense tensors for efficient processing by
    downstream neural components while preserving per-token metadata.
    """

    def __init__(self, tokens: list[SpeechToken]) -> None:
        self.tokens = tokens

    def __len__(self) -> int:
        return len(self.tokens)

    def __getitem__(self, idx: int) -> SpeechToken:
        return self.tokens[idx]

    def to_tensor(self) -> torch.Tensor:
        """Return stacked embeddings as (seq_len, dim) tensor."""
        if not self.tokens:
            return torch.empty(0, self.dim, dtype=torch.float32)
        return torch.stack([t.token_embedding for t in self.tokens], dim=0)

    @classmethod
    def from_tensor(
        cls,
        tensor: torch.Tensor,
        start_ms: float = 0.0,
        duration_ms: float = 20.0,
    ) -> "TokenSequence":
        """Create a TokenSequence from a dense (seq_len, dim) embedding matrix."""
        if tensor.dim() != 2:
            raise ValueError(f"Expected 2D tensor, got {tensor.dim()}D")
        tokens = []
        for i in range(tensor.shape[0]):
            tokens.append(
                SpeechToken(
                    token_id=i,
                    token_embedding=tensor[i].clone().detach(),
                    timestamp_ms=start_ms + i * duration_ms,
                    duration_ms=duration_ms,
                    is_speech=True,
                )
            )
        return cls(tokens)

    @property
    def timestamps(self) -> list[float]:
        return [t.timestamp_ms for t in self.tokens]

    @property
    def durations(self) -> list[float]:
        return [t.duration_ms for t in self.tokens]

    @property
    def dim(self) -> int:
        if self.tokens:
            return int(self.tokens[0].token_embedding.shape[-1])
        return 0

    def slice(self, start: int, end: int) -> "TokenSequence":
        return TokenSequence(self.tokens[start:end])


class CausalSpeechTokenizer(Protocol):
    """Protocol for causal speech tokenizers.

    Produces continuous soft token embeddings from raw audio without
    access to future frames. Supports incremental streaming via state.
    """

    def tokenize(
        self,
        audio_chunk: torch.Tensor,
        state: dict[str, Any] | None = None,
    ) -> "TokenSequence": ...


class AccentTokenTranslator(Protocol):
    """Protocol for accent token translators.

    Maps source token embeddings to target-accent embeddings using a
    causal (or bounded-lookahead) architecture conditioned on accent
    identity and conversion strength. No text required at inference.
    """

    def translate(
        self,
        tokens: "TokenSequence",
        target_accent: torch.Tensor,
        strength: float,
        context: dict[str, Any] | None = None,
    ) -> "TokenSequence": ...


class TokenConditionedSynthesizer(Protocol):
    """Protocol for token-conditioned waveform synthesizers.

    Upsamples token sequences back to waveform using learned speaker
    conditioning. Lightweight architecture suitable for real-time use.
    """

    def synthesize(
        self,
        tokens: "TokenSequence",
        speaker_conditioning: torch.Tensor | None = None,
    ) -> torch.Tensor: ...

