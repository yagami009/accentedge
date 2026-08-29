"""Low-lookahead Streaming AC model modules (reduced defaults)."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from accentedge.models.streaming_ac.interfaces import (
    AccentLatent,
    AudioChunk,
    ContentFrame,
    SpeakerEmbedding,
)
from accentedge.models.streaming_ac.paper_style import (
    AccentBottleneck as PaperStyleAccentBottleneck,
    CausalOrChunkedSynthesizer as PaperStyleCausalOrChunkedSynthesizer,
    ContentProsodyEncoder as PaperStyleContentProsodyEncoder,
    SpeakerEncoder as PaperStyleSpeakerEncoder,
)


class LowLookaheadContentProsodyEncoder(PaperStyleContentProsodyEncoder):
    """Content/prosody encoder with low lookahead defaults."""

    def __init__(
        self,
        input_dim: int = 80,
        hidden_dim: int = 128,
        num_layers: int = 2,
        right_context_ms: int = 0,
        left_context_ms: int = 0,
        frame_ms: int = 20,
        sample_rate: int = 16000,
    ) -> None:
        # Validate configurable range.
        if not 0 <= right_context_ms <= 320:
            raise ValueError("right_context_ms must be between 0 and 320")
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            right_context_ms=right_context_ms,
            left_context_ms=left_context_ms,
            frame_ms=frame_ms,
            sample_rate=sample_rate,
        )


class LowLookaheadAccentBottleneck(PaperStyleAccentBottleneck):
    """Accent bottleneck with low-lookahead compatible dims."""

    def __init__(self, content_dim: int = 128, accent_dim: int = 64) -> None:
        super().__init__(content_dim=content_dim, accent_dim=accent_dim)


class LowLookaheadSpeakerEncoder(PaperStyleSpeakerEncoder):
    """Speaker encoder with low-lookahead compatible dims."""

    def __init__(
        self,
        input_dim: int = 80,
        hidden_dim: int = 128,
        speaker_dim: int = 64,
    ) -> None:
        super().__init__(
            input_dim=input_dim, hidden_dim=hidden_dim, speaker_dim=speaker_dim
        )


class LowLookaheadCausalOrChunkedSynthesizer(PaperStyleCausalOrChunkedSynthesizer):
    """Synthesizer with low-lookahead compatible dims."""

    def __init__(
        self,
        latent_dim: int = 64,
        speaker_dim: int = 64,
        hidden_dim: int = 128,
        output_channels: int = 1,
        hop_length: int = 4,
    ) -> None:
        super().__init__(
            latent_dim=latent_dim,
            speaker_dim=speaker_dim,
            hidden_dim=hidden_dim,
            output_channels=output_channels,
            hop_length=hop_length,
        )

