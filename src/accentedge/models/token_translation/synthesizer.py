"""Token-conditioned synthesizer for Candidate C.

Upsamples token sequences back to waveform using a lightweight transposed
convolution network with learned speaker conditioning. Designed to be
under 1M parameters for real-time inference.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from accentedge.models.token_translation.interfaces import (
    TokenConditionedSynthesizer,
    TokenSequence,
)


class TokenConditionedSynthesizer(nn.Module):
    """Lightweight token-conditioned waveform synthesizer.

    Upsamples token sequences to audio using transposed convolutions
    with learned speaker conditioning injected via FiLM. Target parameter
    count is under 1M for real-time feasibility.

    Architecture:
        1. Project token embeddings to hidden dim.
        2. Inject speaker conditioning via FiLM.
        3. Series of transposed convolutions for upsampling.
        4. Final linear projection to audio waveform.
    """

    def __init__(
        self,
        token_dim: int = 128,
        speaker_dim: int = 64,
        hidden_dim: int = 128,
        hop_length: int = 4,
        output_channels: int = 1,
    ) -> None:
        super().__init__()
        self.token_dim = token_dim
        self.speaker_dim = speaker_dim
        self.hidden_dim = hidden_dim
        self.hop_length = hop_length
        self.output_channels = output_channels

        # Project token embeddings to hidden dimension.
        self.input_proj = nn.Linear(token_dim, hidden_dim)

        # Speaker FiLM conditioning.
        self.speaker_film = nn.Sequential(
            nn.Linear(speaker_dim, hidden_dim * 2),
            nn.Tanh(),
        )

        # Transposed convolution stack for upsampling.
        self.upsample = nn.ModuleList(
            [
                nn.Sequential(
                    nn.ConvTranspose1d(
                        hidden_dim,
                        hidden_dim,
                        kernel_size=hop_length * 2,
                        stride=hop_length,
                        padding=hop_length // 2,
                    ),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(0.1),
                )
                for _ in range(3)
            ]
        )

        # Final output projection.
        self.output_proj = nn.Conv1d(hidden_dim, output_channels, kernel_size=7, padding=3)

        # Learned speaker embedding table.
        self.speaker_embedding = nn.Embedding(64, speaker_dim)

    def forward(
        self,
        tokens: torch.Tensor,
        speaker_conditioning: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Synthesize waveform from token sequence.

        Args:
            tokens: (B, T, token_dim) token embeddings.
            speaker_conditioning: Optional (B, speaker_dim) speaker embeddings.
                If None, uses a learned speaker embedding table with IDs from 0.

        Returns:
            audio: (B, output_channels, T * hop_length) waveform.
        """
        B, T, _ = tokens.shape

        # Project tokens to hidden.
        h = self.input_proj(tokens)  # (B, T, hidden_dim)

        # Speaker conditioning via FiLM.
        if speaker_conditioning is None:
            speaker_ids = torch.zeros(B, device=h.device, dtype=torch.long)
            speaker_conditioning = self.speaker_embedding(speaker_ids)
        spk = self.speaker_film(speaker_conditioning).unsqueeze(1).expand(-1, T, -1)
        gamma, beta = spk.chunk(2, dim=-1)
        h = gamma * h + beta

        # Transpose for conv1d: (B, hidden_dim, T).
        h = h.transpose(1, 2)

        for block in self.upsample:
            conv, norm, gelu, drop = block
            h = conv(h)
            h = norm(h.transpose(1, 2)).transpose(1, 2)
            h = gelu(h)
            h = drop(h)

        # Final projection.
        audio = self.output_proj(h)
        return torch.tanh(audio)

    def synthesize(
        self,
        tokens: TokenSequence,
        speaker_conditioning: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Synthesize waveform from a TokenSequence.

        Args:
            tokens: Input token sequence.
            speaker_conditioning: Optional speaker embedding tensor.

        Returns:
            (1, output_channels, T * hop_length) audio waveform tensor.
        """
        self.eval()
        with torch.no_grad():
            seq_tensor = tokens.to_tensor().unsqueeze(0)
            audio = self.forward(seq_tensor, speaker_conditioning)
        return audio.squeeze(0)

