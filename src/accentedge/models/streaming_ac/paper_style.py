"""Paper-style Streaming AC model modules (baseline)."""

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


class ContentProsodyEncoder(nn.Module):
    """Conformer/Emformer-style encoder for content and prosody."""

    def __init__(
        self,
        input_dim: int = 80,
        hidden_dim: int = 256,
        num_layers: int = 4,
        right_context_ms: int = 640,
        left_context_ms: int = 0,
        frame_ms: int = 20,
        sample_rate: int = 16000,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.right_context_ms = right_context_ms
        self.left_context_ms = left_context_ms
        self.frame_ms = frame_ms
        self.sample_rate = sample_rate

        self.right_context_frames = right_context_ms // frame_ms
        self.left_context_frames = left_context_ms // frame_ms
        self.context_frames = self.left_context_frames + self.right_context_frames

        if self.context_frames < 0:
            raise ValueError("context_frames must be non-negative")

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.GELU(),
                )
                for _ in range(num_layers)
            ]
        )

        self.output_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(
        self, x: torch.Tensor, state: Optional[dict] = None
    ) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0).unsqueeze(-1)
        elif x.dim() == 2:
            x = x.unsqueeze(-1)
        elif x.dim() != 3:
            raise ValueError(f"Expected 1D/2D/3D tensor, got {x.dim()}D")

        B, T, _ = x.shape
        x = x.reshape(B * T, -1)
        x = self.input_proj(x)
        x = self.norm(x)
        x = x.reshape(B, T, -1)

        residual = x
        for layer in self.layers:
            linear, gelu = layer
            x = residual + gelu(linear(x))
            residual = x
        return self.output_proj(x)

    def encode(
        self, audio: torch.Tensor, state: Optional[dict] = None
    ) -> ContentFrame:
        features = self.forward(audio, state)
        return ContentFrame(features=features, state=state or {})


class AccentBottleneck(nn.Module):
    """Accent bottleneck projection."""

    def __init__(self, content_dim: int = 256, accent_dim: int = 64) -> None:
        super().__init__()
        self.content_dim = content_dim
        self.accent_dim = accent_dim
        self.proj = nn.Linear(content_dim, accent_dim)
        self.norm = nn.LayerNorm(accent_dim)

    def forward(self, content: torch.Tensor) -> torch.Tensor:
        return self.norm(self.proj(content))

    def map(self, content: torch.Tensor, speaker: torch.Tensor) -> AccentLatent:
        return AccentLatent(latent=self.forward(content))


class SpeakerEncoder(nn.Module):
    """Small Conv1D stack extracting global speaker embedding."""

    def __init__(
        self,
        input_dim: int = 80,
        hidden_dim: int = 256,
        speaker_dim: int = 64,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.speaker_dim = speaker_dim

        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Linear(hidden_dim, speaker_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0).unsqueeze(-1)
        elif x.dim() == 2:
            x = x.unsqueeze(-1)
        elif x.dim() != 3:
            raise ValueError(f"Expected 1D/2D/3D tensor, got {x.dim()}D")

        x = x.transpose(1, 2)
        x = self.conv(x)
        x = self.pool(x).flatten(1)
        return self.proj(x)

    def encode(
        self, audio_chunk: torch.Tensor, state: Optional[dict] = None
    ) -> SpeakerEmbedding:
        embedding = self.forward(audio_chunk)
        return SpeakerEmbedding(embedding=embedding)


class CausalOrChunkedSynthesizer(nn.Module):
    """HiFi-GAN-inspired lightweight generator."""

    def __init__(
        self,
        latent_dim: int = 64,
        speaker_dim: int = 64,
        hidden_dim: int = 256,
        output_channels: int = 1,
        hop_length: int = 4,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.speaker_dim = speaker_dim
        self.hidden_dim = hidden_dim
        self.output_channels = output_channels
        self.hop_length = hop_length

        self.input_proj = nn.Linear(latent_dim + speaker_dim + 2, hidden_dim)
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=2, dilation=2)
                for _ in range(3)
            ]
        )
        self.final = nn.Conv1d(hidden_dim, output_channels, kernel_size=7, padding=3)

    def forward(
        self,
        latent: torch.Tensor,
        speaker: torch.Tensor,
        f0: torch.Tensor,
        energy: torch.Tensor,
    ) -> torch.Tensor:
        B, T, _ = latent.shape

        speaker_expanded = speaker.unsqueeze(1).expand(-1, T, -1)
        f0 = f0.unsqueeze(-1)
        energy = energy.unsqueeze(-1)

        x = torch.cat([latent, speaker_expanded, f0, energy], dim=-1)
        x = self.input_proj(x)
        x = x.transpose(1, 2)

        for conv in self.convs:
            x = F.gelu(conv(x))

        x = self.final(x)
        return torch.tanh(x)

    def synthesize(
        self,
        latent: torch.Tensor,
        speaker: torch.Tensor,
        f0: torch.Tensor,
        energy: torch.Tensor,
    ) -> AudioChunk:
        return AudioChunk(audio=self.forward(latent, speaker, f0, energy))

