"""Articulatory accent mapper."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from accentedge.models.articulatory_ddsp.interfaces import (
    ArticulatoryAccentMapper,
    ArticulatoryFrame,
    ArticulatoryFrameSequence,
)
from accentedge.models.articulatory_ddsp.streaming_config import (
    ArticulatoryStreamingConfig,
)


class ArticulatoryAccentMapper(nn.Module):
    """Map articulatory frames toward a target accent embedding.

    Implements linear interpolation for F0 toward a learned target template,
    and frame-wise content modulation for formant-related parameters.
    Speaker F0 contour shape is preserved while shifting means.
    """

    def __init__(self, config: ArticulatoryStreamingConfig | None = None) -> None:
        super().__init__()
        self.config = config or ArticulatoryStreamingConfig()
        hidden = self.config.encoder_hidden
        accent_dim = self.config.accent_dim

        # Target accent embedding table.
        self.accent_embedding = nn.Embedding(10, accent_dim)
        self.content_proj = nn.Linear(hidden, hidden)
        self.accent_proj = nn.Linear(accent_dim, hidden)
        self.strength_gate = nn.Linear(1, hidden)
        self.target_f0_template = nn.Embedding(10, 1)

    def forward(
        self,
        source_frames: ArticulatoryFrameSequence,
        target_accent: torch.Tensor,
        strength: float = 0.5,
    ) -> ArticulatoryFrameSequence:
        """Map source frames toward target accent.

        Args:
            source_frames: input articulatory frames.
            target_accent: integer accent IDs (B,) or embedding (B, D).
            strength: interpolation factor in [0, 1].

        Returns:
            Modified ArticulatoryFrameSequence.
        """
        if not source_frames.frames:
            return source_frames

        B = source_frames.frames[0].content_features.shape[0]
        device = source_frames.frames[0].content_features.device

        if target_accent.dim() == 1:
            accent_ids = target_accent.long()
        else:
            accent_ids = target_accent.argmax(dim=-1)

        accent_emb = self.accent_embedding(accent_ids)  # (B, accent_dim)
        strength_tensor = torch.full(
            (B, 1, 1), strength, device=device, dtype=source_frames.frames[0].content_features.dtype
        )
        gate = torch.sigmoid(self.strength_gate(strength_tensor))

        modified: list[ArticulatoryFrame] = []
        for frame in source_frames.frames:
            content = frame.content_features
            f0 = frame.f0
            energy = frame.energy
            voicing = frame.voicing
            timing = frame.timing

            accent_mod = self.accent_proj(accent_emb).unsqueeze(1)
            mapped_content = (1 - gate) * self.content_proj(content) + gate * accent_mod

            target_f0 = self.target_f0_template(accent_ids).unsqueeze(1)
            mapped_f0 = (1 - strength) * f0 + strength * target_f0

            modified.append(
                ArticulatoryFrame(
                    content_features=mapped_content,
                    f0=mapped_f0,
                    voicing=voicing,
                    energy=energy,
                    timing=timing,
                )
            )

        return ArticulatoryFrameSequence(modified)

    def map(
        self,
        source_frames: ArticulatoryFrameSequence,
        target_accent: torch.Tensor,
        strength: float = 0.5,
    ) -> ArticulatoryFrameSequence:
        return self.forward(source_frames, target_accent, strength)

