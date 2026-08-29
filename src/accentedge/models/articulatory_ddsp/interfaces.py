"""Articulatory/DDSP interfaces for Candidate B."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch


@dataclass
class ArticulatoryFrame:
    """Single articulatory frame at ~100 Hz.

    Fields:
        content_features: encoded content/language features (B, T, D)
        f0: fundamental frequency in Hz (B, T, 1)
        voicing: binary voicing probability (B, T, 1)
        energy: frame energy/RMS (B, T, 1)
        timing: phoneme timing alignment / duration hint (B, T, 1)
    """

    content_features: torch.Tensor
    f0: torch.Tensor
    voicing: torch.Tensor
    energy: torch.Tensor
    timing: torch.Tensor

    def to(self, device: str) -> "ArticulatoryFrame":
        return ArticulatoryFrame(
            content_features=self.content_features.to(device),
            f0=self.f0.to(device),
            voicing=self.voicing.to(device),
            energy=self.energy.to(device),
            timing=self.timing.to(device),
        )


class ArticulatoryFrameSequence:
    """Wrapper around list[ArticulatoryFrame] with batch conversion."""

    def __init__(self, frames: list[ArticulatoryFrame]) -> None:
        self.frames = frames

    def to_tensor(self) -> torch.Tensor:
        """Stack content features across time into (B, T, D)."""
        if not self.frames:
            return torch.empty(0)
        return torch.cat([f.content_features for f in self.frames], dim=1)

    def __len__(self) -> int:
        return len(self.frames)


class ArticulatoryEncoder(Protocol):
    """Encode audio into articulatory parameters."""

    def encode(
        self,
        audio: torch.Tensor,
        state: dict | None = None,
    ) -> ArticulatoryFrameSequence: ...


class ArticulatoryAccentMapper(Protocol):
    """Map source articulatory frames toward a target accent."""

    def map(
        self,
        source_frames: ArticulatoryFrameSequence,
        target_accent: torch.Tensor,
        strength: float,
    ) -> ArticulatoryFrameSequence: ...


class DDSPSynthesizer(Protocol):
    """Synthesize audio from articulatory frames + speaker conditioning."""

    def synthesize(
        self,
        frames: ArticulatoryFrameSequence,
        speaker_conditioning: torch.Tensor,
    ) -> torch.Tensor: ...

