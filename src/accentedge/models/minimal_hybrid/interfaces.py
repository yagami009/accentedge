"""MinimalHybrid-specific protocols for Candidate D."""

from __future__ import annotations

from typing import Protocol

import numpy as np
import torch
import torch.nn as nn


class MinimalEncoder(Protocol):
    """Causal waveform encoder: raw audio → per-frame features."""

    input_dim: int

    def encode(
        self,
        x: torch.Tensor,
        state: dict | None = None,
    ) -> torch.Tensor:
        """Encode waveform tensor (B, 1, T) → (B, hidden_dim, T // hop_length)."""
        ...


class MinimalAccentMapper(Protocol):
    """Lightweight mapper: applies accent transformation to frame features."""

    hidden_dim: int

    def map(
        self,
        features: torch.Tensor,
        strength: float,
        target_accent: int = 0,
    ) -> torch.Tensor:
        """Map content features toward target accent.

        Args:
            features: (B, hidden_dim, T_frames)
            strength: conversion strength in [0, 1]
            target_accent: target accent index
        """
        ...


class MinimalSynthesizer(Protocol):
    """Lightweight waveform synthesizer: features → audio."""

    hop_length: int

    def synthesize(self, features: torch.Tensor) -> torch.Tensor:
        """Synthesize waveform from per-frame features.

        Args:
            features: (B, hidden_dim, T_frames)
        Returns:
            audio: (B, 1, T_frames * hop_length)
        """
        ...

