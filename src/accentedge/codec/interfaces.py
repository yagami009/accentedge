"""Phase 1 — Factorized speech codec interface.

AccentEdge does not permanently couple to one upstream codec implementation.
This module defines the internal abstraction.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch


@dataclass
class FactorizedLatents:
    """Container for factorized speech representations.

    Attributes correspond to FAC-FACodec verified factorization:
      content:    z_c1 + z_c2 (quantized content residual)
      content_zc1: z_c1 (8-dim quantized content codebook indices)
      content_zc2: z_c2 (content residual, predicted after denoising)
      prosody:    z_p (prosody codebook indices)
      detail:     z_r (residual acoustic detail codebook indices)
      timbre:     g (global timbre embedding, one vector per utterance)
      metadata:   arbitrary per-frame/per-utterance metadata
    """
    content: torch.Tensor          # [B, C, T] quantized content representation
    content_zc1: torch.Tensor      # [B, 1, T] codebook indices for z_c1
    content_zc2: Optional[torch.Tensor] = None  # [B, C2, T] codebook indices for z_c2
    prosody: Optional[torch.Tensor] = None       # [B, 1, T] codebook indices
    detail: Optional[torch.Tensor] = None        # [B, K, T] codebook indices for residual
    timbre: Optional[torch.Tensor] = None        # [B, D] global timbre vector
    metadata: dict = field(default_factory=dict)


class FactorizedSpeechCodec:
    """Protocol for factorized speech codecs.

    All implementations must:
      1. Freeze codec parameters after initialization
      2. Provide encode/decode round-trip
      3. Expose sample_rate
    """

    sample_rate: int

    def encode(self, waveform: torch.Tensor) -> FactorizedLatents:
        """Encode waveform → factorized latents."""
        ...

    def decode(self, latents: FactorizedLatents) -> torch.Tensor:
        """Decode factorized latents → waveform."""
        ...

    def freeze(self) -> None:
        """Freeze all codec parameters."""
        for param in self.parameters():
            param.requires_grad = False

    def parameters(self):
        """Yield all parameters."""
        ...

