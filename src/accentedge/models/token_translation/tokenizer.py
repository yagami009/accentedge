"""Causal speech tokenizer for Candidate C.

Produces continuous soft token embeddings from raw audio using a small
causal Conv1D encoder operating at 50 Hz by default. Supports incremental
streaming via state carry-over between chunks.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from accentedge.models.token_translation.interfaces import (
    SpeechToken,
    TokenSequence,
)


class CausalConv1D(nn.Module):
    """Causal 1D convolution: output at t depends only on inputs <= t."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.stride = stride
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pad on the left only to ensure causality.
        x = F.pad(x, (self.padding, 0))
        return self.conv(x)


class CausalSpeechTokenizer(nn.Module):
    """Causal speech tokenizer producing continuous soft embeddings.

    Encodes raw audio into a sequence of continuous token embeddings at a
    configurable frame rate. The architecture is fully causal: no future
    audio is used when producing a token. Supports incremental streaming
    via state carry-over.
    """

    def __init__(
        self,
        token_rate_hz: int = 50,
        token_dim: int = 128,
        hidden_dim: int = 256,
        sample_rate: int = 16000,
    ) -> None:
        super().__init__()
        self.token_rate_hz = token_rate_hz
        self.token_dim = token_dim
        self.sample_rate = sample_rate

        # Frame hop in samples.
        self.hop_length = int(sample_rate / token_rate_hz)

        # Project raw audio to hidden representation.
        self.input_proj = nn.Linear(1, hidden_dim)

        # Small causal Conv1D stack.
        self.conv_stack = nn.ModuleList(
            [
                nn.Sequential(
                    CausalConv1D(hidden_dim, hidden_dim, kernel_size=5, dilation=1),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(0.1),
                )
                for _ in range(3)
            ]
        )

        # Project to token embedding dimension.
        self.output_proj = nn.Linear(hidden_dim, token_dim)

        self._state: dict[str, torch.Tensor] | None = None

    @property
    def state(self) -> dict[str, torch.Tensor] | None:
        """Return current streaming state."""
        return self._state

    def _reset_state(self) -> None:
        self._state = None

    def _init_state(self, batch_size: int, device: torch.device) -> None:
        if self._state is None or self._state["h"].shape[0] != batch_size:
            self._state = {
                "h": torch.zeros(batch_size, 256, device=device),
            }

    def forward(
        self, x: torch.Tensor, state: dict[str, torch.Tensor] | None = None
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Encode audio waveform into token embeddings.

        Args:
            x: Input waveform (B, T) or (B, T, 1).
            state: Optional streaming state from previous chunk.

        Returns:
            tokens: (B, T_frames, token_dim) continuous embeddings.
            new_state: Updated state for next chunk.
        """
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        elif x.dim() == 1:
            x = x.unsqueeze(0).unsqueeze(-1)

        B, T, _ = x.shape
        device = x.device

        # Project to hidden.
        h = self.input_proj(x)  # (B, T, hidden_dim)

        # Transpose for conv1d: (B, hidden_dim, T).
        h = h.transpose(1, 2)

        for block in self.conv_stack:
            conv, norm, gelu, drop = block
            h = conv(h)
            # LayerNorm over channels.
            h = norm(h.transpose(1, 2)).transpose(1, 2)
            h = gelu(h)
            h = drop(h)

        # Transpose back: (B, T, hidden_dim).
        h = h.transpose(1, 2)

        # Downsample to token frame rate by strided mean pooling.
        if self.hop_length > 1:
            # Pad to make T divisible by hop_length.
            remainder = T % self.hop_length
            if remainder != 0:
                pad_len = self.hop_length - remainder
                h = F.pad(h, (0, 0, 0, pad_len))
                T_padded = h.shape[1]
            else:
                T_padded = T
            h = h.transpose(1, 2)  # (B, hidden_dim, T_padded)
            h = F.avg_pool1d(h, kernel_size=self.hop_length, stride=self.hop_length)
            h = h.transpose(1, 2)  # (B, T_frames, hidden_dim)

        tokens = self.output_proj(h)
        new_state = {"h": h.detach().clone()}
        return tokens, new_state

    def tokenize(
        self,
        audio_chunk: torch.Tensor,
        state: dict[str, torch.Tensor] | None = None,
    ) -> "TokenSequence":
        """Tokenize a raw audio chunk into a sequence of soft tokens.

        Args:
            audio_chunk: Raw audio waveform (samples,) or (1, samples).
            state: Optional streaming state dictionary from previous chunk.

        Returns:
            TokenSequence with continuous embeddings and per-token metadata.
        """
        self.eval()
        with torch.no_grad():
            if audio_chunk.dim() == 1:
                audio_chunk = audio_chunk.unsqueeze(0)
            tokens_tensor, new_state = self.forward(audio_chunk, state)
            self._state = new_state

        # Convert to TokenSequence.
        seq = TokenSequence([])
        B, T_frames, _ = tokens_tensor.shape
        duration_ms = 1000.0 / self.token_rate_hz
        for t in range(T_frames):
            token = SpeechToken(
                token_id=t,
                token_embedding=tokens_tensor[0, t].clone().detach(),
                timestamp_ms=t * duration_ms,
                duration_ms=duration_ms,
                is_speech=True,
            )
            seq.tokens.append(token)
        return seq

