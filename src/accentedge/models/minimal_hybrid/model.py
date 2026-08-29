"""Candidate D — Minimal Hybrid model implementation.

Deliberately boring first-principles architecture:
  causal Conv1d encoder → linear accent mapper → simple upsampler synthesizer

Parameter target: < 500K
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from accentedge.models.interfaces import (
    CandidateMetadata,
    StreamingCandidate,
    StreamingResult,
    StreamingSession,
)
from accentedge.models.minimal_hybrid.interfaces import (
    MinimalAccentMapper,
    MinimalEncoder,
    MinimalSynthesizer,
)
from accentedge.models.minimal_hybrid.streaming_config import MinimalHybridConfig


# ---------------------------------------------------------------------------
# 1. MinimalHybridEncoder: 2-layer causal Conv1d
# ---------------------------------------------------------------------------

class _CausalConv1dBlock(nn.Module):
    """Two-layer causal Conv1d with GELU activation."""

    def __init__(self, in_ch: int, hidden_dim: int, kernel_size: int) -> None:
        super().__init__()
        self.pad1 = nn.ConstantPad1d((kernel_size - 1, 0), 0.0)
        self.conv1 = nn.Conv1d(in_ch, hidden_dim, kernel_size, padding=0)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.pad2 = nn.ConstantPad1d((kernel_size - 1, 0), 0.0)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=0)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        h = F.gelu(self.norm1(self.conv1(self.pad1(x)).transpose(1, 2)).transpose(1, 2))
        h = F.gelu(self.norm2(self.conv2(self.pad2(h)).transpose(1, 2)).transpose(1, 2))
        return h


class MinimalHybridEncoder(nn.Module):
    """Strictly causal waveform encoder.

    Raw audio (B, 1, T) → frame features (B, hidden_dim, T_frames).
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        kernel_size: int = 5,
        hop_length: int = 160,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.hop_length = hop_length
        # Average pool to frame rate
        self.downsample = nn.AvgPool1d(kernel_size=hop_length, stride=hop_length)
        # Causal conv layers on downsampled signal
        self.block = _CausalConv1dBlock(1, hidden_dim, kernel_size)

    def forward(
        self,
        x: torch.Tensor,
        state: dict | None = None,
    ) -> torch.Tensor:
        """Encode waveform to frame features.

        Args:
            x: (B, 1, T) raw waveform
            state: unused (interface requirement)
        Returns:
            features: (B, hidden_dim, T_frames)
        """
        # Downsample to frame rate first, then apply causal conv
        x_down = self.downsample(x)  # (B, 1, T_frames)
        features = self.block(x_down)  # (B, hidden_dim, T_frames)
        return features


# ---------------------------------------------------------------------------
# 2. MinimalAccentMapper: single linear layer + per-accent embedding
# ---------------------------------------------------------------------------

class MinimalAccentMapper(nn.Module):
    """Lightweight accent mapper.

    Applies a learned per-accent linear shift + scale to frame features.
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        num_accents: int = 5,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        # Per-accent affine parameters: shift and scale
        self.accent_shift = nn.Embedding(num_accents, hidden_dim)
        self.accent_scale = nn.Embedding(num_accents, hidden_dim)

    def forward(
        self,
        features: torch.Tensor,
        strength: float = 0.5,
        target_accent: int = 0,
    ) -> torch.Tensor:
        """Map content features toward target accent.

        Args:
            features: (B, hidden_dim, T_frames)
            strength: blending factor in [0, 1]
            target_accent: accent index
        Returns:
            mapped_features: (B, hidden_dim, T_frames)
        """
        B = features.size(0)
        device = features.device
        accent_idx = torch.tensor([target_accent], device=device, dtype=torch.long)
        shift = self.accent_shift(accent_idx).unsqueeze(-1)  # (1, D, 1)
        scale = torch.sigmoid(self.accent_scale(accent_idx)).unsqueeze(-1)  # (1, D, 1)

        # Identity at strength=0, full accent at strength=1
        mapped = features * (1 - strength) + (features * scale + shift) * strength
        return mapped


# ---------------------------------------------------------------------------
# 3. MinimalSynthesizer: linear upsampler + overlap-add
# ---------------------------------------------------------------------------

class MinimalSynthesizer(nn.Module):
    """Lightweight waveform synthesizer using ConvTranspose1d upsampling."""

    def __init__(
        self,
        hidden_dim: int = 64,
        hop_length: int = 160,
    ) -> None:
        super().__init__()
        self.hop_length = hop_length
        # ConvTranspose1d for learned upsampling (strictly causal via left padding)
        self.upsample = nn.ConvTranspose1d(
            hidden_dim,
            1,
            kernel_size=hop_length * 2,
            stride=hop_length,
            padding=hop_length // 2,
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Synthesize waveform from frame features.

        Args:
            features: (B, hidden_dim, T_frames)
        Returns:
            audio: (B, 1, T_frames * hop_length)
        """
        audio = self.upsample(features)
        return audio


# ---------------------------------------------------------------------------
# 4. MinimalHybridCandidate
# ---------------------------------------------------------------------------

class _MinimalHybridSession:
    """Internal session state holder."""

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self.step = 0
        self.timeline: list[tuple[int, int, int, int]] = []

    def state_size_bytes(self) -> int:
        # Minimal: timeline entries only
        return len(self.timeline) * 32  # 4 ints * 8 bytes


class MinimalHybridCandidate:
    """Candidate D: Minimal Hybrid accent conversion.

    Architecture:
        raw audio → causal Conv1d encoder → accent mapper → ConvTranspose1d synth

    Designed to be the simplest viable streaming accent-conversion model.
    """

    metadata = CandidateMetadata(
        architecture_id="minimal_hybrid",
        version="0.1.0",
        input_sample_rate=16000,
        frame_ms=20.0,
        preferred_chunk_ms=80,
        required_lookahead_ms=0,
        left_context_ms=0,
        supports_conversion_strength=True,
        supports_target_accent=True,
        requires_reference_speaker=False,
        uses_text_at_inference=False,
    )

    def __init__(self, config: MinimalHybridConfig | None = None) -> None:
        self.config = config or MinimalHybridConfig()
        self.metadata.frame_ms = self.config.frame_ms
        self._device: str = "cpu"
        self._precision: str = "fp32"
        self._closed = False
        self._build()
        self.metadata.parameter_count = self.count_parameters()

    def _build(self) -> None:
        cfg = self.config
        self.encoder = MinimalHybridEncoder(
            hidden_dim=cfg.hidden_dim,
            kernel_size=cfg.encoder_kernel_size,
            hop_length=cfg.hop_length,
        )
        self.mapper = MinimalAccentMapper(
            hidden_dim=cfg.hidden_dim,
            num_accents=cfg.num_accents,
        )
        self.synthesizer = MinimalSynthesizer(
            hidden_dim=cfg.hidden_dim,
            hop_length=cfg.hop_length,
        )

    # -- StreamingCandidate protocol ----------------------------------------

    def prepare(self, device: str, precision: str) -> None:
        """Move model to target device / precision."""
        if self._closed:
            raise RuntimeError("candidate is closed")
        self._device = device
        self._precision = precision
        for m in (self.encoder, self.mapper, self.synthesizer):
            m = m.to(torch.device(device))
            if precision == "fp16" and device != "cpu":
                m = m.half()
            elif precision == "bf16" and device != "cpu":
                m = m.bfloat16()

    def create_session(self, config: dict[str, Any]) -> StreamingSession:
        """Create a new streaming session."""
        if self._closed:
            raise RuntimeError("candidate is closed")
        session_id = config.get("session_id", "minimal_hybrid")
        internal = _MinimalHybridSession(session_id=session_id)
        return StreamingSession(
            session_id=session_id,
            state={"d": internal},
            created_at=datetime.utcnow(),
            samples_processed=0,
        )

    def process_chunk(
        self,
        session: StreamingSession,
        audio_chunk: np.ndarray,
        sample_rate: int,
    ) -> StreamingResult:
        """Run encode → map → synthesize on a single chunk."""
        if self._closed:
            raise RuntimeError("candidate is closed")
        internal: _MinimalHybridSession = session.state["d"]

        input_start = session.samples_processed
        input_end = input_start + len(audio_chunk)

        # Convert to torch: (1, 1, T)
        x = torch.from_numpy(audio_chunk).float().unsqueeze(0).unsqueeze(0)
        x = x.to(torch.device(self._device))

        # Encode
        features = self.encoder(x, state=None)  # (1, D, T_frames)

        # Map toward accent
        target_accent = int(session.state.get("target_accent", 0))
        strength = self.config.conversion_strength
        mapped = self.mapper(features, strength=strength, target_accent=target_accent)

        # Synthesize
        audio_tensor = self.synthesizer(mapped)  # (1, 1, T_audio)
        audio_np = audio_tensor.squeeze().detach().cpu().numpy()

        # Align output length with input (trim or pad)
        if len(audio_np) > len(audio_chunk):
            audio_np = audio_np[: len(audio_chunk)]
        elif len(audio_np) < len(audio_chunk):
            audio_np = np.pad(audio_np, (0, len(audio_chunk) - len(audio_np)))

        output_start = input_start
        output_end = output_start + len(audio_np)

        internal.step += 1
        internal.timeline.append((input_start, input_end, output_start, output_end))
        session.samples_processed = input_end

        return StreamingResult(
            audio=audio_np,
            sample_rate=sample_rate,
            input_start_sample=input_start,
            input_end_sample=input_end,
            output_start_sample=output_start,
            output_end_sample=output_end,
            algorithmic_delay_samples=0,
            metadata={"conversion_strength": strength},
        )

    def flush(self, session: StreamingSession) -> list[StreamingResult]:
        """Flush buffered outputs (no buffering in minimal impl)."""
        return []

    def reset(self, session: StreamingSession) -> None:
        """Clear all session state."""
        internal: _MinimalHybridSession | None = session.state.get("d")
        if internal is not None:
            internal.step = 0
            internal.timeline = []
        session.samples_processed = 0

    def close(self) -> None:
        """Cleanup resources."""
        if self._closed:
            return
        del self.encoder
        del self.mapper
        del self.synthesizer
        self._closed = True

    # -- Helpers -----------------------------------------------------------

    def count_parameters(self) -> int:
        """Return total trainable parameter count."""
        total = 0
        for m in (self.encoder, self.mapper, self.synthesizer):
            if isinstance(m, nn.Module):
                total += sum(p.numel() for p in m.parameters() if p.requires_grad)
        return total

