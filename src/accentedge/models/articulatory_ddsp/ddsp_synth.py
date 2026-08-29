"""DDSP-based synthesizer for Candidate B."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from accentedge.models.articulatory_ddsp.interfaces import (
    ArticulatoryFrameSequence,
    DDSPSynthesizer,
)
from accentedge.models.articulatory_ddsp.streaming_config import (
    ArticulatoryStreamingConfig,
)


class DDSPSynthesizer(nn.Module):
    """DDSP synthesizer: harmonic oscillator bank + filtered noise + reverb."""

    def __init__(
        self,
        config: ArticulatoryStreamingConfig | None = None,
        sample_rate: int = 16000,
    ) -> None:
        super().__init__()
        self.config = config or ArticulatoryStreamingConfig()
        self.sample_rate = sample_rate
        self.frame_ms = self.config.frame_ms
        self.samples_per_frame = sample_rate * self.frame_ms // 1000
        self.harmonics = self.config.harmonic_oscillators
        self.noise_bands = self.config.noise_bands
        hidden = self.config.encoder_hidden

        self.harmonic_amp = nn.Linear(hidden, self.harmonics)
        self.harmonic_phase = nn.Linear(hidden, self.harmonics)
        self.noise_filter = nn.Linear(hidden, self.noise_bands)

        ir_len = sample_rate // 100
        self.register_buffer("ir", torch.randn(1, 1, ir_len) * 0.01)

    def _harmonic_oscillators(
        self, f0: torch.Tensor, amps: torch.Tensor, phases: torch.Tensor
    ) -> torch.Tensor:
        B, T, _ = f0.shape
        device = f0.device
        t = torch.arange(self.samples_per_frame, device=device, dtype=f0.dtype) / self.sample_rate
        t = t.view(1, 1, 1, -1)  # (1, 1, 1, S)

        f0_exp = f0.unsqueeze(-1)  # (B, T, 1, 1)
        harmonics = torch.arange(1, self.harmonics + 1, device=device, dtype=f0.dtype).view(1, 1, -1, 1)
        phase = phases.unsqueeze(-1)  # (B, T, H, 1)
        amp = amps.unsqueeze(-1)  # (B, T, H, 1)

        inst_freq = f0_exp * harmonics  # (B, T, H, 1)
        phase_acc = 2 * torch.pi * inst_freq * t + phase
        sine = torch.sin(phase_acc)
        frame_wave = (amp * sine).sum(dim=-2)  # sum over harmonics -> (B, T, S)
        return frame_wave.reshape(B, T * self.samples_per_frame)

    def _filtered_noise(self, content: torch.Tensor, T: int) -> torch.Tensor:
        B = content.shape[0]
        device = content.device
        filters = torch.tanh(self.noise_filter(content))  # (B, T, noise_bands)
        filters_bt = filters.transpose(1, 2)  # (B, noise_bands, T)
        # Interpolate each band independently in 3D.
        filters_up = torch.nn.functional.interpolate(
            filters_bt,
            size=self.samples_per_frame,
            mode="linear",
            align_corners=False,
        )  # (B, noise_bands, S)
        noise = torch.randn(B, T, self.samples_per_frame, device=device)
        shaped = noise * filters_up.sum(dim=1).unsqueeze(1)
        return shaped.reshape(B, -1)

    def forward(
        self, frames: ArticulatoryFrameSequence, speaker_conditioning: torch.Tensor
    ) -> torch.Tensor:
        """Synthesize audio from frames.

        Args:
            frames: articulatory frame sequence.
            speaker_conditioning: speaker embedding (B, D) or (B, 1, D).

        Returns:
            waveform (B, total_samples).
        """
        if not frames.frames:
            return torch.empty(1, 0)

        content, f0, voicing, energy = _to_frames(frames)
        B, T = content.shape[:2]

        amps = torch.sigmoid(self.harmonic_amp(content)) * energy
        phases = self.harmonic_phase(content)

        harmonic = self._harmonic_oscillators(f0, amps, phases)
        noise = self._filtered_noise(content, T)

        mix = harmonic * voicing.squeeze(-1).repeat_interleave(self.samples_per_frame, 1) + noise * (1 - voicing.squeeze(-1).repeat_interleave(self.samples_per_frame, 1))

        ir = self.ir.repeat(B, 1, 1)
        wet = torch.nn.functional.conv1d(
            mix.unsqueeze(1), ir, padding=self.ir.shape[-1] - 1
        ).squeeze(1)
        out = (mix + 0.2 * wet[..., : mix.shape[-1]]).squeeze(1)

        return out

    def synthesize(
        self,
        frames: ArticulatoryFrameSequence,
        speaker_conditioning: torch.Tensor,
    ) -> torch.Tensor:
        return self.forward(frames, speaker_conditioning)


def _to_frames(frames: ArticulatoryFrameSequence) -> tuple[torch.Tensor, ...]:
    if not frames.frames:
        return (torch.empty(0), torch.empty(0), torch.empty(0), torch.empty(0))
    content = torch.cat([f.content_features for f in frames.frames], dim=1)
    f0 = torch.cat([f.f0 for f in frames.frames], dim=1)
    voicing = torch.cat([f.voicing for f in frames.frames], dim=1)
    energy = torch.cat([f.energy for f in frames.frames], dim=1)
    return content, f0, voicing, energy

