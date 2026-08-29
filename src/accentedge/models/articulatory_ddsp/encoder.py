"""Articulatory encoder — audio to articulatory parameter frames."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from accentedge.models.articulatory_ddsp.interfaces import (
    ArticulatoryFrame,
    ArticulatoryFrameSequence,
    ArticulatoryEncoder,
)
from accentedge.models.articulatory_ddsp.streaming_config import (
    ArticulatoryStreamingConfig,
)


class ArticulatoryEncoder(nn.Module):
    """Lightweight articulatory parameter encoder.

    Produces ~100 Hz frame-level parameters from 16 kHz audio using a small
    convolutional stack and dedicated F0/energy/voicing heads.
    """

    def __init__(self, config: ArticulatoryStreamingConfig | None = None) -> None:
        super().__init__()
        self.config = config or ArticulatoryStreamingConfig()
        hidden = self.config.encoder_hidden
        self.sample_rate = 16000
        self.frame_ms = self.config.frame_ms

        # Feature extractor: 16 kHz -> 2 kHz (stride 8).
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
        )

        # Frame projection: 2 kHz -> 100 Hz (stride 20).
        self.frame_proj = nn.Conv1d(
            128, hidden, kernel_size=5, stride=20, padding=2
        )
        self.frame_act = nn.ReLU(inplace=True)

        # F0 estimator at 2 kHz (CREPE-inspired: conv + median pooling).
        self.f0_conv = nn.Conv1d(128, 1, kernel_size=5, padding=2)
        self.f0_pool_size = 20  # 2000 Hz -> 100 Hz via median pooling.

        # Energy estimator: RMS over 10 ms windows.
        self.energy_window = self.sample_rate * self.frame_ms // 1000  # 160

        # Voicing classifier at 100 Hz.
        self.voicing_conv = nn.Conv1d(hidden, 1, kernel_size=5, padding=2)

        # Timing estimator (dummy frame alignment).
        self.timing_proj = nn.Conv1d(hidden, 1, kernel_size=5, padding=2)

        self._frames_per_second = 1000 // self.frame_ms

    def _rms_energy(self, audio: torch.Tensor) -> torch.Tensor:
        """Compute per-frame RMS energy."""
        B = audio.shape[0]
        x = audio.reshape(B, -1)
        window = self.energy_window
        pad = (window - (x.shape[-1] % window)) % window
        if pad > 0:
            x = torch.nn.functional.pad(x, (0, pad))
        x2 = x**2
        windows = x2.unfold(-1, window, window)
        energy = windows.mean(dim=-1).sqrt()
        return energy.unsqueeze(1)

    def _median_pool_1d(self, x: torch.Tensor, pool_size: int) -> torch.Tensor:
        """Median pool over time dimension."""
        if x.shape[-1] < pool_size:
            # Pad to at least pool_size.
            pad = pool_size - x.shape[-1]
            x = torch.nn.functional.pad(x, (0, pad))
        x_unfold = x.unfold(-1, pool_size, pool_size)
        return x_unfold.median(dim=-1).values

    def forward(self, audio: torch.Tensor, state: dict | None = None) -> dict:
        """Forward pass returning articulatory parameter dict.

        Args:
            audio: raw waveform (B, T) or (B, 1, T).
            state: optional state dict (unused; stateless encoder).

        Returns:
            dict with keys: frames, f0, voicing, energy, timing.
        """
        if audio.dim() == 3:
            x = audio
        elif audio.dim() == 2:
            x = audio.unsqueeze(1)
        else:
            raise ValueError(f"audio must be (B,T) or (B,1,T), got {audio.shape}")

        B = x.shape[0]
        features_2k = self.feature_extractor(x)  # (B, 128, T_2k)
        frames_100 = self.frame_act(self.frame_proj(features_2k))  # (B, H, T_100)

        # F0 at 2 kHz -> median pool to 100 Hz.
        f0_logits = self.f0_conv(features_2k)  # (B, 1, T_2k)
        f0_pooled = self._median_pool_1d(f0_logits, self.f0_pool_size)  # (B, 1, T_100)
        f0 = torch.nn.functional.softplus(f0_pooled) + 1e-6

        # Voicing at 100 Hz.
        voicing_logits = self.voicing_conv(frames_100)  # (B, 1, T_100)
        voicing = torch.sigmoid(voicing_logits)

        # Energy at 100 Hz.
        energy = self._rms_energy(x)  # (B, 1, T_100)
        if energy.shape[-1] != frames_100.shape[-1]:
            energy = torch.nn.functional.interpolate(
                energy, size=frames_100.shape[-1], mode="linear", align_corners=False
            )

        # Timing (dummy).
        timing = torch.tanh(self.timing_proj(frames_100))

        # Build frame list.
        T = frames_100.shape[-1]
        content_list = [frames_100[:, :, i].unsqueeze(1) for i in range(T)]
        f0_list = [f0[:, :, i].unsqueeze(1) for i in range(T)]
        voicing_list = [voicing[:, :, i].unsqueeze(1) for i in range(T)]
        energy_list = [energy[:, :, i].unsqueeze(1) for i in range(T)]
        timing_list = [timing[:, :, i].unsqueeze(1) for i in range(T)]

        frames = [
            ArticulatoryFrame(c, f, v, e, t)
            for c, f, v, e, t in zip(
                content_list, f0_list, voicing_list, energy_list, timing_list
            )
        ]

        return {
            "frames": ArticulatoryFrameSequence(frames),
            "f0": f0,
            "voicing": voicing,
            "energy": energy,
            "timing": timing,
        }

    def encode(
        self,
        audio: torch.Tensor,
        state: dict | None = None,
    ) -> ArticulatoryFrameSequence:
        out = self.forward(audio, state)
        return out["frames"]


class ArticulatoryAccentMapper(nn.Module):
    """Map articulatory frames toward a target accent embedding.

    Implements a frame-wise linear projection conditioned on target accent,
    preserving speaker F0 contour while shifting formant-related parameters
    encoded in content_features.
    """

    def __init__(
        self,
        config: ArticulatoryStreamingConfig | None = None,
    ) -> None:
        super().__init__()
        self.config = config or ArticulatoryStreamingConfig()
        hidden = self.config.encoder_hidden
        accent_dim = self.config.accent_dim

        self.accent_embedding = nn.Embedding(10, accent_dim)
        self.content_proj = nn.Linear(hidden, hidden)
        self.accent_proj = nn.Linear(accent_dim, hidden)
        self.strength_gate = nn.Linear(1, hidden)

        # Target F0 template per accent (learnable).
        self.target_f0_template = nn.Embedding(10, 1)

    def forward(
        self,
        source_frames: ArticulatoryFrameSequence,
        target_accent: torch.Tensor,
        strength: float = 0.5,
    ) -> ArticulatoryFrameSequence:
        """Map source frames toward target accent.

        Args:
            source_frames: articulatory frames to transform.
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

        modified: list[ArticulatoryFrame] = []
        for frame in source_frames.frames:
            content = frame.content_features  # (B, 1, H)
            f0 = frame.f0
            energy = frame.energy
            voicing = frame.voicing
            timing = frame.timing

            # Linear interpolation toward accent-modulated content.
            accent_mod = self.accent_proj(accent_emb).unsqueeze(1)  # (B, 1, H)
            strength_tensor = torch.full(
                (B, 1, 1), strength, device=device, dtype=content.dtype
            )
            gate = torch.sigmoid(self.strength_gate(strength_tensor))
            mapped_content = (1 - gate) * self.content_proj(content) + gate * accent_mod

            # Shift F0 toward target template, preserving speaker contour shape.
            target_f0 = self.target_f0_template(accent_ids).unsqueeze(1)  # (B,1,1)
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


class DDSPSynthesizer(nn.Module):
    """Differentiable Digital Signal Processing synthesizer.

    Harmonic oscillator bank + filtered noise + simple IR convolution
    for naturalness. Produces audio from articulatory frames.
    """

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

        # Harmonic amplitudes from content features.
        hidden = self.config.encoder_hidden
        self.harmonic_amp = nn.Linear(hidden, self.harmonics)
        self.harmonic_phase = nn.Linear(hidden, self.harmonics)

        # Noise filter.
        self.noise_filter = nn.Linear(hidden, self.noise_bands)

        # Simple learned reverb IR.
        ir_len = sample_rate // 100  # ~10 ms reverb tail.
        self.register_buffer("ir", torch.randn(1, 1, ir_len) * 0.01)

    def _harmonic_oscillators(
        self, f0: torch.Tensor, amps: torch.Tensor, phases: torch.Tensor
    ) -> torch.Tensor:
        """Generate harmonic sum for each frame.

        Args:
            f0: (B, T, 1) fundamental frequency in Hz.
            amps: (B, T, harmonics) amplitude per harmonic.
            phases: (B, T, harmonics) phase offsets.

        Returns:
            waveform (B, T * samples_per_frame).
        """
        B, T, _ = f0.shape
        device = f0.device
        t = torch.arange(self.samples_per_frame, device=device) / self.sample_rate
        t = t.view(1, 1, -1)  # (1, 1, S)

        f0_exp = f0.unsqueeze(-1)  # (B, T, 1, 1)
        harmonics = torch.arange(1, self.harmonics + 1, device=device).view(1, 1, 1, -1)
        phase = phases.unsqueeze(-1)  # (B, T, 1, H)
        amp = amps.unsqueeze(-1)  # (B, T, 1, H)

        # Instantaneous phase accumulator from f0.
        inst_freq = f0_exp * harmonics  # (B, T, 1, H)
        phase_acc = 2 * torch.pi * inst_freq * t + phase
        sine = torch.sin(phase_acc)
        frame_wave = (amp * sine).sum(dim=-1)  # (B, T, S)

        return frame_wave.reshape(B, T * self.samples_per_frame)

    def _filtered_noise(self, content: torch.Tensor, B: int, T: int) -> torch.Tensor:
        """Generate shaped noise from content features."""
        device = content.device
        filters = torch.tanh(self.noise_filter(content))  # (B, T, noise_bands)
        filters = filters.permute(0, 2, 1)  # (B, noise_bands, T)

        total_samples = T * self.samples_per_frame
        noise = torch.randn(B, total_samples, device=device)

        # Simple FIR filtering per band (depthwise-ish via grouped conv).
        # We convolve noise with learned filters to shape it.
        noise = noise.unsqueeze(1)  # (B, 1, total_samples)
        filters_up = torch.nn.functional.interpolate(
            filters.unsqueeze(-1),
            size=total_samples,
            mode="linear",
            align_corners=False,
        ).squeeze(-1)
        shaped = noise * filters_up
        return shaped.sum(dim=1)  # (B, total_samples)

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

        B = frames.frames[0].content_features.shape[0]
        device = frames.frames[0].content_features.device

        content = torch.cat([f.content_features for f in frames.frames], dim=1)
        f0 = torch.cat([f.f0 for f in frames.frames], dim=1)
        voicing = torch.cat([f.voicing for f in frames.frames], dim=1)
        energy = torch.cat([f.energy for f in frames.frames], dim=1)

        amps = torch.sigmoid(self.harmonic_amp(content)) * energy
        phases = self.harmonic_phase(content)

        harmonic = self._harmonic_oscillators(f0, amps, phases)
        noise = self._filtered_noise(content, B, content.shape[1])

        mix = harmonic * voicing.squeeze(-1) + noise * (1 - voicing.squeeze(-1))

        # Simple learned reverb.
        ir = self.ir.repeat(B, 1, 1)
        wet = torch.nn.functional.conv1d(
            mix.unsqueeze(1), ir, padding=ir.shape[-1] - 1
        ).squeeze(1)
        out = mix + 0.2 * wet[:, : mix.shape[-1]]

        return out

    def synthesize(
        self,
        frames: ArticulatoryFrameSequence,
        speaker_conditioning: torch.Tensor,
    ) -> torch.Tensor:
        return self.forward(frames, speaker_conditioning)


class ArticulatoryEncoderWrapper(nn.Module):
    """Wrapper combining encoder + mapper for convenience."""

    def __init__(
        self,
        encoder: ArticulatoryEncoder,
        mapper: ArticulatoryAccentMapper,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.mapper = mapper

    def forward(
        self,
        audio: torch.Tensor,
        target_accent: torch.Tensor,
        strength: float,
    ) -> ArticulatoryFrameSequence:
        frames = self.encoder.encode(audio, None)
        return self.mapper.map(frames, target_accent, strength)

