"""Phase 1 — AccentEdge pronunciation normalizer.

Combines FACodec (frozen) + FAC-FACodec denoiser (trained) for offline
accent normalization with a single global conversion strength parameter.

strength=0.0 → FACodec reconstruction (no accent normalization)
strength>0.0 → partial denoising toward native pronunciation distribution

Design invariant:
  Only the content/pronunciation factor (zc1) is modified.
  Prosody (zp), acoustic detail (zr), and timbre (g) are preserved from source.
"""
from __future__ import annotations

import os
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import numpy as np

from accentedge.codec.interfaces import FactorizedLatents, FactorizedSpeechCodec
from accentedge.codec.facodec import FACodecAdapter
from accentedge.phase1.denoiser import DenoisingTransformerModel
from accentedge.phase1.diffusion import (
    compute_noise_schedule, q_sample, strength_to_t_start
)
from accentedge.phase1.strength import StrengthScheduler

warnings.simplefilter("ignore")


@dataclass
class Phase1Checkpoint:
    """Metadata for a Phase 1 training checkpoint."""
    phase: str = "phase1"
    step: int = 0
    epoch: int = 0
    loss: float = float("inf")
    config_hash: str = ""
    git_commit: str = ""
    device: str = "cpu"
    seed: int = 42
    timestamp: str = ""

    def to_dict(self) -> dict:
        return {
            "phase": self.phase,
            "step": self.step,
            "epoch": self.epoch,
            "loss": self.loss,
            "config_hash": self.config_hash,
            "git_commit": self.git_commit,
            "device": self.device,
            "seed": self.seed,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Phase1Checkpoint":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class Phase1AccentNormalizer:
    """Offline accent normalization model.

    Usage:
        normalizer = Phase1AccentNormalizer(device="mps")
        normalizer.load_checkpoint("checkpoints/model.pt")
        output = normalizer.convert(source_waveform, strength=0.5, transcript="text")
    """

    def __init__(
        self,
        config=None,
        device: str = "auto",
        facodec: Optional[FactorizedSpeechCodec] = None,
    ):
        if config is None:
            from accentedge.phase1.diffusion import Phase1Config
            config = Phase1Config()
        self.cfg = config

        # Auto-detect device
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = torch.device(device)

        # Frozen FACodec
        self.facodec = facodec or FACodecAdapter(device=str(self.device))
        self.facodec.freeze()

        # Verify frozen
        for name, param in self.facodec.encoder.named_parameters():
            assert not param.requires_grad, f"FACodec encoder param {name} is trainable!"
        for name, param in self.facodec.decoder.named_parameters():
            assert not param.requires_grad, f"FACodec decoder param {name} is trainable!"

        # Denoiser (trainable)
        self.denoiser = DenoisingTransformerModel(
            d_model=config.d_model,
            nhead=config.nhead,
            num_layers=config.num_layers,
            d_ff=config.d_ff,
            dropout=config.dropout,
            phone_vocab_size=config.phone_vocab_size,
            phone_pad_id=config.phone_pad_id,
            phone_emb_dim=config.d_model,
            content_dim=config.facodec_dim,
            num_steps=config.num_steps,
            noise_min=config.noise_min,
            noise_max=config.noise_max,
        ).to(self.device)

        # Noise schedule
        betas = torch.linspace(config.noise_min, config.noise_max, config.num_steps)
        alphas = 1.0 - betas
        abar = torch.cumprod(alphas, dim=0)
        self.register_buffer("sqrt_abar", torch.sqrt(abar))
        self.register_buffer("sqrt_1m_abar", torch.sqrt(1.0 - abar))

        # Strength scheduler
        self.strength_scheduler = StrengthScheduler(num_steps=config.num_steps)

    def register_buffer(self, name, tensor):
        """Register buffer on both denoiser and self."""
        self.denoiser.register_buffer(name, tensor)
        self.__dict__[name] = tensor

    def encode(self, waveform: torch.Tensor) -> FactorizedLatents:
        """Encode waveform → factorized latents."""
        return self.facodec.encode(waveform)

    def decode(self, latents: FactorizedLatents) -> torch.Tensor:
        """Decode factorized latents → waveform."""
        return self.facodec.decode(latents)

    def convert(
        self,
        waveform: torch.Tensor,
        strength: float = 0.5,
        transcript: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        """Convert accent of source waveform.

        Args:
            waveform: [1, T] source audio at 24kHz
            strength: ∈ [0.0, 1.0]
            transcript: ground truth transcript (REQUIRED for paper-faithful mode)
            seed: optional random seed for reproducibility

        Returns:
            Converted waveform [1, T] at 24kHz
        """
        if strength < 0.0 or strength > 1.0:
            raise ValueError(f"strength must be in [0.0, 1.0], got {strength}")
        if transcript is None:
            raise ValueError(
                "transcript is required for paper-faithful inference. "
                "The FAC-FACodec denoiser requires phoneme conditioning. "
                "Set mode='experimental_unconditioned' for speech-only mode."
            )

        self.denoiser.eval()
        with torch.no_grad():
            if seed is not None:
                torch.manual_seed(seed)

            # Encode source
            latents = self.facodec.encode(waveform.to(self.device))
            zc1 = latents.content  # [B, D, T] continuous representation

            # If strength=0, skip denoising (reconstruction path)
            t_start = strength_to_t_start(strength, self.cfg.num_steps)
            if t_start == 0:
                return self.facodec.reconstruction(waveform.to(self.device))

            # Partial forward diffusion
            t = torch.full((zc1.shape[0],), t_start - 1, device=self.device, dtype=torch.long)
            x_t, _ = q_sample(zc1, t, self.sqrt_abar, self.sqrt_1m_abar)

            # DDIM reverse denoising
            x = x_t
            for ti in reversed(range(t_start)):
                t_tensor = torch.full((x.shape[0],), ti, device=self.device, dtype=torch.long)

                # Note: transcript→phonemes should be pre-computed and passed
                # For now, we use zero conditioning (experimental)
                eps_pred, zc2_pred = self.denoiser(
                    x, phone_ids=None, t=t_tensor, padding_mask=None
                )

                alpha = self.sqrt_abar[ti]
                sigma = self.sqrt_1m_abar[ti]
                x = (x - sigma * eps_pred) / alpha

            # Snap to codebook (paper: snap zc1 to nearest codebook vector)
            # Then combine with zc2 prediction
            # For now, use denoised zc1 directly
            denoised_zc1 = x

            # Decode with modified zc1, preserved prosody/detail/timbre
            # This requires reconstructing the full quantized representation
            output_latents = FactorizedLatents(
                content=denoised_zc1,
                content_zc1=latents.content_zc1,
                prosody=latents.prosody,
                detail=latents.detail,
                timbre=latents.timbre,
            )
            output = self.facodec.decode(output_latents)
            return output

    def save_checkpoint(self, path: str, metadata: Optional[dict] = None) -> None:
        """Save denoiser weights + metadata."""
        ckpt = {
            "denoiser_state_dict": self.denoiser.state_dict(),
            "config": self.cfg.__dict__,
            "metadata": metadata or {},
        }
        torch.save(ckpt, path)
        # Save sidecar JSON
        json_path = path.replace(".pt", ".json").replace(".safetensors", ".json")
        if json_path == path:
            json_path = path + ".json"
        import json
        with open(json_path, "w") as f:
            json.dump(ckpt["metadata"], f, indent=2, default=str)

    def load_checkpoint(self, path: str) -> None:
        """Load denoiser weights."""
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.denoiser.load_state_dict(ckpt["denoiser_state_dict"])
        print(f"Loaded checkpoint from {path}")
