#!/usr/bin/env python3
"""Phase 1 — Core implementation of AccentEdge FAC-FACodec reimplementation.

This module implements:
  - FACodec adapter (from Plachta/FAcodec)
  - Official FAC-FACodec denoiser (6-layer Transformer, FiLM conditioning)
  - Linear diffusion noise schedule (T=100)
  - DDIM ODE inference
  - Global strength → t_start mapping
  - Phase1AccentNormalizer (offline utterance conversion)
"""
from __future__ import annotations

import os
import sys
import time
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
import numpy as np
import soundfile as sf
import librosa

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Phase1Config:
    # Device
    device: str = "mps"  # cpu, mps, cuda

    # FACodec
    facodec_ckpt: str = "Plachta/FAcodec"
    facodec_config: str = "Plachta/FAcodec"
    freeze_codec: bool = True

    # Diffusion
    num_steps: int = 100
    noise_min: float = 1e-4
    noise_max: float = 2e-2
    # Linear schedule: beta_t ∈ [noise_min, noise_max]

    # Denoiser (from paper: 6-layer Transformer, 8 heads, dim 1024, FFN 2048)
    d_model: int = 1024
    nhead: int = 8
    num_layers: int = 6
    d_ff: int = 2048
    dropout: float = 0.1
    phone_vocab_size: int = 392
    phone_pad_id: int = 392
    phone_emb_dim: int = 1024

    # Training
    batch_size: int = 4
    lr: float = 3e-4
    epochs: int = 100
    warmup_steps: int = 1000

    # FACodec representation
    facodec_dim: int = 8
    max_seq_len: int = 500  # ~10s at 50fps

    # Strength sweep
    strength_values: list = field(default_factory=lambda: [0.0, 0.25, 0.50, 0.75, 1.0])


# ──────────────────────────────────────────────────────────────────────────────
# Diffusion noise schedule (linear, T=100, β ∈ [1e-4, 2e-2])
# ──────────────────────────────────────────────────────────────────────────────

def compute_noise_schedule(num_steps: int, beta_min: float, beta_max: float, device: torch.device) -> dict:
    """Compute linear noise schedule and return precomputed tensors."""
    betas = torch.linspace(beta_min, beta_max, num_steps, device=device)
    alphas = 1.0 - betas
    abar = torch.cumprod(alphas, dim=0)
    sqrt_abar = torch.sqrt(abar)
    sqrt_1m_abar = torch.sqrt(1.0 - abar)

    return {
        "betas": betas,
        "alphas": alphas,
        "abar": abar,
        "sqrt_abar": sqrt_abar,
        "sqrt_1m_abar": sqrt_1m_abar,
        "num_steps": num_steps,
    }


def q_sample(x0: torch.Tensor, t: torch.Tensor, sqrt_abar: torch.Tensor, sqrt_1m_abar: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Forward diffusion: x_t = sqrt(ā_t) * x0 + sqrt(1-ā_t) * ε."""
    noise = torch.randn_like(x0)
    xt = sqrt_abar[t].view(-1, 1, 1) * x0 + sqrt_1m_abar[t].view(-1, 1, 1) * noise
    return xt, noise


def strength_to_t_start(strength: float, num_steps: int) -> int:
    """Map conversion strength ∈ [0, 1] to diffusion timestep t_start ∈ [0, num_steps].

    strength=0.0 → t_start=0 (no denoising, reconstruction path)
    strength=1.0 → t_start=num_steps (full native denoising)
    """
    strength = max(0.0, min(1.0, strength))
    return int(round(strength * num_steps))


def t_start_to_strength(t_start: int, num_steps: int) -> float:
    """Inverse of strength_to_t_start."""
    return t_start / max(1, num_steps)
