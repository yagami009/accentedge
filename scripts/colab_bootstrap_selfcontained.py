#!/usr/bin/env python3
"""Fully self-contained Colab Phase 1 bootstrap.

All source code is embedded. Just run:
  colab run scripts/colab_bootstrap_selfcontained.py --gpu T4
"""
import subprocess, sys, os

SRC = "/content/accentedge/src/accentedge"
REPO = "https://github.com/yagami009/accentedge.git"

def run(cmd, desc="", check=True, timeout=120):
    print(f"\n>>> {desc or cmd[:80]}")
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
    out = r.stdout.strip()
    if out:
        print(out[:500])
    if check and r.returncode != 0:
        print(f"FAILED: {r.stderr[:500]}")
        sys.exit(1)
    return r

# 1. GPU
run("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader", "GPU", check=False)

# 2. Deps (skip torch)
run("pip install -q numpy soundfile librosa scipy jiwer pyyaml einops huggingface-hub phonemizer speechbrain pytest", "install deps")

# 3. Clone repos
run("git clone --depth 1 https://github.com/Claussss/FAC-FACodec.git /content/FAC-FACodec", "clone FAC-FACodec")
run(f"git clone {REPO} /content/accentedge", "clone accentedge")

# 4. Create dirs
for d in ["codec", "phase1", "evaluation", "tests"]:
    os.makedirs(f"{SRC}/{d}", exist_ok=True)

# 5. Write source files
for pkg in ["", "codec", "phase1", "evaluation"]:
    with open(f"{SRC}/{pkg}/__init__.py", "w") as f:
        pass

# Write each source file

with open(f"{SRC}/phase1/diffusion.py", "w") as f:
    f.write("""
"""Phase 1 -- Diffusion math: noise schedule, sampling, strength mapping."""
from __future__ import annotations

import torch


def compute_noise_schedule(num_steps: int = 100, noise_min: float = 1e-4, noise_max: float = 2e-2):
    betas = torch.linspace(noise_min, noise_max, num_steps)
    alphas = 1.0 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)
    return {
        "betas": betas, "alphas": alphas, "alpha_bar": alpha_bar,
        "sqrt_alpha_bar": torch.sqrt(alpha_bar),
        "sqrt_1m_alpha_bar": torch.sqrt(1.0 - alpha_bar),
    }


def q_sample(x0, t, sqrt_alpha_bar, sqrt_1m_alpha_bar):
    noise = torch.randn_like(x0)
    xt = sqrt_alpha_bar[t].view(-1, 1, 1) * x0 + sqrt_1m_alpha_bar[t].view(-1, 1, 1) * noise
    return xt, noise


def ddim_step(x_t, eps_pred, t, t_prev, sqrt_alpha_bar, sqrt_1m_alpha_bar, eta: float = 0.0):
    bsz = x_t.size(0)
    sqrt_a_t = sqrt_alpha_bar[t].view(-1, 1, 1)
    sqrt_1ma_t = sqrt_1m_alpha_bar[t].view(-1, 1, 1)
    x0_pred = (x_t - sqrt_1ma_t * eps_pred) / sqrt_a_t
    x0_pred = torch.clamp(x0_pred, -10, 10)
    if t_prev is None:
        return x0_pred
    t_prev_int = int(t_prev) if not isinstance(t_prev, int) else t_prev
    if t_prev_int < 0:
        return x0_pred
    sqrt_a_prev = sqrt_alpha_bar[t_prev].view(-1, 1, 1)
    sqrt_1ma_prev = sqrt_1m_alpha_bar[t_prev].view(-1, 1, 1)
    noise = torch.randn_like(x_t)
    x_t = sqrt_a_prev * x0_pred + sqrt_1ma_prev * eps_pred + eta * sqrt_1ma_prev * noise
    return x_t


def strength_to_t_start(strength: float, num_steps: int = 100) -> int:
    strength = max(0.0, min(1.0, strength))
    return int(round(strength * num_steps))


def t_start_to_strength(t_start: int, num_steps: int = 100) -> float:
    return t_start / max(1, num_steps)

""")
print(f"wrote diffusion.py")

with open(f"{SRC}/phase1/denoiser.py", "w") as f:
    f.write("""
"""Phase 1 -- FAC-FACodec denoiser (paper-faithful reimplementation).

Paper: arxiv:2510.10785
Reference impl: Claussss/FAC-FACodec (MIT License)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, List


class StrengthScheduler:
    """Maps strength [0,1] to diffusion timestep."""

    def __init__(self, num_steps: int = 100, schedule: str = "linear"):
        self.num_steps = num_steps
        self.schedule = schedule

    def __call__(self, strength: float) -> int:
        strength = max(0.0, min(1.0, strength))
        return int(round(strength * self.num_steps))

    def validate(self, strength: float) -> bool:
        return 0.0 <= strength <= 1.0

    def available_strengths(self) -> List[float]:
        return [0.0, 0.25, 0.50, 0.75, 1.0]


@dataclass
class DenoisingResult:
    original: torch.Tensor
    denoised: torch.Tensor
    strength: float
    timesteps_used: int
    converged: bool


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embedding for diffusion timesteps."""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


class CondLayerNorm(nn.Module):
    """Conditional LayerNorm with FiLM-style affine modulation.

    cond is [B, T, 2D], split into gamma and beta.
    Output: x_norm * (1 + gamma) + beta
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, elementwise_affine=False)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gamma, beta = cond.chunk(2, dim=-1)
        x_norm = self.norm(x)
        return x_norm * (1 + gamma) + beta


class ConvFeedForward(nn.Module):
    """1D convolution FFN block (matches paper implementation)."""
    def __init__(self, d_model: int = 1024, d_ff: int = 2048, kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(d_model, d_ff, kernel_size, padding=(kernel_size // 2))
        self.conv2 = nn.Conv1d(d_ff, d_model, kernel_size, padding=(kernel_size // 2))
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        return self.dropout(out)


class CustomTransformerEncoderLayer(nn.Module):
    """Transformer encoder layer with ConvFFN and conditional LayerNorm (FiLM)."""
    def __init__(self, d_model: int = 1024, nhead: int = 8, d_ff: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = CondLayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.conv_ff = ConvFeedForward(d_model=d_model, d_ff=d_ff, kernel_size=3, dropout=dropout)

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        src_key_padding_mask: Optional[torch.BoolTensor] = None,
    ) -> torch.Tensor:
        # Self-attention block
        attn_out, _ = self.self_attn(x, x, x, key_padding_mask=src_key_padding_mask)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)

        # Conv feed-forward block
        x_t = x.transpose(1, 2)  # [B, D, T]
        ff_out = self.conv_ff(x_t)
        ff_out = ff_out.transpose(1, 2)  # [B, T, D]
        x = x + self.dropout(ff_out)

        # Conditional LayerNorm with FiLM parameters
        return self.norm2(x, cond)


class CustomTransformerEncoder(nn.Module):
    """Stacks multiple Conditional Transformer encoder layers."""
    def __init__(self, num_layers: int = 6, d_model: int = 1024, nhead: int = 8, d_ff: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            CustomTransformerEncoderLayer(d_model, nhead, d_ff, dropout)
            for _ in range(num_layers)
        ])

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        src_key_padding_mask: Optional[torch.BoolTensor] = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, cond, src_key_padding_mask)
        return x


class DenoisingTransformerModel(nn.Module):
    """FAC-FACodec denoiser: 6-layer Transformer, 8 heads, dim=1024, FFN=2048.

    Predicts epsilon noise and zc2 from denoised x0 estimate.
    Conditioned on phoneme embeddings + timestep via FiLM.
    """
    def __init__(
        self,
        d_model: int = 1024,
        nhead: int = 8,
        num_layers: int = 6,
        d_ff: int = 2048,
        dropout: float = 0.1,
        phone_vocab_size: int = 393,
        phone_pad_id: int = 392,
        num_steps: int = 100,
        facodec_dim: int = 8,
    ):
        super().__init__()
        self.num_steps = num_steps
        self.d_model = d_model
        self.facodec_dim = facodec_dim

        # Timestep embedding (sinusoidal)
        self.t_embed = nn.Sequential(
            SinusoidalPosEmb(d_model),
            nn.Linear(d_model, d_model * 2),
            nn.SiLU(),
            nn.Linear(d_model * 2, d_model * 2),
        )

        # Phoneme embedding
        self.phone_emb = nn.Embedding(phone_vocab_size, d_model, padding_idx=phone_pad_id)

        # Input projection: [B, C, T] -> [B, T, D]
        self.input_proj = nn.Linear(facodec_dim, d_model)

        # Custom encoder with conditional LayerNorm
        self.encoder = CustomTransformerEncoder(
            num_layers=num_layers, d_model=d_model, nhead=nhead,
            d_ff=d_ff, dropout=dropout,
        )

        # Epsilon prediction head
        self.fc_out = nn.Linear(d_model, facodec_dim)

        # Zc2 prediction head
        self.fc_zc2 = nn.Sequential(
            nn.Linear(d_model + facodec_dim, 4 * facodec_dim),
            nn.GELU(),
            nn.Linear(4 * facodec_dim, facodec_dim),
        )

        # FiLM conditioning
        self.phone_proj = nn.Linear(d_model, d_model * 2)
        self.dropout = nn.Dropout(dropout)

        # DDPM schedule buffers
        betas = torch.linspace(1e-4, 0.02, num_steps, dtype=torch.float)
        alphas = 1.0 - betas
        abar = torch.cumprod(alphas, dim=0)
        self.register_buffer("sqrt_abar", torch.sqrt(abar))
        self.register_buffer("sqrt_1mabar", torch.sqrt(1.0 - abar))

    def forward(
        self,
        zc1_noisy: torch.Tensor,        # [B, C, T]
        phone_ids: torch.Tensor,         # [B, T]
        t: torch.Tensor,                 # [B]
        padding_mask: Optional[torch.BoolTensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bsz, feat_dim, seq_len = zc1_noisy.shape

        # Input projection: [B, C, T] -> [B, T, D]
        h = self.input_proj(zc1_noisy.transpose(1, 2))

        # Phoneme conditioning
        phone_emb = self.phone_emb(phone_ids)                        # [B, T, D]
        phone_cond = self.phone_proj(phone_emb)                      # [B, T, 2D]

        # Timestep conditioning
        t_emb = self.t_embed(t)                                      # [B, 2D]
        t_cond = t_emb.unsqueeze(1).expand(-1, seq_len, -1)         # [B, T, 2D]
        cond = self.dropout(phone_cond) + t_cond                     # [B, T, 2D]

        # Encode
        h = self.encoder(h, cond, src_key_padding_mask=padding_mask)

        # Epsilon prediction: [B, T, D] -> [B, C, T]
        eps_pred = self.fc_out(h).transpose(1, 2)

        # Zc2 prediction from denoised x0 estimate
        sa = self.sqrt_abar[t].view(bsz, 1, 1)
        s1a = self.sqrt_1mabar[t].view(bsz, 1, 1)
        x0_hat = (zc1_noisy - s1a * eps_pred.detach()) / sa

        zc2_input = torch.cat([h, x0_hat.detach().transpose(1, 2)], dim=-1)
        zc2_pred = self.fc_zc2(zc2_input).transpose(1, 2)

        # Return shapes [B, C, T]
        return eps_pred, zc2_pred


class Phase1AccentNormalizer(nn.Module):
    """Minimal Phase-1 module: denoiser only.

    Only the denoiser weights are trainable. Everything else is frozen.
    """
    def __init__(self, denoiser: nn.Module, num_steps: int = 100,
                 facodec_dim: int = 8, device: str = "cpu"):
        super().__init__()
        self.denoiser = denoiser
        self.num_steps = num_steps
        self.facodec_dim = facodec_dim
        self.device = torch.device(device)
        self.denoiser.to(self.device)

    def forward(self, zc1: torch.Tensor, phone_ids: torch.Tensor, t: torch.Tensor,
                padding_mask: Optional[torch.BoolTensor] = None):
        return self.denoiser(zc1, phone_ids, t, padding_mask)

""")
print(f"wrote denoiser.py")

with open(f"{SRC}/phase1/strength.py", "w") as f:
    f.write("""
"""Phase 1 — Conversion strength control.

Maps user-facing strength ∈ [0, 1] to paper-native t_start timestep.

strength=0.0 → t_start=0 (no accent normalization)
strength=1.0 → t_start=num_steps (maximum normalization)
"""
from __future__ import annotations

import torch


def strength_to_t_start(strength: float, num_steps: int = 100) -> int:
    """Map conversion strength ∈ [0, 1] to diffusion timestep."""
    strength = max(0.0, min(1.0, strength))
    return int(round(strength * num_steps))


def t_start_to_strength(t_start: int, num_steps: int = 100) -> float:
    """Inverse mapping."""
    return t_start / max(1, num_steps)


class StrengthScheduler:
    """Maps strength to diffusion timestep with configurable schedule."""

    def __init__(self, num_steps: int = 100, schedule: str = "linear"):
        self.num_steps = num_steps
        self.schedule = schedule

    def __call__(self, strength: float) -> int:
        return strength_to_t_start(strength, self.num_steps)

    def validate(self, strength: float) -> bool:
        return 0.0 <= strength <= 1.0

    def available_strengths(self) -> list:
        return [0.0, 0.25, 0.50, 0.75, 1.0]

""")
print(f"wrote strength.py")

with open(f"{SRC}/phase1/interfaces.py", "w") as f:
    f.write("""
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

""")
print(f"wrote interfaces.py")

with open("/content/pyproject.toml", "w") as f:
    f.write("""
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.backends.legacy:build"

[project]
name = "accentedge"
version = "0.1.0-phase1"
description = "Phase 1: FAC-FACodec reimplementation — offline pronunciation normalization baseline"
requires-python = ">=3.11"
dependencies = [
    "torch>=2.2",
    "numpy>=1.24",
    "soundfile>=0.12",
    "librosa>=0.10",
    "scipy>=1.11",
    "jiwer>=3.0",
    "pyyaml>=6.0",
    "einops>=0.7",
    "huggingface-hub>=0.20",
    "phonemizer>=3.0",
    "speechbrain>=0.5.16",
    "faster-whisper>=1.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
]

[tool.pytest.ini_options]
testpaths = ["tests"]

""")

with open("/content/setup.py", "w") as f:
    f.write("from setuptools import setup\nsetup(name='accentedge', packages=['accentedge', 'accentedge.phase1', 'accentedge.codec', 'accentedge.evaluation'])\n")

# Write tests
with open("/content/accentedge/tests/test_phase1.py", "w") as f:
    f.write("""
"""Phase 1 tests."""
import torch
import pytest


class TestDiffusion:
    def test_noise_schedule(self):
        from accentedge.phase1.diffusion import compute_noise_schedule
        s = compute_noise_schedule(100)
        assert s["betas"].shape == (100,)
        assert s["alpha_bar"].shape == (100,)

    def test_q_sample(self):
        from accentedge.phase1.diffusion import q_sample, compute_noise_schedule
        s = compute_noise_schedule(100)
        x0 = torch.randn(2, 8, 20)
        t = torch.tensor([10, 50])
        xt, noise = q_sample(x0, t, s["sqrt_alpha_bar"], s["sqrt_1m_alpha_bar"])
        assert xt.shape == x0.shape

    def test_ddim_step(self):
        from accentedge.phase1.diffusion import ddim_step, compute_noise_schedule
        s = compute_noise_schedule(100)
        x = torch.randn(2, 8, 20)
        eps = torch.randn(2, 8, 20)
        out = ddim_step(x, eps, 10, 5, s["sqrt_alpha_bar"], s["sqrt_1m_alpha_bar"])
        assert out.shape == x.shape


class TestStrength:
    def test_mapping(self):
        from accentedge.phase1.strength import strength_to_t_start, t_start_to_strength
        assert strength_to_t_start(0.0) == 0
        assert strength_to_t_start(1.0) == 100
        assert strength_to_t_start(0.5) == 50
        assert t_start_to_strength(50) == 0.5

    def test_scheduler(self):
        from accentedge.phase1.strength import StrengthScheduler
        s = StrengthScheduler(num_steps=100)
        assert s(0.0) == 0
        assert s(1.0) == 100
        assert s(0.5) == 50


class TestDenoiser:
    def test_sinusoidal_pos_emb(self):
        from accentedge.phase1.denoiser import SinusoidalPosEmb
        pe = SinusoidalPosEmb(64)
        t = torch.tensor([0, 50, 99], dtype=torch.float32)
        out = pe(t)
        assert out.shape == (3, 64)

    def test_cond_layer_norm(self):
        from accentedge.phase1.denoiser import CondLayerNorm
        ln = CondLayerNorm(64)
        x = torch.randn(2, 10, 64)
        cond = torch.randn(2, 10, 128)
        out = ln(x, cond)
        assert out.shape == (2, 10, 64)

    def test_conv_ff(self):
        from accentedge.phase1.denoiser import ConvFeedForward
        ff = ConvFeedForward(d_model=64, d_ff=128)
        x = torch.randn(2, 64, 10)
        out = ff(x)
        assert out.shape == (2, 64, 10)

    def test_transformer_layer(self):
        from accentedge.phase1.denoiser import CustomTransformerEncoderLayer
        layer = CustomTransformerEncoderLayer(d_model=64, nhead=4, d_ff=128)
        x = torch.randn(2, 10, 64)
        cond = torch.randn(2, 10, 128)
        out = layer(x, cond)
        assert out.shape == (2, 10, 64)

    def test_transformer_encoder(self):
        from accentedge.phase1.denoiser import CustomTransformerEncoder
        enc = CustomTransformerEncoder(num_layers=2, d_model=64, nhead=4, d_ff=128)
        x = torch.randn(2, 10, 64)
        cond = torch.randn(2, 10, 128)
        out = enc(x, cond)
        assert out.shape == (2, 10, 64)

    def test_denoising_model_forward(self):
        from accentedge.phase1.denoiser import DenoisingTransformerModel
        model = DenoisingTransformerModel(
            d_model=64, nhead=4, num_layers=2, d_ff=128,
            phone_vocab_size=393, facodec_dim=8
        )
        zc1 = torch.randn(2, 8, 20)
        phone_ids = torch.randint(0, 392, (2, 20))
        t = torch.randint(0, 100, (2,))
        eps, zc2 = model(zc1, phone_ids, t)
        assert eps.shape == (2, 8, 20)
        assert zc2.shape == (2, 8, 20)


class TestCodecInterface:
    def test_interfaces(self):
        from accentedge.codec.interfaces import FactorizedLatents
        latents = FactorizedLatents(
            content=torch.randn(2, 8, 20),
            content_zc1=torch.randint(0, 1024, (2, 1, 20)),
            content_zc2=torch.randn(2, 8, 20),
            prosody=torch.randint(0, 1024, (2, 1, 20)),
            timbre=torch.randn(2, 256),
        )
        assert latents.content.shape == (2, 8, 20)
        assert latents.content_zc1.shape == (2, 1, 20)
        assert latents.content_zc2.shape == (2, 8, 20)
        assert latents.prosody.shape == (2, 1, 20)
        assert latents.timbre.shape == (2, 256)

""")
print("wrote tests")

# Install package
os.chdir("/content/accentedge")
r = subprocess.run([sys.executable, "-m", "pip", "install", "-e", "."], capture_output=True, text=True)
if r.returncode != 0:
    print("pip install failed:", r.stderr[:200])
else:
    print("Package installed")

# Run tests
print("\n=== Running Phase 1 tests ===")
r = subprocess.run(
    [sys.executable, "-m", "pytest", "tests/test_phase1.py", "-v", "--tb=short"],
    capture_output=True, text=True, timeout=120
)
print(r.stdout[-3000:])
if r.stderr:
    print("STDERR:", r.stderr[-2000:])

# Check FACodec
print("\n=== Checking FAC-FACodec ===")
sys.path.insert(0, "/content/FAC-FACodec")
r = subprocess.run([sys.executable, "-c", "from FACodec_AC import models; print('FACodec_AC models loaded')"], capture_output=True, text=True)
print(r.stdout.strip() or r.stderr.strip()[:200])
