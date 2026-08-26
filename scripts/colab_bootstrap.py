#!/usr/bin/env python3
"""Minimal Colab Phase 1 bootstrap.

Creates a fresh Colab T4 VM, installs deps, and writes
the Phase 1 source files inline (no git repo needed).
"""
import subprocess, sys, os

SRC = "/content/accentedge/src/accentedge"

def run(cmd, desc="", check=True):
    print(f"\n>>> {desc or cmd[:80]}")
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    out = r.stdout.strip()
    if out:
        print(out[:500])
    if check and r.returncode != 0:
        print(f"FAILED: {r.stderr[:500]}")
        sys.exit(1)
    return r

# Check GPU
run("nvidia-smi --query-gpu=name --format=csv,noheader", "GPU check")

# Install deps
run("pip install -q numpy soundfile librosa scipy jiwer pyyaml einops huggingface-hub phonemizer speechbrain faster-whisper pytest", "install deps")
run("pip install -q git+https://github.com/open-mmlab/Amphion.git", "install Amphion", check=False)

# Clone FAC-FACodec
run("git clone --depth 1 https://github.com/Claussss/FAC-FACodec.git /content/FAC-FACodec", "clone FAC-FACodec")

# Create directory structure
for d in ["codec", "phase1", "evaluation", "tests"]:
    os.makedirs(f"{SRC}/{d}", exist_ok=True)
os.makedirs("/content/accentedge/configs/phase1", exist_ok=True)

# ── Write diffusion.py ──
run(f"""cat << 'PYEOF' > {SRC}/phase1/diffusion.py
import torch

def compute_noise_schedule(num_steps=100, noise_min=1e-4, noise_max=2e-2):
    betas = torch.linspace(noise_min, noise_max, num_steps)
    alphas = 1.0 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)
    return {{
        "betas": betas, "alphas": alphas, "alpha_bar": alpha_bar,
        "sqrt_alpha_bar": torch.sqrt(alpha_bar),
        "sqrt_1m_alpha_bar": torch.sqrt(1.0 - alpha_bar),
    }}

def q_sample(x0, t, sqrt_alpha_bar, sqrt_1m_alpha_bar):
    noise = torch.randn_like(x0)
    xt = sqrt_alpha_bar[t].view(-1, 1, 1) * x0 + sqrt_1m_alpha_bar[t].view(-1, 1, 1) * noise
    return xt, noise

def ddim_step(x_t, eps_pred, t, t_prev, sqrt_alpha_bar, sqrt_1m_alpha_bar, eta=0.0):
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

def strength_to_t_start(strength, num_steps=100):
    return int(round(max(0.0, min(1.0, strength)) * num_steps))

def t_start_to_strength(t_start, num_steps=100):
    return t_start / max(1, num_steps)
PYEOF""", "write diffusion.py")

# ── Write denoiser.py ──
run(f"""cat << 'PYEOF' > {SRC}/phase1/denoiser.py
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, List
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class StrengthScheduler:
    def __init__(self, num_steps=100):
        self.num_steps = num_steps
    def __call__(self, strength):
        return int(round(max(0.0, min(1.0, strength)) * self.num_steps))
    def validate(self, strength):
        return 0.0 <= strength <= 1.0
    def available_strengths(self):
        return [0.0, 0.25, 0.50, 0.75, 1.0]

@dataclass
class DenoisingResult:
    original: torch.Tensor
    denoised: torch.Tensor
    strength: float
    timesteps_used: int
    converged: bool

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, t):
        half_dim = self.dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

class CondLayerNorm(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, elementwise_affine=False)
    def forward(self, x, cond):
        gamma, beta = cond.chunk(2, dim=-1)
        return self.norm(x) * (1 + gamma) + beta

class ConvFeedForward(nn.Module):
    def __init__(self, d_model=1024, d_ff=2048, kernel_size=3, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(d_model, d_ff, kernel_size, padding=kernel_size//2)
        self.conv2 = nn.Conv1d(d_ff, d_model, kernel_size, padding=kernel_size//2)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
    def forward(self, x):
        return self.dropout(self.conv2(self.dropout(self.relu(self.conv1(x)))))

class CustomTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model=1024, nhead=8, d_ff=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = CondLayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.conv_ff = ConvFeedForward(d_model=d_model, d_ff=d_ff, kernel_size=3, dropout=dropout)
    def forward(self, x, cond, src_key_padding_mask=None):
        attn_out, _ = self.self_attn(x, x, x, key_padding_mask=src_key_padding_mask)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)
        x_t = x.transpose(1, 2)
        ff_out = self.conv_ff(x_t).transpose(1, 2)
        x = x + self.dropout(ff_out)
        return self.norm2(x, cond)

class CustomTransformerEncoder(nn.Module):
    def __init__(self, num_layers=12, d_model=1024, nhead=8, d_ff=2048, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            CustomTransformerEncoderLayer(d_model, nhead, d_ff, dropout)
            for _ in range(num_layers)
        ])
    def forward(self, x, cond, src_key_padding_mask=None):
        for layer in self.layers:
            x = layer(x, cond, src_key_padding_mask)
        return x

class DenoisingTransformerModel(nn.Module):
    def __init__(self, d_model=1024, nhead=8, num_layers=6, d_ff=2048,
                 dropout=0.1, phone_vocab_size=393, num_steps=100, facodec_dim=8):
        super().__init__()
        self.num_steps = num_steps
        self.d_model = d_model
        self.t_embed = nn.Embedding(num_steps, d_model * 2)
        self.phone_emb = nn.Embedding(phone_vocab_size, d_model, padding_idx=phone_vocab_size - 1)
        self.input_proj = nn.Linear(facodec_dim, d_model)
        self.encoder = CustomTransformerEncoder(num_layers, d_model, nhead, d_ff, dropout)
        self.phone_proj = nn.Linear(d_model, d_model * 2)
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(d_model, facodec_dim)
        self.fc_zc2 = nn.Sequential(
            nn.Linear(d_model + facodec_dim, 4 * facodec_dim),
            nn.GELU(),
            nn.Linear(4 * facodec_dim, facodec_dim),
        )
        betas = torch.linspace(1e-4, 0.02, num_steps)
        abar = torch.cumprod(1.0 - betas, dim=0)
        self.register_buffer("sqrt_abar", torch.sqrt(abar))
        self.register_buffer("sqrt_1mabar", torch.sqrt(1.0 - abar))

    def forward(self, zc1_noisy, phone_ids, t, padding_mask=None):
        bsz, C, T = zc1_noisy.shape
        z = zc1_noisy.transpose(1, 2)  # [B,T,C]
        h = self.input_proj(z)
        phone_emb = self.phone_emb(phone_ids)
        seq_len = h.size(1)
        phone_cond = self.phone_proj(phone_emb)
        t_cond = self.t_embed(t).unsqueeze(1).expand(-1, seq_len, -1)
        cond = self.dropout(phone_cond) + t_cond
        h = self.encoder(h, cond, src_key_padding_mask=padding_mask)
        eps_pred = self.fc_out(h).transpose(1, 2)
        sa = self.sqrt_abar[t].view(bsz, 1, 1)
        s1a = self.sqrt_1mabar[t].view(bsz, 1, 1)
        x0_hat = (zc1_noisy - s1a * eps_pred.detach()) / sa
        zc2_input = torch.cat([h, x0_hat.detach().transpose(1, 2)], dim=-1)
        zc2_pred = self.fc_zc2(zc2_input).transpose(1, 2)
        return eps_pred, zc2_pred

class Phase1AccentNormalizer(nn.Module):
    def __init__(self, denoiser, num_steps=100, facodec_dim=8, device="cpu"):
        super().__init__()
        self.denoiser = denoiser
        self.num_steps = num_steps
        self.facodec_dim = facodec_dim
        self.device = torch.device(device)
        self.denoiser.to(self.device)
    def forward(self, zc1, phone_ids, t, padding_mask=None):
        return self.denoiser(zc1, phone_ids, t, padding_mask)
PYEOF""", "write denoiser.py")

# ── Write strength.py ──
run(f"""cat << 'PYEOF' > {SRC}/phase1/strength.py
def strength_to_t_start(strength, num_steps=100):
    return int(round(max(0.0, min(1.0, strength)) * num_steps))

def t_start_to_strength(t_start, num_steps=100):
    return t_start / max(1, num_steps)

class StrengthScheduler:
    def __init__(self, num_steps=100):
        self.num_steps = num_steps
    def __call__(self, strength):
        return strength_to_t_start(strength, self.num_steps)
    def validate(self, strength):
        return 0.0 <= strength <= 1.0
    def available_strengths(self):
        return [0.0, 0.25, 0.50, 0.75, 1.0]
PYEOF""", "write strength.py")

# ── Write __init__.py files ──
for pkg in ["accentedge", "accentedge/codec", "accentedge/phase1", "accentedge/evaluation"]:
    path = f"/content/accentedge/src/{pkg}/__init__.py"
    if not os.path.exists(path):
        open(path, "w").close()

# ── Write pyproject.toml ──
with open("/content/accentedge/pyproject.toml", "w") as f:
    f.write("""[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.backends.legacy:build"

[project]
name = "accentedge"
version = "0.1.0-phase1"
description = "Phase 1: FAC-FACodec reimplementation"
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
dev = ["pytest>=8.0"]
""")

# ── Install package ──
os.chdir("/content/accentedge")
run("pip install -e .", "install accentedge")

# ── Run tests ──
run("python -m pytest tests/ -v", "run tests")

print("\n=== Bootstrap complete ===")
r = run("python -c 'import torch; print(f\"torch {torch.__version__}, CUDA {torch.version.cuda}\")'", "verify torch", check=False)
print(r.stdout.strip())
r = run("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader", "verify GPU", check=False)
print(f"GPU: {r.stdout.strip()}")
