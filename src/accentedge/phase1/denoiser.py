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

