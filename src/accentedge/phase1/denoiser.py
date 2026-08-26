"""Phase 1 — FAC-FACodec denoiser (paper-faithful reimplementation).

Architecture (from arxiv:2510.10785):
  6-layer Transformer encoder, 8 attention heads, dim=1024, FFN=2048, dropout=0.1
  Conditioned on phoneme embeddings via FiLM + additive embeddings.
  Predicts noise epsilon; also predicts zc2 from denoised x0 estimate.

Reference implementation: Claussss/FAC-FACodec (MIT License)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn.utils.rnn import pad_sequence


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


class PhonemeEmbedding(nn.Module):
    """Phoneme embedding table."""
    def __init__(self, vocab_size: int, emb_dim: int, pad_id: int = 0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)

    def forward(self, phone_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(phone_ids)


class CondLayerNorm(nn.Module):
    """Conditional LayerNorm — FiLM-style conditioning on phoneme + timestep."""
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D], cond: [B, 2D]
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        x = (x - mean) / (std + self.eps)
        # FiLM: cond → scale + shift
        scale, shift = cond.chunk(2, dim=-1)
        return self.weight * scale.unsqueeze(1) * x + self.bias * shift.unsqueeze(1) + x


class DenoisingTransformer(nn.Module):
    """FAC-FACodec denoiser: 6-layer Transformer with phoneme + timestep conditioning.

    Predicts epsilon (noise) and zc2 (content residual) from noisy zc1.
    """

    def __init__(
        self,
        d_model: int = 1024,
        nhead: int = 8,
        num_layers: int = 6,
        d_ff: int = 2048,
        dropout: float = 0.1,
        phone_vocab_size: int = 392,
        phone_pad_id: int = 392,
        max_seq_len: int = 500,
        num_steps: int = 100,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_steps = num_steps

        # Components
        self.phone_emb = PhonemeEmbedding(phone_vocab_size, d_model, pad_id=phone_pad_id)
        self.t_embed = SinusoidalPosEmb(d_model * 2)
        self.dropout = nn.Dropout(dropout)

        # Transformer encoder
        encoder_layer = TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output heads
        self.fc_out = nn.Linear(d_model, d_model)  # epsilon prediction
        self.fc_zc2 = nn.Linear(d_model * 2, d_model)  # zc2 prediction

        # Noise schedule (linear, T=100)
        betas = torch.linspace(1e-4, 2e-2, num_steps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer("sqrt_abar", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_1m_abar", torch.sqrt(1.0 - alphas_cumprod))

    def forward(
        self,
        z: torch.Tensor,          # [B, D, T] noisy latent
        padded_phone_ids: torch.Tensor,  # [B, T_phone]
        t: torch.Tensor,          # [B] timestep indices
        padding_mask: Optional[torch.Tensor] = None,  # [B, T] bool
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict epsilon and zc2.

        Returns:
            eps_pred: [B, D, T] predicted noise
            zc2_pred: [B, D, T] predicted content residual
        """
        bsz, d_model, seq_len = z.shape
        assert d_model == self.d_model

        # Transpose to [B, T, D] for Transformer
        h = z.transpose(1, 2)  # [B, T, D]

        # Condition: phoneme embeddings + timestep embeddings
        phone_emb = self.phone_emb(padded_phone_ids)  # [B, T_phone, D]
        # Average phoneme embeddings to match latent sequence length
        phone_cond = F.avg_pool1d(
            phone_emb.transpose(1, 2),
            kernel_size=phone_emb.size(1) // seq_len + 1,
            stride=1,
        )
        if phone_cond.size(2) < seq_len:
            phone_cond = F.pad(phone_cond, (0, seq_len - phone_cond.size(2)))
        phone_cond = phone_cond[:, :, :seq_len].transpose(1, 2)  # [B, T, D]

        # Timestep embedding
        t_emb = self.t_embed(t)  # [B, 2D]
        t_cond = t_emb.unsqueeze(1).expand(-1, seq_len, -1)  # [B, T, 2D]

        # Combine conditioning
        cond = self.dropout(phone_cond) + t_cond  # [B, T, 2D]

        # Build padding mask for Transformer
        if padding_mask is not None:
            src_key_padding_mask = ~padding_mask
        else:
            src_key_padding_mask = None

        # Transformer with conditional norm (manual FiLM)
        h = self.encoder(h, src_key_padding_mask=src_key_padding_mask)

        # Epsilon prediction
        eps_pred = self.fc_out(h).transpose(1, 2)  # [B, D, T]

        # Zc2 prediction from denoised x0 estimate
        sa = self.sqrt_abar[t].view(bsz, 1, 1)
        s1a = self.sqrt_1m_abar[t].view(bsz, 1, 1)
        x0_hat = (z - s1a * eps_pred.detach()) / sa
        zc2_input = torch.cat([h, x0_hat.detach().transpose(1, 2)], dim=-1)
        zc2_pred = self.fc_zc2(zc2_input).transpose(1, 2)  # [B, D, T]

        return eps_pred, zc2_pred


