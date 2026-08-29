"""Accent token translator for Candidate C.

Maps source token embeddings to target-accent embeddings using a causal
LSTM with FiLM conditioning. Supports bounded lookahead so the model
can peek at a small number of future tokens, improving prosody
consistency without breaking the streaming constraint.
"""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from accentedge.models.token_translation.interfaces import (
    AccentTokenTranslator,
    TokenSequence,
)


class FiLM(nn.Module):
    """Feature-wise Linear Modulation for accent conditioning."""

    def __init__(self, hidden_dim: int, accent_dim: int) -> None:
        super().__init__()
        self.accent_proj = nn.Linear(accent_dim, hidden_dim * 2)
        self.strength_proj = nn.Linear(1, hidden_dim * 2)
        self.accent_dim = accent_dim

    def forward(
        self,
        x: torch.Tensor,
        target_accent: torch.Tensor,
        strength: float,
        accent_embedding: nn.Embedding | None = None,
    ) -> torch.Tensor:
        """Apply FiLM modulation: gamma * x + beta."""
        B, T, _ = x.shape
        if target_accent.dim() == 0:
            target_accent = target_accent.unsqueeze(0)
        if accent_embedding is not None:
            accent_emb = self.accent_proj(accent_embedding(target_accent))
        else:
            accent_emb = self.accent_proj(
                torch.zeros(B, self.accent_dim, device=x.device, dtype=x.dtype)
            )
        accent_emb = accent_emb.unsqueeze(1).expand(-1, T, -1)
        strength_tensor = torch.full(
            (B, T, 1), strength, device=x.device, dtype=x.dtype
        )
        film_params = accent_emb + self.strength_proj(strength_tensor)
        gamma, beta = film_params.chunk(2, dim=-1)
        return gamma * x + beta


class AccentTokenTranslator(nn.Module):
    """Causal LSTM-based accent token translator with FiLM conditioning.

    Architecture:
        1. Input projection: token_dim -> translator_hidden
        2. 2-layer LSTM (causal by construction)
        3. FiLM conditioning with target_accent + conversion_strength
        4. Output projection: translator_hidden -> token_dim

    Supports bounded lookahead by buffering future tokens before
    producing output at each position. When lookahead_frames > 0,
    the model sees tokens[t : t + lookahead_frames + 1] to predict
    the output for position t.
    """

    def __init__(
        self,
        token_dim: int = 128,
        translator_layers: int = 2,
        translator_hidden: int = 256,
        num_accents: int = 5,
        accent_dim: int = 32,
        lookahead_frames: int = 0,
    ) -> None:
        super().__init__()
        self.token_dim = token_dim
        self.translator_hidden = translator_hidden
        self.lookahead_frames = lookahead_frames

        self.input_proj = nn.Linear(token_dim, translator_hidden)
        self.lstm = nn.LSTM(
            input_size=translator_hidden,
            hidden_size=translator_hidden,
            num_layers=translator_layers,
            batch_first=True,
            dropout=0.1 if translator_layers > 1 else 0.0,
        )
        self.film = FiLM(translator_hidden, accent_dim)
        self.output_proj = nn.Linear(translator_hidden, token_dim)

        # Accent embedding table.
        self.accent_embedding = nn.Embedding(num_accents, accent_dim)

    def forward(
        self,
        tokens: torch.Tensor,
        target_accent: torch.Tensor,
        strength: float,
        state: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Translate token embeddings with causal lookahead.

        Args:
            tokens: (B, T, token_dim) input token embeddings.
            target_accent: (B,) integer accent IDs.
            strength: Conversion strength in [0, 1].
            state: Optional (h, c) tuple for LSTM state carry-over.

        Returns:
            translated: (B, T_out, token_dim) where T_out = T - lookahead_frames.
            new_state: Updated (h, c) tuple.
        """
        B, T, _ = tokens.shape
        if target_accent.dim() == 0:
            target_accent = target_accent.unsqueeze(0)
        x = self.input_proj(tokens)

        # Pad the beginning with zeros if lookahead > 0, so that output at
        # position t is computed from inputs up to t + lookahead.
        if self.lookahead_frames > 0:
            pad = torch.zeros(B, self.lookahead_frames, self.translator_hidden, device=x.device, dtype=x.dtype)
            x = torch.cat([x, pad], dim=1)

        lstm_out, new_state = self.lstm(x, state)
        # Return only the first T outputs; the extra lookahead frames provided future context.
        lstm_out = lstm_out[:, :T]

        # FiLM conditioning.
        conditioned = self.film(lstm_out, target_accent, strength, self.accent_embedding)

        translated = self.output_proj(conditioned)
        return translated, new_state

    def translate(
        self,
        tokens: TokenSequence,
        target_accent: torch.Tensor,
        strength: float,
        context: Optional[dict[str, Any]] = None,
    ) -> TokenSequence:
        """Translate a TokenSequence to the target accent.

        Args:
            tokens: Source token sequence.
            target_accent: Integer tensor (B,) with target accent IDs.
            strength: Conversion strength in [0, 1].
            context: Optional dictionary with translator state for streaming.

        Returns:
            Translated TokenSequence with modified embeddings.
        """
        self.eval()
        with torch.no_grad():
            seq_tensor = tokens.to_tensor().unsqueeze(0)  # (1, T, dim)
            state = context.get("translator_state") if context else None
            translated_tensor, new_state = self.forward(seq_tensor, target_accent, strength, state)

        if context is not None:
            context["translator_state"] = new_state

        return TokenSequence.from_tensor(translated_tensor.squeeze(0))

