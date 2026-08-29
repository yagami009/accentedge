"""Loss functions for AccentEdge training."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def content_loss(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """MSE on acoustic features (e.g., mel-spectrogram or bottleneck)."""
    return F.mse_loss(predicted, target)


def accent_loss(
    predicted_accent: torch.Tensor, target_accent: torch.Tensor
) -> torch.Tensor:
    """MSE on accent latent embeddings."""
    return F.mse_loss(predicted_accent, target_accent)


def speaker_loss(
    predicted_speaker: torch.Tensor, target_speaker: torch.Tensor
) -> torch.Tensor:
    """Cosine-similarity loss on speaker embeddings (1 - cosine_sim)."""
    # Normalise along the embedding dimension
    pred_norm = F.normalize(predicted_speaker, p=2, dim=-1)
    tgt_norm = F.normalize(target_speaker, p=2, dim=-1)
    cos_sim = (pred_norm * tgt_norm).sum(dim=-1)
    return (1.0 - cos_sim).mean()


def f0_loss(predicted_f0: torch.Tensor, target_f0: torch.Tensor) -> torch.Tensor:
    """MSE on F0 (pitch) contour."""
    return F.mse_loss(predicted_f0, target_f0)


def mel_loss(predicted_mel: torch.Tensor, target_mel: torch.Tensor) -> torch.Tensor:
    """L1 loss on mel spectrogram."""
    return F.l1_loss(predicted_mel, target_mel)


def reconstruction_loss(
    predicted_audio: torch.Tensor, target_audio: torch.Tensor
) -> torch.Tensor:
    """L1 loss on waveform."""
    return F.l1_loss(predicted_audio, target_audio)


def total_loss(
    components: dict[str, torch.Tensor],
    weights: dict[str, float],
) -> torch.Tensor:
    """Weighted sum of named loss components.

    Parameters
    ----------
    components:
        Mapping of ``loss_name -> loss_tensor``.  Only keys present in both
        *components* and *weights* contribute to the total.
    weights:
        Mapping of ``loss_name -> scalar_weight``.
    """
    total = torch.tensor(0.0, device=next(iter(components.values())).device)
    for name, weight in weights.items():
        if weight != 0 and name in components:
            total = total + weight * components[name]
    return total

