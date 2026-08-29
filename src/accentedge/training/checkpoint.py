"""Checkpoint serialization for AccentEdge Phase 1.

Every checkpoint MUST include normalization constants (zc1_mean, zc1_std).
Without them, the checkpoint is unusable for inference.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import torch


def _git_sha() -> str:
    """Return short git SHA of the current checkout, or 'unknown'."""
    try:
        import subprocess
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5
        )
        sha = result.stdout.strip()
        return sha if sha else "unknown"
    except Exception:
        return "unknown"


def _config_hash(config: dict) -> str:
    """Stable SHA-256 of the config dict (deterministic serialization)."""
    import json as _json
    payload = _json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    step: int,
    epoch: int,
    config: dict,
    phone_vocab: list[str] | None,
    facodec_ckpt: str,
    zc1_mean: torch.Tensor,
    zc1_std: torch.Tensor,
    output_path: str,
    extra: dict | None = None,
) -> str:
    """Save a complete checkpoint including normalization constants.

    Args:
        model: Denoiser model (state_dict saved).
        optimizer: Optimizer state.
        scheduler: LR scheduler (may be None).
        step: Global training step.
        epoch: Current epoch number.
        config: Full training config dict (used for config hash).
        phone_vocab: List of phoneme strings (phone_id → label mapping).
        facodec_ckpt: HuggingFace identifier or path for FACodec checkpoint.
        zc1_mean: Per-channel mean of zc1 over training set [C].
        zc1_std: Per-channel std of zc1 over training set [C].
        output_path: Path to save checkpoint.pt.
        extra: Optional additional fields to include.

    Returns:
        The absolute path to the saved checkpoint.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "step": step,
        "epoch": epoch,
        "config_hash": _config_hash(config),
        "git_sha": _git_sha(),
        "phone_vocab": phone_vocab,
        "facodec_ckpt": facodec_ckpt,
        "zc1_mean": zc1_mean.detach().cpu().tolist(),
        "zc1_std": zc1_std.detach().cpu().tolist(),
        "config": config,
    }

    if extra:
        checkpoint["extra"] = extra

    torch.save(checkpoint, str(path))
    return str(path.resolve())


def load_checkpoint(path: str, map_location: str = "cpu") -> dict:
    """Load a checkpoint and return all fields.

    Validates that normalization constants are present.

    Args:
        path: Path to checkpoint.pt.
        map_location: Device to load tensors onto.

    Returns:
        Dict with keys: model_state_dict, optimizer_state_dict,
        scheduler_state_dict, step, epoch, config_hash, git_sha,
        phone_vocab, facodec_ckpt, zc1_mean, zc1_std, config.

    Raises:
        FileNotFoundError: If checkpoint path does not exist.
        ValueError: If normalization constants are missing.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    data = torch.load(str(path), map_location=map_location, weights_only=False)

    # Validate required fields
    required_keys = [
        "model_state_dict", "step", "epoch", "config_hash", "git_sha",
        "facodec_ckpt", "zc1_mean", "zc1_std",
    ]
    missing = [k for k in required_keys if k not in data]
    if missing:
        raise ValueError(
            f"Checkpoint missing required keys: {missing}. "
            "Checkpoint is unusable without normalization constants."
        )

    return data

