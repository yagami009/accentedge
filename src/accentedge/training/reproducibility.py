"""Reproducibility utilities for training."""

from __future__ import annotations

import random
import sys
from typing import Any

import numpy as np

try:
    import torch
    import torch.backends.cudnn

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


def set_seed(seed: int) -> None:
    """Set Python, NumPy, and (if available) PyTorch RNG seeds."""
    random.seed(seed)
    np.random.seed(seed)
    if _TORCH_AVAILABLE:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def enable_deterministic() -> None:
    """Enable deterministic algorithms in PyTorch (with performance trade-off)."""
    if not _TORCH_AVAILABLE:
        return
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def get_rng_state() -> dict[str, Any]:
    """Capture the current state of all relevant RNGs."""
    state: dict[str, Any] = {
        "python_hash_seed": hash("sentinel"),
        "numpy": np.random.get_state(),
    }
    if _TORCH_AVAILABLE:
        state["torch_cpu"] = torch.get_rng_state()
        if torch.cuda.is_available():
            state["torch_cuda"] = torch.cuda.get_rng_state_all()
        else:
            state["torch_cuda"] = []
    return state


def verify_reproducibility(
    model: Any,
    dummy_batch: Any,
    n_runs: int = 3,
) -> bool:
    """Run *model* on *dummy_batch* *n_runs* times with the same seed; all losses
    must match within a tight tolerance."""
    if not _TORCH_AVAILABLE:
        return True  # nothing to verify

    losses: list[float] = []
    for _ in range(n_runs):
        set_seed(42)
        try:
            if isinstance(dummy_batch, dict):
                tensor_input = next(
                    v for v in dummy_batch.values() if isinstance(v, torch.Tensor)
                )
                out = model(tensor_input)
            else:
                out = model(dummy_batch)
            if isinstance(out, dict):
                loss = sum(
                    v for v in out.values() if isinstance(v, torch.Tensor)
                )
            else:
                loss = out
            scalar = loss.sum() if isinstance(loss, torch.Tensor) else float(loss)
            losses.append(float(scalar.detach().cpu()))
        except Exception:
            return False

    return all(
        abs(losses[0] - l) < 1e-9 for l in losses
    )

