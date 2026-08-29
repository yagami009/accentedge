"""Scheduler and optimizer factories."""

from __future__ import annotations

from typing import Any

import torch


def get_optimizer(
    params,
    optimizer_type: str,
    lr: float,
    **kwargs: Any,
) -> torch.optim.Optimizer:
    """Create an optimizer by name.

    Supported types: ``adam``, ``adamw``, ``sgd``.
    """
    optimizer_type = optimizer_type.lower()
    if optimizer_type == "adam":
        return torch.optim.Adam(params, lr=lr, **kwargs)
    if optimizer_type == "adamw":
        return torch.optim.AdamW(params, lr=lr, **kwargs)
    if optimizer_type == "sgd":
        return torch.optim.SGD(params, lr=lr, **kwargs)
    raise ValueError(f"Unsupported optimizer type: {optimizer_type!r}")


def get_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_type: str,
    **kwargs: Any,
) -> torch.optim.lr_scheduler._LRScheduler:
    """Create a learning-rate scheduler by name.

    Supported types: ``linear``, ``cosine``, ``step``, ``constant``.
    """
    scheduler_type = scheduler_type.lower()
    if scheduler_type == "linear":
        return torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=kwargs.get("start_factor", 1.0),
            end_factor=kwargs.get("end_factor", 0.0),
            total_iters=kwargs["total_iters"],
        )
    if scheduler_type == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=kwargs["T_max"],
            eta_min=kwargs.get("eta_min", 0.0),
        )
    if scheduler_type == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=kwargs["step_size"],
            gamma=kwargs.get("gamma", 0.5),
        )
    if scheduler_type == "constant":
        return torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)
    raise ValueError(f"Unsupported scheduler type: {scheduler_type!r}")

