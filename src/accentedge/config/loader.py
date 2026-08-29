"""Configuration loading and validation."""

from pathlib import Path

import yaml

from accentedge.config.schema import Phase2Config


def load_config(path: str | Path) -> Phase2Config:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")
    raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Config file must contain a YAML mapping")
    return Phase2Config(**raw)


def validate_config(config: Phase2Config) -> None:
    if not config.phase2_id:
        raise ValueError("phase2_id is required")
    if config.phase != "2":
        raise ValueError(f"Expected phase='2', got phase={config.phase!r}")
    if config.training.identity_pair_ratio < 0 or config.training.identity_pair_ratio > 1:
        raise ValueError("identity_pair_ratio must be between 0 and 1")
    if config.training.gradient_clip_norm <= 0:
        raise ValueError("gradient_clip_norm must be positive")


def resolve_device(config: Phase2Config) -> str:
    dev = config.training.device
    if dev != "auto":
        return dev
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"

