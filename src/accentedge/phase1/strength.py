"""Phase 1 — Global conversion strength control.

Maps user-facing strength ∈ [0, 1] to paper-native t_start timestep.

strength=0.0 → t_start=0 (no pronunciation normalization, FACodec reconstruction)
strength=1.0 → t_start=num_steps (maximum Phase-1 normalization)

NOTES:
- Perceptual linearity is NOT assumed.
- 'strength 0.5 = 50% American accent' is a false claim.
"""
from __future__ import annotations

import torch


def strength_to_t_start(strength: float, num_steps: int = 100) -> int:
    """Map conversion strength ∈ [0, 1] to diffusion timestep t_start."""
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

    def available_strengths(self) -> list[float]:
        return [0.0, 0.25, 0.50, 0.75, 1.0]
