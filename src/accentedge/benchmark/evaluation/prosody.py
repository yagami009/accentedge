"""Prosody metrics."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class ProsodyResult:
    f0_range_hz: tuple[float, float] = (0.0, 0.0)
    f0_mean: float = 0.0
    energy_mean: float = 0.0
    speech_rate: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


class ProsodyEvaluator:
    def extract_f0(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        try:
            import librosa
            f0, _ = librosa.piptrack(y=audio, sr=sample_rate)
            f0_mean = np.mean(f0[f0 > 0]) if np.any(f0 > 0) else 0.0
            return f0_mean
        except Exception:
            return 0.0

    def extract_energy(self, audio: np.ndarray) -> float:
        return float(np.sqrt(np.mean(audio ** 2)))

    def evaluate(self, audio: np.ndarray, sample_rate: int) -> ProsodyResult:
        f0 = self.extract_f0(audio, sample_rate)
        energy = self.extract_energy(audio)
        return ProsodyResult(f0_mean=float(f0), energy_mean=energy)

