"""Phase 1 — Acoustic quality evaluation.

Honest metrics — no fabricated "speaker similarity" from mel statistics.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def mel_spectrogram(waveform: np.ndarray, sr: int = 24000, n_fft: int = 2048,
                    hop_length: int = 300, n_mels: int = 80) -> np.ndarray:
    """Compute mel spectrogram (numpy, for evaluation only)."""
    import librosa
    mel = librosa.feature.melspectrogram(
        y=waveform, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db


def mel_l1(source: np.ndarray, target: np.ndarray, sr: int = 24000) -> float:
    """Mean absolute difference between mel spectrograms."""
    mel_src = mel_spectrogram(source, sr)
    mel_tgt = mel_spectrogram(target, sr)
    # Align lengths
    min_len = min(mel_src.shape[1], mel_tgt.shape[1])
    diff = np.abs(mel_src[:, :min_len] - mel_tgt[:, :min_len])
    return float(diff.mean())


def duration_ratio(source: np.ndarray, target: np.ndarray, sr: int = 24000) -> float:
    """Ratio of output duration to source duration."""
    return len(target) / max(len(source), 1)


class AcousticEvaluator:
    """Acoustic quality diagnostics."""

    def __init__(self, sr: int = 24000):
        self.sr = sr

    def evaluate(self, source: np.ndarray, target: np.ndarray) -> dict:
        return {
            "mel_l1": mel_l1(source, target, self.sr),
            "duration_ratio": duration_ratio(source, target, self.sr),
        }
