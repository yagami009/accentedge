"""
Audio I/O utilities for Phase 0 research.

Handles loading, saving, and validating audio files for the target
feasibility experiment. No real-time concerns — all offline.
"""

import hashlib
import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Union

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)


@dataclass
class AudioInfo:
    """Metadata for an audio file."""
    path: str
    duration_seconds: float
    sample_rate: int
    channels: int
    samples: int
    dtype: str
    file_hash: str
    format: str

    def to_dict(self) -> dict:
        return asdict(self)


def compute_file_hash(path: Path) -> str:
    """SHA-256 hash of audio file for provenance tracking."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def load_audio(
    path: Union[str, Path],
    expected_sr: int = 22050,
    expected_channels: int = 1,
) -> tuple[np.ndarray, "AudioInfo"]:
    """
    Load an audio file and return the waveform plus metadata.

    Args:
        path: Path to audio file
        expected_sr: Expected sample rate (validation only, no resampling)
        expected_channels: Expected channel count

    Returns:
        (waveform, AudioInfo) tuple
        waveform is float32 numpy array, shape [samples] or [samples, channels]

    Raises:
        ValueError: If sample rate or channels don't match expectations
        FileNotFoundError: If path doesn't exist
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    waveform, sr = sf.read(str(path), dtype=np.float32)
    info = sf.info(str(path))

    # Validate
    if sr != expected_sr:
        raise ValueError(
            f"Sample rate mismatch: expected {expected_sr}, got {sr} "
            f"for {path}"
        )

    channels = info.channels
    if channels != expected_channels:
        raise ValueError(
            f"Channel count mismatch: expected {expected_channels}, "
            f"got {channels} for {path}"
        )

    file_hash = compute_file_hash(path)

    audio_info = AudioInfo(
        path=str(path),
        duration_seconds=info.duration,
        sample_rate=sr,
        channels=channels,
        samples=len(waveform),
        dtype=str(waveform.dtype),
        file_hash=file_hash,
        format=info.format,
    )

    logger.debug(f"Loaded {path}: {info.duration:.2f}s, {sr}Hz, {channels}ch")
    return waveform, audio_info


def save_audio(
    path: Union[str, Path],
    waveform: np.ndarray,
    sample_rate: int = 22050,
    format: str = "WAV",
    subtype: str = "PCM_16",
) -> Path:
    """
    Save waveform to audio file.

    Args:
        path: Output path
        waveform: Audio data as numpy array
        sample_rate: Sample rate in Hz
        format: Audio format (WAV, FLAC, etc.)
        subtype: Audio subtype (PCM_16, PCM_24, FLOAT)

    Returns:
        Path to saved file
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Ensure float32 and mono
    waveform = np.asarray(waveform, dtype=np.float32)
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)

    # Clip to valid range
    waveform = np.clip(waveform, -1.0, 1.0)

    sf.write(str(path), waveform, sample_rate, format=format, subtype=subtype)
    logger.debug(f"Saved {path}: {len(waveform)/sample_rate:.2f}s")
    return path


def validate_audio(
    waveform: np.ndarray,
    sample_rate: int,
) -> list[str]:
    """
    Check waveform for common issues.

    Returns list of warning strings. Empty list = clean.
    """
    warnings = []

    if waveform.dtype != np.float32:
        warnings.append(f"dtype is {waveform.dtype}, expected float32")

    if np.any(np.isnan(waveform)):
        warnings.append("waveform contains NaN values")

    if np.any(np.isinf(waveform)):
        warnings.append("waveform contains Inf values")

    peak = np.max(np.abs(waveform))
    if peak > 1.0:
        warnings.append(f"peak amplitude {peak:.2f} exceeds [-1, 1]")
    elif peak < 0.01:
        warnings.append(f"very low peak amplitude {peak:.6f}")

    if len(waveform) == 0:
        warnings.append("empty waveform")

    return warnings


def duration_seconds(waveform: np.ndarray, sample_rate: int) -> float:
    """Return duration in seconds."""
    return len(waveform) / sample_rate
