"""Audio canonicalization."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..audio.io import load_audio, save_audio, resample_audio
from ..audio.hashing import sha256_file


def canonicalize(
    source_path: str | Path,
    output_path: str | Path,
    target_sr: int = 16000,
    peak_normalize: bool = True,
    peak_target: float = 0.95,
) -> dict:
    """Convert master audio to canonical benchmark format.
    
    Args:
        source_path: Path to master WAV
        output_path: Path for canonical output
        target_sr: Target sample rate (default 16kHz)
        peak_normalize: Whether to peak-normalize
        peak_target: Target peak level for normalization
    
    Returns:
        Dict with output path, hash, and metadata
    """
    audio, sr = load_audio(source_path)
    if sr != target_sr:
        audio = resample_audio(audio, sr, target_sr)
    
    if peak_normalize:
        peak = float(np.max(np.abs(audio))) if len(audio) > 0 else 0.0
        if peak > 0:
            audio = audio * (peak_target / peak)
    
    save_audio(output_path, audio, target_sr)
    output_hash = sha256_file(output_path)
    
    return {
        "output_path": str(output_path),
        "sample_rate": target_sr,
        "duration_ms": (len(audio) / target_sr) * 1000.0,
        "sha256": output_hash,
        "peak_normalized": peak_normalize,
    }
