"""Audio validation."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import numpy as np


@dataclass
class AudioValidationResult:
    valid: bool
    issues: list[str] = field(default_factory=list)
    duration_ms: float = 0.0
    peak_level: float = 0.0
    rms: float = 0.0
    clipping_fraction: float = 0.0
    silence_fraction: float = 0.0


def validate_audio(path, expected_sr=None, max_duration_ms=None):
    """Validate an audio file. Returns AudioValidationResult."""
    issues = []
    try:
        import soundfile as sf
        info = sf.info(str(path))
        if info.frames == 0:
            issues.append("Audio file is empty")
        if expected_sr and info.samplerate != expected_sr:
            issues.append(f"Sample rate {info.samplerate} != expected {expected_sr}")
        # Read and compute stats
        wf, sr = sf.read(str(path), dtype="float32")
        if wf.ndim > 1:
            wf = wf.mean(axis=1)
        duration_ms = (len(wf) / sr) * 1000
        peak = float(np.max(np.abs(wf))) if len(wf) > 0 else 0.0
        rms = float(np.sqrt(np.mean(wf**2))) if len(wf) > 0 else 0.0
        clipping = float(np.mean(np.abs(wf) > 0.99)) if len(wf) > 0 else 0.0
        silence = float(np.mean(np.abs(wf) < 0.001)) if len(wf) > 0 else 0.0
        if not np.all(np.isfinite(wf)):
            issues.append("Audio contains NaN or Inf")
        if peak > 1.0:
            issues.append(f"Peak level {peak:.2f} exceeds 1.0 (clipping)")
        if max_duration_ms and duration_ms > max_duration_ms:
            issues.append(f"Duration {duration_ms:.0f}ms exceeds max {max_duration_ms}ms")
        return AudioValidationResult(
            valid=len(issues) == 0,
            issues=issues,
            duration_ms=duration_ms,
            peak_level=peak,
            rms=rms,
            clipping_fraction=clipping,
            silence_fraction=silence,
        )
    except Exception as e:
        return AudioValidationResult(valid=False, issues=[f"Failed to read: {e}"])
