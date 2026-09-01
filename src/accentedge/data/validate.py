"""Audio QA — validate raw audio files for corruption, format, and quality."""

import os
import wave
import struct
from pathlib import Path

import soundfile as sf
import numpy as np


class ValidationResult:
    def __init__(self, path: Path):
        self.path = path
        self.valid = True
        self.issues = []

    def add_issue(self, severity: str, message: str):
        self.issues.append({"severity": severity, "message": message})
        if severity == "error":
            self.valid = False

    def to_dict(self):
        return {
            "path": str(self.path),
            "valid": self.valid,
            "issues": self.issues,
        }


def validate_wav(path: Path, expected_sr: int = 16000) -> ValidationResult:
    result = ValidationResult(path)

    if not path.exists():
        result.add_issue("error", "File does not exist")
        return result

    if path.stat().st_size == 0:
        result.add_issue("error", "File is empty")
        return result

    try:
        info = sf.info(str(path))
    except Exception as e:
        result.add_issue("error", f"Cannot read audio: {e}")
        return result

    if info.samplerate != expected_sr:
        result.add_issue("warning", f"Sample rate {info.samplerate} != {expected_sr}")

    if info.channels != 1:
        result.add_issue("warning", f"Channels: {info.channels} (expected 1)")

    if info.duration < 0.05:
        result.add_issue("error", f"Duration too short: {info.duration:.2f}s")

    if info.duration > 30:
        result.add_issue("warning", f"Duration long: {info.duration:.1f}s")

    try:
        data, _ = sf.read(str(path), dtype=np.float32)
        if np.any(np.isnan(data)):
            result.add_issue("error", "Contains NaN samples")
        max_val = np.max(np.abs(data))
        if max_val > 0.99:
            result.add_issue("warning", f"Possible clipping: peak={max_val:.3f}")
        rms = np.sqrt(np.mean(data**2))
        if rms < 0.001:
            result.add_issue("warning", f"Very low RMS: {rms:.6f}")
    except Exception as e:
        result.add_issue("error", f"Read error: {e}")

    return result


def validate_dataset(root: Path, pattern: str = "**/*.wav", expected_sr: int = 16000):
    results = []
    for audio_file in root.glob(pattern):
        r = validate_wav(audio_file, expected_sr)
        results.append(r.to_dict())
    return results
