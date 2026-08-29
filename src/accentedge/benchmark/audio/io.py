"""Audio I/O: load, save, resample, mono conversion."""
import numpy as np
from pathlib import Path

import soundfile as sf
import librosa

SAMPLE_RATE_CANONICAL = 16000


def load_audio(path, sr=None, mono=True):
    """Load audio file. Returns (waveform_float32, sample_rate)."""
    wf, sr_actual = sf.read(str(path), dtype="float32")
    if wf.ndim > 1 and mono:
        wf = wf.mean(axis=1)
    if sr is not None and sr_actual != sr:
        wf = librosa.resample(wf, orig_sr=sr_actual, target_sr=sr)
        sr_actual = sr
    return wf.astype(np.float32), sr_actual


def save_audio(path, waveform, sample_rate):
    """Save float32 audio as PCM WAV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), waveform, sample_rate, subtype="PCM_16")


def resample_audio(waveform, orig_sr, target_sr):
    """Resample audio using librosa."""
    if orig_sr == target_sr:
        return waveform
    return librosa.resample(waveform, orig_sr=orig_sr, target_sr=target_sr).astype(
        np.float32
    )


def ensure_mono(waveform):
    """Convert to mono if stereo."""
    if waveform.ndim > 1:
        return waveform.mean(axis=1).astype(np.float32)
    return waveform

