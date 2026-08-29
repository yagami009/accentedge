"""Degradation pipeline."""

import numpy as np

from .canonicalize import canonicalize

__all__ = ["canonicalize", "apply_degradation"]


def apply_degradation(
    audio: np.ndarray,
    sample_rate: int,
    condition: str,
    seed: int = 42,
    nb_target_sr: int = 8000,
    noisy_snr_db: float = 15.0,
) -> np.ndarray:
    """Apply a degradation condition to audio.

    Conditions:
      - ``clean``   — return audio unchanged.
      - ``nb``      — resample to ``nb_target_sr`` then back to ``sample_rate``.
      - ``noisy``   — add zero-mean Gaussian noise at the given SNR.
      - ``nb_noisy`` — resample (as nb) then add noise (as noisy).

    Args:
        audio:        Input waveform (float32).
        sample_rate:  Original sample rate (Hz).
        condition:    One of the DegradationCondition values.
        seed:         RNG seed for deterministic noise.
        nb_target_sr: Sample rate used for the narrowband leg.
        noisy_snr_db: Signal-to-noise ratio for the noise leg.

    Returns:
        Degraded waveform as float32 at *sample_rate*.
    """
    if condition == "clean":
        return audio.astype(np.float32)

    result = audio.astype(np.float32)

    if condition in ("nb", "nb_noisy"):
        rng = np.random.RandomState(seed)
        down = _resample(result, sample_rate, nb_target_sr)
        result = _resample(down, nb_target_sr, sample_rate)

    if condition in ("noisy", "nb_noisy"):
        rng = np.random.RandomState(seed)
        noise = rng.randn(len(result)).astype(np.float32)
        signal_power = float(np.mean(result ** 2))
        noise_power = float(np.mean(noise ** 2))
        if signal_power > 0 and noise_power > 0:
            snr_linear = 10 ** (noisy_snr_db / 10.0)
            scale = np.sqrt(signal_power / (snr_linear * noise_power))
            noise = noise * scale
        else:
            noise = noise * 0.001
        result = result + noise
        max_val = float(np.max(np.abs(result)))
        if max_val > 1.0:
            result = result / max_val

    return result.astype(np.float32)


def _resample(waveform: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample using librosa if available, else simple linear interpolation."""
    if orig_sr == target_sr:
        return waveform
    try:
        import librosa
        return librosa.resample(waveform, orig_sr=orig_sr, target_sr=target_sr).astype(np.float32)
    except ImportError:
        n_samples = int(round(len(waveform) * target_sr / orig_sr))
        return np.interp(
            np.linspace(0, len(waveform), n_samples, endpoint=False),
            np.arange(len(waveform)),
            waveform,
        ).astype(np.float32)
