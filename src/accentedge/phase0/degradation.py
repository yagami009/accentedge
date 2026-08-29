"""
Audio degradation for Phase 0 subtest conditions.

Applies realistic BPO-channel degradations to source audio so that
target generation can be tested on conditions that match real deployment:

    clean       — original high-quality recording (no degradation)
    NB          — narrowband: 8 kHz resample + G.711 mu-law encode/decode
    noisy       — call-centre babble noise mixed in at target SNR
    NB+noisy    — narrowband plus babble noise

Degradations simulate the USB headset + telephony path an Indian-English
BPO agent would use.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DegradationConfig:
    """Parameters for a single degradation condition."""

    # Narrowband parameters
    nb_sample_rate: int = 8000

    # G.711 mu-law codec
    apply_mulaw: bool = True
    mulaw_bit_depth: int = 8

    # Babble noise parameters
    babble_snr_db: float = 10.0  # SNR of babble relative to speech
    babble_source: Optional[Union[str, np.ndarray]] = None
    # Can be a path to a noise file or a pre-loaded noise array

    # Headset frequency-response approximation (simple shelving)
    apply_headset_eq: bool = False
    headset_low_cut_hz: float = 300.0
    headset_high_cut_hz: float = 3400.0

    # Output sample rate (after all processing)
    output_sample_rate: int = 8000

    def __str__(self) -> str:
        parts = []
        if self.apply_mulaw:
            parts.append("NB")
        if self.babble_snr_db is not None:
            parts.append("+noisy")
        if self.apply_headset_eq:
            parts.append("+headset")
        if not parts:
            parts.append("clean")
        return "".join(parts)


def _resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample audio to target rate using scipy or numpy fallback."""
    if orig_sr == target_sr:
        return audio

    try:
        from scipy.signal import resample as _scipy_resample
        n_samples = int(len(audio) * target_sr / orig_sr)
        return _scipy_resample(audio, n_samples)
    except ImportError:
        # Naive decimation / interpolation fallback
        ratio = target_sr / orig_sr
        indices = np.arange(0, len(audio), 1.0 / ratio)
        return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)


def _encode_mulaw(audio: np.ndarray) -> np.ndarray:
    """Encode float32 audio to 8-bit mu-law (G.711)."""
    # mu-law encoding: ITU-T G.711
    # Ensure signal in [-1, 1]
    signal = np.clip(audio, -1.0, 1.0)
    # Sign-magnitude bit
    sign = np.sign(signal)
    magnitude = np.abs(signal)
    # Quantize magnitude to 7 bits (128 levels, with companding)
    # ITU-T G.711 companding curve
    mu = 255
    # Apply companding
    encoded = np.sign(signal) * np.log1p(mu * magnitude) / np.log1p(mu)
    # Quantize to [-127, 127] (0 reserved for silence)
    encoded_int = np.round(encoded * 127).astype(np.int16)
    return encoded_int.astype(np.float32)


def _decode_mulaw(encoded: np.ndarray) -> np.ndarray:
    """Decode 8-bit mu-law back to float32 audio."""
    mu = 255
    # Inverse of the companding curve
    sign = np.sign(encoded)
    magnitude = np.abs(encoded) / 127.0
    decoded = sign * ((1 + mu) ** magnitude - 1) / mu
    return decoded.astype(np.float32)


def _apply_babble(
    speech: np.ndarray,
    babble: np.ndarray,
    snr_db: float,
) -> np.ndarray:
    """Mix babble noise into speech at the specified SNR."""
    # Ensure babble is at least as long as speech
    if len(babble) < len(speech):
        # Tile the babble
        repeats = int(np.ceil(len(speech) / len(babble)))
        babble = np.tile(babble, repeats)[:len(speech)]
    else:
        babble = babble[:len(speech)]

    # Compute RMS levels
    speech_rms = np.sqrt(np.mean(speech ** 2))
    babble_rms = np.sqrt(np.mean(babble ** 2))

    if speech_rms == 0:
        return speech

    # Scale babble to achieve target SNR
    # SNR = 10 * log10(speech_power / noise_power)
    # noise_scale = speech_rms / (10^(SNR/20) * babble_rms)
    target_noise_rms = speech_rms / (10 ** (snr_db / 20.0))
    scale = target_noise_rms / (babble_rms + 1e-10)
    scaled_babble = babble * scale

    mixed = speech + scaled_babble
    # Clip to valid range
    mixed = np.clip(mixed, -1.0, 1.0)
    return mixed.astype(np.float32)


def _apply_headset_eq(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Approximate USB headset frequency response: high-pass + low-pass shelves."""
    try:
        from scipy.signal import butter, sosfilt
        # Design a bandpass filter to approximate headset response
        nyquist = sample_rate / 2.0
        low = max(80.0, _DEGRADE_CONFIGS.get("headset_low_cut_hz", 300.0))
        high = min(nyquist - 100.0, _DEGRADE_CONFIGS.get("headset_high_cut_hz", 3400.0))

        if low >= high:
            return audio

        sos = butter(4, [low / nyquist, high / nyquist], btype="band", output="sos")
        filtered = sosfilt(sos, audio)
        return filtered.astype(np.float32)
    except ImportError:
        logger.warning("scipy not available, skipping headset EQ")
        return audio


# Holds mutable config for headset EQ fallback
_DEGRADE_CONFIGS: dict = {}


def apply_degradation(
    audio: np.ndarray,
    sample_rate: int,
    config: DegradationConfig,
    babble_noise: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Apply degradation chain to audio to simulate BPO channel conditions.

    Processing order:
    1. Headset EQ (optional)
    2. Resample to narrowband (if configured)
    3. G.711 mu-law encode/decode (if configured)
    4. Babble noise mixing (if configured)

    Args:
        audio: Input waveform as float32 numpy array
        sample_rate: Input sample rate
        config: DegradationConfig specifying the degradation chain
        babble_noise: Pre-loaded babble noise array (optional). If None and
            config.babble_source is a path, it will be loaded from that path.

    Returns:
        Degraded audio as float32 numpy array
    """
    result = audio.copy().astype(np.float32)
    cur_sr = sample_rate
    _DEGRADE_CONFIGS["headset_low_cut_hz"] = config.headset_low_cut_hz
    _DEGRADE_CONFIGS["headset_high_cut_hz"] = config.headset_high_cut_hz

    # Step 1: Headset EQ (applied before resampling)
    if config.apply_headset_eq:
        logger.debug("Applying headset EQ approximation")
        result = _apply_headset_eq(result, cur_sr)

    # Step 2: Resample to narrowband if needed
    if config.nb_sample_rate < cur_sr:
        logger.debug("Resampling %d Hz → %d Hz", cur_sr, config.nb_sample_rate)
        result = _resample(result, cur_sr, config.nb_sample_rate)
        cur_sr = config.nb_sample_rate
    elif config.nb_sample_rate > cur_sr:
        logger.warning(
            "Cannot upsample from %d to %d via narrowband; skipping resample",
            cur_sr,
            config.nb_sample_rate,
        )

    # Step 3: G.711 mu-law codec
    if config.apply_mulaw:
        logger.debug("Applying G.711 mu-law encode/decode")
        encoded = _encode_mulaw(result)
        result = _decode_mulaw(encoded)

    # Step 4: Babble noise
    if config.babble_snr_db is not None and config.babble_snr_db >= 0:
        logger.debug("Mixing babble noise at SNR %.1f dB", config.babble_snr_db)
        noise = babble_noise
        if noise is None and config.babble_source is not None:
            if isinstance(config.babble_source, (str, Path)):
                try:
                    import soundfile as sf
                    noise, _ = sf.read(str(config.babble_source), dtype=np.float32)
                except Exception:
                    logger.warning(
                        "Could not load babble source: %s", config.babble_source
                    )
        if noise is not None:
            result = _apply_babble(result, noise, config.babble_snr_db)

    # Final resample to output sample rate if different
    if cur_sr != config.output_sample_rate:
        logger.debug(
            "Resampling %d Hz → %d Hz (output)", cur_sr, config.output_sample_rate
        )
        result = _resample(result, cur_sr, config.output_sample_rate)

    return result.astype(np.float32)


# Pre-defined degradation configurations for Phase 0 subtest
DEGRADATION_PRESETS: dict[str, DegradationConfig] = {
    "clean": DegradationConfig(
        apply_mulaw=False,
        babble_snr_db=None,
        apply_headset_eq=False,
        output_sample_rate=22050,
    ),
    "NB": DegradationConfig(
        apply_mulaw=True,
        babble_snr_db=None,
        apply_headset_eq=False,
        output_sample_rate=8000,
    ),
    "noisy": DegradationConfig(
        apply_mulaw=False,
        babble_snr_db=10.0,
        apply_headset_eq=False,
        output_sample_rate=22050,
    ),
    "NB+noisy": DegradationConfig(
        apply_mulaw=True,
        babble_snr_db=10.0,
        apply_headset_eq=False,
        output_sample_rate=8000,
    ),
}
