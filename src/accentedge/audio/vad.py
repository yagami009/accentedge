"""
Voice Activity Detection using Silero VAD.

Detects speech segments in audio streams and returns timestamps.
Lightweight model (~2MB) that runs on CPU with minimal latency.
"""

import logging
from typing import List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class VADSegment:
    """Represents a detected speech segment."""

    def __init__(self, start_ms: float, end_ms: float, audio: np.ndarray):
        self.start_ms = start_ms
        self.end_ms = end_ms
        self.audio = audio
        self.duration_ms = end_ms - start_ms

    def __repr__(self) -> str:
        return f"VADSegment({self.start_ms:.0f}ms → {self.end_ms:.0f}ms, {self.duration_ms:.0f}ms)"


class VoiceActivityDetector:
    """
    Silero VAD-based voice activity detection.

    Detects speech/non-speech boundaries in audio streams.
    Optimized for call-center audio (8-16 kHz).
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        threshold: float = 0.5,
        min_speech_duration_ms: int = 250,
        min_silence_duration_ms: int = 100,
        speech_pad_ms: int = 30,
    ):
        """
        Initialize VAD.

        Args:
            sample_rate: Audio sample rate
            threshold: VAD confidence threshold (0.0-1.0)
            min_speech_duration_ms: Minimum speech segment duration
            min_silence_duration_ms: Minimum silence to split segments
            speech_pad_ms: Padding around speech segments
        """
        self.sample_rate = sample_rate
        self.threshold = threshold
        self.min_speech_duration_ms = min_speech_duration_ms
        self.min_silence_duration_ms = min_silence_duration_ms
        self.speech_pad_ms = speech_pad_ms

        self._model = None
        self._utils = None
        self._loaded = False

    def _load_model(self) -> None:
        """Lazy-load the Silero VAD model."""
        if self._loaded:
            return

        try:
            import torch

            model, utils = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                force_reload=False,
                trust_repo=True,
            )
            self._model = model
            self._utils = utils
            self._loaded = True
            logger.info("[VAD] Silero VAD loaded")
        except Exception as e:
            logger.error(f"[VAD] Failed to load Silero VAD: {e}")
            raise

    def detect(self, audio: np.ndarray) -> List[VADSegment]:
        """
        Detect speech segments in audio.

        Args:
            audio: Audio samples as numpy array [samples] or [samples, channels]

        Returns:
            List of VADSegment objects with speech boundaries
        """
        self._load_model()

        # Ensure mono
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        # Normalize to float32
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32) / 32768.0 if audio.dtype == np.int16 else audio.astype(np.float32)

        # Clamp to valid range
        audio = np.clip(audio, -1.0, 1.0)

        # Get speech timestamps using Silero's built-in function
        get_speech_timestamps = self._utils[0]
        timestamps = get_speech_timestamps(
            audio,
            self._model,
            threshold=self.threshold,
            min_speech_duration_ms=self.min_speech_duration_ms,
            min_silence_duration_ms=self.min_silence_duration_ms,
            speech_pad_ms=self.speech_pad_ms,
            sampling_rate=self.sample_rate,
        )

        segments = []
        for ts in timestamps:
            start_sample = ts["start"]
            end_sample = ts["end"]
            start_ms = start_sample / self.sample_rate * 1000
            end_ms = end_sample / self.sample_rate * 1000
            segment_audio = audio[start_sample:end_sample]
            segments.append(VADSegment(start_ms, end_ms, segment_audio))

        return segments

    def is_speech(self, audio: np.ndarray) -> bool:
        """
        Quick check if audio contains speech.

        Args:
            audio: Audio samples

        Returns:
            True if speech detected above threshold
        """
        segments = self.detect(audio)
        if not segments:
            return False
        # Return True if total speech duration exceeds minimum
        total_speech_ms = sum(s.duration_ms for s in segments)
        return total_speech_ms >= self.min_speech_duration_ms

