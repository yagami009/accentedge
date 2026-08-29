"""
Audio playback with crossfade support.

Streams converted audio chunks to the speaker output.
Handles crossfading between chunks to avoid audible seams.
"""

import queue
import threading
from typing import Optional

import numpy as np
import sounddevice as sd


class AudioPlayback:
    """
    Streaming speaker output with crossfade.

    Maintains a playback queue and smooths transitions between
    consecutive audio chunks using linear crossfade.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        crossfade_ms: int = 10,
        dtype: str = "float32",
    ):
        """
        Initialize audio playback.

        Args:
            sample_rate: Sample rate in Hz
            channels: Number of audio channels
            crossfade_ms: Crossfade duration in milliseconds
            dtype: Audio data type
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.dtype = dtype
        self.crossfade_samples = int(sample_rate * crossfade_ms / 1000)

        self._queue: queue.Queue[np.ndarray] = queue.Queue()
        self._stream: Optional[sd.OutputStream] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None

        # For crossfading
        self._previous_tail: Optional[np.ndarray] = None

    def _playback_loop(self, stop_event: threading.Event) -> None:
        """Internal playback loop — runs in a separate thread."""
        while not stop_event.is_set() or not self._queue.empty():
            try:
                chunk = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue

            # Apply crossfade with previous chunk's tail
            if self._previous_tail is not None and len(self._previous_tail) > 0:
                fade_len = min(self.crossfade_samples, len(chunk), len(self._previous_tail))
                if fade_len > 0:
                    # Linear fade: previous tail fades out, new chunk head fades in
                    fade_out = np.linspace(1.0, 0.0, fade_len)
                    fade_in = np.linspace(0.0, 1.0, fade_len)
                    chunk[:fade_len] = (
                        self._previous_tail[:fade_len] * fade_out
                        + chunk[:fade_len] * fade_in
                    )

            # Store tail for next crossfade
            self._previous_tail = chunk[-self.crossfade_samples :].copy() if len(chunk) > self.crossfade_samples else chunk.copy()

            # Write to output stream
            if self._stream and self._stream.active:
                self._stream.write(chunk)

    def start(self) -> None:
        """Start playback."""
        if self._running:
            raise RuntimeError("Playback already running")

        self._running = True
        self._stream = sd.OutputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype=self.dtype,
            blocksize=1024,
        )
        self._stream.start()
        print(f"[Playback] Started — {self.sample_rate}Hz, {self.crossfade_samples}-sample crossfade")

    def stop(self) -> None:
        """Stop playback."""
        if not self._running:
            return
        self._running = False
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        self._previous_tail = None
        print("[Playback] Stopped")

    def enqueue(self, chunk: np.ndarray) -> None:
        """
        Queue an audio chunk for playback.

        Args:
            chunk: Audio data as numpy array
        """
        self._queue.put(chunk)

    def flush(self) -> None:
        """Wait for all queued audio to finish playing."""
        self._queue.join()

    @property
    def is_running(self) -> bool:
        return self._running
