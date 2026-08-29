"""
Real-time microphone capture using sounddevice.

Streams 16/24 kHz PCM audio from the default input device.
Provides a callback-based interface for low-latency chunk delivery.
"""

import queue
import threading
from typing import Callable, Optional

import numpy as np
import sounddevice as sd


class AudioCapture:
    """
    Streaming microphone capture.

    Uses sounddevice's callback interface to deliver audio chunks
    as they become available, minimizing latency.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        chunk_duration_ms: int = 200,
        channels: int = 1,
        dtype: str = "float32",
    ):
        """
        Initialize audio capture.

        Args:
            sample_rate: Sample rate in Hz (16000 or 24000)
            chunk_duration_ms: Duration of each audio chunk in milliseconds
            channels: Number of audio channels (1 = mono)
            dtype: Audio data type
        """
        self.sample_rate = sample_rate
        self.chunk_duration_ms = chunk_duration_ms
        self.chunk_size = int(sample_rate * chunk_duration_ms / 1000)
        self.channels = channels
        self.dtype = dtype

        self._queue: queue.Queue[np.ndarray] = queue.Queue()
        self._stream: Optional[sd.InputStream] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def _audio_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info: dict,
        status: sd.CallbackFlags,
    ) -> None:
        """Internal callback — pushes audio chunks to the queue."""
        if status:
            # Drop overflow/underflow warnings for now
            pass
        # Push a copy to the queue
        self._queue.put(indata.copy())

    def start(self) -> None:
        """Start capturing audio from the microphone."""
        if self._running:
            raise RuntimeError("Capture already running")

        self._running = True
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype=self.dtype,
            blocksize=self.chunk_size,
            callback=self._audio_callback,
        )
        self._stream.start()
        print(f"[Capture] Started — {self.sample_rate}Hz, {self.chunk_duration_ms}ms chunks")

    def stop(self) -> None:
        """Stop capturing audio."""
        if not self._running:
            return
        self._running = False
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        print("[Capture] Stopped")

    def read_chunk(self, timeout: Optional[float] = None) -> Optional[np.ndarray]:
        """
        Read the next audio chunk from the queue.

        Args:
            timeout: Max wait time in seconds (None = block indefinitely)

        Returns:
            Audio chunk as numpy array [chunk_size, channels], or None on timeout
        """
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def read_chunks(
        self, stop_event: threading.Event
    ) -> "Generator[np.ndarray, None, None]":
        """
        Generator that yields audio chunks until stop_event is set.

        Args:
            stop_event: Threading event to signal stopping

        Yields:
            Audio chunks as numpy arrays
        """
        while not stop_event.is_set() or not self._queue.empty():
            chunk = self.read_chunk(timeout=0.1)
            if chunk is not None:
                yield chunk

    @property
    def is_running(self) -> bool:
        return self._running
