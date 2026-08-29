"""
Thread-safe ring buffer for streaming audio chunks.

Stores incoming audio in a circular buffer and yields fixed-size
chunks for model inference. Handles variable chunk sizes gracefully.
"""

import threading
from typing import Optional

import numpy as np


class RingBuffer:
    """
    Thread-safe ring buffer for audio data.

    Accumulates audio samples and produces fixed-size chunks.
    """

    def __init__(self, max_duration_seconds: float = 5.0, sample_rate: int = 16000):
        """
        Initialize ring buffer.

        Args:
            max_duration_seconds: Maximum buffer duration
            sample_rate: Sample rate in Hz
        """
        self.max_samples = int(max_duration_seconds * sample_rate)
        self.sample_rate = sample_rate
        self._buffer = np.zeros(self.max_samples, dtype=np.float32)
        self._write_pos = 0
        self._read_pos = 0
        self._filled = 0
        self._lock = threading.Lock()

    def write(self, data: np.ndarray) -> int:
        """
        Write audio data to the buffer.

        Args:
            data: Audio samples as numpy array

        Returns:
            Number of samples actually written
        """
        with self._lock:
            data = data.flatten()
            available = self.max_samples - self._filled
            to_write = min(len(data), available)

            if to_write <= 0:
                return 0  # Buffer full

            # Wrap around if needed
            end_pos = self._write_pos + to_write
            if end_pos <= self.max_samples:
                self._buffer[self._write_pos : end_pos] = data[:to_write]
            else:
                # Wrap
                first_part = self.max_samples - self._write_pos
                self._buffer[self._write_pos :] = data[:first_part]
                self._buffer[: end_pos - self.max_samples] = data[first_part:to_write]

            self._write_pos = (self._write_pos + to_write) % self.max_samples
            self._filled += to_write
            return to_write

    def read_chunk(self, chunk_size: int) -> Optional[np.ndarray]:
        """
        Read a fixed-size chunk from the buffer.

        Args:
            chunk_size: Number of samples to read

        Returns:
            Audio chunk, or None if not enough data available
        """
        with self._lock:
            if self._filled < chunk_size:
                return None

            chunk = np.zeros(chunk_size, dtype=np.float32)
            end_pos = self._read_pos + chunk_size

            if end_pos <= self.max_samples:
                chunk[:] = self._buffer[self._read_pos : end_pos]
            else:
                # Wrap
                first_part = self.max_samples - self._read_pos
                chunk[:first_part] = self._buffer[self._read_pos :]
                chunk[first_part:] = self._buffer[: end_pos - self.max_samples]

            self._read_pos = (self._read_pos + chunk_size) % self.max_samples
            self._filled -= chunk_size
            return chunk

    def available(self) -> int:
        """Return number of samples available to read."""
        with self._lock:
            return self._filled

    def clear(self) -> None:
        """Clear all data from the buffer."""
        with self._lock:
            self._write_pos = 0
            self._read_pos = 0
            self._filled = 0
            self._buffer.fill(0)

    @property
    def capacity(self) -> int:
        return self.max_samples

    @property
    def fill_level(self) -> float:
        """Return fill level as a fraction (0.0 to 1.0)."""
        with self._lock:
            return self._filled / self.max_samples
