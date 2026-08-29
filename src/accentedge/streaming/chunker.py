"""Chunker for streaming audio input."""

from __future__ import annotations

import numpy as np


class Chunker:
    def __init__(
        self,
        chunk_size_ms: int,
        sample_rate: int,
        overlap_ms: int = 0,
    ) -> None:
        self.chunk_size_ms = chunk_size_ms
        self.sample_rate = sample_rate
        self.overlap_ms = overlap_ms

    @property
    def chunk_samples(self) -> int:
        return int(self.chunk_size_ms * self.sample_rate / 1000)

    @property
    def overlap_samples(self) -> int:
        return int(self.overlap_ms * self.sample_rate / 1000)

    def chunk(self, audio: np.ndarray) -> list[np.ndarray]:
        if audio.size == 0:
            return []
        cs = self.chunk_samples
        os = self.overlap_samples
        if cs <= 0:
            raise ValueError("chunk_size_ms must produce a positive chunk size")
        if os >= cs:
            raise ValueError("overlap_ms must be smaller than chunk_size_ms")
        chunks: list[np.ndarray] = []
        i = 0
        while i < len(audio):
            chunk = audio[i : i + cs]
            if chunk.size == 0:
                break
            chunks.append(chunk)
            i += cs - os
        return chunks
