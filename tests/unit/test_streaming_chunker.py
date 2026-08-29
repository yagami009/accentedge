"""Tests for the streaming chunker."""

from __future__ import annotations

import numpy as np
import pytest

from accentedge_lab.streaming.chunker import Chunker


class TestChunker:
    def test_20ms_chunks(self) -> None:
        sr = 16000
        c = Chunker(chunk_size_ms=20, sample_rate=sr)
        audio = np.zeros(800, dtype=np.float32)
        chunks = c.chunk(audio)
        # 20ms = 320 samples, 800 -> 3 chunks (0-320, 320-640, 640-800)
        assert len(chunks) == 3
        assert chunks[0].shape[0] == 320

    def test_80ms_chunks(self) -> None:
        sr = 16000
        c = Chunker(chunk_size_ms=80, sample_rate=sr)
        audio = np.zeros(1600, dtype=np.float32)
        chunks = c.chunk(audio)
        # 80ms = 1280 samples, 1600 -> 2 chunks
        assert len(chunks) == 2
        assert chunks[0].shape[0] == 1280

    def test_overlap(self) -> None:
        sr = 16000
        c = Chunker(chunk_size_ms=40, sample_rate=sr, overlap_ms=20)
        audio = np.zeros(2000, dtype=np.float32)
        chunks = c.chunk(audio)
        # 40ms=640, overlap=20ms=320, step=320 -> 7 chunks for 2000 samples
        assert len(chunks) == 7

    def test_empty_audio(self) -> None:
        c = Chunker(chunk_size_ms=80, sample_rate=16000)
        assert c.chunk(np.zeros(0, dtype=np.float32)) == []
