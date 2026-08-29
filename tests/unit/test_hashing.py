"""Tests for accentedge_benchmark.audio.hashing — SHA-256 file hashing."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

from accentedge_benchmark.audio.hashing import sha256_file, sha256_bytes


class TestSha256File:
    def test_small_file(self, tmp_path):
        content = b"hello world"
        fpath = tmp_path / "test.txt"
        fpath.write_bytes(content)
        expected = hashlib.sha256(content).hexdigest()
        assert sha256_file(fpath) == expected

    def test_large_file(self, tmp_path):
        """Test with a file larger than the chunk size."""
        fpath = tmp_path / "large.bin"
        data = os.urandom(65536)
        fpath.write_bytes(data)
        expected = hashlib.sha256(data).hexdigest()
        assert sha256_file(fpath) == expected

    def test_empty_file(self, tmp_path):
        fpath = tmp_path / "empty.txt"
        fpath.write_bytes(b"")
        expected = hashlib.sha256(b"").hexdigest()
        assert sha256_file(fpath) == expected

    def test_chunk_size_respected(self, tmp_path):
        """Verify that chunk_size parameter is honored."""
        content = b"x" * 1000
        fpath = tmp_path / "chunked.txt"
        fpath.write_bytes(content)
        result = sha256_file(fpath, chunk_size=128)
        expected = hashlib.sha256(content).hexdigest()
        assert result == expected

    def test_deterministic(self, tmp_path):
        """Same file content → same hash."""
        content = b"deterministic test"
        fpath = tmp_path / "det.txt"
        fpath.write_bytes(content)
        h1 = sha256_file(fpath)
        h2 = sha256_file(fpath)
        assert h1 == h2

    def test_streaming_large_file(self, tmp_path):
        """Test streaming with a file much larger than the default chunk_size."""
        content = os.urandom(10_000_000)  # 10 MB
        fpath = tmp_path / "huge.bin"
        fpath.write_bytes(content)
        expected = hashlib.sha256(content).hexdigest()
        result = sha256_file(fpath, chunk_size=8192)
        assert result == expected


class TestSha256Bytes:
    def test_small_bytes(self):
        data = b"hello"
        expected = hashlib.sha256(data).hexdigest()
        assert sha256_bytes(data) == expected

    def test_empty_bytes(self):
        expected = hashlib.sha256(b"").hexdigest()
        assert sha256_bytes(b"") == expected

    def test_deterministic(self):
        data = b"test data"
        h1 = sha256_bytes(data)
        h2 = sha256_bytes(data)
        assert h1 == h2

    def test_returns_hex_string(self):
        result = sha256_bytes(b"data")
        assert isinstance(result, str)
        assert len(result) == 64

    def test_different_inputs_different_hashes(self):
        h1 = sha256_bytes(b"data1")
        h2 = sha256_bytes(b"data2")
        assert h1 != h2
