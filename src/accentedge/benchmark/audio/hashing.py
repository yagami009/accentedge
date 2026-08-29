"""Streaming SHA-256 hashing for audio files."""
import hashlib
from pathlib import Path


def sha256_file(path, chunk_size=8192):
    """Compute SHA-256 hash of a file without loading it entirely into memory."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    """Compute SHA-256 hash of a bytes object."""
    return hashlib.sha256(data).hexdigest()
