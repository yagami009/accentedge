"""Candidate adapter registry."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .base import CandidateAdapter
from .file_output import FileOutputAdapter
from .passthrough import PassthroughAdapter

if TYPE_CHECKING:
    pass

_REGISTRY: dict[str, type[CandidateAdapter]] = {
    "passthrough": PassthroughAdapter,
    "file_output": FileOutputAdapter,
}


def register(name: str, adapter_cls: type[CandidateAdapter]) -> None:
    """Register a candidate adapter class."""
    _REGISTRY[name] = adapter_cls


def get(name: str) -> type[CandidateAdapter]:
    """Get a candidate adapter class by name."""
    if name not in _REGISTRY:
        raise KeyError(f"Unknown candidate: {name}. Available: {list(_REGISTRY)}")
    return _REGISTRY[name]


def available() -> list[str]:
    """List available candidate adapter names."""
    return list(_REGISTRY.keys())
