"""Candidate D — Minimal Hybrid package."""

from __future__ import annotations

from accentedge.models.minimal_hybrid.model import MinimalHybridCandidate
from accentedge.models.minimal_hybrid.streaming_config import MinimalHybridConfig
from accentedge.models.registry import get_registry

get_registry().register("minimal_hybrid", MinimalHybridCandidate)

__all__ = ["MinimalHybridCandidate", "MinimalHybridConfig"]
