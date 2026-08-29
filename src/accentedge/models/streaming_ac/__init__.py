"""Streaming AC candidate package."""

from __future__ import annotations

from accentedge.models.registry import get_registry
from accentedge.models.streaming_ac.streaming_ac import StreamingACCandidate
from accentedge.models.streaming_ac.streaming_config import StreamingACConfig

get_registry().register("streaming_ac", StreamingACCandidate)

__all__ = ["StreamingACCandidate", "StreamingACConfig"]
