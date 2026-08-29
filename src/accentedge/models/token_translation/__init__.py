"""Token Translation candidate package (Candidate C)."""

from __future__ import annotations

from accentedge.models.registry import get_registry
from accentedge.models.token_translation.token_translation_candidate import (
    TokenTranslationCandidate,
)
from accentedge.models.token_translation.streaming_config import TokenTranslationConfig

get_registry().register("token_translation", TokenTranslationCandidate)

__all__ = ["TokenTranslationCandidate", "TokenTranslationConfig"]
