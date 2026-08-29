"""Speaker evaluator registry."""
from __future__ import annotations

from .base import SpeakerEmbedder


class IdentityRegistry:
    def __init__(self):
        self._evaluators: dict[str, SpeakerEmbedder] = {}

    def register(self, name: str, evaluator: SpeakerEmbedder) -> None:
        self._evaluators[name] = evaluator

    def get(self, name: str) -> SpeakerEmbedder | None:
        return self._evaluators.get(name)

    @property
    def names(self) -> list[str]:
        return list(self._evaluators.keys())
