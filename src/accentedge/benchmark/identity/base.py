"""Speaker identity base classes."""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class SpeakerEmbeddingResult:
    embedding: np.ndarray
    model_name: str
    metadata: dict[str, Any] = field(default_factory=dict)


class SpeakerEmbedder(ABC):
    @abstractmethod
    def embed(self, audio: np.ndarray, sample_rate: int) -> np.ndarray: ...

    @abstractmethod
    def distance(self, a: np.ndarray, b: np.ndarray) -> float: ...

    @property
    @abstractmethod
    def name(self) -> str: ...
