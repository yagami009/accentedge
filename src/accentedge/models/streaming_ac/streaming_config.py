"""Streaming AC candidate configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass
class StreamingACConfig:
    """Configuration for Streaming AC candidate."""

    right_context_ms: int = 640
    left_context_ms: int = 0
    chunk_ms: int = 80
    hidden_dim: int = 256
    num_layers: int = 4
    accent_dim: int = 64
    speaker_dim: int = 64
    mode: Literal["paper_style", "low_lookahead"] = "paper_style"
    conversion_strength: float = 0.5

    def __post_init__(self) -> None:
        if self.mode == "low_lookahead":
            paper_style = StreamingACConfig(mode="paper_style")
            if self.right_context_ms == paper_style.right_context_ms:
                object.__setattr__(self, "right_context_ms", 0)
            if self.hidden_dim == paper_style.hidden_dim:
                object.__setattr__(self, "hidden_dim", 128)
            if self.num_layers == paper_style.num_layers:
                object.__setattr__(self, "num_layers", 2)

    @staticmethod
    def _low_lookahead_defaults() -> dict[str, int | float | str]:
        return {
            "right_context_ms": 0,
            "hidden_dim": 128,
            "num_layers": 2,
        }

    def as_dict(self) -> dict[str, object]:
        return {
            "right_context_ms": self.right_context_ms,
            "left_context_ms": self.left_context_ms,
            "chunk_ms": self.chunk_ms,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "accent_dim": self.accent_dim,
            "speaker_dim": self.speaker_dim,
            "mode": self.mode,
            "conversion_strength": self.conversion_strength,
        }
