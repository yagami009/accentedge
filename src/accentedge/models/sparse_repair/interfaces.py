"""Sparse-repair specific protocols and data classes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np


# ---------------------------------------------------------------------------
# Decision
# ---------------------------------------------------------------------------

@dataclass
class DeviationDecision:
    """Output of the deviation detector for a single feature."""

    feature: str = ""
    confidence: float = 0.0
    start_time: float = 0.0          # seconds from session start
    estimated_end_time: float = 0.0  # seconds from session start
    conversion_strength: float = 0.0
    commit_time: float = 0.0         # seconds; decision becomes final after this
    needs_repair: bool = False


# ---------------------------------------------------------------------------
# Controls
# ---------------------------------------------------------------------------

@dataclass
class RepairControls:
    """Instructions for the synthesizer about what to change."""

    feature: str = ""
    strength: float = 0.0
    start_sample: int = 0
    end_sample: int = 0
    fade_samples: int = 256


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------

class StreamingDeviationDetector(Protocol):
    """Detect per-feature accent deviations in streaming audio frames."""

    def detect(
        self,
        features: np.ndarray,
        state: dict,
    ) -> DeviationDecision:
        """Run one step of detection.

        Args:
            features: frame-level features, shape (feature_dim,) or (1, feature_dim)
            state: mutable session state (must not grow unboundedly)

        Returns:
            DeviationDecision for this frame
        """
        ...


class RepairController(Protocol):
    """Convert detector decisions into synthesis control plans."""

    def plan(
        self,
        decision: DeviationDecision,
        context: dict,
    ) -> RepairControls:
        """Convert a deviation decision into synthesis controls.

        Args:
            decision: the detector output
            context: current session context (sample offsets, session time, …)

        Returns:
            RepairControls describing the repair region and parameters
        """
        ...


class SparseSynthesizer(Protocol):
    """Repair only the flagged regions of audio."""

    def repair(
        self,
        audio: np.ndarray,
        controls: RepairControls,
        region: slice,
    ) -> np.ndarray:
        """Apply local resynthesis to the flagged region.

        Args:
            audio: full original audio buffer
            controls: repair parameters
            region: slice of audio to repair

        Returns:
            modified audio (same shape as input)
        """
        ...
