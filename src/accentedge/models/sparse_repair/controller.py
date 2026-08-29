"""Repair controller: converts detector decisions into synthesis controls."""

from __future__ import annotations

from typing import Any

import numpy as np

from accentedge.models.sparse_repair.interfaces import (
    DeviationDecision,
    RepairControls,
)


class RepairControllerImpl:
    """Convert detector decisions into synthesis control plans.

    Handles:
    - commit_time gating (decision becomes final after commit_time)
    - Fade-in/fade-out to avoid clicks at repair boundaries
    - Min-repair-duration enforcement
    """

    def __init__(
        self,
        sr: int = 16000,
        min_repair_duration_ms: int = 50,
        fade_samples: int = 256,
        conversion_strength: float = 0.7,
    ) -> None:
        self.sr = sr
        self.min_repair_duration_ms = min_repair_duration_ms
        self.min_repair_samples = max(1, int(sr * min_repair_duration_ms / 1000))
        self.fade_samples = fade_samples
        self.conversion_strength = conversion_strength

    def plan(
        self,
        decision: DeviationDecision,
        context: dict[str, Any],
    ) -> RepairControls:
        """Convert a deviation decision into synthesis controls.

        Args:
            decision: the detector output
            context: must contain:
                - 'current_sample': int — current sample position in audio
                - 'current_time': float — current time in seconds

        Returns:
            RepairControls describing the repair region and parameters
        """
        current_sample = int(context.get("current_sample", 0))
        current_time = float(context.get("current_time", 0.0))

        # Gate on commit_time: decision only becomes actionable after commit_time
        if current_time < decision.commit_time:
            return RepairControls(
                feature=decision.feature,
                strength=0.0,
                start_sample=current_sample,
                end_sample=current_sample,
                fade_samples=self.fade_samples,
            )

        if not decision.needs_repair:
            return RepairControls(
                feature=decision.feature,
                strength=0.0,
                start_sample=current_sample,
                end_sample=current_sample,
                fade_samples=self.fade_samples,
            )

        # Enforce minimum repair duration
        start_sample = current_sample
        end_sample = max(
            current_sample + self.min_repair_samples,
            int(decision.estimated_end_time * self.sr),
        )

        return RepairControls(
            feature=decision.feature,
            strength=decision.conversion_strength
            if decision.conversion_strength > 0
            else self.conversion_strength,
            start_sample=start_sample,
            end_sample=end_sample,
            fade_samples=self.fade_samples,
        )
