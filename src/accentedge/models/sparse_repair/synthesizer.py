"""Sparse synthesizer: only reprocess flagged regions with local OLA resynthesis."""

from __future__ import annotations

from typing import Any

import numpy as np

from accentedge.models.sparse_repair.interfaces import RepairControls


class SparseSynthesizerImpl:
    """Local overlap-add resynthesizer for flagged regions.

    Only modifies the region specified by RepairControls.
    Unmodified regions are returned bit-exact.
    """

    def __init__(self, sr: int = 16000, frame_ms: float = 10.0) -> None:
        self.sr = sr
        self.frame_samples = max(1, int(sr * frame_ms / 1000))

    def repair(
        self,
        audio: np.ndarray,
        controls: RepairControls,
        region: slice,
    ) -> np.ndarray:
        """Apply local resynthesis to the flagged region.

        Args:
            audio: full original audio buffer, shape (T,)
            controls: repair parameters
            region: slice of audio to repair

        Returns:
            modified audio (same shape as input)
        """
        if controls.strength <= 0.0:
            # No repair requested — return original exactly
            return audio.copy()

        start = max(region.start, 0)
        end = min(region.stop, len(audio))

        if start >= end:
            return audio.copy()

        # Extract the region
        region_audio = audio[start:end].copy()

        # Simple local "repair": apply a filtered version
        # In production this would call an actual accent-conversion model;
        # here we use a light IIR-like filter + mix to simulate resynthesis.
        repaired = self._apply_local_resynthesis(region_audio, controls)

        # Apply fade-in / fade-out to avoid clicks
        repaired = self._apply_fades(repaired, controls.fade_samples)

        # Blend with original based on strength
        mixed = (1.0 - controls.strength) * region_audio + controls.strength * repaired

        # Build output: copy original, then patch the region
        output = audio.copy()
        output[start:end] = mixed
        return output

    def _apply_local_resynthesis(
        self, audio: np.ndarray, controls: RepairControls
    ) -> np.ndarray:
        """Simulate local resynthesis with a simple spectral transform.

        Uses a first-order IIR filter + small amount of noise to
        simulate the effect of accent conversion on the flagged region.
        """
        # Simple one-pole lowpass to simulate spectral change
        alpha = 0.3 + 0.4 * controls.strength  # more strength → more filtering
        filtered = np.zeros_like(audio)
        filtered[0] = audio[0]
        for i in range(1, len(audio)):
            filtered[i] = alpha * filtered[i - 1] + (1 - alpha) * audio[i]

        # Add subtle noise to simulate accent texture change
        noise = (np.random.rand(len(audio)) - 0.5) * 0.01 * controls.strength
        return filtered + noise

    def _apply_fades(self, audio: np.ndarray, fade_samples: int) -> np.ndarray:
        """Apply fade-in and fade-out to avoid clicks at boundaries."""
        N = min(fade_samples, len(audio) // 2)
        if N <= 0:
            return audio

        # Fade-in
        fade_in = np.linspace(0.0, 1.0, N)
        audio[:N] *= fade_in

        # Fade-out
        fade_out = np.linspace(1.0, 0.0, N)
        audio[-N:] *= fade_out

        return audio
