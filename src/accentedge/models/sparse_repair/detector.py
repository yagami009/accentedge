"""Lightweight causal deviation detector for sparse-repair streaming."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from accentedge.models.sparse_repair.interfaces import DeviationDecision


# ---------------------------------------------------------------------------
# Detector model: single-layer linear classifier over frame features
# ---------------------------------------------------------------------------

class _FrameDeviationClassifier(nn.Module):
    """1-layer linear classifier for per-frame deviation detection.

    Input:  (B, feature_dim) — frame-level features
    Output: (B, 1) — deviation confidence in [0, 1]
    """

    def __init__(self, feature_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(feature_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, feature_dim) → (B, 1)
        return torch.sigmoid(self.linear(x))


# ---------------------------------------------------------------------------
# StreamingDeviationDetector implementation
# ---------------------------------------------------------------------------

class StreamingDeviationDetectorImpl:
    """Causal frame-level deviation detector.

    Operates on one frame at a time, maintains bounded internal state.
    """

    def __init__(
        self,
        feature_dim: int = 2,
        hidden_dim: int = 8,
        detection_threshold: float = 0.5,
        device: str = "cpu",
    ) -> None:
        self.model = _FrameDeviationClassifier(feature_dim, hidden_dim)
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.detection_threshold = detection_threshold
        self._device = device
        self._closed = False

    def prepare(self, device: str, precision: str) -> None:
        """Move model to target device and precision."""
        if self._closed:
            raise RuntimeError("detector is closed")
        self._device = device
        self.model = self.model.to(torch.device(device))
        if precision == "fp16" and device != "cpu":
            self.model = self.model.half()
        elif precision == "bf16" and device != "cpu":
            self.model = self.model.bfloat16()

    @property
    def metadata(self) -> type:
        """Metadata placeholder — not used directly."""
        return None

    def detect(
        self,
        features: np.ndarray,
        state: dict[str, Any],
    ) -> DeviationDecision:
        """Detect deviation for a single frame.

        Args:
            features: frame features, shape (feature_dim,) or (1, feature_dim)
            state: mutable dict; must grow O(1) per call

        Returns:
            DeviationDecision
        """
        if self._closed:
            raise RuntimeError("detector is closed")

        # Bounded state increment: only track frame count
        state.setdefault("frame_count", 0)
        state["frame_count"] += 1

        # Ensure features is a 2-D array of shape (1, feature_dim)
        feats = np.asarray(features, dtype=np.float32).reshape(-1)
        if feats.shape[-1] != self.feature_dim:
            # Zero-pad or truncate to match expected dim
            padded = np.zeros(self.feature_dim, dtype=np.float32)
            padded[: min(feats.shape[-1], self.feature_dim)] = feats[: self.feature_dim]
            feats = padded

        x = torch.from_numpy(feats).unsqueeze(0)  # (1, feature_dim)
        x = x.to(torch.device(self._device))

        self.model.eval()
        with torch.no_grad():
            confidence = float(self.model(x).item())

        session_time = float(state["frame_count"]) * 0.01  # 10 ms per frame
        commit_time = session_time + 0.05  # 50 ms commit horizon

        return DeviationDecision(
            feature="frame",
            confidence=confidence,
            start_time=session_time,
            estimated_end_time=session_time + 0.01,
            conversion_strength=0.0,
            commit_time=commit_time,
            needs_repair=confidence >= self.detection_threshold,
        )

    def reset(self, state: dict[str, Any]) -> None:
        """Reset bounded state."""
        state.clear()

    def close(self) -> None:
        """Cleanup resources."""
        if self._closed:
            return
        del self.model
        self._closed = True

