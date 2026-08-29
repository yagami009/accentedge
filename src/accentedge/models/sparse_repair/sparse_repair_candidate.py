"""SparseRepair candidate — detector → repair streaming architecture.

Architecture: streaming_deviation_detector → repair_controller → sparse_synthesizer
Only reprocesses flagged regions instead of full S2S transformation.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
import torch

from accentedge.models.interfaces import (
    CandidateMetadata,
    StreamingCandidate,
    StreamingResult,
    StreamingSession,
)
from accentedge.models.sparse_repair.controller import RepairControllerImpl
from accentedge.models.sparse_repair.detector import StreamingDeviationDetectorImpl
from accentedge.models.sparse_repair.interfaces import (
    DeviationDecision,
    RepairControls,
)
from accentedge.models.sparse_repair.streaming_config import SparseRepairConfig
from accentedge.models.sparse_repair.synthesizer import SparseSynthesizerImpl


# ---------------------------------------------------------------------------
# SparseRepairCandidate
# ---------------------------------------------------------------------------

class SparseRepairCandidate:
    """Sparse-repair streaming candidate.

    Implements detector → controller → synthesizer pipeline:
    1. Lightweight causal detector identifies deviation frames
    2. Controller converts decisions to repair controls
    3. Sparse synthesizer reprocesses only flagged regions

    Only activates if Phase 0 returned SPARSE_REPAIR.
    """

    metadata = CandidateMetadata(
        architecture_id="sparse_repair",
        version="0.1.0",
        input_sample_rate=16000,
        frame_ms=10.0,
        preferred_chunk_ms=80,
        required_lookahead_ms=0,
        left_context_ms=0,
        supports_conversion_strength=True,
        supports_target_accent=False,
        requires_reference_speaker=False,
        uses_text_at_inference=False,
    )

    def __init__(self, config: SparseRepairConfig | None = None) -> None:
        self.config = config or SparseRepairConfig()
        self._device: str = "cpu"
        self._precision: str = "fp32"
        self._closed = False
        self._build()

    def _build(self) -> None:
        cfg = self.config
        self.detector = StreamingDeviationDetectorImpl(
            feature_dim=cfg.feature_dim,
            hidden_dim=cfg.hidden_dim,
            detection_threshold=cfg.detection_threshold,
            device=self._device,
        )
        self.controller = RepairControllerImpl(
            sr=cfg.sr,
            min_repair_duration_ms=cfg.min_repair_duration_ms,
            fade_samples=cfg.fade_samples,
            conversion_strength=cfg.conversion_strength,
        )
        self.synthesizer = SparseSynthesizerImpl(
            sr=cfg.sr,
            frame_ms=cfg.frame_ms,
        )

    # -- StreamingCandidate protocol ----------------------------------------

    def prepare(self, device: str, precision: str) -> None:
        """Move models to target device / precision."""
        if self._closed:
            raise RuntimeError("candidate is closed")
        self._device = device
        self._precision = precision
        self.detector.prepare(device, precision)

    def create_session(self, config: dict[str, Any]) -> StreamingSession:
        """Create a new streaming session."""
        if self._closed:
            raise RuntimeError("candidate is closed")
        session_id = config.get("session_id", "sparse_repair")
        internal_state: dict[str, Any] = {
            "detector_state": {"frame_count": 0},
            "pending_repairs": [],  # list of DeviationDecision
            "total_samples": 0,
            "current_time": 0.0,
            "step": 0,
        }
        return StreamingSession(
            session_id=session_id,
            state=internal_state,
            created_at=datetime.utcnow(),
            samples_processed=0,
        )

    def process_chunk(
        self,
        session: StreamingSession,
        audio_chunk: np.ndarray,
        sample_rate: int,
    ) -> StreamingResult:
        """Process one streaming chunk: detect → plan → repair committed decisions."""
        if self._closed:
            raise RuntimeError("candidate is closed")

        state = session.state
        detector_state = state["detector_state"]
        pending = state["pending_repairs"]
        total_samples = state["total_samples"]
        current_time = state["current_time"]

        input_start = session.samples_processed
        input_end = input_start + len(audio_chunk)

        # --- Step 1: commit any pending repairs whose commit_time has passed ---
        repaired_audio = np.array(audio_chunk, dtype=np.float32)
        frame_samples = self.config.frame_samples
        frame_duration = self.config.frame_ms / 1000.0

        # Check pending repairs and apply committed ones
        pending = state["pending_repairs"]
        still_pending = []
        for decision in pending:
            if current_time >= decision.commit_time and decision.needs_repair:
                controls = self.controller.plan(decision, {
                    "current_sample": total_samples,
                    "current_time": current_time,
                })
                if controls.strength > 0.0 and controls.end_sample > controls.start_sample:
                    region = slice(
                        controls.start_sample - total_samples,
                        controls.end_sample - total_samples,
                    )
                    repaired_audio = self.synthesizer.repair(
                        repaired_audio, controls, region
                    )
            else:
                still_pending.append(decision)
        state["pending_repairs"] = still_pending

        # --- Step 2: extract frame-level features and run detector ------------
        pending = state["pending_repairs"]  # re-fetch after Step 1 replacement
        n_frames = max(1, len(audio_chunk) // frame_samples)
        for fi in range(n_frames):
            frame_start = fi * frame_samples
            frame_end = min(frame_start + frame_samples, len(audio_chunk))
            frame_data = audio_chunk[frame_start:frame_end]

            # Simple frame feature: mean + std (2 features)
            features = np.array(
                [np.mean(frame_data), np.std(frame_data)], dtype=np.float32
            )

            decision = self.detector.detect(features, detector_state)
            # Set conversion_strength from config
            decision.conversion_strength = self.config.conversion_strength
            pending.append(decision)

            current_time += frame_duration

        # Update session state
        state["current_time"] = current_time
        state["total_samples"] = total_samples + len(audio_chunk)
        state["step"] += 1
        session.samples_processed = input_end

        output_start = input_start
        output_end = input_start + len(repaired_audio)

        return StreamingResult(
            audio=repaired_audio,
            sample_rate=sample_rate,
            input_start_sample=input_start,
            input_end_sample=input_end,
            output_start_sample=output_start,
            output_end_sample=output_end,
            algorithmic_delay_samples=0,
            metadata={"conversion_strength": self.config.conversion_strength},
        )

    def flush(self, session: StreamingSession) -> list[StreamingResult]:
        """Flush: commit any remaining repairs at session end."""
        state = session.state
        pending = state["pending_repairs"]
        total_samples = state["total_samples"]
        current_time = state["current_time"]

        # Force-commit all remaining repairs
        for decision in pending:
            if decision.needs_repair:
                controls = self.controller.plan(decision, {
                    "current_sample": total_samples,
                    "current_time": current_time,
                })
                # At flush, we just record — actual audio was already output
        state["pending_repairs"] = []
        return []

    def reset(self, session: StreamingSession) -> None:
        """Clear all session state."""
        session.state = {
            "detector_state": {"frame_count": 0},
            "pending_repairs": [],
            "total_samples": 0,
            "current_time": 0.0,
            "step": 0,
        }
        session.samples_processed = 0

    def close(self) -> None:
        """Cleanup resources."""
        if self._closed:
            return
        del self.detector
        del self.controller
        del self.synthesizer
        self._closed = True

