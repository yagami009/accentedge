"""Articulatory/DDSP candidate — Candidate B for Phase 2 bake-off."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from accentedge.models.articulatory_ddsp.ddsp_synth import DDSPSynthesizer
from accentedge.models.articulatory_ddsp.encoder import ArticulatoryEncoder as EncoderModule
from accentedge.models.articulatory_ddsp.interfaces import (
    ArticulatoryAccentMapper,
    ArticulatoryEncoder,
    ArticulatoryFrameSequence,
)
from accentedge.models.articulatory_ddsp.mapper import ArticulatoryAccentMapper as MapperModule
from accentedge.models.articulatory_ddsp.streaming_config import ArticulatoryStreamingConfig
from accentedge.models.interfaces import (
    CandidateMetadata,
    StreamingCandidate,
    StreamingResult,
    StreamingSession,
)


@dataclass
class ArticulatoryDDSPSession:
    """Session state for Candidate B."""

    session_id: str
    encoder_state: dict = field(default_factory=dict)
    mapper_state: dict = field(default_factory=dict)
    synth_state: dict = field(default_factory=dict)
    timeline: list[tuple[int, int, int, int]] = field(default_factory=list)
    samples_processed: int = 0


class ArticulatoryDDSPCandidate:
    """Articulatory/DDSP candidate for architecture bake-off.

    Hypothesis: accent is controllable in articulatory/phonetic parameter
    space, enabling low-latency synthesis without a large neural decoder.
    """

    metadata = CandidateMetadata(
        architecture_id="articulatory_ddsp",
        version="0.1.0",
        input_sample_rate=16000,
        frame_ms=10.0,
        preferred_chunk_ms=40,
        required_lookahead_ms=0,
        left_context_ms=0,
        supports_conversion_strength=True,
        supports_target_accent=True,
        requires_reference_speaker=False,
        uses_text_at_inference=False,
    )

    def __init__(self, config: ArticulatoryStreamingConfig | None = None) -> None:
        self.config = config or ArticulatoryStreamingConfig()
        self._device: str = "cpu"
        self._precision: str = "fp32"
        self._closed = False
        self._build()

    def _build(self) -> None:
        self.encoder: ArticulatoryEncoder = EncoderModule(config=self.config)
        self.mapper: ArticulatoryAccentMapper = MapperModule(config=self.config)
        self.synthesizer = DDSPSynthesizer(config=self.config)

    def prepare(self, device: str, precision: str) -> None:
        """Move model to target device/precision."""
        if self._closed:
            raise RuntimeError("candidate is closed")
        self._device = device
        self._precision = precision
        modules: list[nn.Module] = [self.encoder, self.mapper, self.synthesizer]
        for m in modules:
            m = m.to(torch.device(device))
            if precision == "fp16" and device != "cpu":
                m = m.half()
            elif precision == "bf16" and device != "cpu":
                m = m.bfloat16()

    def create_session(self, config: dict[str, Any]) -> StreamingSession:
        """Create a new streaming session."""
        if self._closed:
            raise RuntimeError("candidate is closed")
        session_id = config.get("session_id", "articulatory_ddsp")
        b_state = ArticulatoryDDSPSession(session_id=session_id)
        return StreamingSession(
            session_id=session_id,
            state={"b": b_state},
            created_at=datetime.utcnow(),
            samples_processed=0,
        )

    def process_chunk(
        self,
        session: StreamingSession,
        audio_chunk: np.ndarray,
        sample_rate: int,
    ) -> StreamingResult:
        """Run encode -> map -> synthesize on a single chunk."""
        if self._closed:
            raise RuntimeError("candidate is closed")
        b_state: ArticulatoryDDSPSession = session.state["b"]

        input_start = session.samples_processed
        input_end = input_start + len(audio_chunk)

        # Convert to torch tensor: (B, T).
        x = torch.from_numpy(audio_chunk).float()
        if x.dim() == 1:
            x = x.unsqueeze(0)
        elif x.dim() == 2:
            x = x.T
        x = x.to(torch.device(self._device))

        # Encode.
        frames = self.encoder.encode(x, b_state.encoder_state)

        # Map.
        B = x.shape[0]
        target_accent = torch.zeros(B, device=self._device, dtype=torch.long)
        strength = self.config.conversion_strength
        mapped = self.mapper.map(frames, target_accent, strength)

        # Synthesize.
        speaker = torch.zeros(B, self.config.accent_dim, device=self._device)
        audio = self.synthesizer.synthesize(mapped, speaker)
        audio_np = audio.detach().cpu().numpy()
        if audio_np.shape[0] == 1:
            audio_np = audio_np.squeeze(0)

        output_len = audio_np.shape[-1]
        output_start = input_start
        output_end = output_start + output_len

        b_state.timeline.append((input_start, input_end, output_start, output_end))
        b_state.samples_processed = input_end
        session.samples_processed = input_end

        return StreamingResult(
            audio=audio_np,
            sample_rate=sample_rate,
            input_start_sample=input_start,
            input_end_sample=input_end,
            output_start_sample=output_start,
            output_end_sample=output_end,
            algorithmic_delay_samples=0,
            metadata={"conversion_strength": strength},
        )

    def flush(self, session: StreamingSession) -> list[StreamingResult]:
        """Flush any buffered outputs."""
        return []

    def reset(self, session: StreamingSession) -> None:
        """Reset session state."""
        b_state: ArticulatoryDDSPSession = session.state.get("b")
        if b_state is not None:
            b_state.encoder_state = {}
            b_state.mapper_state = {}
            b_state.synth_state = {}
            b_state.timeline = []
            b_state.samples_processed = 0
        session.samples_processed = 0

    def close(self) -> None:
        """Cleanup resources."""
        if self._closed:
            return
        del self.encoder
        del self.mapper
        del self.synthesizer
        self._closed = True

