"""Streaming AC candidate — architecture bake-off Candidate A."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from accentedge.models.interfaces import (
    CandidateMetadata,
    StreamingCandidate,
    StreamingResult,
    StreamingSession,
)
from accentedge.models.streaming_ac.interfaces import (
    AccentLatent,
    AccentBottleneck,
    AudioChunk,
    ContentFrame,
    ContentProsodyEncoder,
    CausalOrChunkedSynthesizer,
    SpeakerEncoder,
    SpeakerEmbedding,
)
from accentedge.models.streaming_ac.low_lookahead import (
    LowLookaheadAccentBottleneck,
    LowLookaheadCausalOrChunkedSynthesizer,
    LowLookaheadContentProsodyEncoder,
    LowLookaheadSpeakerEncoder,
)
from accentedge.models.streaming_ac.paper_style import (
    AccentBottleneck as PaperStyleAccentBottleneck,
    CausalOrChunkedSynthesizer as PaperStyleCausalOrChunkedSynthesizer,
    ContentProsodyEncoder as PaperStyleContentProsodyEncoder,
    SpeakerEncoder as PaperStyleSpeakerEncoder,
)
from accentedge.models.streaming_ac.state import StreamingACSession as ACState
from accentedge.models.streaming_ac.streaming_config import StreamingACConfig


class StreamingACCandidate:
    """Streaming accent-conversion candidate (Candidate A).

    Paper-style baseline with configurable lookahead and low-lookahead mode.
    """

    metadata = CandidateMetadata(
        architecture_id="streaming_ac",
        version="0.1.0",
        input_sample_rate=16000,
        frame_ms=20.0,
        preferred_chunk_ms=80,
        required_lookahead_ms=640,
        left_context_ms=0,
        supports_conversion_strength=True,
        supports_target_accent=True,
        requires_reference_speaker=False,
        uses_text_at_inference=False,
    )

    def __init__(self, config: StreamingACConfig | None = None) -> None:
        self.config = config or StreamingACConfig()
        self.metadata.required_lookahead_ms = self.config.right_context_ms
        self._device: str = "cpu"
        self._precision: str = "fp32"
        self._closed = False
        self._build()

    def _build(self) -> None:
        mode = self.config.mode
        if mode == "paper_style":
            enc_cls = PaperStyleContentProsodyEncoder
            bot_cls = PaperStyleAccentBottleneck
            spk_cls = PaperStyleSpeakerEncoder
            syn_cls = PaperStyleCausalOrChunkedSynthesizer
        elif mode == "low_lookahead":
            enc_cls = LowLookaheadContentProsodyEncoder
            bot_cls = LowLookaheadAccentBottleneck
            spk_cls = LowLookaheadSpeakerEncoder
            syn_cls = LowLookaheadCausalOrChunkedSynthesizer
        else:
            raise ValueError(f"Unknown mode: {mode}")

        self.encoder = enc_cls(
            hidden_dim=self.config.hidden_dim,
            num_layers=self.config.num_layers,
            right_context_ms=self.config.right_context_ms,
            left_context_ms=self.config.left_context_ms,
        )
        self.bottleneck = bot_cls(
            content_dim=self.config.hidden_dim,
            accent_dim=self.config.accent_dim,
        )
        self.speaker_encoder = spk_cls(
            hidden_dim=self.config.hidden_dim,
            speaker_dim=self.config.speaker_dim,
        )
        self.synthesizer = syn_cls(
            latent_dim=self.config.accent_dim,
            speaker_dim=self.config.speaker_dim,
            hidden_dim=self.config.hidden_dim,
        )

    def prepare(self, device: str, precision: str) -> None:
        """Move model to the target device and precision."""
        if self._closed:
            raise RuntimeError("candidate is closed")
        self._device = device
        self._precision = precision
        modules = [self.encoder, self.bottleneck, self.speaker_encoder, self.synthesizer]
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
        session_id = config.get("session_id", "streaming_ac")
        ac_state = ACState(session_id=session_id)
        return StreamingSession(
            session_id=session_id,
            state={"ac": ac_state},
            created_at=datetime.utcnow(),
            samples_processed=0,
        )

    def process_chunk(
        self,
        session: StreamingSession,
        audio_chunk: np.ndarray,
        sample_rate: int,
    ) -> StreamingResult:
        """Run forward pass on a single audio chunk."""
        if self._closed:
            raise RuntimeError("candidate is closed")
        ac_state: ACState = session.state["ac"]

        input_start = session.samples_processed
        input_end = input_start + len(audio_chunk)

        # Convert numpy audio to torch tensor of shape [B, T, input_dim].
        x = torch.from_numpy(audio_chunk).float()
        if x.dim() == 1:
            input_dim = getattr(self.encoder, "input_dim", 1)
            remainder = x.numel() % input_dim
            if remainder != 0:
                x = x[: x.numel() - remainder]
            x = x.reshape(1, -1, input_dim)
        elif x.dim() == 2:
            x = x.unsqueeze(0)
        x = x.to(torch.device(self._device))

        # Encode content.
        content = self.encoder.encode(x, state=None).features

        # Speaker encoding.
        speaker_emb = self.speaker_encoder.encode(x, state=None)
        ac_state.speaker_state["embedding"] = (
            speaker_emb.embedding.detach().cpu().numpy()
        )
        ac_state.speaker_state["confidence"] = float(speaker_emb.confidence)
        ac_state.speaker_state["updated_at_sample"] = input_end

        # Accent bottleneck.
        accent = self.bottleneck.map(content, speaker_emb.embedding)

        # Dummy f0 and energy.
        B, T, _ = content.shape
        f0 = torch.zeros(B, T, device=content.device)
        energy = torch.zeros(B, T, device=content.device)

        audio = self.synthesizer.synthesize(
            accent.latent, speaker_emb.embedding, f0, energy
        )
        audio_np = audio.audio.squeeze(1).detach().cpu().numpy()

        output_len = audio_np.shape[-1]
        output_start = max(0, input_start - self.config.right_context_ms * sample_rate // 1000)
        output_end = output_start + output_len

        ac_state.decoder_state["buffer"].append(audio_np)
        ac_state.decoder_state["step"] += 1
        ac_state.step += 1
        ac_state.timeline.append((input_start, input_end, output_start, output_end))
        session.samples_processed = input_end

        return StreamingResult(
            audio=audio_np,
            sample_rate=sample_rate,
            input_start_sample=input_start,
            input_end_sample=input_end,
            output_start_sample=output_start,
            output_end_sample=output_end,
            algorithmic_delay_samples=self.config.right_context_ms
            * sample_rate
            // 1000,
            metadata={"conversion_strength": self.config.conversion_strength},
        )

    def flush(self, session: StreamingSession) -> list[StreamingResult]:
        """Flush buffered outputs."""
        ac_state: ACState = session.state["ac"]
        results: list[StreamingResult] = []
        while ac_state.decoder_state.get("buffer"):
            chunk = ac_state.decoder_state["buffer"].pop(0)
            results.append(
                StreamingResult(
                    audio=chunk,
                    sample_rate=self.metadata.input_sample_rate,
                    input_start_sample=0,
                    input_end_sample=0,
                    output_start_sample=0,
                    output_end_sample=0,
                )
            )
        return results

    def reset(self, session: StreamingSession) -> None:
        """Clear all session state."""
        ac_state: ACState = session.state.get("ac")
        if ac_state is not None:
            ac_state.encoder_cache = None
            ac_state.speaker_state = {
                "embedding": None,
                "confidence": 0.0,
                "updated_at_sample": 0,
            }
            ac_state.decoder_state = {"buffer": [], "step": 0}
            ac_state.timeline = []
            ac_state.step = 0
        session.samples_processed = 0

    def close(self) -> None:
        """Cleanup resources."""
        if self._closed:
            return
        del self.encoder
        del self.bottleneck
        del self.speaker_encoder
        del self.synthesizer
        self._closed = True

