"""Token Translation Candidate C — main candidate class.

Hypothesis: mapping structured content representations (phonetic/acoustic
tokens) is easier for accent transformation than direct waveform regression.
Inspired by PHONOS (40ms lookahead, 241ms GPU latency).
"""

from __future__ import annotations

from dataclasses import dataclass, field
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
from accentedge.models.token_translation.interfaces import (
    TokenSequence,
)
from accentedge.models.token_translation.streaming_config import TokenTranslationConfig
from accentedge.models.token_translation.synthesizer import TokenConditionedSynthesizer
from accentedge.models.token_translation.tokenizer import CausalSpeechTokenizer
from accentedge.models.token_translation.translator import AccentTokenTranslator


@dataclass
class TokenTranslationSession:
    """Session state for Candidate C."""

    session_id: str
    tokenizer_state: dict[str, Any] = field(default_factory=dict)
    translator_state: dict[str, Any] = field(default_factory=dict)
    synth_state: dict[str, Any] = field(default_factory=dict)
    timeline: list[tuple[int, int, int, int]] = field(default_factory=list)
    samples_processed: int = 0

    def state_size_bytes(self) -> int:
        total = 0
        for v in self.tokenizer_state.values():
            if isinstance(v, np.ndarray):
                total += v.nbytes
        for v in self.translator_state.values():
            if isinstance(v, np.ndarray):
                total += v.nbytes
        for v in self.synth_state.values():
            if isinstance(v, np.ndarray):
                total += v.nbytes
        return total


class TokenTranslationCandidate:
    """Candidate C: Causal token translation for accent transformation.

    Architecture:
        tokenize -> translate -> synthesize

    - tokenize: raw audio -> continuous soft token embeddings (causal Conv1D)
    - translate: source token sequence -> target-accent token sequence (LSTM + FiLM)
    - synthesize: target token sequence -> waveform (transposed conv + speaker FiLM)
    """

    metadata = CandidateMetadata(
        architecture_id="token_translation",
        version="0.1.0",
        input_sample_rate=16000,
        frame_ms=20.0,
        preferred_chunk_ms=80,
        required_lookahead_ms=0,
        left_context_ms=0,
        supports_conversion_strength=True,
        supports_target_accent=True,
        requires_reference_speaker=False,
        uses_text_at_inference=False,
    )

    def __init__(self, config: TokenTranslationConfig | None = None) -> None:
        self.config = config or TokenTranslationConfig()
        self.metadata.required_lookahead_ms = int(self.config.lookahead_ms)
        self.metadata.frame_ms = self.config.frame_ms
        self._device: str = "cpu"
        self._precision: str = "fp32"
        self._closed = False
        self._build()

    def _build(self) -> None:
        cfg = self.config
        self.tokenizer = CausalSpeechTokenizer(
            token_rate_hz=cfg.token_rate_hz,
            token_dim=cfg.token_dim,
            hidden_dim=cfg.translator_hidden,
        )
        self.translator = AccentTokenTranslator(
            token_dim=cfg.token_dim,
            translator_layers=cfg.translator_layers,
            translator_hidden=cfg.translator_hidden,
            num_accents=cfg.num_accents,
            accent_dim=cfg.accent_dim,
            lookahead_frames=cfg.lookahead_frames,
        )
        self.synthesizer = TokenConditionedSynthesizer(
            token_dim=cfg.token_dim,
            speaker_dim=cfg.speaker_dim,
            hidden_dim=min(128, cfg.translator_hidden),
            hop_length=max(1, cfg.token_rate_hz // 50),
        )

    def prepare(self, device: str, precision: str) -> None:
        """Move model to target device/precision."""
        if self._closed:
            raise RuntimeError("candidate is closed")
        self._device = device
        self._precision = precision
        modules: list[nn.Module] = [self.tokenizer, self.translator, self.synthesizer]
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
        session_id = config.get("session_id", "token_translation")
        tt_state = TokenTranslationSession(session_id=session_id)
        return StreamingSession(
            session_id=session_id,
            state={"c": tt_state},
            created_at=datetime.utcnow(),
            samples_processed=0,
        )

    def process_chunk(
        self,
        session: StreamingSession,
        audio_chunk: np.ndarray,
        sample_rate: int,
    ) -> StreamingResult:
        """Run encode -> translate -> synthesize on a single chunk."""
        if self._closed:
            raise RuntimeError("candidate is closed")
        tt_state: TokenTranslationSession = session.state["c"]

        input_start = session.samples_processed
        input_end = input_start + len(audio_chunk)

        # Convert to torch tensor: (1, T).
        x = torch.from_numpy(audio_chunk).float()
        if x.dim() == 1:
            x = x.unsqueeze(0)
        elif x.dim() == 2:
            x = x.T
        x = x.to(torch.device(self._device))

        # 1. Tokenize.
        token_seq = self.tokenizer.tokenize(x, tt_state.tokenizer_state or None)
        tt_state.tokenizer_state = self.tokenizer.state if hasattr(self.tokenizer, "state") else {}

        # 2. Translate.
        target_accent = torch.zeros(1, device=self._device, dtype=torch.long)
        strength = self.config.conversion_strength
        translator_ctx = tt_state.translator_state
        translated = self.translator.translate(token_seq, target_accent, strength, translator_ctx)
        tt_state.translator_state = translator_ctx

        # 3. Synthesize.
        audio_tensor = self.synthesizer.synthesize(translated, speaker_conditioning=None)
        audio_np = audio_tensor.squeeze(0).detach().cpu().numpy()

        output_len = audio_np.shape[-1]
        output_start = input_start
        output_end = output_start + output_len

        tt_state.timeline.append((input_start, input_end, output_start, output_end))
        tt_state.samples_processed = input_end
        session.samples_processed = input_end

        return StreamingResult(
            audio=audio_np,
            sample_rate=sample_rate,
            input_start_sample=input_start,
            input_end_sample=input_end,
            output_start_sample=output_start,
            output_end_sample=output_end,
            algorithmic_delay_samples=self.metadata.required_lookahead_ms * sample_rate // 1000,
            metadata={
                "conversion_strength": strength,
                "token_rate_hz": self.config.token_rate_hz,
                "lookahead_frames": self.config.lookahead_frames,
            },
        )

    def flush(self, session: StreamingSession) -> list[StreamingResult]:
        """Flush any buffered outputs."""
        return []

    def reset(self, session: StreamingSession) -> None:
        """Reset session state."""
        tt_state: TokenTranslationSession | None = session.state.get("c")
        if tt_state is not None:
            tt_state.tokenizer_state = {}
            tt_state.translator_state = {}
            tt_state.synth_state = {}
            tt_state.timeline = []
            tt_state.samples_processed = 0
        session.samples_processed = 0

    def count_parameters(self) -> int:
        """Return total trainable parameter count."""
        total = 0
        for attr in ("tokenizer", "translator", "synthesizer"):
            mod = getattr(self, attr, None)
            if isinstance(mod, nn.Module):
                total += sum(p.numel() for p in mod.parameters() if p.requires_grad)
        return total

    def close(self) -> None:
        """Cleanup resources."""
        if self._closed:
            return
        del self.tokenizer
        del self.translator
        del self.synthesizer
        self._closed = True

