"""Content evaluation (WER, CER)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import jiwer


@dataclass
class ContentResult:
    """Result of content evaluation for a single utterance."""

    reference_text: str
    recognized_text: str
    wer: float = 0.0
    cer: float = 0.0
    word_count: int = 0
    errors: int = 0
    asr_metadata: dict[str, Any] = field(default_factory=dict)


class ASRBackend(Protocol):
    """Protocol for abstracted ASR backends."""

    def transcribe(self, audio_path: str) -> str:
        """Transcribe an audio file and return the recognized text."""
        ...


class ContentEvaluator:
    """Evaluates transcript accuracy using WER and CER metrics.

    When an ASR backend is provided, the evaluator will transcribe the audio
    and compare against the reference text.  Without a backend, it falls back
    to treating the reference as the recognized output (WER/CER = 0.0).
    """

    def __init__(self, asr_backend: ASRBackend | None = None) -> None:
        self._asr = asr_backend

    def evaluate(self, audio_path: str, reference_text: str) -> ContentResult:
        """Evaluate content accuracy for a single utterance.

        Args:
            audio_path: Path to the output audio file.
            reference_text: Ground-truth transcript text.

        Returns:
            ContentResult with WER, CER, and related statistics.
        """
        text = self._transcribe(audio_path) if self._asr else reference_text
        wer = jiwer.wer(reference_text, text)
        cer = jiwer.cer(reference_text, text)
        return ContentResult(
            reference_text=reference_text,
            recognized_text=text,
            wer=wer,
            cer=cer,
            word_count=len(reference_text.split()),
            errors=sum(
                1 for r, t in zip(reference_text.split(), text.split()) if r != t
            ),
        )

    def _transcribe(self, path: str) -> str:
        """Run ASR transcription through the configured backend."""
        return self._asr.transcribe(path) if self._asr else ""
