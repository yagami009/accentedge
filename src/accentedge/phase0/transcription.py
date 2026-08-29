"""
Audio transcription for Phase 0.

Provides:
- TranscriptionResult: text, words with timestamps, confidence
- transcribe_audio(): interface to whisperx (scaffold)
- export_transcript() / import_transcript() for JSON interchange
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Union

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class WordSegment:
    """One word in a transcription result."""
    word: str
    start: float   # seconds
    end: float     # seconds
    confidence: Optional[float] = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "WordSegment":
        return cls(**d)


@dataclass
class TranscriptionResult:
    """Result of an audio transcription pass."""
    utterance_id: str
    text: str
    words: list[WordSegment] = field(default_factory=list)
    source: str = "unknown"
    language: str = "en"
    overall_confidence: Optional[float] = None

    def to_dict(self) -> dict:
        d = {
            "utterance_id": self.utterance_id,
            "text": self.text,
            "source": self.source,
            "language": self.language,
            "overall_confidence": self.overall_confidence,
            "words": [w.to_dict() for w in self.words],
        }
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "TranscriptionResult":
        d = d.copy()
        d["words"] = [WordSegment.from_dict(w) for w in d.get("words", [])]
        return cls(**d)

    def word_count(self) -> int:
        return len(self.words)

    def duration_seconds(self) -> float:
        if not self.words:
            return 0.0
        return self.words[-1].end - self.words[0].start


# ---------------------------------------------------------------------------
# Transcription interface
# ---------------------------------------------------------------------------

def transcribe_audio(
    audio: Union[np.ndarray, str, Path],
    sample_rate: int = 22050,
    utterance_id: str = "unknown",
    model_name: str = "medium",
    device: str = "cpu",
    language: str = "en",
) -> Optional[TranscriptionResult]:
    """Transcribe an audio file or waveform using whisperx.

    Args:
        audio: waveform (float32 numpy array) or path to audio file
        sample_rate: sample rate if audio is a numpy array
        utterance_id: identifier for this utterance
        model_name: whisperx model name
        device: "cpu" or "cuda"
        language: ISO 639-1 language code

    Returns:
        TranscriptionResult, or None if whisperx is unavailable or fails.
    """
    try:
        import whisperx  # type: ignore
    except ImportError:
        logger.warning("whisperx not available; cannot transcribe")
        return None

    logger.info("Transcribing utterance %s with whisperx (%s)", utterance_id, model_name)

    try:
        # Load model
        model = whisperx.load_model(model_name, device)

        # Handle audio input
        if isinstance(audio, (str, Path)):
            audio_array = whisperx.load_audio(str(audio))
        else:
            audio_array = np.asarray(audio, dtype=np.float32)

        result = model.transcribe(audio_array, language=language)

        segments = result.get("segments", [])
        detected_language = result.get("language", language)

        # Build word-level segments
        words: list[WordSegment] = []
        full_text_parts: list[str] = []

        for seg in segments:
            seg_text = seg.get("text", "").strip()
            if seg_text:
                full_text_parts.append(seg_text)
            for w in seg.get("words", []):
                words.append(WordSegment(
                    word=w["word"],
                    start=float(w["start"]),
                    end=float(w["end"]),
                    confidence=w.get("score"),
                ))

        full_text = " ".join(full_text_parts).strip()

        # Compute overall confidence from word scores
        scores = [w.confidence for w in words if w.confidence is not None]
        overall_conf = float(np.mean(scores)) if scores else None

        return TranscriptionResult(
            utterance_id=utterance_id,
            text=full_text,
            words=words,
            source="whisperx",
            language=detected_language,
            overall_confidence=overall_conf,
        )

    except Exception as exc:
        logger.error("whisperx transcription failed: %s", exc)
        return None


def mock_transcribe(
    utterance_id: str,
    text: str,
    words: Optional[list[dict]] = None,
) -> TranscriptionResult:
    """Create a TranscriptionResult from plain text (testing scaffold).

    If *words* is provided, each dict should have word, start, end, confidence.
    Otherwise a simple word list with uniform confidence is generated.
    """
    if words is None:
        tokens = text.split()
        n = len(tokens)
        step = 0.3  # seconds per word
        words = [
            WordSegment(
                word=tok,
                start=i * step,
                end=(i + 1) * step,
                confidence=0.9,
            )
            for i, tok in enumerate(tokens)
        ]
    else:
        words = [WordSegment(**w) for w in words]

    return TranscriptionResult(
        utterance_id=utterance_id,
        text=text,
        words=words,
        source="mock",
    )


# ---------------------------------------------------------------------------
# JSON interchange
# ---------------------------------------------------------------------------

def export_transcript(result: TranscriptionResult, path: Path) -> None:
    """Write a TranscriptionResult to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(result.to_dict(), f, indent=2)
    logger.info("Transcript exported to %s", path)


def import_transcript(path: Path) -> TranscriptionResult:
    """Read a TranscriptionResult from JSON."""
    with open(path) as f:
        data = json.load(f)
    return TranscriptionResult.from_dict(data)
