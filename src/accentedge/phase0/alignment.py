"""
Forced alignment for Phase 0.

Provides:
- ForcedAlignmentResult: holds phones and words with start/end times
- AlignmentCorrector: tracks original vs corrected alignments and diffs
- validate_alignment(): checks for gaps, overlaps, phones outside word bounds
- A simple forced-alignment interface (whisperx / MFA / pre-computed)
"""

import logging
import warnings
from dataclasses import dataclass, field, asdict
from typing import Optional, Union

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class ForcedAlignmentResult:
    """Result of a forced-alignment pass.

    All times are in milliseconds relative to the start of the utterance.
    Phone sequences are flat; words reference phone indices.
    """
    utterance_id: str
    phones: list[tuple[str, float, float]]  # (phone, start_ms, end_ms)
    words: list[tuple[str, float, float]]   # (word, start_ms, end_ms)
    source: str = "unknown"   # e.g. "whisperx", "mfa", "manual"
    confidence: Optional[float] = None  # overall alignment confidence

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ForcedAlignmentResult":
        return cls(**d)

    def phone_durations_ms(self) -> np.ndarray:
        """Return per-phone durations in ms."""
        return np.array([end - start for _, start, end in self.phones])

    def word_durations_ms(self) -> np.ndarray:
        """Return per-word durations in ms."""
        return np.array([end - start for _, start, end in self.words])


# ---------------------------------------------------------------------------
# Alignment correction
# ---------------------------------------------------------------------------

@dataclass
class CorrectionDiff:
    """A single boundary correction."""
    token_index: int          # index into phones (or words)
    token_type: str           # "phone" or "word"
    old_start: float
    old_end: float
    new_start: float
    new_end: float


@dataclass
class AlignmentCorrector:
    """Stores original and corrected alignments, tracks correction diffs."""
    utterance_id: str
    original: ForcedAlignmentResult
    corrected: ForcedAlignmentResult
    corrections: list[CorrectionDiff] = field(default_factory=list)
    annotator_notes: str = ""

    def add_correction(self, diff: CorrectionDiff) -> None:
        """Register a boundary change."""
        self.corrections.append(diff)

    def total_boundary_shift_ms(self) -> float:
        """Sum of absolute boundary shifts across all corrections."""
        return sum(
            abs(d.new_start - d.old_start) + abs(d.new_end - d.old_end)
            for d in self.corrections
        )

    def correction_count(self) -> int:
        return len(self.corrections)

    def to_dict(self) -> dict:
        return {
            "utterance_id": self.utterance_id,
            "original": self.original.to_dict(),
            "corrected": self.corrected.to_dict(),
            "corrections": [asdict(c) for c in self.corrections],
            "annotator_notes": self.annotator_notes,
            "total_boundary_shift_ms": self.total_boundary_shift_ms(),
        }


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class AlignmentValidationError(Exception):
    """Raised when an alignment fails structural checks."""
    pass


def validate_alignment(
    result: ForcedAlignmentResult,
    gap_threshold_ms: float = 5.0,
    overlap_threshold_ms: float = 2.0,
) -> list[str]:
    """Check alignment for structural problems.

    Args:
        result: ForcedAlignmentResult to validate
        gap_threshold_ms: flag gaps larger than this (within phone sequence)
        overlap_threshold_ms: flag overlaps larger than this

    Returns:
        List of human-readable warning strings. Empty list = clean.
    """
    warnings: list[str] = []

    phones = result.phones
    words = result.words

    # --- Phone sequence: gaps and overlaps ---
    for i in range(len(phones) - 1):
        _, start_i, end_i = phones[i]
        _, start_j, _ = phones[i + 1]
        gap = start_j - end_i
        if gap > gap_threshold_ms:
            warnings.append(
                f"Gap between phone {i} and {i+1}: {gap:.1f} ms "
                f"('{phones[i][0]}' -> '{phones[i+1][0]}')"
            )
        elif gap < -overlap_threshold_ms:
            warnings.append(
                f"Overlap between phone {i} and {i+1}: {abs(gap):.1f} ms "
                f"('{phones[i][0]}' -> '{phones[i+1][0]}')"
            )

    # --- Word-level: check each phone falls within some word's bounds ---
    # For each phone, find the nearest word. Flag only if the phone center
    # is more than 20 ms outside all word spans (tolerant of boundary phones).
    if words and phones:
        for p_idx, (phone, p_start, p_end) in enumerate(phones):
            p_center = (p_start + p_end) / 2.0
            near_any_word = any(
                (w_start - 20.0) <= p_center <= (w_end + 20.0)
                for _, w_start, w_end in words
            )
            if not near_any_word:
                # Find nearest word for context
                nearest_word = min(
                    words, key=lambda w: abs((w[1] + w[2]) / 2.0 - p_center)
                )
                warnings.append(
                    f"Phone '{phone}' ({p_start:.0f}-{p_end:.0f} ms) "
                    f"center {p_center:.0f} ms far from any word "
                    f"(nearest: '{nearest_word[0]}' {nearest_word[1]:.0f}-{nearest_word[2]:.0f} ms)"
                )

    # --- Word sequence: gaps and overlaps ---
    for i in range(len(words) - 1):
        _, _, end_i = words[i]
        _, start_j, _ = words[i + 1]
        gap = start_j - end_i
        if gap > gap_threshold_ms * 3:
            warnings.append(
                f"Gap between word '{words[i][0]}' and '{words[i+1][0]}': "
                f"{gap:.1f} ms"
            )
        elif gap < -overlap_threshold_ms * 3:
            warnings.append(
                f"Overlap between word '{words[i][0]}' and '{words[i+1][0]}': "
                f"{abs(gap):.1f} ms"
            )

    # --- Monotonicity ---
    for i, (phone, start, end) in enumerate(phones):
        if start < 0:
            warnings.append(f"Negative start for phone {i}: {start}")
        if start >= end:
            warnings.append(
                f"Phone {i} '{phone}': start {start:.1f} >= end {end:.1f}"
            )
    for i, (word, start, end) in enumerate(words):
        if start < 0:
            warnings.append(f"Negative start for word {i}: {start}")
        if start >= end:
            warnings.append(
                f"Word {i} '{word}': start {start:.1f} >= end {end:.1f}"
            )

    return warnings


# ---------------------------------------------------------------------------
# Alignment interface
# ---------------------------------------------------------------------------

def align_with_whisperx(
    audio_path: str,
    transcript: str,
) -> Optional[ForcedAlignmentResult]:
    """Run whisperx forced alignment if available.

    Returns ForcedAlignmentResult or None if whisperx is not installed.
    """
    try:
        import whisperx  # type: ignore
    except ImportError:
        logger.warning("whisperx not available; returning None")
        return None

    logger.info("Running whisperx alignment on %s", audio_path)
    try:
        device = "cpu"
        model = whisperx.load_model("medium", device)
        audio = whisperx.load_audio(audio_path)
        result = model.transcribe(audio)
        model_a, metadata = whisperx.load_align_model(
            language_code=result.get("language", "en")
        )
        result = whisperx.align(
            result["segments"], model_a, metadata, audio, device
        )

        # Build flat phone list from segments
        phones_flat: list[tuple[str, float, float]] = []
        words_flat: list[tuple[str, float, float]] = []

        utt_start = 0.0  # seconds offset
        for seg in result["segments"]:
            for w in seg.get("words", []):
                w_start = w.get("start", 0.0) * 1000.0
                w_end = w.get("end", 0.0) * 1000.0
                words_flat.append((w["word"], w_start, w_end))
                for p in w.get("phones", []):
                    p_start = p.get("start", w_start / 1000.0) * 1000.0
                    p_end = p.get("end", w_end / 1000.0) * 1000.0
                    phones_flat.append((p["phone"], p_start, p_end))

        return ForcedAlignmentResult(
            utterance_id="",
            phones=phones_flat,
            words=words_flat,
            source="whisperx",
        )
    except Exception as exc:
        logger.error("whisperx alignment failed: %s", exc)
        return None


def align_with_mfa(
    audio_path: str,
    transcript: str,
    acoustic_model: str = "english_us_arpa",
    dictionary_path: Optional[str] = None,
) -> Optional[ForcedAlignmentResult]:
    """Run Montreal Forced Aligner if available.

    Returns ForcedAlignmentResult or None if MFA is not installed.
    """
    try:
        from montreal_forced_aligner import align  # type: ignore
    except ImportError:
        logger.warning("MFA not available; returning None")
        return None

    logger.info("Running MFA alignment on %s", audio_path)
    try:
        # MFA CLI-style invocation (scaffold)
        warnings.warn(
            "MFA alignment scaffold — integrate montreal-forced-aligner "
            "command or API properly.",
            stacklevel=2,
        )
        return None
    except Exception as exc:
        logger.error("MFA alignment failed: %s", exc)
        return None


def align_audio(
    audio_path: str,
    transcript: str,
    method: str = "auto",
    precomputed: Optional[ForcedAlignmentResult] = None,
) -> ForcedAlignmentResult:
    """High-level alignment interface.

    Priority:
    1. If *precomputed* is given, use it.
    2. Try whisperx, then MFA.
    3. If *method == "precomputed"* and nothing was given, raise.

    Args:
        audio_path: path to audio file
        transcript: text transcript
        method: "auto", "whisperx", "mfa", "precomputed"
        precomputed: a ForcedAlignmentResult to use directly

    Returns:
        ForcedAlignmentResult

    Raises:
        ValueError: if method == "precomputed" with no precomputed result
    """
    if precomputed is not None:
        logger.info("Using pre-computed alignment")
        return precomputed

    if method == "precomputed":
        raise ValueError("method='precomputed' requires a precomputed alignment")

    if method in ("auto", "whisperx"):
        result = align_with_whisperx(audio_path, transcript)
        if result is not None:
            return result

    if method in ("auto", "mfa"):
        result = align_with_mfa(audio_path, transcript)
        if result is not None:
            return result

    # Fallback: return an empty result with a warning
    logger.warning(
        "No alignment backend available; returning empty result for %s",
        audio_path,
    )
    return ForcedAlignmentResult(
        utterance_id="",
        phones=[],
        words=[],
        source="none",
    )
