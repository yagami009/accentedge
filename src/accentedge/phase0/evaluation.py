"""
Evaluation metrics for Phase 0 target assessment.

Research probes only. Not production metrics.

Extended with:
- AccentShiftEvaluator  — probe-based accent shift scoring
- IdentityEvaluator     — speaker-embedding identity preservation
- TimingEvaluator       — duration and timing distribution comparison
- ContentEvaluator      — WER via WhisperX ASR
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import numpy as np

logger = logging.getLogger(__name__)


# ── Base result class ──────────────────────────────────────────────────────

@dataclass
class EvaluationResult:
    utterance_id: str
    strategy: str
    critical_entities_correct: Optional[bool] = None
    word_error_rate: Optional[float] = None
    accent_shift_score: Optional[float] = None
    correction_rate: Optional[float] = None
    damage_rate: Optional[float] = None
    identity_score: Optional[float] = None
    duration_ratio: Optional[float] = None
    naturalness_mos: Optional[float] = None
    metrics: dict = field(default_factory=dict)

    def to_dict(self):
        return {
            'utterance_id': self.utterance_id,
            'strategy': self.strategy,
            'critical_entities_correct': self.critical_entities_correct,
            'word_error_rate': self.word_error_rate,
            'accent_shift_score': self.accent_shift_score,
            'correction_rate': self.correction_rate,
            'damage_rate': self.damage_rate,
            'identity_score': self.identity_score,
            'duration_ratio': self.duration_ratio,
            'naturalness_mos': self.naturalness_mos,
            'metrics': self.metrics,
        }


# ── Audio quality helpers ──────────────────────────────────────────────────

def compute_snr_estimate(signal):
    noise_floor = np.percentile(np.abs(signal), 10)
    signal_level = np.percentile(np.abs(signal), 90)
    if noise_floor == 0:
        return 100.0
    return float(10 * np.log10(signal_level / noise_floor))


def validate_audio_quality(signal):
    checks = {
        'snr_db': compute_snr_estimate(signal),
        'peak_amplitude': float(np.max(np.abs(signal))),
        'rms': float(np.sqrt(np.mean(signal ** 2))),
    }
    return checks


# ── AccentShiftEvaluator ───────────────────────────────────────────────────

class AccentShiftEvaluator:
    """
    Runs accent probes on source and target audio segments and computes
    an accent-shift score.

    The score is the proportion of probe dimensions where the target
    is classified more target-like than the source.

    Expects centroids to be pre-built and probe objects to be configured
    per dimension.
    """

    def __init__(self, sample_rate: int = 22050):
        self.sample_rate = sample_rate

    def evaluate(
        self,
        source_audio: np.ndarray,
        target_audio: np.ndarray,
        probe_results_source: dict[str, "ProbeResult"],
        probe_results_target: dict[str, "ProbeResult"],
    ) -> EvaluationResult:
        """
        Compute accent shift score from pre-computed probe results.

        Args:
            source_audio: Source waveform (informational, not used directly).
            target_audio: Target waveform (informational).
            probe_results_source: dict mapping dimension → ProbeResult for source.
            probe_results_target: dict mapping dimension → ProbeResult for target.

        Returns:
            EvaluationResult with accent_shift_score filled.
        """
        if not probe_results_source or not probe_results_target:
            return EvaluationResult(
                utterance_id="",
                strategy="",
                accent_shift_score=None,
                metrics={"error": "no probe results provided"},
            )

        improved = 0
        total = 0
        dimension_scores = {}

        for dim_key in probe_results_source:
            if dim_key not in probe_results_target:
                continue
            src = probe_results_source[dim_key]
            tgt = probe_results_target[dim_key]

            # A dimension "improved" if target distance_to_native < source distance_to_native
            # AND target classification is "target" (or at least closer)
            src_dist = src.distance_to_native
            tgt_dist = tgt.distance_to_native
            improvement = src_dist - tgt_dist

            dimension_scores[dim_key] = {
                "source_distance": float(src_dist),
                "target_distance": float(tgt_dist),
                "improvement": float(improvement),
                "source_class": src.classification,
                "target_class": tgt.classification,
            }

            total += 1
            if tgt_dist < src_dist:
                improved += 1

        shift_score = improved / total if total > 0 else 0.0

        result = EvaluationResult(
            utterance_id="",
            strategy="",
            accent_shift_score=float(shift_score),
            metrics={
                "dimensions_evaluated": total,
                "dimensions_improved": improved,
                "dimension_scores": dimension_scores,
            },
        )
        return result

    def evaluate_from_segments(
        self,
        source_segments: dict[str, np.ndarray],
        target_segments: dict[str, np.ndarray],
        probe: "AccentProbe",
    ) -> EvaluationResult:
        """
        Convenience method: run probes on pre-segmented audio.

        Args:
            source_segments: dict mapping dimension → waveform segment from source.
            target_segments: dict mapping dimension → waveform segment from target.
            probe: AccentProbe configured with centroids for a single dimension.

        Returns:
            EvaluationResult with accent_shift_score.
        """
        from accentedge.phase0.probes import AccentProbe, ProbeDimension

        # If probe is for a single dimension, wrap per-dimension
        src_results = {}
        tgt_results = {}

        for dim_key, src_wf in source_segments.items():
            try:
                dim_enum = ProbeDimension(dim_key)
            except ValueError:
                dim_enum = ProbeDimension.STR  # default fallback
            single_probe = AccentProbe(dimension=dim_enum)
            single_probe.set_centroids(
                probe._native_centroids, probe._indian_centroids
            )
            src_results[dim_key] = single_probe.classify(src_wf, dim_key)
            if dim_key in target_segments:
                tgt_results[dim_key] = single_probe.classify(
                    target_segments[dim_key], dim_key
                )

        return self.evaluate(
            source_audio=np.concatenate(list(source_segments.values())),
            target_audio=np.concatenate(list(target_segments.values())),
            probe_results_source=src_results,
            probe_results_target=tgt_results,
        )


# ── IdentityEvaluator ──────────────────────────────────────────────────────

class IdentityEvaluator:
    """
    Computes speaker identity preservation using multiple speaker encoders.

    Compares source and target audio embeddings to verify that accent
    transformation does not alter speaker identity beyond natural variation.
    """

    def __init__(self, sample_rate: int = 22050):
        self.sample_rate = sample_rate
        self._encoder: Optional["SpeakerEncoder"] = None

    def _get_encoder(self) -> "SpeakerEncoder":
        if self._encoder is None:
            from accentedge.phase0.identity import SpeakerEncoder
            self._encoder = SpeakerEncoder(sample_rate=self.sample_rate)
        return self._encoder

    def evaluate(
        self,
        source_audio: np.ndarray,
        target_audio: np.ndarray,
    ) -> tuple[EvaluationResult, "SpeakerEncoderResult"]:
        """
        Compute identity preservation between source and target.

        Returns:
            (EvaluationResult, SpeakerEncoderResult) tuple.
        """
        from accentedge.phase0.identity import SpeakerEncoderResult as SER

        try:
            encoder = self._get_encoder()
            ser = encoder.compare(source_audio, target_audio)
            id_score = ser.mean_similarity
        except ImportError as exc:
            logger.warning(f"Speaker encoder unavailable: {exc}")
            ser = SER(
                encoder_names=[],
                distances=[],
                similarities=[],
                mean_distance=1.0,
                mean_similarity=0.0,
            )

        # Identity score: mean similarity across encoders, scaled 0..1
        identity_score = ser.mean_similarity if ser.encoder_names else None

        result = EvaluationResult(
            utterance_id="",
            strategy="",
            identity_score=identity_score,
            metrics={
                "encoder_results": ser.individual_results,
                "mean_distance": ser.mean_distance,
                "mean_similarity": ser.mean_similarity,
            },
        )
        return result, ser


# ── TimingEvaluator ────────────────────────────────────────────────────────

class TimingEvaluator:
    """
    Measures timing preservation: duration ratio, phone-level timing
    changes, and word-level timing changes.

    Uses librosa for onset detection and tempo estimation.
    """

    def __init__(self, sample_rate: int = 22050):
        self.sample_rate = sample_rate

    def evaluate(
        self,
        source_audio: np.ndarray,
        target_audio: np.ndarray,
    ) -> EvaluationResult:
        """
        Compute timing metrics comparing source and target.

        Args:
            source_audio: Source waveform.
            target_audio: Target waveform.

        Returns:
            EvaluationResult with duration_ratio and timing metrics.
        """
        metrics = {}

        # Duration ratio
        src_dur = len(source_audio) / self.sample_rate
        tgt_dur = len(target_audio) / self.sample_rate
        duration_ratio = tgt_dur / src_dur if src_dur > 0 else 1.0
        metrics["source_duration_s"] = float(src_dur)
        metrics["target_duration_s"] = float(tgt_dur)
        metrics["duration_ratio"] = float(duration_ratio)

        # Onset-based timing change
        try:
            import librosa  # type: ignore
            src_onsets = self._detect_onsets(source_audio)
            tgt_onsets = self._detect_onsets(target_audio)
            onset_change = self._compare_timing(
                src_onsets, tgt_onsets, src_dur, tgt_dur
            )
            metrics["onset_count_source"] = len(src_onsets)
            metrics["onset_count_target"] = len(tgt_onsets)
            metrics["onset_count_change"] = onset_change["count_change"]
            metrics["onset_timing_shift_s"] = onset_change["mean_shift_s"]
        except ImportError:
            metrics["onset_error"] = "librosa not available"
        except Exception as exc:
            metrics["onset_error"] = str(exc)

        # Tempo estimation
        try:
            import librosa  # type: ignore
            src_tempo = self._estimate_tempo(source_audio)
            tgt_tempo = self._estimate_tempo(target_audio)
            if src_tempo and tgt_tempo:
                metrics["source_tempo_bpm"] = float(src_tempo)
                metrics["target_tempo_bpm"] = float(tgt_tempo)
                metrics["tempo_ratio"] = float(tgt_tempo / src_tempo)
        except ImportError:
            metrics["tempo_error"] = "librosa not available"
        except Exception as exc:
            metrics["tempo_error"] = str(exc)

        # Word-level timing: estimate from onset spacing
        try:
            import librosa  # type: ignore
            src_onsets = self._detect_onsets(source_audio)
            tgt_onsets = self._detect_onsets(target_audio)
            if len(src_onsets) > 1 and len(tgt_onsets) > 1:
                src_intervals = np.diff(src_onsets)
                tgt_intervals = np.diff(tgt_onsets)
                # Compare distributions via coefficient of variation
                src_cv = float(np.std(src_intervals) / np.mean(src_intervals)) if len(src_intervals) > 0 else 0
                tgt_cv = float(np.std(tgt_intervals) / np.mean(tgt_intervals)) if len(tgt_intervals) > 0 else 0
                metrics["source_onset_cv"] = src_cv
                metrics["target_onset_cv"] = tgt_cv
                metrics["cv_change"] = tgt_cv - src_cv
        except Exception:
            pass

        return EvaluationResult(
            utterance_id="",
            strategy="",
            duration_ratio=float(duration_ratio),
            metrics=metrics,
        )

    # ── private helpers ──────────────────────────────────────────────────

    def _detect_onsets(self, waveform: np.ndarray) -> np.ndarray:
        """Detect onset frames and return times in seconds."""
        import librosa  # type: ignore
        wf = np.asarray(waveform, dtype=np.float32)
        if wf.ndim > 1:
            wf = wf.mean(axis=1)
        onset_frames = librosa.onset.onset_detect(y=wf, sr=self.sample_rate)
        onset_times = librosa.frames_to_time(onset_frames, sr=self.sample_rate)
        return onset_times

    def _estimate_tempo(self, waveform: np.ndarray) -> Optional[float]:
        """Estimate tempo in BPM."""
        import librosa  # type: ignore
        wf = np.asarray(waveform, dtype=np.float32)
        if wf.ndim > 1:
            wf = wf.mean(axis=1)
        tempo, _ = librosa.beat.beat_track(y=wf, sr=self.sample_rate)
        return float(tempo) if tempo and tempo > 0 else None

    @staticmethod
    def _compare_timing(
        src_onsets: np.ndarray, tgt_onsets: np.ndarray,
        src_dur: float, tgt_dur: float,
    ) -> dict:
        """Compare two onset sequences, returning shift statistics."""
        if len(src_onsets) == 0 and len(tgt_onsets) == 0:
            return {"count_change": 0, "mean_shift_s": 0.0}
        if len(src_onsets) == 0:
            return {"count_change": len(tgt_onsets), "mean_shift_s": tgt_dur}
        if len(tgt_onsets) == 0:
            return {"count_change": -len(src_onsets), "mean_shift_s": src_dur}

        # Align by relative position in utterance
        src_rel = src_onsets / src_dur if src_dur > 0 else src_onsets
        tgt_rel = tgt_onsets / tgt_dur if tgt_dur > 0 else tgt_onsets

        # Normalize to min length
        n = min(len(src_rel), len(tgt_rel))
        if n == 0:
            return {"count_change": len(tgt_onsets) - len(src_onsets), "mean_shift_s": 0.0}

        shifts = np.abs(tgt_rel[:n] - src_rel[:n])
        mean_shift_abs = float(np.mean(shifts)) if n > 0 else 0.0
        # Convert back to seconds using average duration
        avg_dur = (src_dur + tgt_dur) / 2
        mean_shift_s = mean_shift_abs * avg_dur

        return {
            "count_change": len(tgt_onsets) - len(src_onsets),
            "mean_shift_s": mean_shift_s,
        }


# ── ContentEvaluator ───────────────────────────────────────────────────────

class ContentEvaluator:
    """
    Computes word error rate (WER) using WhisperX ASR.

    Special handling for critical entities (numbers, dates, names):
    separately computes entity WER vs. non-entity WER.
    """

    # Patterns for critical BPO entities
    _CRITICAL_PATTERNS = [
        r'\b\d+\.?\d*\b',           # numbers: 30, 13.5, etc.
        r'\b(?:January|February|March|April|May|June|July|August|'
        r'September|October|November|December)\b',  # month names
        r'\b\d{1,2}(?:st|nd|rd|th)?\b',  # ordinal dates
        r'\b[A-Z][a-z]+\b',         # capitalized names (heuristic)
    ]

    def __init__(self, sample_rate: int = 22050, whisperx_model: str = "large-v3"):
        self.sample_rate = sample_rate
        self.whisperx_model = whisperx_model
        self._asr_model = None
        self._align_model = None
        self._align_metadata = None

    def _lazy_load_whisperx(self):
        """Lazy-load WhisperX model and align model."""
        if self._asr_model is not None:
            return
        try:
            import whisperx  # type: ignore
            import torch  # noqa: F401
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._asr_model = whisperx.load_model(
                self.whisperx_model, device=device
            )
            self._align_model, self._align_metadata = whisperx.load_align_model(
                language_code="en", device=device
            )
        except ImportError:
            raise ImportError(
                "whisperx is required for ContentEvaluator. "
                "Install it with: pip install whisperx"
            )

    def evaluate(
        self,
        source_audio: np.ndarray,
        target_audio: np.ndarray,
        reference_text: str = "",
    ) -> EvaluationResult:
        """
        Compute WER between reference and both source/target ASR outputs.

        Args:
            source_audio: Source waveform.
            target_audio: Target waveform.
            reference_text: Ground-truth transcript. If empty, source ASR is
                used as reference (assumes source is correct).

        Returns:
            EvaluationResult with word_error_rate and entity-specific metrics.
        """
        metrics = {}

        # Transcribe both
        src_text = self._transcribe(source_audio)
        tgt_text = self._transcribe(target_audio)

        if not reference_text:
            reference_text = src_text

        # Compute overall WER (target vs reference)
        try:
            import jiwer  # type: ignore
            wer = jiwer.wer(reference_text, tgt_text)
            metrics["wer"] = float(wer)
            metrics["reference"] = reference_text
            metrics["target_transcript"] = tgt_text
        except ImportError:
            raise ImportError(
                "jiwer is required for WER computation. "
                "Install it with: pip install jiwer"
            )

        # Entity-specific WER
        entity_metrics = self._entity_wer(reference_text, tgt_text)
        metrics["entity_wer"] = entity_metrics
        metrics["critical_entities_correct"] = entity_metrics["entity_wer"] < 0.15

        return EvaluationResult(
            utterance_id="",
            strategy="",
            word_error_rate=float(wer),
            critical_entities_correct=bool(metrics["critical_entities_correct"]),
            metrics=metrics,
        )

    def transcribe(self, audio: np.ndarray) -> str:
        """Transcribe a single audio segment. Public utility method."""
        return self._transcribe(audio)

    def _transcribe(self, audio: np.ndarray) -> str:
        """Run WhisperX ASR on a waveform."""
        try:
            self._lazy_load_whisperx()
        except ImportError:
            return ""  # gracefully return empty if model unavailable

        wf = np.asarray(audio, dtype=np.float32)
        if wf.ndim > 1:
            wf = wf.mean(axis=1)

        try:
            import whisperx  # type: ignore
            result = self._asr_model.transcribe(wf, language="en")
            segments = result.get("segments", [])
            text = " ".join(seg.get("text", "") for seg in segments).strip()
            return text
        except Exception as exc:
            logger.warning(f"WhisperX transcription failed: {exc}")
            return ""

    def _entity_wer(self, reference: str, hypothesis: str) -> dict:
        """Compute WER restricted to critical entity tokens."""
        import re
        try:
            import jiwer  # type: ignore
        except ImportError:
            return {"entity_wer": None, "entity_ref_count": 0}

        # Extract entity tokens from reference
        pattern = r'|'.join(self._CRITICAL_PATTERNS)
        ref_entities = re.findall(pattern, reference)
        hyp_entities = re.findall(pattern, hypothesis)

        if not ref_entities:
            return {"entity_wer": 0.0, "entity_ref_count": 0, "entity_tokens": []}

        ref_str = " ".join(ref_entities)
        hyp_str = " ".join(hyp_entities)
        entity_wer = jiwer.wer(ref_str, hyp_str)

        return {
            "entity_wer": float(entity_wer),
            "entity_ref_count": len(ref_entities),
            "entity_hyp_count": len(hyp_entities),
            "entity_tokens_ref": ref_entities,
            "entity_tokens_hyp": hyp_entities,
        }

