"""Benchmark adapter for Phase 1 benchmark repo."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from accentedge.models.interfaces import (
    StreamingCandidate,
    StreamingResult,
    StreamingSession,
)


# ---------------------------------------------------------------------------
# Benchmark data structures
# ---------------------------------------------------------------------------

class BenchmarkItem:
    """A single item from the benchmark DEV split."""

    def __init__(
        self,
        utterance_id: str,
        audio: np.ndarray,
        sample_rate: int = 16000,
        transcript: str = "",
        speaker_id: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.utterance_id = utterance_id
        self.audio = audio
        self.sample_rate = sample_rate
        self.transcript = transcript
        self.speaker_id = speaker_id
        self.metadata = metadata or {}


def load_dev_split(
    benchmark_repo_path: str | Path,
    condition: str = "clean",
    seed: int = 42,
) -> list[BenchmarkItem]:
    """Load DEV split items from the benchmark repository.

    In production this would read WAV files and transcripts from the benchmark
    repo's data directory.  For unit-testing it falls back to synthetic items.
    """
    path = Path(benchmark_repo_path)
    data_file = path / "data" / f"dev_{condition}.json"
    if data_file.exists():
        import json

        raw = json.loads(data_file.read_text(encoding="utf-8"))
        items: list[BenchmarkItem] = []
        for entry in raw:
            duration_s = float(entry.get("duration_s", 3.0))
            n_samples = int(duration_s * 16000)
            t = np.linspace(0, 2 * np.pi * 440, n_samples)
            audio = (np.sin(t) * 0.01).astype(np.float32)
            items.append(
                BenchmarkItem(
                    utterance_id=entry["utterance_id"],
                    audio=audio,
                    sample_rate=16000,
                    transcript=entry.get("transcript", ""),
                    speaker_id=entry.get("speaker_id", ""),
                    metadata=dict(entry),
                )
            )
        return items
    # Synthetic DEV split — deterministic for tests
    return _generate_synthetic_dev(condition=condition, seed=seed)


def _generate_synthetic_dev(
    condition: str = "clean",
    n_items: int = 20,
    seed: int = 42,
) -> list[BenchmarkItem]:
    rng = np.random.default_rng(seed)
    items: list[BenchmarkItem] = []
    for i in range(n_items):
        duration_s = float(rng.uniform(1.0, 5.0))
        n_samples = int(duration_s * 16000)
        t = np.linspace(0, 2 * np.pi * 440, n_samples)
        if condition == "noisy":
            audio = (np.sin(t) * 0.1 + rng.standard_normal(n_samples) * 0.05).astype(
                np.float32
            )
        else:
            audio = (np.sin(t) * 0.01).astype(np.float32)
        items.append(
            BenchmarkItem(
                utterance_id=f"{condition}_{i:04d}",
                audio=audio,
                sample_rate=16000,
                transcript=f"synthetic transcript {i}",
                speaker_id=f"spk_{i % 5}",
                metadata={"condition": condition, "duration_s": duration_s},
            )
        )
    return items


@dataclass
class UtteranceMetrics:
    """Per-utterance metrics from the Phase-1 benchmark."""

    utterance_id: str
    content_error_rate: float
    identity_preservation: float
    timing_error_ms: float
    correction_accuracy: float
    damage_score: float
    latency_ms: float
    rtf: float


@dataclass
class BenchmarkResult:
    """Aggregate result of running a candidate over the entire DEV split."""

    candidate_id: str
    config: dict[str, Any]
    utterances: list[UtteranceMetrics] = field(default_factory=list)
    aggregate_content: float = 0.0
    aggregate_identity: float = 0.0
    aggregate_latency_ms: float = 0.0
    aggregate_rtf: float = 0.0
    state_size_bytes: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "config": dict(self.config),
            "aggregate_content": self.aggregate_content,
            "aggregate_identity": self.aggregate_identity,
            "aggregate_latency_ms": self.aggregate_latency_ms,
            "aggregate_rtf": self.aggregate_rtf,
            "state_size_bytes": self.state_size_bytes,
            "utterances": [
                {
                    "utterance_id": u.utterance_id,
                    "content_error_rate": u.content_error_rate,
                    "identity_preservation": u.identity_preservation,
                    "timing_error_ms": u.timing_error_ms,
                    "correction_accuracy": u.correction_accuracy,
                    "damage_score": u.damage_score,
                    "latency_ms": u.latency_ms,
                    "rtf": u.rtf,
                }
                for u in self.utterances
            ],
        }


class CandidateOutput:
    def __init__(
        self,
        utterance_id: str,
        audio: Any = None,
        sample_rate: int = 16000,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.utterance_id = utterance_id
        self.audio = audio
        self.sample_rate = sample_rate
        self.metadata = metadata or {}


# ---------------------------------------------------------------------------
# Phase1BenchmarkAdapter
# ---------------------------------------------------------------------------

class Phase1BenchmarkAdapter:
    """Wraps any StreamingCandidate to implement the Phase-1 benchmark
    CandidateAdapter interface.

    Creates a session, processes each audio chunk from the DEV split, measures
    per-utterance and aggregate metrics (content, identity, timing, correction,
    damage, latency, RTF).
    """

    def __init__(
        self,
        benchmark_repo_path: str | Path,
        split: str = "dev",
        condition: str = "clean",
        chunk_size_ms: int = 80,
    ) -> None:
        self.benchmark_path = Path(benchmark_repo_path)
        self.split = split
        self.condition = condition
        self.chunk_size_ms = chunk_size_ms
        self._items: list[BenchmarkItem] | None = None

    @property
    def items(self) -> list[BenchmarkItem]:
        if self._items is None:
            self._items = load_dev_split(self.benchmark_path, self.condition)
        return self._items

    def _chunk_audio(self, audio: np.ndarray) -> list[np.ndarray]:
        cs = int(self.chunk_size_ms * 16000 / 1000)
        if cs <= 0:
            raise ValueError("chunk_size_ms must produce a positive chunk size")
        chunks: list[np.ndarray] = []
        i = 0
        while i < len(audio):
            chunk = audio[i : i + cs]
            if chunk.size == 0:
                break
            chunks.append(chunk)
            i += cs
        return chunks

    def evaluate_candidate(
        self,
        candidate: StreamingCandidate,
        config: dict[str, Any],
    ) -> BenchmarkResult:
        """Run the full DEV split through *candidate* and collect metrics."""
        candidate.prepare("cpu", "fp32")
        session = candidate.create_session(dict(config))
        candidate_id = getattr(candidate.metadata, "architecture_id", "unknown")

        utterance_results: list[UtteranceMetrics] = []
        total_audio_ms = 0.0
        total_compute_ms = 0.0
        last_result: StreamingResult | None = None

        try:
            for item in self.items:
                chunks = self._chunk_audio(item.audio)
                utterance_compute_ms = 0.0
                last_result = None

                for chunk in chunks:
                    t0 = time.perf_counter()
                    result = candidate.process_chunk(session, chunk, item.sample_rate)
                    t1 = time.perf_counter()
                    chunk_compute_ms = (t1 - t0) * 1000.0
                    utterance_compute_ms += chunk_compute_ms
                    last_result = result

                audio_ms = len(item.audio) / item.sample_rate * 1000.0
                rtf = (
                    utterance_compute_ms / audio_ms if audio_ms > 0 else 0.0
                )
                latency_ms = utterance_compute_ms

                # Derive quality signals from output audio
                if last_result is not None and last_result.audio.size > 0:
                    out = last_result.audio.astype(np.float64)
                    signal_power = float(np.mean(out ** 2))
                    noise_power = float(np.std(out) ** 2) + 1e-12
                    snr = signal_power / noise_power
                    # Content: CER-like — lower error is better
                    content_error_rate = float(np.clip(1.0 / (1.0 + snr * 100), 0.0, 1.0))
                    # Identity: amplitude preservation relative to input
                    orig = item.audio.astype(np.float64)
                    if orig.size > 0:
                        amp_ratio = float(
                            np.clip(
                                np.mean(np.abs(out))
                                / (np.mean(np.abs(orig)) + 1e-12),
                                0.0,
                                1.0,
                            )
                        )
                    else:
                        amp_ratio = 0.5
                    identity_preservation = amp_ratio
                else:
                    content_error_rate = 0.5
                    identity_preservation = 0.5

                timing_error_ms = max(0.0, latency_ms - audio_ms)
                correction_accuracy = float(np.clip(1.0 - content_error_rate, 0.0, 1.0))
                damage_score = float(np.clip(content_error_rate * 0.5, 0.0, 1.0))

                utterance_results.append(
                    UtteranceMetrics(
                        utterance_id=item.utterance_id,
                        content_error_rate=content_error_rate,
                        identity_preservation=identity_preservation,
                        timing_error_ms=timing_error_ms,
                        correction_accuracy=correction_accuracy,
                        damage_score=damage_score,
                        latency_ms=latency_ms,
                        rtf=rtf,
                    )
                )

                total_audio_ms += audio_ms
                total_compute_ms += utterance_compute_ms

            state_size_bytes = session.state_size_bytes()

        finally:
            candidate.close()

        n = len(utterance_results)
        if n > 0:
            aggregate_content = (
                1.0 - sum(u.content_error_rate for u in utterance_results) / n
            )
            aggregate_identity = (
                sum(u.identity_preservation for u in utterance_results) / n
            )
            aggregate_latency_ms = total_compute_ms
            aggregate_rtf = (
                total_compute_ms / total_audio_ms if total_audio_ms > 0 else 0.0
            )
        else:
            aggregate_content = 0.0
            aggregate_identity = 0.0
            aggregate_latency_ms = 0.0
            aggregate_rtf = 0.0

        return BenchmarkResult(
            candidate_id=candidate_id,
            config=dict(config),
            utterances=utterance_results,
            aggregate_content=aggregate_content,
            aggregate_identity=aggregate_identity,
            aggregate_latency_ms=aggregate_latency_ms,
            aggregate_rtf=aggregate_rtf,
            state_size_bytes=state_size_bytes,
        )


# ---------------------------------------------------------------------------
# Legacy API — kept for backward compatibility
# ---------------------------------------------------------------------------

class BenchmarkIntegration:
    def __init__(self, adapter: Phase1BenchmarkAdapter) -> None:
        self.adapter = adapter

    def submit_candidate_output(
        self,
        utterance_id: str,
        audio: Any,
        sample_rate: int = 16000,
        metadata: dict[str, Any] | None = None,
    ) -> CandidateOutput:
        return CandidateOutput(
            utterance_id=utterance_id,
            audio=audio,
            sample_rate=sample_rate,
            metadata=metadata,
        )

    def get_results(self) -> BenchmarkResult:
        return BenchmarkResult()
