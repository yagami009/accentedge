"""Sweep utilities for running candidates across chunk sizes and lookahead values."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from accentedge.models.interfaces import StreamingCandidate
from accentedge.benchmark.adapter import BenchmarkResult, Phase1BenchmarkAdapter


# ---------------------------------------------------------------------------
# SweepResult
# ---------------------------------------------------------------------------

@dataclass
class SweepResult:
    """Result of a single (candidate, config) evaluation within a sweep."""

    candidate_id: str
    chunk_size_ms: int
    lookahead_ms: int
    metrics: dict[str, Any] = field(default_factory=dict)
    latency_ms: float = 0.0
    rtf: float = 0.0
    state_size_bytes: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "chunk_size_ms": self.chunk_size_ms,
            "lookahead_ms": self.lookahead_ms,
            "metrics": dict(self.metrics),
            "latency_ms": self.latency_ms,
            "rtf": self.rtf,
            "state_size_bytes": self.state_size_bytes,
        }


# ---------------------------------------------------------------------------
# Standard sweep axes
# ---------------------------------------------------------------------------

CHUNK_SIZES_MS: list[int] = [20, 40, 80, 160]
LOOKAHEAD_SIZES_MS: list[int] = [0, 20, 40, 80, 160, 320]


# ---------------------------------------------------------------------------
# Base sweep runner
# ---------------------------------------------------------------------------

class BaseSweep:
    """Common base that holds a candidate factory and an adapter."""

    def __init__(
        self,
        candidate_factory: Callable[[dict[str, Any]], StreamingCandidate],
        benchmark_repo_path: str,
        condition: str = "clean",
    ) -> None:
        self._factory = candidate_factory
        self._adapter = Phase1BenchmarkAdapter(
            benchmark_repo_path=benchmark_repo_path,
            condition=condition,
        )


# ---------------------------------------------------------------------------
# ChunkSweep
# ---------------------------------------------------------------------------

class ChunkSweep(BaseSweep):
    """Run a candidate at multiple chunk sizes, keeping lookahead fixed."""

    UNSUPPORTED_SIZES: list[int] = [1600, 3200]  # clearly too large

    def __init__(
        self,
        candidate_factory: Callable[[dict[str, Any]], StreamingCandidate],
        benchmark_repo_path: str,
        chunk_sizes: list[int] | None = None,
        lookahead_ms: int = 0,
        condition: str = "clean",
    ) -> None:
        super().__init__(candidate_factory, benchmark_repo_path, condition)
        self.chunk_sizes = list(chunk_sizes) if chunk_sizes is not None else list(CHUNK_SIZES_MS)
        self.lookahead_ms = lookahead_ms

    def run(
        self,
        base_config: dict[str, Any] | None = None,
    ) -> list[SweepResult]:
        """Run the candidate for every chunk size and return SweepResults."""
        base_config = base_config or {}
        results: list[SweepResult] = []
        for chunk_size in self.chunk_sizes:
            if chunk_size in self.UNSUPPORTED_SIZES:
                continue
            config = dict(base_config)
            config["chunk_size_ms"] = chunk_size
            config["lookahead_ms"] = self.lookahead_ms

            candidate = self._factory(config)
            bench_result = self._adapter.evaluate_candidate(candidate, config)

            # Average latency and RTF across all utterances
            n = len(bench_result.utterances)
            avg_latency = (
                bench_result.aggregate_latency_ms / n if n > 0 else 0.0
            )
            avg_rtf = bench_result.aggregate_rtf

            results.append(
                SweepResult(
                    candidate_id=bench_result.candidate_id,
                    chunk_size_ms=chunk_size,
                    lookahead_ms=self.lookahead_ms,
                    metrics={
                        "content": bench_result.aggregate_content,
                        "identity": bench_result.aggregate_identity,
                        "state_size_bytes": bench_result.state_size_bytes,
                    },
                    latency_ms=avg_latency,
                    rtf=avg_rtf,
                    state_size_bytes=bench_result.state_size_bytes,
                )
            )
        return results

    def validate_sizes(self, sizes: list[int]) -> list[int]:
        """Return sizes that are supported (reject unsupported sizes cleanly)."""
        return [s for s in sizes if s not in self.UNSUPPORTED_SIZES]


# ---------------------------------------------------------------------------
# LookaheadSweep
# ---------------------------------------------------------------------------

class LookaheadSweep(BaseSweep):
    """Run a candidate at multiple lookahead values, keeping chunk size fixed."""

    def __init__(
        self,
        candidate_factory: Callable[[dict[str, Any]], StreamingCandidate],
        benchmark_repo_path: str,
        chunk_size_ms: int = 80,
        lookahead_values: list[int] | None = None,
        condition: str = "clean",
    ) -> None:
        super().__init__(candidate_factory, benchmark_repo_path, condition)
        self.chunk_size_ms = chunk_size_ms
        self.lookahead_values = (
            list(lookahead_values)
            if lookahead_values is not None
            else list(LOOKAHEAD_SIZES_MS)
        )

    def run(
        self,
        base_config: dict[str, Any] | None = None,
    ) -> list[SweepResult]:
        """Run the candidate for every lookahead value and return SweepResults."""
        base_config = base_config or {}
        results: list[SweepResult] = []
        for lookahead in self.lookahead_values:
            config = dict(base_config)
            config["chunk_size_ms"] = self.chunk_size_ms
            config["lookahead_ms"] = lookahead

            candidate = self._factory(config)
            bench_result = self._adapter.evaluate_candidate(candidate, config)

            n = len(bench_result.utterances)
            avg_latency = (
                bench_result.aggregate_latency_ms / n if n > 0 else 0.0
            )
            avg_rtf = bench_result.aggregate_rtf

            results.append(
                SweepResult(
                    candidate_id=bench_result.candidate_id,
                    chunk_size_ms=self.chunk_size_ms,
                    lookahead_ms=lookahead,
                    metrics={
                        "content": bench_result.aggregate_content,
                        "identity": bench_result.aggregate_identity,
                        "state_size_bytes": bench_result.state_size_bytes,
                    },
                    latency_ms=avg_latency,
                    rtf=avg_rtf,
                    state_size_bytes=bench_result.state_size_bytes,
                )
            )
        return results


# ---------------------------------------------------------------------------
# CombinedSweep
# ---------------------------------------------------------------------------

class CombinedSweep(BaseSweep):
    """Run a candidate across both dimensions (chunk size × lookahead)."""

    def __init__(
        self,
        candidate_factory: Callable[[dict[str, Any]], StreamingCandidate],
        benchmark_repo_path: str,
        chunk_sizes: list[int] | None = None,
        lookahead_values: list[int] | None = None,
        condition: str = "clean",
    ) -> None:
        super().__init__(candidate_factory, benchmark_repo_path, condition)
        self.chunk_sizes = list(chunk_sizes) if chunk_sizes is not None else list(CHUNK_SIZES_MS)
        self.lookahead_values = (
            list(lookahead_values)
            if lookahead_values is not None
            else list(LOOKAHEAD_SIZES_MS)
        )

    def run(
        self,
        base_config: dict[str, Any] | None = None,
    ) -> list[SweepResult]:
        """Run the candidate at every (chunk_size, lookahead) pair."""
        base_config = base_config or {}
        results: list[SweepResult] = []
        for chunk_size in self.chunk_sizes:
            for lookahead in self.lookahead_values:
                config = dict(base_config)
                config["chunk_size_ms"] = chunk_size
                config["lookahead_ms"] = lookahead

                candidate = self._factory(config)
                bench_result = self._adapter.evaluate_candidate(candidate, config)

                n = len(bench_result.utterances)
                avg_latency = (
                    bench_result.aggregate_latency_ms / n if n > 0 else 0.0
                )
                avg_rtf = bench_result.aggregate_rtf

                results.append(
                    SweepResult(
                        candidate_id=bench_result.candidate_id,
                        chunk_size_ms=chunk_size,
                        lookahead_ms=lookahead,
                        metrics={
                            "content": bench_result.aggregate_content,
                            "identity": bench_result.aggregate_identity,
                            "state_size_bytes": bench_result.state_size_bytes,
                        },
                        latency_ms=avg_latency,
                        rtf=avg_rtf,
                        state_size_bytes=bench_result.state_size_bytes,
                    )
                )
        return results
