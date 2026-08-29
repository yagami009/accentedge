"""Streaming simulator with virtual time and backlog tracking."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from accentedge.models.models.interfaces import (
    StreamingCandidate,
    StreamingResult,
)
from accentedge.models.streaming.chunker import Chunker


@dataclass
class SimulatorEvent:
    event_type: Literal[
        "chunk_arrival",
        "processing_start",
        "processing_end",
        "output_ready",
    ]
    timestamp_ms: float
    chunk_index: int
    backlog_ms: float = 0.0


@dataclass
class SimulatorReport:
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    latency_p99_ms: float = 0.0
    max_backlog_ms: float = 0.0
    avg_backlog_ms: float = 0.0
    total_chunks: int = 0
    total_output_samples: int = 0
    rtf_p50: float = 0.0
    rtf_p95: float = 0.0
    rtf_p99: float = 0.0


class StreamingSimulator:
    def __init__(
        self,
        candidate: StreamingCandidate,
        chunk_size_ms: int = 80,
        lookahead_ms: int = 0,
        sample_rate: int = 16000,
    ) -> None:
        self.candidate = candidate
        self.chunk_size_ms = chunk_size_ms
        self.lookahead_ms = lookahead_ms
        self.sample_rate = sample_rate
        self.chunker = Chunker(
            chunk_size_ms=chunk_size_ms,
            sample_rate=sample_rate,
            overlap_ms=lookahead_ms,
        )
        self.virtual_time_ms: float = 0.0
        self.backlog_ms: float = 0.0
        self.max_backlog_ms: float = 0.0
        self.events: list[SimulatorEvent] = []
        self._chunk_index: int = 0
        self._results: list[StreamingResult] = []

    async def feed(self, audio: np.ndarray) -> list[StreamingResult]:
        session = self.candidate.create_session({"simulator": True})
        chunks = self.chunker.chunk(audio)
        try:
            for chunk in chunks:
                self._chunk_index += 1
                arrival = self.virtual_time_ms
                self.events.append(
                    SimulatorEvent(
                        event_type="chunk_arrival",
                        timestamp_ms=arrival,
                        chunk_index=self._chunk_index,
                        backlog_ms=self.backlog_ms,
                    )
                )
                self.events.append(
                    SimulatorEvent(
                        event_type="processing_start",
                        timestamp_ms=self.virtual_time_ms,
                        chunk_index=self._chunk_index,
                        backlog_ms=self.backlog_ms,
                    )
                )
                start = asyncio.get_event_loop().time()
                result = self.candidate.process_chunk(session, chunk, self.sample_rate)
                elapsed_ms = (asyncio.get_event_loop().time() - start) * 1000.0
                chunk_duration_ms = len(chunk) / self.sample_rate * 1000.0
                self.backlog_ms = max(0.0, self.backlog_ms + elapsed_ms - chunk_duration_ms)
                self.max_backlog_ms = max(self.max_backlog_ms, self.backlog_ms)
                self.virtual_time_ms += chunk_duration_ms
                self.events.append(
                    SimulatorEvent(
                        event_type="processing_end",
                        timestamp_ms=self.virtual_time_ms,
                        chunk_index=self._chunk_index,
                        backlog_ms=self.backlog_ms,
                    )
                )
                self.events.append(
                    SimulatorEvent(
                        event_type="output_ready",
                        timestamp_ms=self.virtual_time_ms,
                        chunk_index=self._chunk_index,
                        backlog_ms=self.backlog_ms,
                    )
                )
                if result is not None and getattr(result, "audio", None) is not None:
                    self._results.append(result)
        finally:
            flush = self.candidate.flush(session)
            self._results.extend(flush)
            self.candidate.close()
        return self._results

    def report(self) -> SimulatorReport:
        rpt = SimulatorReport(total_chunks=self._chunk_index)
        if self.events:
            starts = [e for e in self.events if e.event_type == "processing_start"]
            ends = [e for e in self.events if e.event_type == "processing_end"]
            latencies = [
                e2.timestamp_ms - e1.timestamp_ms
                for e1, e2 in zip(starts, ends)
            ]
            if latencies:
                rpt.latency_p50_ms = float(np.percentile(latencies, 50))
                rpt.latency_p95_ms = float(np.percentile(latencies, 95))
                rpt.latency_p99_ms = float(np.percentile(latencies, 99))
        backlogs = [e.backlog_ms for e in self.events]
        if backlogs:
            rpt.max_backlog_ms = float(np.max(backlogs))
            rpt.avg_backlog_ms = float(np.mean(backlogs))
        rpt.total_output_samples = sum(r.audio.size for r in self._results if r.audio is not None)
        rtf_vals = []
        for e1, e2 in zip(
            [e for e in self.events if e.event_type == "processing_start"],
            [e for e in self.events if e.event_type == "processing_end"],
        ):
            audio_ms = (e2.timestamp_ms - e1.timestamp_ms)
            compute_ms = (e2.timestamp_ms - e1.timestamp_ms)
            if audio_ms > 0:
                rtf_vals.append(compute_ms / audio_ms)
        if rtf_vals:
            rpt.rtf_p50 = float(np.percentile(rtf_vals, 50))
            rpt.rtf_p95 = float(np.percentile(rtf_vals, 95))
            rpt.rtf_p99 = float(np.percentile(rtf_vals, 99))
        return rpt
