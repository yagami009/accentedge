"""Integration tests for streaming pipeline."""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

from accentedge_lab.benchmark.adapter import Phase1BenchmarkAdapter
from accentedge_lab.streaming.causality import prefix_invariance_test
from accentedge_lab.streaming.simulator import StreamingSimulator
from tests.fixtures.fake_causal_model import FakeCausalModel


class TestFullStreamingPipeline:
    def test_pipeline_runs(self) -> None:
        model = FakeCausalModel()
        model.prepare("cpu", "fp32")
        try:
            sim = StreamingSimulator(candidate=model, chunk_size_ms=80, sample_rate=16000)
            audio = np.ones(3200, dtype=np.float32)
            results = asyncio.run(sim.feed(audio))
            report = sim.report()
            assert len(results) > 0
            assert report.total_chunks > 0
        finally:
            model.close()


class TestCausalityIntegration:
    def test_end_to_end(self) -> None:
        model = FakeCausalModel()
        model.prepare("cpu", "fp32")
        try:
            audio = np.sin(np.linspace(0, 4 * np.pi, 4000, dtype=np.float32))
            result = prefix_invariance_test(model, audio, 0.0, tolerance_db=60.0)
            assert result.passed is True
        finally:
            model.close()


class TestBenchmarkAdapterIntegration:
    def test_adapter_construction(self) -> None:
        adapter = Phase1BenchmarkAdapter(benchmark_repo_path="accentedge-benchmark")
        assert adapter.benchmark_path is not None
