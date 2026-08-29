"""Tests for latency profiling."""

from __future__ import annotations

import numpy as np

from accentedge_lab.profiling.latency import algorithmic_latency_from_config, measure_chunk_latency
from tests.fixtures.fake_causal_model import FakeCausalModel


class TestLatencyBreakdown:
    def test_measure(self) -> None:
        model = FakeCausalModel()
        model.prepare("cpu", "fp32")
        try:
            sess = model.create_session({})
            chunk = np.zeros(1280, dtype=np.float32)
            lb = measure_chunk_latency(model, sess, chunk, 16000)
            assert lb.total_ms >= 0
        finally:
            model.close()

    def test_algorithmic(self) -> None:
        lb = algorithmic_latency_from_config(
            frame_ms=10.0, lookahead_ms=0, model_frames=2, buffer_frames=1, sample_rate=16000
        )
        assert lb.total_ms >= 0
