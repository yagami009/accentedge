"""Tests for memory profiling."""

from __future__ import annotations

import numpy as np

from accentedge_lab.profiling.memory import measure_model_memory, measure_session_memory, profile_inference_memory
from accentedge_lab.streaming.session import StreamingSessionState
from tests.fixtures.fake_causal_model import FakeCausalModel


class TestMemoryProfiling:
    def test_model_memory(self) -> None:
        mem = measure_model_memory(FakeCausalModel())
        assert mem == 0

    def test_session_memory(self) -> None:
        sess = StreamingSessionState(session_id="s", candidate_id="c")
        assert measure_session_memory(sess) == 0

    def test_profile_inference(self) -> None:
        model = FakeCausalModel()
        model.prepare("cpu", "fp32")
        sess = model.create_session({})
        try:
            profile = profile_inference_memory(model, sess, np.zeros(1600, dtype=np.float32))
            assert "total_bytes" in profile
        finally:
            model.close()
