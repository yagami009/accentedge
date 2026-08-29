"""Tests for RTF measurement."""

from __future__ import annotations

import numpy as np
import pytest

from accentedge_lab.profiling.rtf import RTFMeasurement, measure_rtf
from tests.fixtures.fake_causal_model import FakeCausalModel


class TestRTFMeasurement:
    def test_formula(self) -> None:
        m = RTFMeasurement(audio_ms=1000.0, compute_ms=500.0)
        assert abs(m.rtf - 0.5) < 1e-9

    def test_zero_audio(self) -> None:
        m = RTFMeasurement(audio_ms=0.0, compute_ms=100.0)
        assert m.rtf == 0.0

    def test_realtime(self) -> None:
        model = FakeCausalModel()
        model.prepare("cpu", "fp32")
        try:
            sess = model.create_session({})
            chunks = [np.zeros(1600, dtype=np.float32) for _ in range(4)]
            ms = measure_rtf(model, sess, chunks, 16000)
            assert all(m.rtf <= 10.0 for m in ms)
        finally:
            model.close()
