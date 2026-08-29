"""Tests for causality harness."""

from __future__ import annotations

import numpy as np
import pytest

from accentedge_lab.streaming.causality import (
    prefix_invariance_test,
    measure_causality,
)
from tests.fixtures.fake_causal_model import FakeCausalModel
from tests.fixtures.fake_leak_model import FakeLeakModel


class TestCausalityHarness:
    def test_causal_model_passes(self) -> None:
        model = FakeCausalModel()
        model.prepare("cpu", "fp32")
        try:
            audio = np.sin(np.linspace(0, 4 * np.pi, 4000, dtype=np.float32))
            result = measure_causality(model, audio, 0.0, tolerance_db=60.0)
            assert result.passed is True
            assert result.verdict == "PASS"
        finally:
            model.close()

    def test_leak_model_fails(self) -> None:
        model = FakeLeakModel()
        model.prepare("cpu", "fp32")
        try:
            sess_a = model.create_session({})
            sess_b = model.create_session({})
            audio_prefix = np.zeros(4000, dtype=np.float32)
            audio_future = np.zeros(200, dtype=np.float32)
            audio_future[-1] = 1.0
            audio_full = np.concatenate([audio_prefix, audio_future])
            r_a = model.process_chunk(sess_a, audio_prefix, 16000)
            r_b = model.process_chunk(sess_b, audio_full, 16000)
            assert np.max(r_a.audio) == 0.0
            assert np.max(r_b.audio) == 1.0
        finally:
            model.close()

    def test_prefix_invariance(self) -> None:
        model = FakeCausalModel()
        model.prepare("cpu", "fp32")
        try:
            audio = np.sin(np.linspace(0, 4 * np.pi, 4000, dtype=np.float32))
            result = measure_causality(model, audio, 0.0, tolerance_db=60.0)
            assert result.passed is True
        finally:
            model.close()
