"""Tests for the streaming simulator."""

import asyncio

import numpy as np
import pytest

from accentedge_lab.streaming.simulator import StreamingSimulator
from tests.fixtures.fake_causal_model import FakeCausalModel
from tests.fixtures.fake_slow_model import FakeSlowModel


def _run(coro):
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(asyncio.run, coro)
        return future.result()


class TestSimulatorBasic:
    def test_feed_audio(self) -> None:
        model = FakeCausalModel()
        model.prepare("cpu", "fp32")
        try:
            sim = StreamingSimulator(candidate=model, chunk_size_ms=80, sample_rate=16000)
            audio = np.ones(3200, dtype=np.float32)
            results = _run(sim.feed(audio))
            assert len(results) > 0
        finally:
            model.close()

    def test_empty_audio(self) -> None:
        model = FakeCausalModel()
        model.prepare("cpu", "fp32")
        try:
            sim = StreamingSimulator(candidate=model, chunk_size_ms=80, sample_rate=16000)
            results = _run(sim.feed(np.zeros(0, dtype=np.float32)))
            assert results == []
        finally:
            model.close()


class TestSimulatorBacklog:
    def test_slow_model_backlog(self) -> None:
        model = FakeSlowModel()
        model.prepare("cpu", "fp32")
        try:
            sim = StreamingSimulator(candidate=model, chunk_size_ms=80, sample_rate=16000)
            audio = np.ones(3200, dtype=np.float32)
            _run(sim.feed(audio))
            report = sim.report()
            assert report.max_backlog_ms > 0
        finally:
            model.close()
