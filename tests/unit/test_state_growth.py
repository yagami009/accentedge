"""Tests for state growth measurement."""

import asyncio

import numpy as np
import pytest

from accentedge_lab.streaming.state_growth import measure_state_growth
from tests.fixtures.fake_causal_model import FakeCausalModel
from tests.fixtures.fake_growing_state_model import FakeGrowingStateModel


def _run(coro):
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(asyncio.run, coro)
        return future.result()


class TestStateGrowthBounded:
    def test_bounded(self) -> None:
        result = _run(measure_state_growth(lambda: FakeCausalModel(), duration_seconds=61))
        assert result.verdict == "BOUNDED"


class TestStateGrowthCatchesGrowth:
    def test_flagged(self) -> None:
        result = _run(measure_state_growth(lambda: FakeGrowingStateModel(), duration_seconds=61))
        assert result.verdict == "LINEAR_GROWTH"
