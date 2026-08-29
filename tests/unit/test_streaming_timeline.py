"""Tests for timeline."""

import numpy as np
import pytest

from accentedge_lab.streaming.timeline import Timeline


class TestTimeline:
    def test_add_and_lookup(self) -> None:
        tl = Timeline()
        tl.add(0, 1000, 0, 800)
        out = tl.get_output_for_input(500)
        assert out == 400

    def test_drift(self) -> None:
        tl = Timeline()
        tl.add(0, 1000, 0, 900)
        assert tl.drift_samples() == -100

    def test_empty(self) -> None:
        tl = Timeline()
        assert tl.total_input_samples() == 0
        assert tl.total_output_samples() == 0
