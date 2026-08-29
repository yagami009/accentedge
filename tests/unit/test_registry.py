"""Tests for model registry."""

from __future__ import annotations

import pytest

from accentedge_lab.models.registry import ModelRegistry, get_registry
from tests.fixtures.fake_causal_model import FakeCausalModel


class TestModelRegistry:
    def test_register_and_get(self) -> None:
        reg = ModelRegistry()
        reg.register("fake_causal", FakeCausalModel)
        assert reg.get("fake_causal") is FakeCausalModel

    def test_list_available(self) -> None:
        reg = ModelRegistry()
        reg.register("fake_causal", FakeCausalModel)
        assert "fake_causal" in reg.list_available()

    def test_create(self) -> None:
        reg = ModelRegistry()
        reg.register("fake_causal", FakeCausalModel)
        instance = reg.create("fake_causal", {})
        assert instance is not None

    def test_missing_raises(self) -> None:
        reg = ModelRegistry()
        with pytest.raises(KeyError):
            reg.get("missing")

    def test_duplicate_raises(self) -> None:
        reg = ModelRegistry()
        reg.register("fake_causal", FakeCausalModel)
        with pytest.raises(ValueError):
            reg.register("fake_causal", FakeCausalModel)
