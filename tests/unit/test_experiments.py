"""Tests for experiment registry."""

from __future__ import annotations

from datetime import datetime

import pytest

from accentedge_lab.experiments.registry import ExperimentRecord, ExperimentRegistry


class TestExperimentRecord:
    def test_fields(self) -> None:
        rec = ExperimentRecord(
            experiment_id="exp1",
            architecture="arch_a",
            architecture_version="v1",
            config_hash="abc",
            benchmark_version="v1",
            dev_split_hash="def",
            status="completed",
        )
        assert rec.experiment_id == "exp1"
        assert rec.status == "completed"


class TestExperimentRegistry:
    def test_register_and_get(self, tmp_path) -> None:
        reg = ExperimentRegistry(path=str(tmp_path / "reg.json"))
        rec = ExperimentRecord(
            experiment_id="e1",
            architecture="a",
            architecture_version="v1",
            config_hash="x",
            benchmark_version="v1",
            dev_split_hash="y",
        )
        reg.register(rec)
        assert reg.get("e1").experiment_id == "e1"

    def test_list_by_architecture(self, tmp_path) -> None:
        reg = ExperimentRegistry(path=str(tmp_path / "reg.json"))
        for i in range(3):
            reg.register(
                ExperimentRecord(
                    experiment_id=f"e{i}",
                    architecture="arch_a" if i < 2 else "arch_b",
                    architecture_version="v1",
                    config_hash="x",
                    benchmark_version="v1",
                    dev_split_hash="y",
                )
            )
        assert len(reg.list_by_architecture("arch_a")) == 2

    def test_list_completed(self, tmp_path) -> None:
        reg = ExperimentRegistry(path=str(tmp_path / "reg.json"))
        reg.register(
            ExperimentRecord(
                experiment_id="e1",
                architecture="a",
                architecture_version="v1",
                config_hash="x",
                benchmark_version="v1",
                dev_split_hash="y",
                status="completed",
            )
        )
        reg.register(
            ExperimentRecord(
                experiment_id="e2",
                architecture="a",
                architecture_version="v1",
                config_hash="x",
                benchmark_version="v1",
                dev_split_hash="y",
                status="running",
            )
        )
        assert len(reg.list_completed()) == 1
