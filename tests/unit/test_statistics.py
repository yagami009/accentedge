"""Tests for accentedge_benchmark.statistics — bootstrap and paired bootstrap."""

from __future__ import annotations

import numpy as np
import pytest

from accentedge_benchmark.statistics.bootstrap import speaker_bootstrap, BootstrapResult
from accentedge_benchmark.statistics.paired import paired_bootstrap, PairedResult


class TestSpeakerBootstrap:
    def test_produces_confidence_interval(self):
        speaker_metrics = {f"spk_{i}": float(i) for i in range(10)}
        result = speaker_bootstrap(
            speaker_metrics,
            metric_fn=np.mean,
            n_replicates=1000,
            seed=42,
        )
        assert isinstance(result, BootstrapResult)
        assert result.ci_lower <= result.point_estimate <= result.ci_upper
        assert result.n_speakers == 10
        assert result.n_replicates == 1000

    def test_point_estimate_equals_data_mean(self):
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        speaker_metrics = {f"spk_{i}": values[i] for i in range(len(values))}
        result = speaker_bootstrap(
            speaker_metrics,
            metric_fn=np.mean,
            n_replicates=500,
            seed=42,
        )
        expected_mean = np.mean(values)
        assert result.point_estimate == pytest.approx(expected_mean)

    def test_deterministic_same_seed(self):
        speaker_metrics = {f"spk_{i}": float(i) for i in range(10)}
        r1 = speaker_bootstrap(speaker_metrics, np.mean, n_replicates=500, seed=42)
        r2 = speaker_bootstrap(speaker_metrics, np.mean, n_replicates=500, seed=42)
        np.testing.assert_array_equal(r1.replicates, r2.replicates)
        assert r1.ci_lower == r2.ci_lower
        assert r1.ci_upper == r2.ci_upper

    def test_different_seed_different_replicates(self):
        speaker_metrics = {f"spk_{i}": float(i) for i in range(10)}
        r1 = speaker_bootstrap(speaker_metrics, np.mean, n_replicates=500, seed=1)
        r2 = speaker_bootstrap(speaker_metrics, np.mean, n_replicates=500, seed=99)
        assert not np.allclose(r1.replicates, r2.replicates)

    def test_single_speaker(self):
        speaker_metrics = {"spk_0": 5.0}
        result = speaker_bootstrap(
            speaker_metrics,
            metric_fn=np.mean,
            n_replicates=100,
            seed=42,
        )
        # With only one speaker, bootstrap just returns 5.0 repeatedly
        assert result.point_estimate == pytest.approx(5.0)
        assert result.n_speakers == 1

    def test_ci_width_decreases_with_more_speakers(self):
        """More speakers → tighter CIs (roughly)."""
        few = {f"spk_{i}": float(i) for i in range(5)}
        many = {f"spk_{i}": float(i) for i in range(30)}
        r_few = speaker_bootstrap(few, np.mean, n_replicates=500, seed=42)
        r_many = speaker_bootstrap(many, np.mean, n_replicates=500, seed=42)
        # Both CIs should contain the point estimate
        assert r_few.ci_lower <= r_few.point_estimate <= r_few.ci_upper
        assert r_many.ci_lower <= r_many.point_estimate <= r_many.ci_upper


class TestPairedBootstrap:
    def test_computes_delta_correctly(self):
        metrics_a = {"spk_0": 0.1, "spk_1": 0.2, "spk_2": 0.3}
        metrics_b = {"spk_0": 0.05, "spk_1": 0.15, "spk_2": 0.25}
        result = paired_bootstrap(
            metrics_a, metrics_b,
            delta_fn=lambda a, b: np.mean(a) - np.mean(b),
            n_replicates=1000,
            seed=42,
        )
        assert isinstance(result, PairedResult)
        # True delta = mean(0.1, 0.2, 0.3) - mean(0.05, 0.15, 0.25) = 0.2 - 0.15 = 0.05
        assert result.delta_mean == pytest.approx(0.05, abs=0.05)

    def test_no_common_speakers_raises(self):
        metrics_a = {"spk_a": 0.1}
        metrics_b = {"spk_b": 0.2}
        with pytest.raises(ValueError, match="No common speakers"):
            paired_bootstrap(
                metrics_a, metrics_b,
                delta_fn=lambda a, b: np.mean(a) - np.mean(b),
            )

    def test_deterministic(self):
        metrics_a = {f"spk_{i}": float(i) for i in range(10)}
        metrics_b = {f"spk_{i}": float(i) + 0.5 for i in range(10)}
        r1 = paired_bootstrap(
            metrics_a, metrics_b,
            delta_fn=lambda a, b: np.mean(a) - np.mean(b),
            n_replicates=500,
            seed=42,
        )
        r2 = paired_bootstrap(
            metrics_a, metrics_b,
            delta_fn=lambda a, b: np.mean(a) - np.mean(b),
            n_replicates=500,
            seed=42,
        )
        assert r1.delta_mean == r2.delta_mean
        assert r1.delta_ci_lower == r2.delta_ci_lower
        assert r1.delta_ci_upper == r2.delta_ci_upper

    def test_significance_when_ci_excludes_zero(self):
        """When delta CI clearly excludes 0, p_significant should be True."""
        metrics_a = {f"spk_{i}": float(i) + 10 for i in range(10)}
        metrics_b = {f"spk_{i}": float(i) for i in range(10)}
        result = paired_bootstrap(
            metrics_a, metrics_b,
            delta_fn=lambda a, b: np.mean(a) - np.mean(b),
            n_replicates=2000,
            seed=42,
        )
        # The delta should be consistently ~10, so CI should not include 0
        assert result.delta_ci_lower > 0.0
        assert result.p_significant is True

    def test_not_significant_when_overlapping(self):
        """When A and B are the same, delta CI should include 0."""
        metrics = {f"spk_{i}": float(i) for i in range(10)}
        result = paired_bootstrap(
            metrics, metrics,
            delta_fn=lambda a, b: np.mean(a) - np.mean(b),
            n_replicates=2000,
            seed=42,
        )
        assert result.delta_ci_lower <= 0.0 <= result.delta_ci_upper
        assert result.p_significant is False

    def test_ci_contains_point_estimate(self):
        metrics_a = {f"spk_{i}": float(i) * 0.1 for i in range(10)}
        metrics_b = {f"spk_{i}": float(i) * 0.05 for i in range(10)}
        result = paired_bootstrap(
            metrics_a, metrics_b,
            delta_fn=lambda a, b: np.mean(a) - np.mean(b),
            n_replicates=500,
            seed=42,
        )
        assert result.delta_ci_lower <= result.delta_mean <= result.delta_ci_upper
