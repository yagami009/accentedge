"""Paired A-vs-B comparison bootstrap."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class PairedResult:
    delta_mean: float
    delta_ci_lower: float
    delta_ci_upper: float
    p_significant: bool
    n_replicates: int
    confidence_level: float


def paired_bootstrap(
    metrics_a: dict[str, float],
    metrics_b: dict[str, float],
    delta_fn: Callable[[list[float], list[float]], float],
    n_replicates: int = 10000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> PairedResult:
    """Paired bootstrap comparing two models.

    Samples speakers WITH replacement, includes all their utterances,
    computes delta(A-B) for each replicate.

    Args:
        metrics_a: {speaker_id: metric_value} for model A
        metrics_b: {speaker_id: metric_value} for model B
        delta_fn: Function (values_a, values_b) -> float
        n_replicates: Number of bootstrap replicates
        confidence_level: CI coverage
        seed: RNG seed

    Returns:
        PairedResult with delta mean, CI, significance
    """
    common_speakers = sorted(set(metrics_a) & set(metrics_b))
    if not common_speakers:
        raise ValueError("No common speakers between A and B")

    vals_a = [metrics_a[s] for s in common_speakers]
    vals_b = [metrics_b[s] for s in common_speakers]
    n = len(common_speakers)
    rng = np.random.RandomState(seed)

    deltas = np.empty(n_replicates, dtype=np.float64)
    for i in range(n_replicates):
        idx = rng.randint(0, n, size=n)
        sample_a = [vals_a[j] for j in idx]
        sample_b = [vals_b[j] for j in idx]
        deltas[i] = delta_fn(sample_a, sample_b)

    delta_mean = float(np.mean(deltas))
    alpha = 1.0 - confidence_level
    ci_lower = float(np.percentile(deltas, 100 * alpha / 2))
    ci_upper = float(np.percentile(deltas, 100 * (1 - alpha / 2)))

    # Significance: CI does not include zero
    p_significant = not (ci_lower <= 0.0 <= ci_upper)

    return PairedResult(
        delta_mean=delta_mean,
        delta_ci_lower=ci_lower,
        delta_ci_upper=ci_upper,
        p_significant=p_significant,
        n_replicates=n_replicates,
        confidence_level=confidence_level,
    )
