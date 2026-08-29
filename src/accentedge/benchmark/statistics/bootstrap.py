"""Speaker-level bootstrap confidence intervals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class BootstrapResult:
    point_estimate: float
    ci_lower: float
    ci_upper: float
    replicates: np.ndarray
    n_speakers: int
    n_replicates: int


def speaker_bootstrap(
    speaker_metrics: dict[str, float],
    metric_fn: Callable[[list[float]], float],
    n_replicates: int = 10000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> BootstrapResult:
    """Bootstrap CIs at speaker level."""
    speakers = list(speaker_metrics.keys())
    values = [speaker_metrics[s] for s in speakers]
    rng = np.random.RandomState(seed)

    replicates = np.empty(n_replicates, dtype=np.float64)
    n = len(speakers)
    for i in range(n_replicates):
        idx = rng.randint(0, n, size=n)
        sample = [values[j] for j in idx]
        replicates[i] = metric_fn(sample)

    point = metric_fn(values)
    alpha = 1.0 - confidence_level
    ci_lower = float(np.percentile(replicates, 100 * alpha / 2))
    ci_upper = float(np.percentile(replicates, 100 * (1 - alpha / 2)))

    return BootstrapResult(
        point_estimate=point,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        replicates=replicates,
        n_speakers=n,
        n_replicates=n_replicates,
    )
