"""Speaker-aware metric aggregation."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class SpeakerMetric:
    speaker_id: str
    metric_name: str
    value: float
    partition: str


@dataclass
class AggregatedMetric:
    metric_name: str
    point_estimate: float
    std: float
    speaker_count: int
    item_count: int
    slice_key: str | None = None
    details: dict[str, Any] = None

    def __post_init__(self):
        if self.details is None:
            self.details = {}


def aggregate_by_speaker(
    speaker_metrics: list[SpeakerMetric],
    metric_name: str,
    slice_key: str | None = None,
) -> AggregatedMetric:
    """Aggregate metrics at speaker level (not utterance level)."""
    values = [m.value for m in speaker_metrics if m.metric_name == metric_name]
    if not values:
        return AggregatedMetric(
            metric_name=metric_name,
            point_estimate=0.0,
            std=0.0,
            speaker_count=0,
            item_count=0,
            slice_key=slice_key,
        )
    return AggregatedMetric(
        metric_name=metric_name,
        point_estimate=float(np.mean(values)),
        std=float(np.std(values)),
        speaker_count=len(set(m.speaker_id for m in speaker_metrics if m.metric_name == metric_name)),
        item_count=len(values),
        slice_key=slice_key,
    )


def aggregate_all(
    speaker_metrics: list[SpeakerMetric],
) -> dict[str, AggregatedMetric]:
    """Aggregate all unique metric names."""
    names = set(m.metric_name for m in speaker_metrics)
    return {name: aggregate_by_speaker(speaker_metrics, name) for name in names}
