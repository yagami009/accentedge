"""Robustness reporting — slice metrics by condition."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any

from .content import ContentResult


@dataclass
class RobustnessSlice:
    condition: str
    content_wer: float | None = None
    content_cer: float | None = None
    identity_distance: float | None = None
    timing_ratio: float | None = None
    artifact_errors: int = 0
    n_items: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


class RobustnessEvaluator:
    def __init__(self, slices: list[str] | None = None):
        self.slices = slices or ["clean", "nb", "noisy", "nb_noisy"]
        self._results: dict[str, list[Any]] = {s: [] for s in self.slices}

    def add_result(self, condition: str, result: Any) -> None:
        if condition in self._results:
            self._results[condition].append(result)

    def report(self) -> list[RobustnessSlice]:
        reports = []
        for condition, results in self._results.items():
            n = len(results)
            if n == 0:
                continue
            avg_wer = np.mean([r.wer for r in results if hasattr(r, 'wer') and r.wer is not None]) if any(hasattr(r, 'wer') and r.wer is not None for r in results) else None
            reports.append(RobustnessSlice(condition=condition, content_wer=avg_wer, n_items=n))
        return reports
