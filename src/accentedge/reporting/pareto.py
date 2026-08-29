"""Pareto frontier analysis."""

from __future__ import annotations

from typing import Literal


class ParetoTable:
    def __init__(
        self,
        candidates: list[str],
        metrics: list[str],
        data: dict[str, dict[str, float]],
    ) -> None:
        self.candidates = candidates
        self.metrics = metrics
        self.data = data

    def dominated_candidates(self) -> list[str]:
        dominated: list[str] = []
        for a in self.candidates:
            for b in self.candidates:
                if a == b:
                    continue
                if self._dominates(b, a):
                    dominated.append(a)
                    break
        return dominated

    def frontier(self, metric_x: str, metric_y: str) -> list[tuple[str, float, float]]:
        return [
            (c, self.data[c].get(metric_x, 0.0), self.data[c].get(metric_y, 0.0))
            for c in self.candidates
            if not any(
                self._dominates(other, c) and other != c
                for other in self.candidates
            )
        ]

    def _dominates(self, a: str, b: str) -> bool:
        if a not in self.data or b not in self.data:
            return False
        for m in self.metrics:
            if m in ("wer", "cer", "latency_p50_ms", "rtf_p50"):
                if self.data[a][m] >= self.data[b][m]:
                    return False
            else:
                if self.data[a][m] <= self.data[b][m]:
                    return False
        return True
