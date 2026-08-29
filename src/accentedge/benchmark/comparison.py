"""Pareto frontier analysis for comparing benchmark sweep results."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from accentedge.benchmark.sweeps import SweepResult


# ---------------------------------------------------------------------------
# ParetoFrontier
# ---------------------------------------------------------------------------

@dataclass
class ParetoFrontier:
    """A Pareto frontier of non-dominated SweepResults."""

    non_dominated: list[SweepResult] = field(default_factory=list)
    dominated: list[SweepResult] = field(default_factory=list)

    def summary_table(self) -> list[dict[str, Any]]:
        """Return a table-like list of dicts for the non-dominated frontier."""
        return [
            {
                "candidate_id": r.candidate_id,
                "chunk_size_ms": r.chunk_size_ms,
                "lookahead_ms": r.lookahead_ms,
                **r.metrics,
                "latency_ms": r.latency_ms,
                "rtf": r.rtf,
                "state_size_bytes": r.state_size_bytes,
                "status": "non-dominated",
            }
            for r in self.non_dominated
        ]

    def dominated_table(self) -> list[dict[str, Any]]:
        """Return a table-like list of dicts for dominated configurations."""
        return [
            {
                "candidate_id": r.candidate_id,
                "chunk_size_ms": r.chunk_size_ms,
                "lookahead_ms": r.lookahead_ms,
                **r.metrics,
                "latency_ms": r.latency_ms,
                "rtf": r.rtf,
                "state_size_bytes": r.state_size_bytes,
                "status": "dominated",
            }
            for r in self.dominated
        ]


# ---------------------------------------------------------------------------
# Dominance helpers
# ---------------------------------------------------------------------------

def _dominates(a: SweepResult, b: SweepResult) -> bool:
    """A dominates B if A >= B on all quality axes and A < B on all cost axes.

    Quality axes (higher is better):
      - content
      - identity
      - rtf (lower is better, so we compare inverse: 1/rtf)

    Cost axes (lower is better):
      - latency_ms
      - rtf
      - state_size_bytes
    """
    # Quality scores from metrics dict (default to 0 if missing)
    a_content = a.metrics.get("content", 0.0)
    a_identity = a.metrics.get("identity", 0.0)
    b_content = b.metrics.get("content", 0.0)
    b_identity = b.metrics.get("identity", 0.0)

    # A >= B on all quality axes
    a_better_quality = (a_content >= b_content) and (a_identity >= b_identity)

    # A < B on all cost axes (strictly less on at least one)
    a_lower_latency = a.latency_ms < b.latency_ms
    a_lower_rtf = a.rtf < b.rtf
    a_lower_state = a.state_size_bytes < b.state_size_bytes
    a_better_cost = a_lower_latency or a_lower_rtf or a_lower_state

    # No cost regression (A <= B on all cost axes)
    no_cost_regression = (
        a.latency_ms <= b.latency_ms
        and a.rtf <= b.rtf
        and a.state_size_bytes <= b.state_size_bytes
    )

    return a_better_quality and a_better_cost and no_cost_regression


def _equal(a: SweepResult, b: SweepResult) -> bool:
    """Two results are equal if they are identical on every measured axis."""
    return (
        a.candidate_id == b.candidate_id
        and a.chunk_size_ms == b.chunk_size_ms
        and a.lookahead_ms == b.lookahead_ms
        and a.metrics == b.metrics
        and a.latency_ms == b.latency_ms
        and a.rtf == b.rtf
        and a.state_size_bytes == b.state_size_bytes
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compare_candidates(
    candidates: dict[str, list[SweepResult]],
) -> ParetoFrontier:
    """Compare all sweep results across multiple candidates and return the
    Pareto frontier of non-dominated configurations.

    Args:
        candidates: mapping from candidate_id → list of SweepResults from
                    ChunkSweep / LookaheadSweep / CombinedSweep.

    Returns:
        ParetoFrontier with non_dominated and dominated lists.
    """
    all_results: list[SweepResult] = []
    for results in candidates.values():
        all_results.extend(results)

    return compute_pareto_frontier(all_results)


def compute_pareto_frontier(
    results: list[SweepResult],
) -> ParetoFrontier:
    """Compute the Pareto frontier from a flat list of SweepResults.

    A result is non-dominated if no other result strictly dominates it.
    """
    non_dominated: list[SweepResult] = []
    dominated: list[SweepResult] = []

    for candidate in results:
        is_dominated = False
        # Check if any other result dominates this one
        for other in results:
            if other is candidate or _equal(candidate, other):
                continue
            if _dominates(other, candidate):
                is_dominated = True
                break

        if is_dominated:
            dominated.append(candidate)
        else:
            non_dominated.append(candidate)

    return ParetoFrontier(non_dominated=non_dominated, dominated=dominated)


def report_per_candidate(
    candidates: dict[str, list[SweepResult]],
) -> dict[str, Any]:
    """Generate a per-candidate summary with best/worst content and latency."""
    summary: dict[str, Any] = {}
    for candidate_id, results in candidates.items():
        if not results:
            summary[candidate_id] = {"count": 0}
            continue
        contents = [r.metrics.get("content", 0.0) for r in results]
        latencies = [r.latency_ms for r in results]
        summary[candidate_id] = {
            "count": len(results),
            "best_content": max(contents),
            "worst_content": min(contents),
            "best_latency_ms": min(latencies),
            "worst_latency_ms": max(latencies),
            "mean_content": sum(contents) / len(contents),
            "mean_latency_ms": sum(latencies) / len(latencies),
        }
    return summary


def generate_frontier_report(
    candidates: dict[str, list[SweepResult]],
) -> dict[str, Any]:
    """Full report: per-candidate summary + Pareto table + dominated list."""
    frontier = compare_candidates(candidates)
    return {
        "per_candidate": report_per_candidate(candidates),
        "pareto_frontier": frontier.summary_table(),
        "dominated": frontier.dominated_table(),
        "n_non_dominated": len(frontier.non_dominated),
        "n_dominated": len(frontier.dominated),
        "n_total": len(frontier.non_dominated) + len(frontier.dominated),
    }
