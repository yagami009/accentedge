"""Sparse-repair candidate package."""

from __future__ import annotations

from accentedge.models.registry import get_registry
from accentedge.models.sparse_repair.sparse_repair_candidate import (
    SparseRepairCandidate,
)
from accentedge.models.sparse_repair.streaming_config import SparseRepairConfig

get_registry().register("sparse_repair", SparseRepairCandidate)

__all__ = ["SparseRepairCandidate", "SparseRepairConfig"]
