"""AccentEdge Model Lab — Reporting package.

Provides:
    - Architecture Decision Record (ADR) generation
    - Phase 2 report aggregation
    - Pareto frontier analysis
"""

from accentedge.reporting.adr import (
    build_default_adr_data,
    generate_adr,
)
from accentedge.reporting.pareto import ParetoTable
from accentedge.reporting.phase2_report import Phase2Report

__all__ = [
    "ParetoTable",
    "Phase2Report",
    "generate_adr",
    "build_default_adr_data",
]
