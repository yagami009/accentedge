"""Metric table generators."""
from __future__ import annotations
from typing import Any
import numpy as np

def summary_table(results: list[dict[str, Any]]) -> str:
    """Generate a markdown summary table from metric results."""
    lines = ["| Metric | Value | CI Lower | CI Upper | Count |", "|---|---|---|---|---|"]
    for r in results:
        name = r.get("metric_name", "unknown")
        val = r.get("value", 0.0)
        ci_l = r.get("confidence_interval", (None, None))[0]
        ci_u = r.get("confidence_interval", (None, None))[1]
        count = r.get("count", 0)
        lines.append(f"| {name} | {val:.4f} | {ci_l:.4f} | {ci_u:.4f} | {count} |")
    return "\n".join(lines)

def robustness_table(slice_metrics: dict[str, dict[str, float]]) -> str:
    lines = ["| Slice | WER | CER | Identity | Damage |", "|---|---|---|---|---|"]
    for slice_name, m in sorted(slice_metrics.items()):
        wer = m.get("wer", 0.0)
        cer = m.get("cer", 0.0)
        identity = m.get("identity", 0.0)
        damage = m.get("damage", 0.0)
        lines.append(f"| {slice_name} | {wer:.4f} | {cer:.4f} | {identity:.4f} | {damage:.4f} |")
    return "\n".join(lines)
