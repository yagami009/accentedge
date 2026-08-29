"""JSON report generation."""
from __future__ import annotations
from pathlib import Path
import json
from dataclasses import asdict

def generate_json_report(
    results: dict[str, Any],
    output_path: str | Path,
    run_manifest: dict[str, Any] | None = None,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "benchmark_version": "1.0.0",
        "run_manifest": run_manifest or {},
        "summary": results,
    }
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
