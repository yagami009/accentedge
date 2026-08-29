"""Benchmark runner."""
from __future__ import annotations
from pathlib import Path
from typing import Any

import pandas as pd

from ..schemas import DatasetItem, RunManifest
from ..audio.io import load_audio
from ..candidates.base import CandidateAdapter, BenchmarkContext, CandidateOutput
from ..runner.run_manifest import create_run_manifest
from ..runner.failures import FailureRecord, record_failure


class BenchmarkRunner:
    def __init__(
        self,
        candidate: CandidateAdapter,
        split: str = "dev",
        condition: str = "clean",
        output_dir: str = "runs/",
        conversion_strength: float | None = None,
        max_parallel: int = 4,
    ):
        self.candidate = candidate
        self.split = split
        self.condition = condition
        self.output_dir = Path(output_dir)
        self.conversion_strength = conversion_strength
        self.max_parallel = max_parallel
        self.failures: list[FailureRecord] = []
        self.outputs: list[CandidateOutput] = []

    def run(self, manifest: list[DatasetItem]) -> dict[str, Any]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        items = [item for item in manifest if item.partition.value == self.split]
        results = []
        for item in items:
            try:
                audio, sr = load_audio(item.canonical_path)
                ctx = BenchmarkContext(
                    target_accent="en-US-neutral",
                    conversion_strength=self.conversion_strength,
                    utterance_id=item.utterance_id,
                    speaker_id=item.speaker_id,
                )
                output = self.candidate.process(audio, sr, ctx)
                self.outputs.append(output)
                results.append({"utterance_id": item.utterance_id, "status": "ok"})
            except Exception as exc:
                self.failures.append(record_failure(
                    item.utterance_id, self.candidate.metadata.name, exc
                ))
                results.append({"utterance_id": item.utterance_id, "status": "failed"})
        return {
            "candidate": self.candidate.metadata.name,
            "split": self.split,
            "condition": self.condition,
            "total_items": len(items),
            "succeeded": sum(1 for r in results if r["status"] == "ok"),
            "failed": len(self.failures),
            "results": results,
        }
