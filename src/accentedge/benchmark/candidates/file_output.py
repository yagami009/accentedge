"""File output adapter — loads pre-computed WAV files."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..audio.io import load_audio
from .base import BenchmarkContext, CandidateAdapter, CandidateMetadata, CandidateOutput


class FileOutputAdapter(CandidateAdapter):
    """Loads pre-generated candidate outputs from a directory."""

    def __init__(self, output_dir: str | Path) -> None:
        self._output_dir = Path(output_dir)
        self._meta = CandidateMetadata(
            name="file_output",
            version="1.0.0",
            description=f"Pre-generated outputs from {output_dir}",
            target_accent="unknown",
        )

    @property
    def metadata(self) -> CandidateMetadata:
        return self._meta

    def process(
        self,
        audio: np.ndarray,
        sample_rate: int,
        context: BenchmarkContext,
    ) -> CandidateOutput:
        if context.utterance_id is None:
            raise ValueError("utterance_id required in context for file output adapter")
        output_path = self._output_dir / f"{context.utterance_id}.wav"
        if not output_path.exists():
            raise FileNotFoundError(f"Candidate output not found: {output_path}")
        wf, sr = load_audio(output_path, sr=sample_rate)
        return CandidateOutput(
            audio=wf,
            sample_rate=sr,
            metadata={"source_path": str(output_path)},
        )
