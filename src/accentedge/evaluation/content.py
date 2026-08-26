"""Phase 1 — Content evaluation.

Metrics:
  - WER (via Faster-Whisper ASR)
  - CER (character error rate)
  - Content preservation ratio
"""
from __future__ import annotations

import numpy as np
from jiwer import wer, cer


def compute_wer(reference: str, hypothesis: str) -> float:
    return wer(reference, hypothesis)


def compute_cer(reference: str, hypothesis: str) -> float:
    return cer(reference, hypothesis)


def compute_content_metrics(reference: str, hypothesis: str) -> dict:
    """Compute content preservation metrics for a single utterance."""
    return {
        "wer": compute_wer(reference, hypothesis),
        "cer": compute_cer(reference, hypothesis),
        "reference": reference,
        "hypothesis": hypothesis,
    }


class ContentEvaluator:
    """Batch content evaluation using WER/CER."""

    def __init__(self):
        self.results: list[dict] = []

    def evaluate(self, reference: str, hypothesis: str) -> dict:
        result = compute_content_metrics(reference, hypothesis)
        self.results.append(result)
        return result

    def summary(self) -> dict:
        if not self.results:
            return {"mean_wer": 0.0, "mean_cer": 0.0, "n_samples": 0}
        return {
            "mean_wer": float(np.mean([r["wer"] for r in self.results])),
            "mean_cer": float(np.mean([r["cer"] for r in self.results])),
            "n_samples": len(self.results),
        }
