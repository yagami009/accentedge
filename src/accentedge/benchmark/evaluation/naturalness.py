"""Naturalness evaluation (automatic MOS screen)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class NaturalnessResult:
    predicted_mos: float | None = None
    human_mos: float | None = None
    evaluator_name: str = "unavailable"
    is_auto: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


class NaturalnessEvaluator:
    """Automatic naturalness screen (e.g., Distill-MOS)."""

    def __init__(self, model_name: str = "microsoft/distill-mos"):
        self.model_name = model_name
        self._model = None

    def _load(self):
        if self._model is None:
            try:
                from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
                self._feature_extractor = AutoFeatureExtractor.from_pretrained(self.model_name)
                self._model = AutoModelForAudioClassification.from_pretrained(self.model_name)
            except (ImportError, OSError):
                pass  # Model unavailable — return without evaluation

    def evaluate(self, audio_path: str) -> NaturalnessResult:
        self._load()
        if self._model is None:
            return NaturalnessResult(evaluator_name="unavailable")
        try:
            import librosa
            audio, sr = librosa.load(audio_path, sr=16000, mono=True)
            import torch
            inputs = self._feature_extractor(audio, sampling_rate=sr, return_tensors="pt")
            with torch.no_grad():
                outputs = self._model(**inputs)
            logits = outputs.logits
            mos = float(torch.sigmoid(logits).mean().item())
            return NaturalnessResult(predicted_mos=mos * 5.0, evaluator_name="distill-mos", is_auto=True)
        except Exception:
            return NaturalnessResult(evaluator_name="error")

