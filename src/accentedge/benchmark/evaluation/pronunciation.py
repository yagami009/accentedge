"""Pronunciation evaluation: correction, damage, off-target."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..schemas import PronunciationToken, SourceStatus


@dataclass
class PronunciationResult:
    """Per-feature pronunciation evaluation result."""
    feature: str
    corrected: int = 0
    eligible_correction: int = 0
    damaged: int = 0
    eligible_damage: int = 0
    off_target: int = 0
    ambiguous: int = 0
    correction_rate: float = 0.0
    damage_rate: float = 0.0


class PronunciationEvaluator:
    def evaluate(self, source_tokens, output_tokens):
        from collections import defaultdict
        source_by_feature = defaultdict(list)
        for tok in source_tokens:
            source_by_feature.setdefault(tok.feature, []).append(tok)
        output_by_feature = defaultdict(list)
        for tok in output_tokens:
            output_by_feature.setdefault(tok.feature, []).append(tok)
        results = {}
        for feat, src_toks in source_by_feature.items():
            result = PronunciationResult(feature=feat)
            for tok in src_toks:
                if tok.source_status == SourceStatus.DEVIANT:
                    result.eligible_correction += 1
                elif tok.source_status == SourceStatus.ALREADY_TARGET:
                    result.eligible_damage += 1
                elif tok.source_status == SourceStatus.AMBIGUOUS:
                    result.ambiguous += 1
            out_toks = output_by_feature.get(feat, [])
            for src_tok, out_tok in zip(src_toks, out_toks):
                if src_tok.source_status == SourceStatus.DEVIANT and out_tok.source_status == SourceStatus.ALREADY_TARGET:
                    result.corrected += 1
                if src_tok.source_status == SourceStatus.ALREADY_TARGET and out_tok.source_status == SourceStatus.DEVIANT:
                    result.damaged += 1
            result.correction_rate = (result.corrected / result.eligible_correction) if result.eligible_correction > 0 else 0.0
            result.damage_rate = (result.damaged / result.eligible_damage) if result.eligible_damage > 0 else 0.0
            results[feat] = result
        return list(results.values())
