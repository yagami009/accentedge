"""
Reporting for Phase 0 — gate artifacts, markdown reports, and decision memo.

Implements:
- GateArtifact: one gate's required artifact (spec section 56)
- Phase0Report: accumulates all gate artifacts into the final report
- Report templates for each Phase-0 gate outcome
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Union

from accentedge.phase0.evaluation import EvaluationResult

logger = logging.getLogger(__name__)


class GateOutcome:
    """Phase 0 gate outcomes (from spec)."""
    PENDING = "PENDING"
    PASS = "PASS"
    FAIL = "FAIL"
    FULL_S2S_PASS = "FULL-S2S PASS"
    SPARSE_REPAIR_PASS = "SPARSE-REPAIR PASS"
    TEACHER_FAIL_GOLD_PASS = "TEACHER FAIL / GOLD PASS"
    FUNDAMENTAL_FAIL = "FUNDAMENTAL FAIL"

    ALL = [
        FULL_S2S_PASS,
        SPARSE_REPAIR_PASS,
        TEACHER_FAIL_GOLD_PASS,
        FUNDAMENTAL_FAIL,
    ]


OUTCOME_DESCRIPTIONS = {
    GateOutcome.FULL_S2S_PASS: (
        "Good whole-speech targets can be created; "
        "proceed toward causal direct S2S."
    ),
    GateOutcome.SPARSE_REPAIR_PASS: (
        "Full transformation damages identity, but controlled "
        "pronunciation repair works."
    ),
    GateOutcome.TEACHER_FAIL_GOLD_PASS: (
        "Humans demonstrate the desired transformation, but our "
        "synthetic supervision cannot reproduce it yet."
    ),
    GateOutcome.FUNDAMENTAL_FAIL: (
        "Even natural same-speaker cross-accent behavior cannot "
        "satisfy the intended product contract."
    ),
}


@dataclass
class GateArtifact:
    """
    Represents one gate's required artifact (spec section 56).

    Each gate produces artifacts that feed into the next gate
    and the final decision.
    """
    gate_name: str
    gate_number: str  # e.g. "-1A", "0", "1A", "2", "1B", "DECISION"
    outcome: str = "PENDING"
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    artifacts: dict = field(default_factory=dict)
    notes: str = ""
    elapsed_seconds: float = 0.0

    def to_dict(self) -> dict:
        return {
            "gate_name": self.gate_name,
            "gate_number": self.gate_number,
            "outcome": self.outcome,
            "timestamp": self.timestamp,
            "artifacts": self.artifacts,
            "notes": self.notes,
            "elapsed_seconds": self.elapsed_seconds,
        }


class Phase0Report:
    """
    Accumulates all gate artifacts into the final Phase 0 report.

    Generates:
    - Per-gate markdown reports
    - One-page decision memo
    - Complete Phase 0 report with all artifacts
    """

    def __init__(self, experiment_id: str, config: dict = None):
        self.experiment_id = experiment_id
        self.config = config or {}
        self.artifacts: list[GateArtifact] = []
        self.evaluation_results: list[EvaluationResult] = []
        self.final_decision: Optional[str] = None
        self.decision_rationale: str = ""

    def add_gate_artifact(self, artifact: GateArtifact) -> None:
        """Add a gate artifact to the report."""
        self.artifacts.append(artifact)
        logger.info(
            "Gate artifact added: %s (%s)", artifact.gate_name, artifact.outcome
        )

    def add_evaluation_result(self, result: EvaluationResult) -> None:
        """Add an evaluation result."""
        self.evaluation_results.append(result)

    def generate_gate_report(self, artifact: GateArtifact) -> str:
        """
        Generate formatted markdown for one gate's report.

        Includes gate metadata, outcome, artifacts, and notes.
        """
        lines = [
            f"## {artifact.gate_name}",
            "",
            f"**Gate number:** {artifact.gate_number}  ",
            f"**Outcome:** {artifact.outcome}  ",
            f"**Timestamp:** {artifact.timestamp}  ",
            f"**Elapsed:** {artifact.elapsed_seconds:.1f}s  ",
            "",
        ]

        if artifact.outcome in OUTCOME_DESCRIPTIONS:
            lines.append(f"*{OUTCOME_DESCRIPTIONS[artifact.outcome]}*")
            lines.append("")

        if artifact.artifacts:
            lines.append("### Artifacts")
            lines.append("")
            lines.append("| Key | Value |")
            lines.append("|-----|-------|")
            for key, value in artifact.artifacts.items():
                lines.append(f"| {key} | {value} |")
            lines.append("")

        if artifact.notes:
            lines.append("### Notes")
            lines.append("")
            lines.append(artifact.notes)
            lines.append("")

        return "\n".join(lines)

    def generate_decision_memo(self) -> str:
        """
        Generate the one-page Phase-0 decision memo.

        Summarizes gate outcomes, key metrics, and the final decision
        with rationale.
        """
        lines = [
            "# Phase-0 Decision Memo",
            "",
            f"**Experiment ID:** {self.experiment_id}  ",
            f"**Generated:** {datetime.now(timezone.utc).isoformat()}  ",
            "",
            "---",
            "",
            "## Executive Summary",
            "",
            self.decision_rationale or (
                "No final decision recorded. Complete all gates before generating the decision memo."
            ),
            "",
            f"**Final Decision:** {self.final_decision or 'PENDING'}  ",
            "",
            "---",
            "",
            "## Gate Outcomes",
            "",
            "| Gate | Number | Outcome | Elapsed (s) |",
            "|------|--------|---------|-------------|",
        ]

        for art in self.artifacts:
            lines.append(
                f"| {art.gate_name} | {art.gate_number} | {art.outcome} | {art.elapsed_seconds:.1f} |"
            )

        lines.append("")

        # Key metrics summary
        if self.evaluation_results:
            lines.append("## Key Metrics")
            lines.append("")
            lines.append("| Utterance | Strategy | WER | Identity | Duration Ratio |")
            lines.append("|-----------|----------|-----|----------|----------------|")
            for r in self.evaluation_results:
                wer = f"{r.word_error_rate:.3f}" if r.word_error_rate is not None else "—"
                identity = f"{r.identity_score:.3f}" if r.identity_score is not None else "—"
                dur = f"{r.duration_ratio:.3f}" if r.duration_ratio is not None else "—"
                lines.append(
                    f"| {r.utterance_id} | {r.strategy} | {wer} | {identity} | {dur} |"
                )
            lines.append("")

        # Decision interpretation
        if self.final_decision and self.final_decision in OUTCOME_DESCRIPTIONS:
            lines.append("## Decision Interpretation")
            lines.append("")
            lines.append(f"**{self.final_decision}**")
            lines.append("")
            lines.append(OUTCOME_DESCRIPTIONS[self.final_decision])
            lines.append("")

        lines.extend([
            "---",
            "",
            "*Generated by AccentEdge Phase 0 experiment harness.*",
            f"*Experiment: {self.experiment_id}*",
        ])

        return "\n".join(lines)

    def export_full_report(self, output_dir: Union[str, Path]) -> Path:
        """
        Generate the complete Phase 0 report with all gate artifacts.

        Writes:
        - full_report.md: the complete report
        - gate_artifacts/: individual gate artifact files
        - decision_memo.md: the one-page decision memo

        Returns path to full_report.md.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Full report
        lines = [
            f"# Phase 0 Full Report — {self.experiment_id}",
            "",
            f"**Generated:** {datetime.now(timezone.utc).isoformat()}  ",
            f"**Config:** {self.config.get('description', '—')}  ",
            "",
            "---",
            "",
        ]

        # Table of contents
        lines.append("## Table of Contents")
        lines.append("")
        for art in self.artifacts:
            anchor = art.gate_name.lower().replace(" ", "_").replace("/", "_")
            lines.append(f"- [{art.gate_name}](#{anchor})")
        lines.append("- [Decision Memo](#decision-memo)")
        lines.append("")

        # Individual gate reports
        for art in self.artifacts:
            lines.append(self.generate_gate_report(art))
            lines.append("---")
            lines.append("")

        # Decision memo
        lines.append("<a name='decision-memo'></a>")
        lines.append(self.generate_decision_memo())
        lines.append("")

        # Save individual gate artifacts
        gate_dir = output_dir / "gate_artifacts"
        gate_dir.mkdir(exist_ok=True)
        import json as _json
        for art in self.artifacts:
            artifact_path = gate_dir / f"{art.gate_name.replace('/', '_')}.json"
            with open(artifact_path, "w") as f:
                _json.dump(art.to_dict(), f, indent=2)

        # Save full report
        report_path = output_dir / "full_report.md"
        with open(report_path, "w") as f:
            f.write("\n".join(lines))

        # Save decision memo separately
        memo_path = output_dir / "decision_memo.md"
        with open(memo_path, "w") as f:
            f.write(self.generate_decision_memo())

        logger.info("Full report exported: %s", report_path)
        return report_path
