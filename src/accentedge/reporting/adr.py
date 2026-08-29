"""Architecture Decision Record generator.

Provides templates and rendering for Phase 2 ADR documents.
"""

from __future__ import annotations

from typing import Any


# ---------------------------------------------------------------------------
# Section templates
# ---------------------------------------------------------------------------

def _section_decision(data: dict[str, Any]) -> str:
    chosen = data.get("chosen_architecture", "UNKNOWN")
    backup = data.get("backup_architecture", "NONE")
    return f"""\
## Decision

**{chosen}** is selected as the architecture to advance to Phase 3.

**Backup**: {backup}
"""


def _section_context(data: dict[str, Any]) -> str:
    context = data.get("context", "Phase 2 architecture bake-off.")
    return f"""\
## Context

{context}
"""


def _section_candidates(data: dict[str, Any]) -> str:
    candidates = data.get("candidates", {})
    lines = ["## Candidates Evaluated", ""]
    for arch_id, info in candidates.items():
        desc = info.get("description", "")
        status = info.get("status", "UNKNOWN")
        lines.append(f"### {arch_id}")
        lines.append("")
        lines.append(f"- **Status**: {status}")
        if desc:
            lines.append(f"- {desc}")
        lines.append("")
    return "\n".join(lines)


def _section_data(data: dict[str, Any]) -> str:
    datasets = data.get("data_used", [])
    lines = ["## Data Used", ""]
    if not datasets:
        lines.append("No data records provided.")
    else:
        lines.append("| Dataset | Description | Purpose |")
        lines.append("|---------|-------------|---------|")
        for ds in datasets:
            lines.append(
                f"| {ds.get('name', '—')} "
                f"| {ds.get('description', '—')} "
                f"| {ds.get('purpose', '—')} |"
            )
    lines.append("")
    return "\n".join(lines)


def _section_quality_gates(data: dict[str, Any]) -> str:
    gates = data.get("quality_gates", [])
    candidates = data.get("candidates", {})
    arch_ids = list(candidates.keys())

    lines = ["## Quality Gates", ""]
    if not gates:
        lines.append("No quality gates defined.")
        return "\n".join(lines)

    header = "| Gate | Description |" + "".join(f" {a} |" for a in arch_ids)
    sep = "|------|-------------|" + "".join("------|" for _ in arch_ids)
    lines.append(header)
    lines.append(sep)

    for gate in gates:
        name = gate.get("name", "—")
        desc = gate.get("description", "—")
        row = f"| {name} | {desc} |"
        for arch_id in arch_ids:
            result = candidates.get(arch_id, {}).get("quality_gate_results", {}).get(
                name, "TBD"
            )
            row += f" {result} |"
        lines.append(row)

    lines.append("")
    return "\n".join(lines)


def _section_streaming_gates(data: dict[str, Any]) -> str:
    gates = data.get("streaming_gates", [])
    candidates = data.get("candidates", {})
    arch_ids = list(candidates.keys())

    lines = ["## Streaming Gates", ""]
    if not gates:
        lines.append("No streaming gates defined.")
        return "\n".join(lines)

    header = "| Gate | Description |" + "".join(f" {a} |" for a in arch_ids)
    sep = "|------|-------------|" + "".join("------|" for _ in arch_ids)
    lines.append(header)
    lines.append(sep)

    for gate in gates:
        name = gate.get("name", "—")
        desc = gate.get("description", "—")
        row = f"| {name} | {desc} |"
        for arch_id in arch_ids:
            result = candidates.get(arch_id, {}).get("streaming_gate_results", {}).get(
                name, "TBD"
            )
            row += f" {result} |"
        lines.append(row)

    lines.append("")
    return "\n".join(lines)


def _section_latency(data: dict[str, Any]) -> str:
    candidates = data.get("candidates", {})

    lines = ["## Latency Results", ""]

    # Algorithmic latency table
    lines.append("### Algorithmic Latency (from config, ms)")
    arch_ids = list(candidates.keys())
    if arch_ids:
        header = "| Candidate | Frame Accumulation | Lookahead | Model Structural | Output Buffer | Total |"
        sep =    "|-----------|-------------------|-----------|------------------|---------------|-------|"
        lines.append(header)
        lines.append(sep)
        for arch_id in arch_ids:
            lat = candidates[arch_id].get("algorithmic_latency_ms", {})
            fa = lat.get("frame_accumulation_ms", _TBD())
            la = lat.get("lookahead_ms", _TBD())
            ms = lat.get("model_structural_ms", _TBD())
            ob = lat.get("output_buffer_ms", _TBD())
            total = lat.get("total_ms", _TBD())
            lines.append(
                f"| {arch_id} | {fa} | {la} | {ms} | {ob} | **{total}** |"
            )
    lines.append("")

    # Compute latency table
    lines.append("### Compute Latency (measured, ms/chunk at 16 kHz)")
    if arch_ids:
        header = "| Candidate | P50 (ms) | P95 (ms) | E2E P50 (ms) | RTF |"
        sep =    "|-----------|----------|----------|--------------|-----|"
        lines.append(header)
        lines.append(sep)
        for arch_id in arch_ids:
            comp = candidates[arch_id].get("compute_latency_ms", {})
            p50 = comp.get("p50_ms", _TBD())
            p95 = comp.get("p95_ms", _TBD())
            e2e = comp.get("e2e_p50_ms", _TBD())
            rtf = comp.get("rtf_p50", _TBD())
            lines.append(f"| {arch_id} | {p50} | {p95} | {e2e} | {rtf} |")
    lines.append("")
    lines.append("*RTF = real-time factor. Target: < 1.0 on consumer CPU, < 0.3 on consumer GPU.*")
    lines.append("")
    return "\n".join(lines)


def _section_resources(data: dict[str, Any]) -> str:
    candidates = data.get("candidates", {})
    arch_ids = list(candidates.keys())

    lines = ["## Resource Results", ""]

    # Parameter / memory table
    lines.append("### Parameters and Memory")
    if arch_ids:
        header = "| Candidate | Parameter Count | Model Memory (fp32) | Session State (per hr) |"
        sep =    "|-----------|----------------|---------------------|----------------------|"
        lines.append(header)
        lines.append(sep)
        for arch_id in arch_ids:
            res = candidates[arch_id].get("resources", {})
            params = res.get("parameter_count", _TBD())
            mem = res.get("model_memory_bytes", _TBD())
            state = res.get("session_state_per_hour", _TBD())
            lines.append(f"| {arch_id} | {params} | {mem} | {state} |")
    lines.append("")

    # Training cost table
    lines.append("### Training Cost")
    if arch_ids:
        header = "| Candidate | Training GPU-Hours (est.) | Training Cost (USD est.) |"
        sep =    "|-----------|--------------------------|--------------------------|"
        lines.append(header)
        lines.append(sep)
        for arch_id in arch_ids:
            res = candidates[arch_id].get("resources", {})
            gpu_hrs = res.get("training_gpu_hours", _TBD())
            cost = res.get("training_cost_usd", _TBD())
            lines.append(f"| {arch_id} | {gpu_hrs} | {cost} |")
    lines.append("")
    return "\n".join(lines)


def _section_rejected(data: dict[str, Any]) -> str:
    rejected = data.get("rejected_alternatives", [])
    lines = ["## Rejected Alternatives", ""]
    if not rejected:
        lines.append("No rejected alternatives recorded.")
        return "\n".join(lines)

    for entry in rejected:
        arch_id = entry.get("architecture_id", "—")
        reason = entry.get("reason", "No reason provided.")
        lines.append(f"### Why {arch_id} was rejected")
        lines.append("")
        lines.append(f"- {reason}")
        lines.append("")

    return "\n".join(lines)


def _section_chosen(data: dict[str, Any]) -> str:
    chosen = data.get("chosen_architecture", "UNKNOWN")
    chosen_info = data.get("candidates", {}).get(chosen, {})
    rationale = chosen_info.get("rationale", "No rationale provided.")
    limitations = chosen_info.get("known_limitations", [])

    lines = [
        "## Chosen Architecture",
        "",
        f"**{chosen}**",
        "",
        "### Rationale",
        "",
    ]
    for point in rationale:
        lines.append(f"- {point}")
    lines.append("")

    if limitations:
        lines.append("### Known Limitations")
        lines.append("")
        for lim in limitations:
            lines.append(f"- {lim}")
        lines.append("")

    return "\n".join(lines)


def _section_backup(data: dict[str, Any]) -> str:
    backup = data.get("backup_architecture", "NONE")
    if backup == "NONE":
        return "## Backup Architecture\n\nNo backup architecture selected.\n"

    backup_info = data.get("candidates", {}).get(backup, {})
    rationale = backup_info.get("rationale", [])
    risks = backup_info.get("risks", [])

    lines = [
        "## Backup Architecture",
        "",
        f"**{backup}**",
        "",
        "### Rationale",
        "",
    ]
    if rationale:
        for point in rationale:
            lines.append(f"- {point}")
    else:
        lines.append("- Selected as runner-up in Phase 2 evaluation.")
    lines.append("")

    if risks:
        lines.append("### Risks")
        lines.append("")
        for risk in risks:
            lines.append(f"- {risk}")
        lines.append("")

    return "\n".join(lines)


def _section_risks(data: dict[str, Any]) -> str:
    risks = data.get("known_risks", [])
    lines = ["## Known Risks", ""]
    if not risks:
        lines.append("No risks recorded.")
        return "\n".join(lines)

    lines.append("| Risk | Likelihood | Impact | Mitigation |")
    lines.append("|------|-----------|--------|------------|")
    for risk in risks:
        lines.append(
            f"| {risk.get('description', '—')} "
            f"| {risk.get('likelihood', '—')} "
            f"| {risk.get('impact', '—')} "
            f"| {risk.get('mitigation', '—')} |"
        )
    lines.append("")
    return "\n".join(lines)


def _section_phase3(data: dict[str, Any]) -> str:
    tasks = data.get("phase3_tasks", [])
    lines = ["## What Phase 3 Must Solve", ""]
    if not tasks:
        lines.append("No Phase 3 tasks recorded.")
        return "\n".join(lines)

    for i, task in enumerate(tasks, 1):
        lines.append(f"{i}. **{task.get('title', 'Task')}**: {task.get('description', '')}")
    lines.append("")
    return "\n".join(lines)


def _section_criteria(data: dict[str, Any]) -> str:
    criteria = data.get("decision_criteria", [])
    scores = data.get("pareto_scores", {})
    lines = [
        "## Appendix: Decision Criteria",
        "",
        "The following criteria were used with equal weighting (1/5 each):",
        "",
    ]
    if criteria:
        lines.append("| # | Criterion | Weight | Description |")
        lines.append("|---|-----------|--------|-------------|")
        for i, c in enumerate(criteria, 1):
            lines.append(
                f"| {i} | {c.get('name', '—')} "
                f"| {c.get('weight', '—')} "
                f"| {c.get('description', '—')} |"
            )
        lines.append("")

    if scores:
        arch_ids = list(scores.keys())
        crit_names = [c.get("name", f"C{i}") for i, c in enumerate(criteria)]
        header = "| Candidate |" + "".join(f" {n} |" for n in crit_names) + " **Score** |"
        sep =    "|-----------|" + "".join("------|" for _ in crit_names) + "---------|"
        lines.append(header)
        lines.append(sep)
        for arch_id in arch_ids:
            row = f"| **{arch_id}** |"
            s = scores[arch_id]
            total = 0.0
            for i, c in enumerate(criteria):
                w = c.get("weight", 0.2)
                val = s.get(c.get("name", f"C{i}"), 0)
                row += f" {val}/5 |"
                total += val * w
            row += f" **{total:.1f}** |"
            lines.append(row)

    lines.append("")
    lines.append("*Scores are preliminary design-phase estimates. Final scores will be computed from benchmark data.*")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _TBD() -> str:
    return "_TBD_"


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------

def generate_adr(phase2_results: dict[str, Any]) -> str:
    """Render a complete Architecture Decision Record from Phase 2 results.

    Args:
        phase2_results: Dictionary containing all Phase 2 evaluation data.
            Expected keys:
                - context: str
                - chosen_architecture: str
                - backup_architecture: str
                - candidates: dict[str, dict]
                - data_used: list[dict]
                - quality_gates: list[dict]
                - streaming_gates: list[dict]
                - known_risks: list[dict]
                - phase3_tasks: list[dict]
                - decision_criteria: list[dict]
                - pareto_scores: dict[str, dict]

    Returns:
        Full ADR document as a Markdown string.
    """
    sections = [
        "# Architecture Decision Record — Phase 2 Bake-Off",
        "",
        f"**Version:** 1.0  ",
        f"**Date:** {phase2_results.get('date', '2026-08-24')}  ",
        f"**Status:** {phase2_results.get('status', 'Accepted')}  ",
        "",
        "---",
        "",
        _section_decision(phase2_results),
        _section_context(phase2_results),
        _section_candidates(phase2_results),
        _section_data(phase2_results),
        _section_quality_gates(phase2_results),
        _section_streaming_gates(phase2_results),
        _section_latency(phase2_results),
        _section_resources(phase2_results),
        _section_rejected(phase2_results),
        _section_chosen(phase2_results),
        _section_backup(phase2_results),
        _section_risks(phase2_results),
        _section_phase3(phase2_results),
        _section_criteria(phase2_results),
        "---",
        "",
        "*This ADR was generated by `src/accentedge.models/reporting/adr.py`.*",
        "",
    ]
    return "\n".join(sections)


def build_default_adr_data() -> dict[str, Any]:
    """Return a populated ADR data dict with placeholder values.

    Use this as a starting template; populate actual benchmark numbers
    after training completes.
    """
    return {
        "date": "2026-08-24",
        "status": "Accepted",
        "context": (
            "Phase 2 of the AccentEdge model-lab was a structured architecture "
            "bake-off. Four candidate families were implemented with full streaming "
            "interfaces, and a fifth (Sparse Repair) was scaffolded for future "
            "evaluation. The goal was to identify which architecture best satisfies "
            "AccentEdge's hard constraints: streaming-first, real-time capable, "
            "compact, and quality-floor."
        ),
        "chosen_architecture": "minimal_hybrid",
        "backup_architecture": "articulatory_ddsp",
        "candidates": {
            "streaming_ac": {
                "description": (
                    "Paper-style baseline with configurable lookahead and "
                    "low-lookahead mode. Content encoder → speaker encoder → "
                    "accent bottleneck → synthesizer."
                ),
                "status": "Rejected",
                "quality_gate_results": {
                    "Content Preservation": "TBD",
                    "Identity Preservation": "TBD",
                    "Damage Prevention": "TBD",
                    "Critical Entities": "TBD",
                },
                "streaming_gate_results": {
                    "Causality": "TBD",
                    "State-Boundedness": "TBD",
                    "Prefix Invariance": "TBD",
                },
                "algorithmic_latency_ms": {
                    "frame_accumulation_ms": 80,
                    "lookahead_ms": 640,
                    "model_structural_ms": 80,
                    "output_buffer_ms": 80,
                    "total_ms": 880,
                },
                "compute_latency_ms": {
                    "p50_ms": "_TBD_",
                    "p95_ms": "_TBD_",
                    "e2e_p50_ms": "_TBD_",
                    "rtf_p50": "_TBD_",
                },
                "resources": {
                    "parameter_count": "_TBD_",
                    "model_memory_bytes": "_TBD_",
                    "session_state_per_hour": "_TBD_",
                    "training_gpu_hours": "_TBD_",
                    "training_cost_usd": "_TBD_",
                },
                "rationale": [
                    "Paper-style mode: 640 ms lookahead exceeds conversational latency budget.",
                    "Low-lookahead mode: marginal benefit over D does not justify complexity.",
                    "Encoder cache state grows unboundedly with session length.",
                ],
            },
            "articulatory_ddsp": {
                "description": (
                    "Articulatory/DDSP candidate. Waveform encoder → articulatory "
                    "feature mapper → DDSP harmonic+noise synthesizer. 10 ms frames, "
                    "0 ms lookahead."
                ),
                "status": "Backup",
                "quality_gate_results": {
                    "Content Preservation": "TBD",
                    "Identity Preservation": "TBD",
                    "Damage Prevention": "TBD",
                    "Critical Entities": "TBD",
                },
                "streaming_gate_results": {
                    "Causality": "TBD",
                    "State-Boundedness": "TBD",
                    "Prefix Invariance": "TBD",
                },
                "algorithmic_latency_ms": {
                    "frame_accumulation_ms": 10,
                    "lookahead_ms": 0,
                    "model_structural_ms": 10,
                    "output_buffer_ms": 10,
                    "total_ms": 30,
                },
                "compute_latency_ms": {
                    "p50_ms": "_TBD_",
                    "p95_ms": "_TBD_",
                    "e2e_p50_ms": "_TBD_",
                    "rtf_p50": "_TBD_",
                },
                "resources": {
                    "parameter_count": "_TBD_",
                    "model_memory_bytes": "_TBD_",
                    "session_state_per_hour": "_TBD_",
                    "training_gpu_hours": "_TBD_",
                    "training_cost_usd": "_TBD_",
                },
                "rationale": [
                    "Passes streaming gates (0 ms lookahead, causal, bounded).",
                    "Physically interpretable control space.",
                    "If D's quality ceiling is insufficient, B offers a different trade-off.",
                ],
                "risks": [
                    "DDSP synthesis quality ceiling on complex accents.",
                    "Articulatory parameter estimation error compounds.",
                ],
            },
            "token_translation": {
                "description": (
                    "Token translation candidate. Causal speech tokenizer → LSTM "
                    "accent translator (with FiLM) → token-conditioned synthesizer. "
                    "0-frame lookahead."
                ),
                "status": "Rejected",
                "quality_gate_results": {
                    "Content Preservation": "TBD",
                    "Identity Preservation": "TBD",
                    "Damage Prevention": "TBD",
                    "Critical Entities": "TBD",
                },
                "streaming_gate_results": {
                    "Causality": "TBD",
                    "State-Boundedness": "TBD",
                    "Prefix Invariance": "TBD",
                },
                "algorithmic_latency_ms": {
                    "frame_accumulation_ms": 20,
                    "lookahead_ms": 0,
                    "model_structural_ms": 20,
                    "output_buffer_ms": 20,
                    "total_ms": 60,
                },
                "compute_latency_ms": {
                    "p50_ms": "_TBD_",
                    "p95_ms": "_TBD_",
                    "e2e_p50_ms": "_TBD_",
                    "rtf_p50": "_TBD_",
                },
                "resources": {
                    "parameter_count": "_TBD_",
                    "model_memory_bytes": "_TBD_",
                    "session_state_per_hour": "_TBD_",
                    "training_gpu_hours": "_TBD_",
                    "training_cost_usd": "_TBD_",
                },
                "rationale": [
                    "LSTM state grows linearly with session duration.",
                    "Tokenizer quality is a prerequisite — creates training dependency chain.",
                    "No parameter target documented.",
                ],
            },
            "minimal_hybrid": {
                "description": (
                    "Minimal Hybrid (Candidate D). Causal Conv1d encoder → per-accent "
                    "affine mapper → ConvTranspose1d synthesizer. 20 ms frames, "
                    "80 ms chunks, < 500K parameters, 0 ms lookahead."
                ),
                "status": "Chosen",
                "quality_gate_results": {
                    "Content Preservation": "TBD",
                    "Identity Preservation": "TBD",
                    "Damage Prevention": "TBD",
                    "Critical Entities": "TBD",
                },
                "streaming_gate_results": {
                    "Causality": "TBD",
                    "State-Boundedness": "TBD",
                    "Prefix Invariance": "TBD",
                },
                "algorithmic_latency_ms": {
                    "frame_accumulation_ms": 20,
                    "lookahead_ms": 0,
                    "model_structural_ms": 20,
                    "output_buffer_ms": 20,
                    "total_ms": 60,
                },
                "compute_latency_ms": {
                    "p50_ms": "_TBD_",
                    "p95_ms": "_TBD_",
                    "e2e_p50_ms": "_TBD_",
                    "rtf_p50": "_TBD_",
                },
                "resources": {
                    "parameter_count": "< 500K",
                    "model_memory_bytes": "_TBD_",
                    "session_state_per_hour": "O(chunk_count) — bounded",
                    "training_gpu_hours": "_TBD_",
                    "training_cost_usd": "_TBD_",
                },
                "rationale": [
                    "Passes all hard gates by design: 0 ms lookahead, strict causality, bounded state.",
                    "Simplest viable gradient path: only 3 modules. Fewer failure modes.",
                    "< 500K parameters — deployable on edge devices.",
                    "Per-accent affine mapper naturally supports strength ∈ [0, 1].",
                    "Lowest Phase 3 risk: architecture fully specified, minimal.",
                ],
                "known_limitations": [
                    "Linear mapper may be too simple for complex accent transformations.",
                    "No speaker disentanglement — Phase 3 will add speaker conditioning.",
                    "ConvTranspose1d upsampler may produce artifacts vs. learned generators.",
                    "Single-pass processing: no iterative refinement.",
                ],
            },
        },
        "data_used": [
            {
                "name": "Training Corpus",
                "description": "AccentEdge curated multi-speaker, multi-accent speech",
                "purpose": "Model training for all candidates",
            },
            {
                "name": "Phase-1 DEV Benchmark",
                "description": "Held-out speaker-disjoint dev set",
                "purpose": "Quality gate evaluation (WER/CER)",
            },
        ],
        "quality_gates": [
            {"name": "Content Preservation", "description": "WER ≤ baseline + 5% on Phase-1 DEV"},
            {"name": "Identity Preservation", "description": "Speaker similarity (SIM) ≥ 0.85 on Phase-1 DEV"},
            {"name": "Damage Prevention", "description": "MOS ≥ 3.5 on clean reference resynthesis"},
            {"name": "Critical Entities", "description": "Proper nouns / numbers transcribed correctly (CER ≤ 3%)"},
        ],
        "streaming_gates": [
            {"name": "Causality", "description": "Output at time t depends only on inputs ≤ t + declared lookahead"},
            {"name": "State-Boundedness", "description": "Per-step memory growth O(1); total state O(T) with small constant"},
            {"name": "Prefix Invariance", "description": "Output prefix identical when given prefix-only input"},
        ],
        "rejected_alternatives": [
            {
                "architecture_id": "Candidate A (Streaming AC)",
                "reason": (
                    "Paper-style mode: 640 ms lookahead exceeds conversational latency budget. "
                    "Low-lookahead mode: marginal benefit over D does not justify complexity "
                    "of 4-module stack. Encoder cache state grows unboundedly with session length."
                ),
            },
            {
                "architecture_id": "Candidate C (Token Translation)",
                "reason": (
                    "LSTM-based translator state grows linearly with session duration. "
                    "Tokenizer quality is a prerequisite creating a training dependency chain. "
                    "No documented parameter target."
                ),
            },
            {
                "architecture_id": "Sparse Repair",
                "reason": (
                    "Only interfaces and configuration exist. No concrete model implementation. "
                    "Complementary approach — better evaluated as Phase 3 enhancement."
                ),
            },
        ],
        "known_risks": [
            {
                "description": "D's linear mapper too simple for quality targets",
                "likelihood": "High",
                "impact": "High",
                "mitigation": "Phase 3: deepen mapper to 2-3 layer MLP with attention.",
            },
            {
                "description": "D's ConvTranspose1d upsampler produces audible artifacts",
                "likelihood": "Medium",
                "impact": "Medium",
                "mitigation": "Phase 3: replace with window-based overlap-add or LSTM post-filter.",
            },
            {
                "description": "D lacks speaker disentanglement → identity loss",
                "likelihood": "Medium",
                "impact": "High",
                "mitigation": "Phase 3: add speaker encoder and FiLM conditioning.",
            },
            {
                "description": "Benchmark numbers unavailable at ADR time",
                "likelihood": "Certain",
                "impact": "Low",
                "mitigation": "ADR is design-phase analysis; numbers populated post-training.",
            },
            {
                "description": "Sparse Repair could improve D with low cost",
                "likelihood": "Medium",
                "impact": "Low",
                "mitigation": "Evaluate in Phase 3 as a post-hoc enhancement layer.",
            },
        ],
        "phase3_tasks": [
            {
                "title": "Quality ceiling expansion",
                "description": "Replace linear mapper with deeper network (2-3 layer MLP or light transformer) while maintaining streaming causality.",
            },
            {
                "title": "Speaker preservation",
                "description": "Add speaker encoder branch and inject speaker conditioning into the mapper via FiLM layers.",
            },
            {
                "title": "Loss function design",
                "description": "Develop accent-discriminative and speaker-discriminative losses for streaming training.",
            },
            {
                "title": "Synthesizer quality",
                "description": "Evaluate and potentially replace ConvTranspose1d with higher-quality waveform generator.",
            },
            {
                "title": "Full training and benchmarking",
                "description": "Train all candidates on full corpus, evaluate on Phase-1 DEV benchmark, populate placeholder values.",
            },
            {
                "title": "Sparse Repair evaluation",
                "description": "Implement detector, controller, and synthesizer; evaluate as post-processing layer on D's output.",
            },
            {
                "title": "Sweep optimization",
                "description": "Run hyperparameter sweeps over hidden_dim, kernel_size, hop_length, and mapper depth.",
            },
        ],
        "decision_criteria": [
            {"name": "Streaming Latency", "weight": 0.2, "description": "Algorithmic + compute latency; target E2E P50 < 200 ms"},
            {"name": "Quality Potential", "weight": 0.2, "description": "Expected ceiling based on architecture expressivity"},
            {"name": "Parameter Efficiency", "weight": 0.2, "description": "Parameters per dB of quality improvement"},
            {"name": "Implementation Risk", "weight": 0.2, "description": "Complexity, state management, known failure modes"},
            {"name": "Phase 3 Extensibility", "weight": 0.2, "description": "How easily the architecture can be enhanced in Phase 3"},
        ],
        "pareto_scores": {
            "streaming_ac (paper)": {
                "Streaming Latency": 1,
                "Quality Potential": 4,
                "Parameter Efficiency": 2,
                "Implementation Risk": 2,
                "Phase 3 Extensibility": 3,
            },
            "streaming_ac (low-look)": {
                "Streaming Latency": 3,
                "Quality Potential": 3,
                "Parameter Efficiency": 3,
                "Implementation Risk": 3,
                "Phase 3 Extensibility": 3,
            },
            "articulatory_ddsp": {
                "Streaming Latency": 5,
                "Quality Potential": 3,
                "Parameter Efficiency": 3,
                "Implementation Risk": 3,
                "Phase 3 Extensibility": 3,
            },
            "token_translation": {
                "Streaming Latency": 4,
                "Quality Potential": 3,
                "Parameter Efficiency": 3,
                "Implementation Risk": 2,
                "Phase 3 Extensibility": 4,
            },
            "minimal_hybrid": {
                "Streaming Latency": 5,
                "Quality Potential": 3,
                "Parameter Efficiency": 5,
                "Implementation Risk": 5,
                "Phase 3 Extensibility": 5,
            },
        },
    }
