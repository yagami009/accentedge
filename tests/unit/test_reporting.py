"""Tests for the Phase 2 reporting modules."""

from __future__ import annotations

from accentedge_lab.reporting.adr import (
    build_default_adr_data,
    generate_adr,
)
from accentedge_lab.reporting.pareto import ParetoTable
from accentedge_lab.reporting.phase2_report import Phase2Report


# ---------------------------------------------------------------------------
# TestADR
# ---------------------------------------------------------------------------

class TestGenerateADR:
    def test_returns_nonempty_string(self) -> None:
        data = build_default_adr_data()
        adr = generate_adr(data)
        assert isinstance(adr, str)
        assert len(adr) > 0

    def test_contains_decision_section(self) -> None:
        data = build_default_adr_data()
        adr = generate_adr(data)
        assert "## Decision" in adr
        assert "minimal_hybrid" in adr

    def test_contains_backup_section(self) -> None:
        data = build_default_adr_data()
        adr = generate_adr(data)
        assert "## Backup Architecture" in adr
        assert "articulatory_ddsp" in adr

    def test_contains_chosen_section(self) -> None:
        data = build_default_adr_data()
        adr = generate_adr(data)
        assert "## Chosen Architecture" in adr
        assert "minimal_hybrid" in adr

    def test_contains_candidates_section(self) -> None:
        data = build_default_adr_data()
        adr = generate_adr(data)
        assert "## Candidates Evaluated" in adr
        assert "streaming_ac" in adr
        assert "token_translation" in adr
        assert "articulatory_ddsp" in adr

    def test_contains_quality_gates_table(self) -> None:
        data = build_default_adr_data()
        adr = generate_adr(data)
        assert "## Quality Gates" in adr
        assert "Content Preservation" in adr

    def test_contains_streaming_gates_table(self) -> None:
        data = build_default_adr_data()
        adr = generate_adr(data)
        assert "## Streaming Gates" in adr
        assert "Causality" in adr

    def test_contains_latency_section(self) -> None:
        data = build_default_adr_data()
        adr = generate_adr(data)
        assert "## Latency Results" in adr
        assert "Algorithmic Latency" in adr

    def test_contains_resource_section(self) -> None:
        data = build_default_adr_data()
        adr = generate_adr(data)
        assert "## Resource Results" in adr
        assert "Parameter Count" in adr

    def test_contains_rejected_section(self) -> None:
        data = build_default_adr_data()
        adr = generate_adr(data)
        assert "## Rejected Alternatives" in adr
        assert "streaming_ac" in adr

    def test_contains_risks_section(self) -> None:
        data = build_default_adr_data()
        adr = generate_adr(data)
        assert "## Known Risks" in adr

    def test_contains_phase3_section(self) -> None:
        data = build_default_adr_data()
        adr = generate_adr(data)
        assert "## What Phase 3 Must Solve" in adr

    def test_contains_criteria_appendix(self) -> None:
        data = build_default_adr_data()
        adr = generate_adr(data)
        assert "## Appendix: Decision Criteria" in adr

    def test_overrides_chosen_architecture(self) -> None:
        data = build_default_adr_data()
        data["chosen_architecture"] = "articulatory_ddsp"
        adr = generate_adr(data)
        assert "articulatory_ddsp" in adr

    def test_rejected_section_has_three_alternatives(self) -> None:
        data = build_default_adr_data()
        adr = generate_adr(data)
        # Should mention three rejected architectures
        assert data["rejected_alternatives"][0]["architecture_id"] in adr
        assert data["rejected_alternatives"][1]["architecture_id"] in adr
        assert data["rejected_alternatives"][2]["architecture_id"] in adr

    def test_pareto_scores_rendered_in_appendix(self) -> None:
        data = build_default_adr_data()
        adr = generate_adr(data)
        assert "minimal_hybrid" in adr
        assert "Streaming Latency" in adr

    def test_header_metadata(self) -> None:
        data = build_default_adr_data()
        data["date"] = "2026-08-24"
        data["status"] = "Accepted"
        adr = generate_adr(data)
        assert "2026-08-24" in adr
        assert "Accepted" in adr

    def test_empty_candidates_handled(self) -> None:
        data = build_default_adr_data()
        data["candidates"] = {}
        adr = generate_adr(data)
        assert "## Candidates Evaluated" in adr

    def test_empty_gates_handled(self) -> None:
        data = build_default_adr_data()
        data["quality_gates"] = []
        data["streaming_gates"] = []
        adr = generate_adr(data)
        assert "No quality gates defined" in adr
        assert "No streaming gates defined" in adr


# ---------------------------------------------------------------------------
# TestBuildDefaultADRData
# ---------------------------------------------------------------------------

class TestBuildDefaultADRData:
    def test_returns_dict(self) -> None:
        data = build_default_adr_data()
        assert isinstance(data, dict)

    def test_has_required_keys(self) -> None:
        data = build_default_adr_data()
        required = [
            "context",
            "chosen_architecture",
            "backup_architecture",
            "candidates",
            "data_used",
            "quality_gates",
            "streaming_gates",
            "rejected_alternatives",
            "known_risks",
            "phase3_tasks",
            "decision_criteria",
            "pareto_scores",
        ]
        for key in required:
            assert key in data, f"Missing key: {key}"

    def test_five_candidates(self) -> None:
        data = build_default_adr_data()
        # Include sparse_repair alongside the 4 main candidates
        data["candidates"]["sparse_repair"] = {
            "description": "Sparse repair candidate.",
            "status": "Not evaluated",
            "quality_gate_results": {},
            "streaming_gate_results": {},
            "algorithmic_latency_ms": {"total_ms": 0},
            "compute_latency_ms": {},
            "resources": {},
        }
        assert len(data["candidates"]) == 5

    def test_chosen_is_minimal_hybrid(self) -> None:
        data = build_default_adr_data()
        assert data["chosen_architecture"] == "minimal_hybrid"

    def test_backup_is_articulatory_ddsp(self) -> None:
        data = build_default_adr_data()
        assert data["backup_architecture"] == "articulatory_ddsp"

    def test_pareto_scores_have_all_candidates(self) -> None:
        data = build_default_adr_data()
        scores = data["pareto_scores"]
        expected = [
            "streaming_ac (paper)",
            "streaming_ac (low-look)",
            "articulatory_ddsp",
            "token_translation",
            "minimal_hybrid",
        ]
        for arch in expected:
            assert arch in scores, f"Missing Pareto score for: {arch}"

    def test_pareto_scores_have_all_criteria(self) -> None:
        data = build_default_adr_data()
        criteria_names = [c["name"] for c in data["decision_criteria"]]
        for arch, scores in data["pareto_scores"].items():
            for crit in criteria_names:
                assert crit in scores, f"Missing criterion '{crit}' for {arch}"

    def test_four_quality_gates(self) -> None:
        data = build_default_adr_data()
        assert len(data["quality_gates"]) == 4

    def test_three_streaming_gates(self) -> None:
        data = build_default_adr_data()
        assert len(data["streaming_gates"]) == 3

    def test_three_rejected(self) -> None:
        data = build_default_adr_data()
        assert len(data["rejected_alternatives"]) == 3

    def test_minimal_hybrid_has_known_limitations(self) -> None:
        data = build_default_adr_data()
        assert "known_limitations" in data["candidates"]["minimal_hybrid"]
        assert len(data["candidates"]["minimal_hybrid"]["known_limitations"]) > 0


# ---------------------------------------------------------------------------
# TestParetoTable
# ---------------------------------------------------------------------------

class TestParetoTable:
    def test_dominated_candidates_empty_when_all_pareto(self) -> None:
        # Two candidates: A better on both metrics (lower is better)
        pt = ParetoTable(
            candidates=["A", "B"],
            metrics=["wer", "rtf_p50"],
            data={
                "A": {"wer": 0.1, "rtf_p50": 0.3},
                "B": {"wer": 0.2, "rtf_p50": 0.5},
            },
        )
        dominated = pt.dominated_candidates()
        assert "B" in dominated
        assert "A" not in dominated

    def test_frontier_returns_non_dominated(self) -> None:
        pt = ParetoTable(
            candidates=["A", "B", "C"],
            metrics=["wer", "rtf_p50"],
            data={
                "A": {"wer": 0.1, "rtf_p50": 0.3},
                "B": {"wer": 0.2, "rtf_p50": 0.5},
                "C": {"wer": 0.15, "rtf_p50": 0.4},
            },
        )
        frontier = pt.frontier("rtf_p50", "wer")
        assert len(frontier) == 1
        assert frontier[0][0] == "A"

    def test_single_candidate_is_own_frontier(self) -> None:
        pt = ParetoTable(
            candidates=["X"],
            metrics=["wer"],
            data={"X": {"wer": 0.5}},
        )
        frontier = pt.frontier("wer", "wer")
        assert len(frontier) == 1
        assert frontier[0][0] == "X"


# ---------------------------------------------------------------------------
# TestPhase2Report
# ---------------------------------------------------------------------------

class TestPhase2Report:
    def test_generate_experiment_report(self) -> None:
        report = Phase2Report()
        exp = type("Obj", (), {"experiment_id": "e1", "status": "completed"})()
        results = {"wer": 0.05, "rtf_p50": 0.3}
        out = report.generate_experiment_report(exp, results)
        assert out["experiment_id"] == "e1"
        assert out["status"] == "completed"
        assert out["results"]["wer"] == 0.05

    def test_generate_architecture_comparison(self) -> None:
        report = Phase2Report()
        arch_data = {
            "minimal_hybrid": {
                "status": "Chosen",
                "summary": {"wer": 0.05},
                "algorithmic_latency_ms": {"total_ms": 60},
                "compute_latency_ms": {"p50_ms": 5.0},
                "resources": {"parameter_count": 450_000},
                "streaming_gate_results": {"Causality": "PASS"},
                "quality_gate_results": {"Content Preservation": "PASS"},
            },
        }
        out = report.generate_architecture_comparison(arch_data)
        assert "architectures" in out
        assert "minimal_hybrid" in out["architectures"]
        assert out["architectures"]["minimal_hybrid"]["status"] == "Chosen"

    def test_generate_report_basic(self) -> None:
        report = Phase2Report()
        all_results = {
            "candidates": {
                "minimal_hybrid": {
                    "status": "Chosen",
                    "summary": {"wer": 0.05},
                    "algorithmic_latency_ms": {"total_ms": 60},
                    "compute_latency_ms": {"p50_ms": 5.0},
                    "resources": {"parameter_count": 450_000},
                    "streaming_gate_results": {},
                    "quality_gate_results": {},
                },
                "streaming_ac": {
                    "status": "Rejected",
                    "summary": {"wer": 0.08},
                    "algorithmic_latency_ms": {"total_ms": 880},
                    "compute_latency_ms": {"p50_ms": 50.0},
                    "resources": {"parameter_count": 2_000_000},
                    "streaming_gate_results": {},
                    "quality_gate_results": {},
                },
            },
        }
        out = report.generate_report(all_results)
        assert "candidates" in out
        assert "pareto" in out
        assert "latency" in out
        assert "recommendations" in out
        assert "adr_markdown" in out
        assert out["recommendations"]["chosen"] == "minimal_hybrid"

    def test_generate_report_with_pareto(self) -> None:
        report = Phase2Report()
        pt = ParetoTable(
            candidates=["A", "B"],
            metrics=["wer", "rtf_p50"],
            data={"A": {"wer": 0.1, "rtf_p50": 0.3}, "B": {"wer": 0.2, "rtf_p50": 0.5}},
        )
        all_results = {
            "candidates": {},
            "experiments": [],
            "pareto_metrics": ["wer", "rtf_p50"],
        }
        out = report.generate_report(all_results, pareto_data=pt)
        assert out["pareto"]["dominated"] == ["B"]
        assert out["recommendations"]["chosen"] == "A"

    def test_generate_report_with_sweeps(self) -> None:
        report = Phase2Report()
        sweep_data = {
            "hidden_dim_sweep": [
                {"config": {"hidden_dim": 64}, "metrics": {"wer": 0.05}},
                {"config": {"hidden_dim": 128}, "metrics": {"wer": 0.04}},
            ]
        }
        all_results = {
            "candidates": {},
            "experiments": [],
            "pareto_metrics": ["wer"],
        }
        out = report.generate_report(all_results, sweep_data=sweep_data)
        assert "sweeps" in out
        assert "hidden_dim_sweep" in out["sweeps"]
        assert len(out["sweeps"]["hidden_dim_sweep"]["rows"]) == 2

    def test_adr_markdown_contains_key_sections(self) -> None:
        report = Phase2Report()
        all_results = {
            "candidates": {
                "minimal_hybrid": {
                    "status": "Chosen",
                    "summary": {},
                    "algorithmic_latency_ms": {"total_ms": 60},
                    "compute_latency_ms": {},
                    "resources": {},
                    "streaming_gate_results": {},
                    "quality_gate_results": {},
                },
            },
            "experiments": [],
            "pareto_metrics": ["wer"],
        }
        out = report.generate_report(all_results)
        adr = out["adr_markdown"]
        assert "# Architecture Decision Record" in adr
        assert "## Decision" in adr
        assert "## Chosen Architecture" in adr
        assert "minimal_hybrid" in adr

    def test_empty_candidates_handled(self) -> None:
        report = Phase2Report()
        out = report.generate_report({"candidates": {}})
        assert out["recommendations"]["chosen"] == "UNKNOWN"

    def test_latency_analysis_keys(self) -> None:
        report = Phase2Report()
        all_results = {
            "candidates": {
                "test_arch": {
                    "status": "Running",
                    "summary": {},
                    "algorithmic_latency_ms": {
                        "total_ms": 100,
                        "frame_accumulation_ms": 20,
                        "lookahead_ms": 0,
                        "model_structural_ms": 20,
                        "output_buffer_ms": 20,
                    },
                    "compute_latency_ms": {
                        "p50_ms": 10.0,
                        "p95_ms": 15.0,
                        "e2e_p50_ms": 110.0,
                        "rtf_p50": 0.5,
                    },
                    "resources": {},
                    "streaming_gate_results": {},
                    "quality_gate_results": {},
                },
            },
            "experiments": [],
            "pareto_metrics": ["wer"],
        }
        out = report.generate_report(all_results)
        latency = out["latency"]["test_arch"]
        assert latency["algorithmic_total_ms"] == 100
        assert latency["breakdown"]["frame_accumulation_ms"] == 20
        assert latency["compute_p50_ms"] == 10.0
        assert latency["rtf_p50"] == 0.5
