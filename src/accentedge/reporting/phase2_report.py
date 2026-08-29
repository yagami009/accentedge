"""Phase 2 report generation.

Aggregates experiment results, sweep tables, Pareto analysis, latency
measurements, and produces a recommendation for Phase 3 advancement.
"""

from __future__ import annotations

from typing import Any

from accentedge.reporting.adr import generate_adr, build_default_adr_data
from accentedge.reporting.pareto import ParetoTable


class Phase2Report:
    """Aggregates all Phase 2 experiment results into structured reports."""

    # ------------------------------------------------------------------
    # Experiment report
    # ------------------------------------------------------------------

    def generate_experiment_report(
        self, experiment: Any, results: dict[str, Any]
    ) -> dict[str, Any]:
        """Wrap a single experiment's results into a report dict.

        Args:
            experiment: An ExperimentRecord or object with experiment_id, status.
            results: Raw result dict from the benchmark runner.

        Returns:
            Dict with experiment_id, status, results, and derived summary.
        """
        exp_id = getattr(experiment, "experiment_id", "unknown")
        status = getattr(experiment, "status", "unknown")

        # Derive a simple quality summary if the data is present
        summary = {}
        for key in ("wer", "cer", "latency_p50_ms", "rtf_p50"):
            if key in results:
                summary[key] = results[key]

        return {
            "experiment_id": exp_id,
            "status": status,
            "results": results,
            "summary": summary,
        }

    # ------------------------------------------------------------------
    # Pareto table
    # ------------------------------------------------------------------

    def generate_pareto_table(
        self,
        experiments: list[Any],
        metrics: list[str],
    ) -> ParetoTable:
        """Build a ParetoTable from experiment records.

        Args:
            experiments: List of ExperimentRecord objects.
            metrics: Metric names to include (e.g. ["wer", "rtf_p50", "parameter_count"]).

        Returns:
            ParetoTable populated with candidate names and metric values.
        """
        candidates: list[str] = []
        data: dict[str, dict[str, float]] = {}

        for exp in experiments:
            arch = getattr(exp, "architecture", "unknown")
            if arch not in candidates:
                candidates.append(arch)

            if arch not in data:
                data[arch] = {}

            # Pull metrics from experiment results if available
            exp_results = {}
            if hasattr(exp, "benchmark_results") and exp.benchmark_results:
                exp_results = exp.benchmark_results

            for metric in metrics:
                if metric in exp_results:
                    data[arch][metric] = float(exp_results[metric])
                else:
                    data[arch][metric] = 0.0

        return ParetoTable(candidates=candidates, metrics=metrics, data=data)

    # ------------------------------------------------------------------
    # Architecture comparison
    # ------------------------------------------------------------------

    def generate_architecture_comparison(
        self, arch_results: dict[str, Any]
    ) -> dict[str, Any]:
        """Return a structured comparison dict keyed by architecture id.

        Args:
            arch_results: Dict mapping architecture_id -> result dict.

        Returns:
            Dict with normalized architecture data.
        """
        comparison: dict[str, Any] = {}
        for arch_id, raw in arch_results.items():
            comparison[arch_id] = {
                "status": raw.get("status", "UNKNOWN"),
                "quality_summary": raw.get("summary", {}),
                "algorithmic_latency_ms": raw.get("algorithmic_latency_ms", {}),
                "compute_latency_ms": raw.get("compute_latency_ms", {}),
                "resources": raw.get("resources", {}),
                "streaming_gate_results": raw.get("streaming_gate_results", {}),
                "quality_gate_results": raw.get("quality_gate_results", {}),
            }
        return {"architectures": comparison}

    # ------------------------------------------------------------------
    # Full report
    # ------------------------------------------------------------------

    def generate_report(
        self,
        all_results: dict[str, Any],
        sweep_data: dict[str, Any] | None = None,
        pareto_data: ParetoTable | None = None,
    ) -> dict[str, Any]:
        """Produce the complete Phase 2 report.

        Args:
            all_results: Dict of all experiment and candidate results.
                Expected shape:
                {
                    "experiments": list[ExperimentRecord],
                    "candidates": dict[str, dict],  # arch_id -> result data
                    "sweeps": dict (optional),
                    ...
                }
            sweep_data: Optional structured sweep results.
            pareto_data: Optional pre-built ParetoTable.

        Returns:
            Dict with keys: candidates, sweeps, pareto, latency, recommendations.
        """
        candidates = all_results.get("candidates", {})

        # Build Pareto table from experiments if not provided
        pareto_table = pareto_data
        if pareto_table is None:
            experiments = all_results.get("experiments", [])
            metrics = all_results.get("pareto_metrics", ["wer", "rtf_p50"])
            pareto_table = self.generate_pareto_table(experiments, metrics)

        # Latency analysis
        latency_analysis = self._analyze_latency(candidates)

        # Recommendations
        recommendations = self._generate_recommendations(candidates, pareto_table)

        # ADR
        adr_data = self._build_adr_data(all_results, recommendations)
        adr_markdown = generate_adr(adr_data)

        # Sweep tables
        sweep_tables = {}
        if sweep_data:
            sweep_tables = self._build_sweep_tables(sweep_data)

        return {
            "candidates": self._build_candidate_summaries(candidates),
            "sweeps": sweep_tables,
            "pareto": self._serialize_pareto(pareto_table),
            "latency": latency_analysis,
            "recommendations": recommendations,
            "adr_markdown": adr_markdown,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_candidate_summaries(
        self, candidates: dict[str, Any]
    ) -> dict[str, dict[str, Any]]:
        """Build a one-line summary per candidate."""
        summaries = {}
        for arch_id, data in candidates.items():
            params = data.get("resources", {}).get("parameter_count", "—")
            lat = data.get("algorithmic_latency_ms", {}).get("total_ms", "—")
            summaries[arch_id] = {
                "status": data.get("status", "UNKNOWN"),
                "parameters": params,
                "algorithmic_latency_ms": lat,
                "quality_summary": data.get("summary", {}),
                "streaming_status": data.get("streaming_gate_results", {}),
                "quality_status": data.get("quality_gate_results", {}),
            }
        return summaries

    def _analyze_latency(
        self, candidates: dict[str, Any]
    ) -> dict[str, dict[str, Any]]:
        """Aggregate latency data across candidates."""
        analysis: dict[str, dict[str, Any]] = {}
        for arch_id, data in candidates.items():
            algo = data.get("algorithmic_latency_ms", {})
            comp = data.get("compute_latency_ms", {})
            analysis[arch_id] = {
                "algorithmic_total_ms": algo.get("total_ms"),
                "breakdown": {
                    "frame_accumulation_ms": algo.get("frame_accumulation_ms"),
                    "lookahead_ms": algo.get("lookahead_ms"),
                    "model_structural_ms": algo.get("model_structural_ms"),
                    "output_buffer_ms": algo.get("output_buffer_ms"),
                },
                "compute_p50_ms": comp.get("p50_ms"),
                "compute_p95_ms": comp.get("p95_ms"),
                "e2e_p50_ms": comp.get("e2e_p50_ms"),
                "rtf_p50": comp.get("rtf_p50"),
            }
        return analysis

    def _generate_recommendations(
        self, candidates: dict[str, Any], pareto: ParetoTable | None
    ) -> dict[str, Any]:
        """Produce Phase 3 advancement recommendation."""
        # Determine chosen: prefer the one explicitly marked as chosen,
        # otherwise use Pareto frontier
        chosen = None
        for arch_id, data in candidates.items():
            if data.get("status") == "Chosen":
                chosen = arch_id
                break

        if chosen is None and pareto is not None:
            frontier = pareto.frontier("rtf_p50", "wer")
            if frontier:
                chosen = frontier[0][0]

        if chosen is None:
            chosen = list(candidates.keys())[0] if candidates else "UNKNOWN"

        # Find backup: next best on Pareto frontier
        backup = None
        if pareto is not None:
            frontier = pareto.frontier("rtf_p50", "wer")
            for name, _, _ in frontier:
                if name != chosen:
                    backup = name
                    break

        dominated = pareto.dominated_candidates() if pareto else []

        return {
            "chosen": chosen,
            "backup": backup,
            "dominated_candidates": dominated,
            "rationale": (
                f"{chosen} selected based on Pareto frontier analysis "
                f"across latency, quality, and efficiency metrics."
            ),
        }

    def _build_adr_data(
        self, all_results: dict[str, Any], recommendations: dict[str, Any]
    ) -> dict[str, Any]:
        """Merge all_results into the ADR template data."""
        base = build_default_adr_data()

        # Override chosen/backup from recommendations
        if recommendations.get("chosen"):
            base["chosen_architecture"] = recommendations["chosen"]
        if recommendations.get("backup"):
            base["backup_architecture"] = recommendations["backup"]

        # Merge candidate data from all_results
        candidates_in = all_results.get("candidates", {})
        for arch_id, merged in base["candidates"].items():
            if arch_id in candidates_in:
                merged.update(candidates_in[arch_id])

        return base

    def _build_sweep_tables(
        self, sweep_data: dict[str, Any]
    ) -> dict[str, Any]:
        """Convert sweep results into table-friendly dicts."""
        tables = {}
        for sweep_name, sweep_results in sweep_data.items():
            rows = []
            if isinstance(sweep_results, list):
                for entry in sweep_results:
                    rows.append({
                        "config": entry.get("config", {}),
                        "metrics": entry.get("metrics", {}),
                        "pareto_rank": entry.get("pareto_rank", "—"),
                    })
            tables[sweep_name] = {"rows": rows}
        return tables

    def _serialize_pareto(self, pareto: ParetoTable | None) -> dict[str, Any]:
        """Serialize ParetoTable to a JSON-friendly dict."""
        if pareto is None:
            return {"candidates": [], "metrics": [], "data": {}, "dominated": []}

        return {
            "candidates": pareto.candidates,
            "metrics": pareto.metrics,
            "data": pareto.data,
            "dominated": pareto.dominated_candidates(),
        }
