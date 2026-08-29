"""Tests for benchmark integration, sweeps, and Pareto analysis."""

from __future__ import annotations

import numpy as np
import pytest

from accentedge_lab.models.interfaces import (
    CandidateMetadata,
    StreamingCandidate,
    StreamingResult,
    StreamingSession,
)
from accentedge_lab.benchmark.adapter import (
    BenchmarkItem,
    BenchmarkResult,
    Phase1BenchmarkAdapter,
    UtteranceMetrics,
    load_dev_split,
)
from accentedge_lab.benchmark.sweeps import (
    CHUNK_SIZES_MS,
    LOOKAHEAD_SIZES_MS,
    ChunkSweep,
    CombinedSweep,
    LookaheadSweep,
    SweepResult,
)
from accentedge_lab.benchmark.comparison import (
    ParetoFrontier,
    compare_candidates,
    compute_pareto_frontier,
    generate_frontier_report,
    report_per_candidate,
    _dominates,
    _equal,
)


# ---------------------------------------------------------------------------
# Fake candidate for testing
# ---------------------------------------------------------------------------

class FakeBenchmarkCandidate:
    metadata = CandidateMetadata(
        architecture_id="fake_bench",
        version="0.0.0",
        input_sample_rate=16000,
        frame_ms=10.0,
        preferred_chunk_ms=80,
        required_lookahead_ms=0,
    )

    def prepare(self, device: str, precision: str) -> None:
        pass

    def create_session(self, config: dict) -> StreamingSession:
        return StreamingSession(
            session_id=f"session_{id(config)}",
            state={"array": np.zeros(100, dtype=np.float32)},
        )

    def process_chunk(
        self,
        session: StreamingSession,
        audio_chunk: np.ndarray,
        sample_rate: int,
    ) -> StreamingResult:
        session.samples_processed += len(audio_chunk)
        return StreamingResult(
            audio=audio_chunk.copy(),
            sample_rate=sample_rate,
            input_start_sample=session.samples_processed - len(audio_chunk),
            input_end_sample=session.samples_processed,
            output_start_sample=session.samples_processed - len(audio_chunk),
            output_end_sample=session.samples_processed,
        )

    def flush(self, session: StreamingSession) -> list[StreamingResult]:
        return []

    def reset(self, session: StreamingSession) -> None:
        session.state = {"array": np.zeros(100, dtype=np.float32)}
        session.samples_processed = 0

    def close(self) -> None:
        pass


def _fake_factory(config: dict) -> StreamingCandidate:
    return FakeBenchmarkCandidate()


# ---------------------------------------------------------------------------
# Adapter tests
# ---------------------------------------------------------------------------

class TestPhase1BenchmarkAdapter:
    """Phase1BenchmarkAdapter wraps candidate correctly."""

    def test_adapter_evaluates_candidate(self, tmp_path) -> None:
        adapter = Phase1BenchmarkAdapter(
            benchmark_repo_path=str(tmp_path),
            condition="clean",
            chunk_size_ms=80,
        )
        candidate = FakeBenchmarkCandidate()
        result = adapter.evaluate_candidate(candidate, {})
        assert result.candidate_id == "fake_bench"
        assert isinstance(result, BenchmarkResult)
        assert len(result.utterances) > 0

    def test_adapter_produces_per_utterance_metrics(self, tmp_path) -> None:
        adapter = Phase1BenchmarkAdapter(
            benchmark_repo_path=str(tmp_path),
            condition="clean",
            chunk_size_ms=80,
        )
        candidate = FakeBenchmarkCandidate()
        result = adapter.evaluate_candidate(candidate, {})
        for u in result.utterances:
            assert isinstance(u, UtteranceMetrics)
            assert 0.0 <= u.content_error_rate <= 1.0
            assert 0.0 <= u.identity_preservation <= 1.0
            assert u.latency_ms >= 0.0
            assert u.rtf >= 0.0

    def test_adapter_aggregate_content_range(self, tmp_path) -> None:
        adapter = Phase1BenchmarkAdapter(
            benchmark_repo_path=str(tmp_path),
            condition="clean",
            chunk_size_ms=80,
        )
        result = adapter.evaluate_candidate(FakeBenchmarkCandidate(), {})
        assert 0.0 <= result.aggregate_content <= 1.0

    def test_adapter_aggregate_identity_range(self, tmp_path) -> None:
        adapter = Phase1BenchmarkAdapter(
            benchmark_repo_path=str(tmp_path),
            condition="clean",
            chunk_size_ms=80,
        )
        result = adapter.evaluate_candidate(FakeBenchmarkCandidate(), {})
        assert 0.0 <= result.aggregate_identity <= 1.0

    def test_adapter_latency_positive(self, tmp_path) -> None:
        adapter = Phase1BenchmarkAdapter(
            benchmark_repo_path=str(tmp_path),
            condition="clean",
            chunk_size_ms=80,
        )
        result = adapter.evaluate_candidate(FakeBenchmarkCandidate(), {})
        assert result.aggregate_latency_ms >= 0.0

    def test_adapter_rtf_non_negative(self, tmp_path) -> None:
        adapter = Phase1BenchmarkAdapter(
            benchmark_repo_path=str(tmp_path),
            condition="clean",
            chunk_size_ms=80,
        )
        result = adapter.evaluate_candidate(FakeBenchmarkCandidate(), {})
        assert result.aggregate_rtf >= 0.0

    def test_adapter_close_called_even_on_error(self, tmp_path) -> None:
        adapter = Phase1BenchmarkAdapter(
            benchmark_repo_path=str(tmp_path),
            condition="clean",
            chunk_size_ms=80,
        )

        class FailCandidate:
            metadata = CandidateMetadata(
                architecture_id="fail",
                version="0.0",
                input_sample_rate=16000,
                frame_ms=10.0,
            )

            def prepare(self, device: str, precision: str) -> None:
                pass

            def create_session(self, config: dict) -> StreamingSession:
                return StreamingSession(session_id="s", state={})

            def process_chunk(self, session, audio_chunk, sample_rate):
                raise RuntimeError("boom")

            def flush(self, session): return []

            def reset(self, session): pass

            def close(self):
                self.closed = True

        c = FailCandidate()
        with pytest.raises(RuntimeError):
            adapter.evaluate_candidate(c, {})
        assert getattr(c, "closed", False) is True

    def test_adapter_config_preserved(self, tmp_path) -> None:
        adapter = Phase1BenchmarkAdapter(
            benchmark_repo_path=str(tmp_path),
            condition="clean",
            chunk_size_ms=80,
        )
        config = {"chunk_size_ms": 80, "lookahead_ms": 0}
        result = adapter.evaluate_candidate(FakeBenchmarkCandidate(), config)
        assert result.config == config

    def test_adapter_chunking_audio(self, tmp_path) -> None:
        adapter = Phase1BenchmarkAdapter(
            benchmark_repo_path=str(tmp_path),
            condition="clean",
            chunk_size_ms=20,
        )
        # 640 samples / (20ms * 16000/1000=320samples) = 2 chunks of 320
        chunks = adapter._chunk_audio(np.zeros(640, dtype=np.float32))
        assert len(chunks) == 2
        assert all(c.size == 320 for c in chunks)


# ---------------------------------------------------------------------------
# DEV split loading
# ---------------------------------------------------------------------------

class TestLoadDevSplit:
    """DEV split loading."""

    def test_loads_synthetic_dev(self, tmp_path) -> None:
        items = load_dev_split(str(tmp_path), condition="clean")
        assert len(items) == 20
        assert all(isinstance(i, BenchmarkItem) for i in items)
        assert all(i.sample_rate == 16000 for i in items)

    def test_dev_items_have_transcripts(self, tmp_path) -> None:
        items = load_dev_split(str(tmp_path), condition="clean")
        assert all(i.transcript != "" for i in items)

    def test_dev_deterministic(self, tmp_path) -> None:
        items_a = load_dev_split(str(tmp_path), condition="clean", seed=42)
        items_b = load_dev_split(str(tmp_path), condition="clean", seed=42)
        assert len(items_a) == len(items_b)
        # same seed → same first utterance id
        assert items_a[0].utterance_id == items_b[0].utterance_id

    def test_dev_condition_noisy(self, tmp_path) -> None:
        items = load_dev_split(str(tmp_path), condition="noisy")
        assert all(i.metadata.get("condition") == "noisy" for i in items)

    def test_dev_json_file_reads_correctly(self, tmp_path) -> None:
        import json
        data = [
            {
                "utterance_id": "u1",
                "duration_s": 2.0,
                "transcript": "hello",
                "speaker_id": "spk1",
            }
        ]
        (tmp_path / "data").mkdir()
        (tmp_path / "data" / "dev_clean.json").write_text(
            json.dumps(data), encoding="utf-8"
        )
        items = load_dev_split(str(tmp_path), condition="clean")
        assert len(items) == 1
        assert items[0].utterance_id == "u1"
        assert items[0].transcript == "hello"


# ---------------------------------------------------------------------------
# SweepResult structure tests
# ---------------------------------------------------------------------------

class TestSweepResultStructure:
    """Sweep results have correct structure."""

    def test_sweep_result_fields(self) -> None:
        r = SweepResult(
            candidate_id="cand_x",
            chunk_size_ms=80,
            lookahead_ms=20,
            metrics={"content": 0.9, "identity": 0.85},
            latency_ms=50.0,
            rtf=0.5,
            state_size_bytes=4096,
        )
        assert r.candidate_id == "cand_x"
        assert r.chunk_size_ms == 80
        assert r.lookahead_ms == 20
        assert r.metrics["content"] == 0.9
        assert r.latency_ms == 50.0
        assert r.rtf == 0.5
        assert r.state_size_bytes == 4096

    def test_sweep_result_to_dict(self) -> None:
        r = SweepResult(
            candidate_id="c", chunk_size_ms=40, lookahead_ms=0, latency_ms=10.0
        )
        d = r.to_dict()
        assert d["candidate_id"] == "c"
        assert d["chunk_size_ms"] == 40
        assert d["latency_ms"] == 10.0

    def test_sweep_result_default_metrics(self) -> None:
        r = SweepResult(candidate_id="x", chunk_size_ms=1, lookahead_ms=1)
        assert r.metrics == {}
        assert r.latency_ms == 0.0
        assert r.rtf == 0.0
        assert r.state_size_bytes == 0


# ---------------------------------------------------------------------------
# ChunkSweep tests
# ---------------------------------------------------------------------------

class TestChunkSweep:
    """ChunkSweep: all standard sizes, rejects unsupported sizes cleanly."""

    def test_all_standard_sizes_run(self, tmp_path) -> None:
        sweep = ChunkSweep(
            candidate_factory=_fake_factory,
            benchmark_repo_path=str(tmp_path),
            chunk_sizes=[20, 40, 80, 160],
            lookahead_ms=0,
        )
        results = sweep.run()
        assert len(results) == 4
        sizes = {r.chunk_size_ms for r in results}
        assert sizes == {20, 40, 80, 160}

    def test_result_chunk_size_correct(self, tmp_path) -> None:
        sweep = ChunkSweep(
            candidate_factory=_fake_factory,
            benchmark_repo_path=str(tmp_path),
            chunk_sizes=[20, 80],
            lookahead_ms=0,
        )
        results = sweep.run()
        for r in results:
            assert r.chunk_size_ms in (20, 80)
            assert r.lookahead_ms == 0

    def test_rejects_unsupported_sizes(self) -> None:
        sweep = ChunkSweep(
            candidate_factory=_fake_factory,
            benchmark_repo_path="/nonexistent",
        )
        valid = sweep.validate_sizes([20, 40, 1600, 3200, 80])
        assert 1600 not in valid
        assert 3200 not in valid
        assert valid == [20, 40, 80]

    def test_unsupported_sizes_skipped_in_run(self, tmp_path) -> None:
        sweep = ChunkSweep(
            candidate_factory=_fake_factory,
            benchmark_repo_path=str(tmp_path),
            chunk_sizes=[20, 1600, 80, 3200],
            lookahead_ms=0,
        )
        results = sweep.run()
        sizes = {r.chunk_size_ms for r in results}
        assert 1600 not in sizes
        assert 3200 not in sizes
        assert sizes == {20, 80}

    def test_default_chunk_sizes(self, tmp_path) -> None:
        sweep = ChunkSweep(
            candidate_factory=_fake_factory,
            benchmark_repo_path=str(tmp_path),
        )
        assert sweep.chunk_sizes == CHUNK_SIZES_MS

    def test_lookahead_fixed(self, tmp_path) -> None:
        sweep = ChunkSweep(
            candidate_factory=_fake_factory,
            benchmark_repo_path=str(tmp_path),
            chunk_sizes=[20, 40],
            lookahead_ms=10,
        )
        results = sweep.run()
        for r in results:
            assert r.lookahead_ms == 10

    def test_metrics_have_content_and_identity(self, tmp_path) -> None:
        sweep = ChunkSweep(
            candidate_factory=_fake_factory,
            benchmark_repo_path=str(tmp_path),
            chunk_sizes=[80],
            lookahead_ms=0,
        )
        results = sweep.run()
        assert "content" in results[0].metrics
        assert "identity" in results[0].metrics

    def test_result_latency_non_negative(self, tmp_path) -> None:
        sweep = ChunkSweep(
            candidate_factory=_fake_factory,
            benchmark_repo_path=str(tmp_path),
            chunk_sizes=[80],
            lookahead_ms=0,
        )
        results = sweep.run()
        for r in results:
            assert r.latency_ms >= 0.0
            assert r.rtf >= 0.0

    def test_result_candidate_id_set(self, tmp_path) -> None:
        sweep = ChunkSweep(
            candidate_factory=_fake_factory,
            benchmark_repo_path=str(tmp_path),
            chunk_sizes=[40],
            lookahead_ms=0,
        )
        results = sweep.run()
        assert results[0].candidate_id == "fake_bench"


# ---------------------------------------------------------------------------
# LookaheadSweep tests
# ---------------------------------------------------------------------------

class TestLookaheadSweep:
    """LookaheadSweep: all standard values."""

    def test_all_standard_values_run(self, tmp_path) -> None:
        sweep = LookaheadSweep(
            candidate_factory=_fake_factory,
            benchmark_repo_path=str(tmp_path),
            chunk_size_ms=80,
            lookahead_values=[0, 20, 40, 80, 160, 320],
        )
        results = sweep.run()
        assert len(results) == 6
        lookaheads = {r.lookahead_ms for r in results}
        assert lookaheads == {0, 20, 40, 80, 160, 320}

    def test_chunk_size_fixed(self, tmp_path) -> None:
        sweep = LookaheadSweep(
            candidate_factory=_fake_factory,
            benchmark_repo_path=str(tmp_path),
            chunk_size_ms=40,
            lookahead_values=[0, 20],
        )
        results = sweep.run()
        for r in results:
            assert r.chunk_size_ms == 40

    def test_default_lookahead_values(self, tmp_path) -> None:
        sweep = LookaheadSweep(
            candidate_factory=_fake_factory,
            benchmark_repo_path=str(tmp_path),
            chunk_size_ms=80,
        )
        assert sweep.lookahead_values == LOOKAHEAD_SIZES_MS

    def test_custom_lookahead_values(self, tmp_path) -> None:
        sweep = LookaheadSweep(
            candidate_factory=_fake_factory,
            benchmark_repo_path=str(tmp_path),
            chunk_size_ms=80,
            lookahead_values=[0, 100],
        )
        results = sweep.run()
        assert len(results) == 2
        assert {r.lookahead_ms for r in results} == {0, 100}


# ---------------------------------------------------------------------------
# CombinedSweep tests
# ---------------------------------------------------------------------------

class TestCombinedSweep:
    """CombinedSweep: produces quality/latency frontier data."""

    def test_product_of_axes(self, tmp_path) -> None:
        sweep = CombinedSweep(
            candidate_factory=_fake_factory,
            benchmark_repo_path=str(tmp_path),
            chunk_sizes=[20, 40],
            lookahead_values=[0, 20],
        )
        results = sweep.run()
        assert len(results) == 4  # 2 * 2
        pairs = {(r.chunk_size_ms, r.lookahead_ms) for r in results}
        assert pairs == {(20, 0), (20, 20), (40, 0), (40, 20)}

    def test_default_chunk_sizes(self, tmp_path) -> None:
        sweep = CombinedSweep(
            candidate_factory=_fake_factory,
            benchmark_repo_path=str(tmp_path),
        )
        assert sweep.chunk_sizes == CHUNK_SIZES_MS

    def test_default_lookahead_values(self, tmp_path) -> None:
        sweep = CombinedSweep(
            candidate_factory=_fake_factory,
            benchmark_repo_path=str(tmp_path),
        )
        assert sweep.lookahead_values == LOOKAHEAD_SIZES_MS


# ---------------------------------------------------------------------------
# Pareto frontier tests
# ---------------------------------------------------------------------------

class TestParetoFrontier:
    """Pareto: dominated vs non-dominated."""

    def _make_result(
        self,
        candidate_id: str,
        content: float,
        latency_ms: float,
        rtf: float = 0.5,
        state_bytes: int = 1000,
    ) -> SweepResult:
        return SweepResult(
            candidate_id=candidate_id,
            chunk_size_ms=80,
            lookahead_ms=0,
            metrics={"content": content, "identity": content},
            latency_ms=latency_ms,
            rtf=rtf,
            state_size_bytes=state_bytes,
        )

    def test_dominated_vs_non_dominated(self) -> None:
        # A: high content, low latency
        a = self._make_result("cand_a", content=0.9, latency_ms=10.0, rtf=0.5)
        # B: lower content, higher latency → dominated
        b = self._make_result("cand_b", content=0.5, latency_ms=20.0, rtf=0.8)
        frontier = compute_pareto_frontier([a, b])
        assert len(frontier.non_dominated) == 1
        assert frontier.non_dominated[0].candidate_id == "cand_a"
        assert len(frontier.dominated) == 1
        assert frontier.dominated[0].candidate_id == "cand_b"

    def test_equal_candidates_both_survive(self) -> None:
        a = self._make_result("cand_a", content=0.8, latency_ms=10.0)
        b = self._make_result("cand_b", content=0.8, latency_ms=10.0)
        frontier = compute_pareto_frontier([a, b])
        # Both are on the frontier — neither strictly dominates the other
        assert len(frontier.non_dominated) == 2
        assert len(frontier.dominated) == 0

    def test_dominance_check(self) -> None:
        a = self._make_result("cand_a", content=0.9, latency_ms=10.0, rtf=0.5)
        b = self._make_result("cand_b", content=0.7, latency_ms=20.0, rtf=0.8)
        assert _dominates(a, b) is True
        assert _dominates(b, a) is False

    def test_empty_results_handled(self) -> None:
        frontier = compute_pareto_frontier([])
        assert len(frontier.non_dominated) == 0
        assert len(frontier.dominated) == 0

    def test_single_result_non_dominated(self) -> None:
        a = self._make_result("cand_a", content=0.8, latency_ms=10.0)
        frontier = compute_pareto_frontier([a])
        assert len(frontier.non_dominated) == 1
        assert len(frontier.dominated) == 0

    def test_multiple_candidates_comparison(self) -> None:
        candidates = {
            "cand_a": [
                self._make_result("cand_a", content=0.9, latency_ms=10.0),
                self._make_result("cand_a", content=0.7, latency_ms=30.0),
            ],
            "cand_b": [
                self._make_result("cand_b", content=0.8, latency_ms=15.0),
                self._make_result("cand_b", content=0.6, latency_ms=50.0),
            ],
        }
        frontier = compare_candidates(candidates)
        # cand_a at 0.9/10ms should be non-dominated
        assert any(r.candidate_id == "cand_a" for r in frontier.non_dominated)
        assert len(frontier.non_dominated) >= 1

    def test_frontier_tables_correct(self) -> None:
        a = self._make_result("cand_a", content=0.9, latency_ms=10.0)
        b = self._make_result("cand_b", content=0.5, latency_ms=20.0)
        frontier = compute_pareto_frontier([a, b])
        table = frontier.summary_table()
        assert len(table) == 1
        assert table[0]["candidate_id"] == "cand_a"
        dominated = frontier.dominated_table()
        assert len(dominated) == 1
        assert dominated[0]["candidate_id"] == "cand_b"


# ---------------------------------------------------------------------------
# Report generation tests
# ---------------------------------------------------------------------------

class TestReportGeneration:
    """Report generation."""

    def test_per_candidate_summary(self) -> None:
        candidates = {
            "cand_a": [
                SweepResult(
                    candidate_id="cand_a",
                    chunk_size_ms=80,
                    lookahead_ms=0,
                    metrics={"content": 0.9, "identity": 0.85},
                    latency_ms=10.0,
                ),
                SweepResult(
                    candidate_id="cand_a",
                    chunk_size_ms=40,
                    lookahead_ms=0,
                    metrics={"content": 0.7, "identity": 0.65},
                    latency_ms=5.0,
                ),
            ],
        }
        summary = report_per_candidate(candidates)
        assert summary["cand_a"]["count"] == 2
        assert summary["cand_a"]["best_content"] == 0.9
        assert summary["cand_a"]["worst_content"] == 0.7
        assert summary["cand_a"]["best_latency_ms"] == 5.0

    def test_frontier_report_structure(self) -> None:
        candidates = {
            "cand_a": [
                SweepResult(
                    candidate_id="cand_a",
                    chunk_size_ms=80,
                    lookahead_ms=0,
                    metrics={"content": 0.9, "identity": 0.9},
                    latency_ms=10.0,
                ),
            ],
            "cand_b": [
                SweepResult(
                    candidate_id="cand_b",
                    chunk_size_ms=80,
                    lookahead_ms=0,
                    metrics={"content": 0.5, "identity": 0.5},
                    latency_ms=30.0,
                ),
            ],
        }
        report = generate_frontier_report(candidates)
        assert "per_candidate" in report
        assert "pareto_frontier" in report
        assert "dominated" in report
        assert report["n_non_dominated"] >= 1
        assert report["n_dominated"] >= 1
        assert report["n_total"] == report["n_non_dominated"] + report["n_dominated"]

    def test_pareto_table_fields(self) -> None:
        candidates = {
            "cand_a": [
                SweepResult(
                    candidate_id="cand_a",
                    chunk_size_ms=80,
                    lookahead_ms=0,
                    metrics={"content": 0.95, "identity": 0.9},
                    latency_ms=5.0,
                    rtf=0.3,
                    state_size_bytes=2048,
                ),
            ],
        }
        report = generate_frontier_report(candidates)
        row = report["pareto_frontier"][0]
        assert "content" in row
        assert "identity" in row
        assert "latency_ms" in row
        assert "status" in row

    def test_no_composite_scores_in_report(self) -> None:
        """Report should not contain any composite score field."""
        candidates = {
            "cand_a": [
                SweepResult(
                    candidate_id="cand_a",
                    chunk_size_ms=80,
                    lookahead_ms=0,
                    metrics={"content": 0.9, "identity": 0.85},
                    latency_ms=10.0,
                ),
            ],
        }
        report = generate_frontier_report(candidates)
        # No composite score field anywhere
        report_str = str(report)
        assert "composite_score" not in report_str
        assert "weighted_score" not in report_str
        assert "overall_score" not in report_str

    def test_empty_candidates(self) -> None:
        report = generate_frontier_report({})
        assert report["n_total"] == 0
        assert report["n_non_dominated"] == 0
        assert report["n_dominated"] == 0
