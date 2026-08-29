"""
Experiment controller and runner for Phase 0.

Provides:
- ExperimentConfig: dataclass for all experiment parameters
- Experiment: top-level controller registering sources/golds/targets
- ExperimentRunner: executes an experiment from config
"""

import json
import logging
import os
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Union

import numpy as np
import yaml

from accentedge.phase0.annotations import AnnotationDB, Utterance, TokenAnnotation, TokenLabel, Alignment
from accentedge.phase0.audio_io import AudioInfo, save_audio, validate_audio
from accentedge.phase0.degradation import (
    DegradationConfig,
    DEGRADATION_PRESETS,
    apply_degradation,
)
from accentedge.phase0.evaluation import EvaluationResult
from accentedge.phase0.provenance import (
    ProvenanceRecord,
    create_experiment_id,
    compute_audio_hash,
)
from accentedge.phase0.target_generation import TargetStrategy, get_strategy
from accentedge.phase0.reporting import Phase0Report, GateOutcome

logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """All parameters for a Phase 0 experiment run."""

    # Identity
    experiment_id: str = ""
    description: str = ""
    experimenter: str = ""

    # Speaker and utterance selection
    speaker_ids: list[str] = field(default_factory=list)
    utterance_ids: list[str] = field(default_factory=list)

    # Strategies and strengths to evaluate
    strategies: list[str] = field(default_factory=lambda: ["strategy_a", "strategy_b", "strategy_c"])
    strengths: list[float] = field(default_factory=lambda: [0.3, 0.6, 1.0])

    # Gate sequence: list of gate names in execution order
    gate_sequence: list[str] = field(default_factory=lambda: [
        "gate_neg1a", "gate_neg1b", "gate_0", "gate_1a",
        "gate_2", "contract_checkpoint", "gate_1b", "phase_0_decision"
    ])

    # Pass criteria (gate name -> threshold dict)
    pass_criteria: dict = field(default_factory=dict)

    # Time constraint (Phase 0: 10-week side-project timebox)
    time_limit_weeks: int = 10

    # Audio parameters
    sample_rate: int = 22050
    output_sample_rate: int = 22050

    # Degradation presets to test
    degradation_presets: list[str] = field(
        default_factory=lambda: ["clean", "NB", "noisy", "NB+noisy"]
    )

    # Output directories
    output_dir: str = "results"
    gate_artifacts_dir: str = "gate_artifacts"

    # Cheapest-disqualifying-first: order strategies by risk
    strategy_test_order: list[str] = field(
        default_factory=lambda: ["strategy_c", "strategy_b", "strategy_a"]
    )

    def __post_init__(self):
        if not self.experiment_id:
            self.experiment_id = create_experiment_id()

    def get_degradation_config(self, preset_name: str) -> DegradationConfig:
        """Return a degradation config by preset name."""
        if preset_name not in DEGRADATION_PRESETS:
            raise ValueError(
                f"Unknown degradation preset '{preset_name}'. "
                f"Available: {list(DEGRADATION_PRESETS.keys())}"
            )
        cfg = DEGRADATION_PRESETS[preset_name]
        # Override output sample rate with experiment config
        cfg.output_sample_rate = self.output_sample_rate
        return cfg


class Experiment:
    """
    Top-level controller for a Phase 0 experiment.

    Tracks all audio sources (source recordings, gold cross-accent
    recordings, and generated targets) with full provenance.

    Usage:
        exp = Experiment(config)
        exp.register_source(utterance, audio_path)
        exp.register_gold(utterance, audio_path)
        exp.register_target(utterance_id, strategy, audio, provenance)
        problems = exp.verify_provenance()
    """

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.experiment_id = config.experiment_id
        self.created_at = datetime.now(timezone.utc).isoformat()

        # Registered audio
        self._sources: dict[str, tuple[Utterance, Path, AudioInfo]] = {}
        self._golds: dict[str, tuple[Utterance, Path, AudioInfo]] = {}
        self._targets: dict[str, tuple[ProvenanceRecord, Path, AudioInfo]] = {}

        # Provenance records indexed by (utterance_id, strategy)
        self._provenance: dict[tuple[str, str], ProvenanceRecord] = {}

        # Annotations
        self.annotation_db = AnnotationDB()

        # Results accumulator
        self.results: list[EvaluationResult] = []

        logger.info(
            "Experiment %s initialized (%s)", self.experiment_id, config.description
        )

    def register_source(
        self, utterance: Utterance, audio_path: Union[str, Path]
    ) -> AudioInfo:
        """
        Register a source recording.

        Loads audio, computes hash, stores with annotation.
        """
        from accentedge.phase0.audio_io import load_audio
        audio_path = Path(audio_path)
        waveform, info = load_audio(audio_path, expected_sr=self.config.sample_rate)
        info.file_hash = compute_audio_hash(audio_path)
        self._sources[utterance.utterance_id] = (utterance, audio_path, info)
        self.annotation_db.add(utterance)
        logger.info(
            "Source registered: %s (speaker %s, %.2fs)",
            utterance.utterance_id, utterance.speaker_id, info.duration_seconds
        )
        return info

    def register_gold(
        self, utterance: Utterance, audio_path: Union[str, Path]
    ) -> AudioInfo:
        """
        Register a gold (natural cross-accent) recording.

        This is the same speaker producing the target pronunciation naturally.
        """
        from accentedge.phase0.audio_io import load_audio
        audio_path = Path(audio_path)
        waveform, info = load_audio(audio_path, expected_sr=self.config.sample_rate)
        info.file_hash = compute_audio_hash(audio_path)
        self._golds[utterance.utterance_id] = (utterance, audio_path, info)
        self.annotation_db.add(utterance)
        logger.info(
            "Gold registered: %s (speaker %s, %.2fs)",
            utterance.utterance_id, utterance.speaker_id, info.duration_seconds
        )
        return info

    def register_target(
        self,
        utterance_id: str,
        strategy: str,
        output_path: Union[str, Path],
        provenance: ProvenanceRecord,
    ) -> AudioInfo:
        """
        Register a generated target audio file.
        """
        from accentedge.phase0.audio_io import load_audio
        output_path = Path(output_path)
        waveform, info = load_audio(
            output_path, expected_sr=self.config.output_sample_rate
        )
        info.file_hash = compute_audio_hash(output_path)
        key = (utterance_id, strategy)
        self._targets[key] = (provenance, output_path, info)
        self._provenance[key] = provenance
        logger.info(
            "Target registered: %s / %s (%.2fs)",
            utterance_id, strategy, info.duration_seconds
        )
        return info

    def get_provenance(
        self, utterance_id: str, strategy: Optional[str] = None
    ) -> Optional[ProvenanceRecord]:
        """Retrieve provenance for a generated file."""
        key = (utterance_id, strategy) if strategy else None
        if key:
            return self._provenance.get(key)
        # Return all provenance records for this utterance
        results = {k: v for k, v in self._provenance.items() if k[0] == utterance_id}
        if not results:
            return None
        if len(results) == 1:
            return next(iter(results.values()))
        return results  # type: ignore

    def verify_provenance(self) -> list[str]:
        """
        Check that all registered files have valid provenance chains.

        Returns list of problem descriptions. Empty = all valid.
        """
        problems = []

        # Check targets have matching provenance
        for (utt_id, strategy), (prov, path, info) in self._targets.items():
            # Verify the provenance output hash matches actual file hash
            if prov.output_hash != info.file_hash:
                problems.append(
                    f"{utt_id}/{strategy}: output_hash mismatch "
                    f"(provenance={prov.output_hash}, actual={info.file_hash})"
                )

            # Verify source source_hash matches if we have the source
            if utt_id in self._sources:
                src_info = self._sources[utt_id][2]
                if prov.source_hash != src_info.file_hash:
                    problems.append(
                        f"{utt_id}/{strategy}: source_hash mismatch "
                        f"(provenance={prov.source_hash}, actual={src_info.file_hash})"
                    )

        # Check all gates have results (placeholder — gates populate results)
        registered_utterances = set(self._sources.keys()) | set(self._golds.keys())
        for utt_id in registered_utterances:
            # Every source/gold utterance should have at least one target
            has_target = any(k[0] == utt_id for k in self._targets)
            if not has_target:
                logger.warning("Utterance %s has no generated targets", utt_id)

        if problems:
            logger.error("Provenance verification found %d problems", len(problems))
        else:
            logger.info("Provenance verification: all chains valid")

        return problems

    def get_summary(self) -> dict:
        """Return a summary of the experiment state."""
        return {
            "experiment_id": self.experiment_id,
            "description": self.config.description,
            "created_at": self.created_at,
            "sources": len(self._sources),
            "golds": len(self._golds),
            "targets": len(self._targets),
            "utterances": len(self.annotation_db.utterances),
            "speakers": sorted(self.annotation_db.speakers),
            "results_count": len(self.results),
            "time_limit_weeks": self.config.time_limit_weeks,
        }


class ExperimentRunner:
    """
    Executes a Phase 0 experiment from a config file.

    Handles loading YAML/JSON configs, running gates, and exporting results.
    """

    def __init__(self, output_dir: str = "results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.gate_artifacts_dir = self.output_dir / "gate_artifacts"
        self.gate_artifacts_dir.mkdir(exist_ok=True)
        self.experiment: Optional[Experiment] = None

    def load_config(self, config_path: Union[str, Path]) -> ExperimentConfig:
        """
        Read YAML or JSON experiment config.

        Returns ExperimentConfig populated from the file.
        """
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")

        with open(config_path) as f:
            raw = yaml.safe_load(f) if config_path.suffix in (".yaml", ".yml") else json.load(f)

        # Map dict to ExperimentConfig fields
        kwargs = {}
        known_fields = {f.name for f in ExperimentConfig.__dataclass_fields__.values()}
        for key, value in raw.items():
            if key in known_fields:
                kwargs[key] = value

        config = ExperimentConfig(**kwargs)
        logger.info("Config loaded from %s (experiment: %s)", config_path, config.experiment_id)
        return config

    def run_gate(
        self,
        experiment: Experiment,
        gate_name: str,
        gate_func=None,
    ) -> tuple[GateOutcome, dict]:
        """
        Execute a single gate.

        If gate_func is provided, it is called as:
            gate_func(experiment, gate_name) -> (outcome, artifacts)

        Otherwise, a placeholder gate result is recorded.

        Returns (GateOutcome, artifacts_dict).
        """
        logger.info("Running gate: %s", gate_name)
        start_time = time.time()

        if gate_func is not None:
            outcome, artifacts = gate_func(experiment, gate_name)
        else:
            # Placeholder: record gate execution without evaluation
            outcome = GateOutcome.PENDING
            artifacts = {"gate_name": gate_name, "timestamp": datetime.now(timezone.utc).isoformat()}
            logger.info("Gate %s: PENDING (no gate function provided)", gate_name)

        elapsed = time.time() - start_time

        # Save gate artifacts
        gate_dir = self.gate_artifacts_dir / gate_name
        gate_dir.mkdir(exist_ok=True)

        import json as _json
        with open(gate_dir / "artifacts.json", "w") as f:
            _json.dump({"outcome": outcome, "artifacts": artifacts, "elapsed_seconds": elapsed}, f, indent=2, default=str)

        logger.info("Gate %s complete: %s (%.1fs)", gate_name, outcome, elapsed)
        return outcome, artifacts

    def export_results(self, experiment: Experiment) -> Path:
        """
        Generate the gate artifact table (spec section 56).

        Writes results/gate_artifact_table.json and returns its path.
        """
        table = []
        gate_dir = self.gate_artifacts_dir

        for gate_name in experiment.config.gate_sequence:
            artifacts_path = gate_dir / gate_name / "artifacts.json"
            if artifacts_path.exists():
                import json as _json
                with open(artifacts_path) as f:
                    data = _json.load(f)
                table.append({
                    "gate": gate_name,
                    "outcome": data.get("outcome", "UNKNOWN"),
                    "artifacts": data.get("artifacts", {}),
                    "elapsed_seconds": data.get("elapsed_seconds", 0),
                })
            else:
                table.append({"gate": gate_name, "outcome": "NOT_RUN", "artifacts": {}, "elapsed_seconds": 0})

        output_path = self.output_dir / "gate_artifact_table.json"
        with open(output_path, "w") as f:
            import json as _json
            _json.dump({"experiment_id": experiment.experiment_id, "gates": table}, f, indent=2)

        logger.info("Gate artifact table written: %s", output_path)
        return output_path
