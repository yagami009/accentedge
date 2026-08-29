"""
Study configuration and pre-registration for Gate 1B listening studies.

Pre-registration ensures the study design is documented and frozen
before data collection begins, preventing p-hacking and HARKing.

Usage:
    config = StudyConfig.default_gate_1b()
    prereg = PreRegistration("gate_1b_strategy_compare", config)
    prereg.freeze()
    # ... cannot be modified after freeze ...
"""

import copy
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import IntEnum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class RatingScale(IntEnum):
    """Rating scale anchors."""
    SAME_PERSON = 1       # 1 = different person, 5 = same person
    ACCENT_SHIFT = 1      # 1 = still Indian, 5 = neutral US
    NATURALNESS = 1       # 1 = robotic, 5 = natural
    CONTENT_PRESERVED = 0 # categorical: yes/no/partial


@dataclass
class ExclusionRule:
    """Encapsulates a single pre-registered exclusion criterion."""

    name: str
    description: str
    check_fn: callable
    threshold: float = 0.0

    def check(self, rater_data: dict) -> bool:
        """Return True if the rater should be EXCLUDED."""
        return self.check_fn(rater_data)


@dataclass
class StudyConfig:
    """
    Complete configuration for a Gate 1B listening study.

    All parameters that govern the study design are captured here so
    they can be frozen into the pre-registration.
    """

    # Panel size
    min_raters: int = 3
    min_trials_per_rater: int = 20

    # Stimulus design
    anchor_inclusion_rate: float = 0.15  # fraction of trials that are anchors
    n_target_candidates: int = 3          # A/B/C
    include_gold: bool = True

    # Rating scales
    rating_scale_min: int = 1
    rating_scale_max: int = 5
    content_preserved_options: list = field(
        default_factory=lambda: ["yes", "no", "partial"]
    )

    # Pre-registered exclusion rules
    exclusion_rules: list[dict] = field(default_factory=list)

    # Study metadata
    study_id: str = ""
    study_name: str = ""
    description: str = ""
    hypotheses: str = ""

    # Analysis plan
    primary_metric: str = "accent_shift"
    primary_threshold: float = 3.0
    secondary_metrics: list[str] = field(default_factory=list)
    secondary_thresholds: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.secondary_metrics:
            self.secondary_metrics = ["same_person", "naturalness"]
        if not self.secondary_thresholds:
            self.secondary_thresholds = {
                "same_person": 4.0,
                "naturalness": 3.0,
                "accent_shift": 3.0,
            }

    def add_exclusion_rule(self, name: str, description: str, check_fn: callable,
                           threshold: float = 0.0):
        """Register a pre-registered exclusion rule."""
        rule = ExclusionRule(name, description, check_fn, threshold)
        self.exclusion_rules.append({
            "name": rule.name,
            "description": rule.description,
            "threshold": threshold,
        })

    def to_dict(self) -> dict:
        """Export configuration as a dictionary."""
        return {
            "study_id": self.study_id,
            "study_name": self.study_name,
            "description": self.description,
            "hypotheses": self.hypotheses,
            "min_raters": self.min_raters,
            "min_trials_per_rater": self.min_trials_per_rater,
            "anchor_inclusion_rate": self.anchor_inclusion_rate,
            "n_target_candidates": self.n_target_candidates,
            "include_gold": self.include_gold,
            "rating_scale_min": self.rating_scale_min,
            "rating_scale_max": self.rating_scale_max,
            "content_preserved_options": self.content_preserved_options,
            "exclusion_rules": self.exclusion_rules,
            "primary_metric": self.primary_metric,
            "primary_threshold": self.primary_threshold,
            "secondary_metrics": self.secondary_metrics,
            "secondary_thresholds": self.secondary_thresholds,
        }

    @classmethod
    def load(cls, path: Path) -> "StudyConfig":
        """Load configuration from JSON."""
        with open(path) as f:
            data = json.load(f)
        return cls(**data)

    def get_default_rules(self):
        """Add the standard Gate 1B exclusion rules."""
        self.add_exclusion_rule(
            name="low_agreement",
            description=(
                "Exclude raters whose same_person ratings correlate "
                "< 0.3 with the panel median across anchor trials."
            ),
            check_fn=lambda d: d.get("anchor_agreement", 1.0) < 0.3,
            threshold=0.3,
        )
        self.add_exclusion_rule(
            name="random_responding",
            description=(
                "Exclude raters who always respond with the same rating "
                "(zero variance across all trials)."
            ),
            check_fn=lambda d: d.get("rating_variance", 1.0) < 0.01,
            threshold=0.01,
        )
        self.add_exclusion_rule(
            name="identity_catch_failure",
            description=(
                "Exclude raters whose hit rate on identity catch trials "
                "is < 0.4 or > 0.95 (suggests misunderstanding or guessing)."
            ),
            check_fn=lambda d: (
                d.get("identity_catch_hit_rate", 0.5) < 0.4
                or d.get("identity_catch_hit_rate", 0.5) > 0.95
            ),
        )
        self.add_exclusion_rule(
            name="insufficient_trials",
            description=(
                f"Exclude raters who completed fewer than "
                f"{self.min_trials_per_rater} trials."
            ),
            check_fn=lambda d: d.get("n_trials_completed", 0) < self.min_trials_per_rater,
        )


def default_gate_1b_config() -> StudyConfig:
    """Create the default Gate 1B study configuration."""
    config = StudyConfig(
        min_raters=3,
        min_trials_per_rater=20,
        anchor_inclusion_rate=0.15,
        n_target_candidates=3,
        include_gold=True,
    )
    config.get_default_rules()
    return config


# ---------------------------------------------------------------------------
# Pre-registration
# ---------------------------------------------------------------------------

@dataclass
class PreRegistration:
    """
    A frozen study design document.

    Prevents p-hacking by capturing hypotheses, methods, and analysis
    plans before data collection begins. Cannot be modified after freezing.
    """

    prereg_id: str
    config: StudyConfig = field(repr=False)
    frozen: bool = False
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    frozen_at: str = ""
    version: str = "1.0"

    # Pre-registered design sections
    procedures: str = ""
    speaker_rules: str = ""
    metrics_section: str = ""
    listening_design: str = ""
    exclusion_rules_section: str = ""
    analysis_plan: str = ""
    sample_size_justification: str = ""
    stopping_rules: str = ""
    other_notes: str = ""

    def freeze(self):
        """Freeze the pre-registration — no further modifications allowed."""
        if self.frozen:
            raise RuntimeError(
                f"PreRegistration '{self.prereg_id}' is already frozen."
            )
        object.__setattr__(self, 'frozen', True)
        object.__setattr__(self, 'frozen_at', datetime.now(timezone.utc).isoformat())
        logger.info(f"PreRegistration '{self.prereg_id}' frozen at {self.frozen_at}")

    def unfreeze(self):
        """Unfreeze — always raises RuntimeError (pre-registrations are immutable once registered)."""
        raise RuntimeError(
            f"Cannot unfreeze PreRegistration '{self.prereg_id}'. "
            "Pre-registrations are immutable after freezing."
        )

    def __setattr__(self, name, value):
        """Prevent modification of fields after freezing."""
        if getattr(self, 'frozen', False) and name in self.__dict__:
            raise RuntimeError(
                f"Cannot modify '{name}' on frozen PreRegistration "
                f"'{self.prereg_id}'. Pre-registrations are immutable."
            )
        super().__setattr__(name, value)

    def to_dict(self) -> dict:
        """Export the full pre-registration as a dictionary."""
        return {
            "prereg_id": self.prereg_id,
            "version": self.version,
            "created_at": self.created_at,
            "frozen": self.frozen,
            "frozen_at": self.frozen_at,
            "config": self.config.to_dict(),
            "procedures": self.procedures,
            "speaker_rules": self.speaker_rules,
            "metrics_section": self.metrics_section,
            "listening_design": self.listening_design,
            "exclusion_rules_section": self.exclusion_rules_section,
            "analysis_plan": self.analysis_plan,
            "sample_size_justification": self.sample_size_justification,
            "stopping_rules": self.stopping_rules,
            "other_notes": self.other_notes,
        }

    def save(self, path: Path) -> None:
        """Export pre-registration to JSON for archival."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"PreRegistration saved: {path}")

    @classmethod
    def load(cls, path: Path) -> "PreRegistration":
        """Load a pre-registration from JSON."""
        with open(path) as f:
            data = json.load(f)

        config_data = data.pop("config", {})
        config = StudyConfig(**config_data)

        return cls(
            prereg_id=data["prereg_id"],
            config=config,
            frozen=data.get("frozen", False),
            created_at=data.get("created_at", ""),
            frozen_at=data.get("frozen_at", ""),
            version=data.get("version", "1.0"),
            procedures=data.get("procedures", ""),
            speaker_rules=data.get("speaker_rules", ""),
            metrics_section=data.get("metrics_section", ""),
            listening_design=data.get("listening_design", ""),
            exclusion_rules_section=data.get("exclusion_rules_section", []),
            analysis_plan=data.get("analysis_plan", ""),
            sample_size_justification=data.get("sample_size_justification", ""),
            stopping_rules=data.get("stopping_rules", ""),
            other_notes=data.get("other_notes", ""),
        )
