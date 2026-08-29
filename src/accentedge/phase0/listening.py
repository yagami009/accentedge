"""
Gate 1B listening study framework.

Implements the controlled listening experiment for adjudicating generated
accent transformation targets against gold-standard cross-accent speech.

Key components:
- Stimulus / StimulusSet: trial audio with blinding codes
- ListeningPanel: rater management and eligibility screening
- ListeningTrial: individual rater responses
- ListeningStudy: full study orchestration

Blinding: Listeners see opaque codes (e.g., "Sample A7K2") instead of
condition labels, preventing response bias.

Identity catch trials: Same-speaker pairs embedded in the stimulus set
to verify that raters can discriminate same vs. different speakers.
"""

import copy
import json
import logging
import random
import string
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from accentedge.phase0.study_config import PreRegistration, StudyConfig
from accentedge.phase0.stats import compute_dprime

logger = logging.getLogger(__name__)


# ───────────────────────��──────────────────────────────────────────
# Stimulus
# ──────────────────────────────────────────────────────────────────

@dataclass
class Stimulus:
    """
    One listening trial: a single audio file presented to a rater.

    The blinding code hides the condition identity from the listener.
    The code is deterministic for a given stimulus so it remains
    consistent across raters and analyses.
    """

    audio_path: str
    condition: str  # "source", "target_a", "target_b", "target_c", "gold_low", "gold_high"
    utterance_id: str
    is_anchor: bool = False
    is_identity_catch: bool = False
    blinding_code: str = ""
    speaker_id: str = ""
    text: str = ""

    def __post_init__(self):
        if not self.blinding_code:
            self.blinding_code = _generate_blinding_code(
                self.audio_path, self.condition, self.utterance_id
            )

    def to_dict(self) -> dict:
        return {
            "audio_path": self.audio_path,
            "condition": self.condition,
            "utterance_id": self.utterance_id,
            "is_anchor": self.is_anchor,
            "is_identity_catch": self.is_identity_catch,
            "blinding_code": self.blinding_code,
            "speaker_id": self.speaker_id,
            "text": self.text,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Stimulus":
        return cls(**d)


def _generate_blinding_code(audio_path: str, condition: str, utterance_id: str) -> str:
    """
    Generate a deterministic, human-friendly blinding code.

    Format: "<prefix><3-char hash>" e.g., "A7K2", "B3X9"
    The prefix encodes the broad condition category so the experimenter
    can still track things (but listeners cannot decode it):
      S = source, T = target, G = gold, A = anchor, I = identity catch
    """
    raw = f"{audio_path}:{condition}:{utterance_id}"
    h = hash(raw)
    if h < 0:
        h = -h + 0x7FFFFFFFFFFFFFFF

    cond_lower = condition.lower()
    if "identity" in cond_lower or "catch" in cond_lower:
        prefix = "I"
    elif "anchor" in cond_lower:
        prefix = "A"
    elif "gold" in cond_lower:
        prefix = "G"
    elif "target" in cond_lower:
        prefix = "T"
    elif "source" in cond_lower:
        prefix = "S"
    else:
        prefix = "X"

    chars = string.ascii_uppercase + string.digits
    code = ""
    h_val = h
    for _ in range(3):
        code += chars[h_val % len(chars)]
        h_val //= len(chars)

    return f"Sample {prefix}{code}"


def _generate_blinding_code(audio_path: str, condition: str, utterance_id: str) -> str:
    """
    Generate a deterministic, human-friendly blinding code.

    Format: "Sample <prefix><3-char hash>" e.g., "Sample A7K2", "Sample T3X9"
    The prefix encodes the broad condition category so the experimenter
    can still track things (but listeners cannot decode meaning).

    Prefixes: S = source, T = target, G = gold, A = anchor, I = identity catch
    """
    raw = f"{audio_path}:{condition}:{utterance_id}"
    h = hash(raw)
    if h < 0:
        h = -h + 0x7FFFFFFFFFFFFFFF

    cond_lower = condition.lower()
    if "identity" in cond_lower or "catch" in cond_lower:
        prefix = "I"
    elif "anchor" in cond_lower:
        prefix = "A"
    elif "gold" in cond_lower:
        prefix = "G"
    elif "target" in cond_lower:
        prefix = "T"
    elif "source" in cond_lower:
        prefix = "S"
    else:
        prefix = "X"

    chars = string.ascii_uppercase + string.digits
    code = ""
    h_val = h
    for _ in range(3):
        code += chars[h_val % len(chars)]
        h_val //= len(chars)

    return f"Sample {prefix}{code}"

@dataclass
class StimulusSet:
    """
    Complete set of stimuli for one evaluation round.

    Contains source, target candidates (A/B/C), gold references,
    and anchor items. Supports randomized presentation order and
    blinding code generation.
    """

    stimuli: list = field(default_factory=list)
    source_condition: str = "source"
    target_conditions: List[str] = field(default_factory=lambda: ["target_a", "target_b", "target_c"])
    gold_conditions: List[str] = field(default_factory=lambda: ["gold"])
    presentation_order: List[str] = field(default_factory=list)
    randomized: bool = False

    def add_stimulus(self, stimulus: Stimulus):
        """Add a stimulus to the set."""
        self.stimuli.append(stimulus)
        self.presentation_order = []

    def add_from_paths(
        self,
        audio_path: str,
        condition: str,
        utterance_id: str,
        is_anchor: bool = False,
        is_identity_catch: bool = False,
        speaker_id: str = "",
        text: str = "",
    ):
        """Convenience: add a stimulus from path + metadata."""
        s = Stimulus(
            audio_path=audio_path,
            condition=condition,
            utterance_id=utterance_id,
            is_anchor=is_anchor,
            is_identity_catch=is_identity_catch,
            speaker_id=speaker_id,
            text=text,
        )
        self.add_stimulus(s)
        return s

    def get_by_condition(self, condition: str) -> List[Stimulus]:
        """Get all stimuli for a given condition string."""
        return [s for s in self.stimuli if s.condition == condition]

    def get_anchors(self) -> List[Stimulus]:
        """Get anchor stimuli."""
        return [s for s in self.stimuli if s.is_anchor]

    def get_identity_catches(self) -> List[Stimulus]:
        """Get identity catch trial stimuli."""
        return [s for s in self.stimuli if s.is_identity_catch]

    def get_targets(self) -> List[Stimulus]:
        """Get all target candidate stimuli."""
        return [s for s in self.stimuli if s.condition in self.target_conditions]

    def get_gold(self) -> List[Stimulus]:
        """Get all gold reference stimuli."""
        return [s for s in self.stimuli if s.condition in self.gold_conditions]

    def get_by_utterance(self, utterance_id: str) -> List[Stimulus]:
        """Get all stimuli for a specific utterance."""
        return [s for s in self.stimuli if s.utterance_id == utterance_id]

    def randomize(self, seed: Optional[int] = None) -> List[str]:
        """Randomize the presentation order. Returns blinding codes."""
        codes = [s.blinding_code for s in self.stimuli]
        if seed is not None:
            rng = random.Random(seed)
            rng.shuffle(codes)
        else:
            random.shuffle(codes)
        self.presentation_order = codes
        self.randomized = True
        logger.info(f"StimulusSet randomized: {len(codes)} items")
        return codes

    def get_presentation_sequence(self) -> List[dict]:
        """Return the randomized presentation sequence with audio info."""
        if not self.presentation_order:
            if not self.randomized:
                self.randomize()
            else:
                self.presentation_order = [s.blinding_code for s in self.stimuli]

        code_to_stimulus = {s.blinding_code: s for s in self.stimuli}
        return [
            {
                "blinding_code": code,
                "audio_path": code_to_stimulus[code].audio_path,
                "condition": code_to_stimulus[code].condition,
                "is_anchor": code_to_stimulus[code].is_anchor,
                "utterance_id": code_to_stimulus[code].utterance_id,
            }
            for code in self.presentation_order
            if code in code_to_stimulus
        ]

    @property
    def n_stimuli(self) -> int:
        return len(self.stimuli)

    @property
    def n_anchors(self) -> int:
        return len(self.get_anchors())

    def to_dict(self) -> dict:
        return {
            "stimuli": [s.to_dict() for s in self.stimuli],
            "presentation_order": self.presentation_order,
            "randomized": self.randomized,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "StimulusSet":
        ss = cls(
            stimuli=[Stimulus.from_dict(s) for s in d.get("stimuli", [])],
            presentation_order=d.get("presentation_order", []),
            randomized=d.get("randomized", False),
        )
        return ss


# ──────────────────────────────────────────────────────────────────
# ListeningPanel
# ──────────────────────────────────────────────────────────────────

@dataclass
class RaterProfile:
    """Profile for a listening panel rater."""

    rater_id: str
    native_language: str = "en-US"
    accent_expertise: str = "none"
    screening_score: float = 0.0
    eligible: bool = True
    excluded: bool = False
    exclusion_reason: str = ""
    notes: str = ""
    metadata: dict = field(default_factory=dict)

    def is_eligible(self) -> bool:
        """Check if the rater is eligible to participate."""
        return self.eligible and not self.excluded

    def to_dict(self) -> dict:
        return {
            "rater_id": self.rater_id,
            "native_language": self.native_language,
            "accent_expertise": self.accent_expertise,
            "screening_score": self.screening_score,
            "eligible": self.eligible,
            "excluded": self.excluded,
            "exclusion_reason": self.exclusion_reason,
            "notes": self.notes,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "RaterProfile":
        return cls(**d)


class ListeningPanel:
    """Manages a group of raters for a listening study."""

    def __init__(self):
        self.raters: Dict[str, RaterProfile] = {}

    def add_rater(
        self,
        rater_id: str,
        native_language: str = "en-US",
        accent_expertise: str = "none",
        screening_score: float = 0.0,
        **kwargs,
    ) -> RaterProfile:
        """Register a new rater."""
        profile = RaterProfile(
            rater_id=rater_id,
            native_language=native_language,
            accent_expertise=accent_expertise,
            screening_score=screening_score,
            **kwargs,
        )
        self.raters[rater_id] = profile
        logger.info(f"Rater added: {rater_id} (lang={native_language}, expertise={accent_expertise})")
        return profile

    def remove_rater(self, rater_id: str, reason: str = ""):
        """Remove a rater from the panel."""
        if rater_id not in self.raters:
            raise KeyError(f"Rater '{rater_id}' not found in panel")
        profile = self.raters[rater_id]
        profile.excluded = True
        profile.exclusion_reason = reason
        del self.raters[rater_id]
        logger.info(f"Rater removed: {rater_id} (reason: {reason})")

    def get_eligible_raters(self, config: Optional[StudyConfig] = None) -> List[RaterProfile]:
        """
        Return list of raters eligible to participate.

        Filters by native language (exclude en-IN) and accent expertise
        when no config is provided. When config is given, uses its
        thresholds and excluded languages.
        """
        excluded_langs = {"en-IN"}
        min_score = 0.8
        require_expertise = True

        if config is not None:
            if hasattr(config, 'excluded_native_languages'):
                excluded_langs = set(config.excluded_native_languages)
            if hasattr(config, 'min_screening_score'):
                min_score = config.min_screening_score
            if hasattr(config, 'accent_expertise_required'):
                require_expertise = config.accent_expertise_required

        eligible = []
        for rater in self.raters.values():
            if not rater.is_eligible():
                continue
            if rater.native_language in excluded_langs:
                continue
            if rater.screening_score < min_score:
                continue
            if require_expertise and rater.accent_expertise == "none":
                continue
            eligible.append(rater)

        return eligible

    def get_rater(self, rater_id: str) -> Optional[RaterProfile]:
        """Get a rater profile by ID."""
        return self.raters.get(rater_id)

    @property
    def total_raters(self) -> int:
        return len(self.raters)

    @property
    def eligible_count(self) -> int:
        return len(self.get_eligible_raters())

    def to_dict(self) -> dict:
        return {
            "raters": {rid: r.to_dict() for rid, r in self.raters.items()},
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ListeningPanel":
        panel = cls()
        for rid, rdata in d.get("raters", {}).items():
            panel.raters[rid] = RaterProfile.from_dict(rdata)
        return panel


# ──────────────────────────────────────────────────────────────────
# ListeningTrial
# ──────────────────────────────────────────────────────────────────

@dataclass
class ListeningTrial:
    """One rater's response to one stimulus."""

    trial_id: str = ""
    rater_id: str = ""
    stimulus: Optional[Stimulus] = None
    same_person: Optional[int] = None
    accent_shift: Optional[int] = None
    naturalness: Optional[int] = None
    content_preserved: Optional[str] = None
    response_time_ms: Optional[float] = None
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    notes: str = ""

    def __post_init__(self):
        if not self.trial_id:
            self.trial_id = f"trial_{self.rater_id}_{id(self)}"

    def is_complete(self) -> bool:
        """Check if all required responses are present."""
        return all([
            self.same_person is not None,
            self.accent_shift is not None,
            self.naturalness is not None,
            self.content_preserved is not None,
        ])

    def responses(self) -> dict:
        """Return all response values as a dictionary."""
        return {
            "same_person": self.same_person,
            "accent_shift": self.accent_shift,
            "naturalness": self.naturalness,
            "content_preserved": self.content_preserved,
        }

    @property
    def time_ms(self) -> Optional[float]:
        """Alias for response_time_ms (backward compatibility)."""
        return self.response_time_ms

    def to_dict(self) -> dict:
        return {
            "trial_id": self.trial_id,
            "rater_id": self.rater_id,
            "stimulus": self.stimulus.to_dict(),
            "same_person": self.same_person,
            "accent_shift": self.accent_shift,
            "naturalness": self.naturalness,
            "content_preserved": self.content_preserved,
            "response_time_ms": self.response_time_ms,
            "timestamp": self.timestamp,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ListeningTrial":
        stimulus = Stimulus.from_dict(d.pop("stimulus"))
        return cls(stimulus=stimulus, **d)


# ─────────────────────────��────────────────────────────────────────
# ListeningStudy
# ──────────────────────────────────────────────────────────────────

class ListeningStudy:
    """
    Orchestrates the full Gate 1B listening study.

    Manages stimulus creation, trial assignment, response collection,
    rater screening, and result export.
    """

    def __init__(
        self,
        study_id: str,
        config: StudyConfig,
        panel: Optional[ListeningPanel] = None,
        stimulus_set: Optional[StimulusSet] = None,
    ):
        self.study_id = study_id
        self.config = config
        self.panel = panel or ListeningPanel()
        self.stimulus_set = stimulus_set or StimulusSet()
        self.trials: List[ListeningTrial] = []
        self._trial_counter = 0
        self._assignments: Dict[str, List[str]] = {}
        self.prereg: Optional[PreRegistration] = None

    def register_prereg(self, prereg: PreRegistration):
        """
        Register a frozen pre-registration for this study.

        The pre-registration must be frozen before it can be registered.
        """
        if not prereg.frozen:
            raise RuntimeError(
                "Cannot register an unfrozen PreRegistration. "
                "Freeze the pre-registration before registering."
            )
        self.prereg = prereg
        logger.info(f"PreRegistration '{prereg.prereg_id}' registered for study '{self.study_id}'")

    def create_stimulus_set(
        self,
        source_paths: Dict[str, str],
        target_paths: Dict[str, Dict[str, str]],
        gold_paths: Dict[str, str],
        anchor_paths: Optional[List[dict]] = None,
        identity_catch_paths: Optional[List[dict]] = None,
    ) -> StimulusSet:
        """Build the stimulus set from generated targets and gold references."""
        ss = StimulusSet()

        for utt_id, path in source_paths.items():
            ss.add_from_paths(audio_path=path, condition="source", utterance_id=utt_id)

        for strategy, utt_dict in target_paths.items():
            for utt_id, path in utt_dict.items():
                ss.add_from_paths(
                    audio_path=path, condition=f"target_{strategy}", utterance_id=utt_id
                )

        for condition, path in gold_paths.items():
            ss.add_from_paths(
                audio_path=path, condition=f"gold_{condition}", utterance_id=f"gold_{condition}"
            )

        if anchor_paths:
            for a in anchor_paths:
                s = Stimulus(
                    audio_path=a["audio_path"],
                    condition=a.get("condition", "anchor"),
                    utterance_id=a["utterance_id"],
                    is_anchor=True,
                    speaker_id=a.get("speaker_id", ""),
                    text=a.get("text", ""),
                )
                ss.add_stimulus(s)

        if identity_catch_paths:
            for c in identity_catch_paths:
                s = Stimulus(
                    audio_path=c["audio_path"],
                    condition=c.get("condition", "identity_catch"),
                    utterance_id=c["utterance_id"],
                    is_identity_catch=True,
                    speaker_id=c.get("speaker_id", ""),
                    text=c.get("text", ""),
                )
                ss.add_stimulus(s)

        self.stimulus_set = ss
        logger.info(
            f"StimulusSet created: {ss.n_stimuli} stimuli, {ss.n_anchors} anchors"
        )
        return ss

    def assign_trials(self, seed: Optional[int] = None) -> Dict[str, List[str]]:
        """Assign stimuli to raters in a balanced design."""
        eligible = self.panel.get_eligible_raters()
        if not eligible:
            raise RuntimeError("No eligible raters available for assignment")

        if self.stimulus_set.n_stimuli == 0:
            raise RuntimeError("StimulusSet is empty")

        assignments = {}
        for rater in eligible:
            rater_seed = seed
            if seed is not None:
                rater_seed = seed + hash(rater.rater_id) % 100000
            codes = self.stimulus_set.randomize(seed=rater_seed)
            assignments[rater.rater_id] = codes

        self._assignments = assignments
        logger.info(
            f"Trials assigned to {len(assignments)} raters "
            f"({self.stimulus_set.n_stimuli} stimuli each)"
        )
        return assignments

    def collect_response(
        self,
        rater_id: str,
        blinding_code: str,
        same_person: int,
        accent_shift: int,
        naturalness: int,
        content_preserved: str,
        response_time_ms: Optional[float] = None,
        notes: str = "",
    ) -> ListeningTrial:
        """Record a rater's response to a stimulus."""
        for name, val in [
            ("same_person", same_person),
            ("accent_shift", accent_shift),
            ("naturalness", naturalness),
        ]:
            if not (1 <= val <= 5):
                raise ValueError(f"{name} must be 1-5, got {val}")

        if content_preserved not in ("yes", "no", "partial"):
            raise ValueError(f"content_preserved must be yes/no/partial, got '{content_preserved}'")

        code_to_stim = {s.blinding_code: s for s in self.stimulus_set.stimuli}
        if blinding_code not in code_to_stim:
            raise KeyError(f"Blinding code '{blinding_code}' not found")

        self._trial_counter += 1
        trial = ListeningTrial(
            rater_id=rater_id,
            stimulus=code_to_stim[blinding_code],
            same_person=same_person,
            accent_shift=accent_shift,
            naturalness=naturalness,
            content_preserved=content_preserved,
            response_time_ms=response_time_ms,
            notes=notes,
        )
        self.trials.append(trial)
        return trial

    def get_trials_for_rater(self, rater_id: str) -> List[ListeningTrial]:
        """Get all trials completed by a specific rater."""
        return [t for t in self.trials if t.rater_id == rater_id]

    def get_trials_for_condition(self, condition: str) -> List[ListeningTrial]:
        """Get all trials for a given condition."""
        return [t for t in self.trials if t.stimulus.condition == condition]

    def compute_rater_reliability(self, rater_id: str) -> dict:
        """Compute reliability metrics for a single rater."""
        rater_trials = self.get_trials_for_rater(rater_id)

        # Anchor agreement
        anchor_trials = [t for t in rater_trials if t.stimulus.is_anchor]
        anchor_agreement = 1.0
        if len(anchor_trials) >= 2:
            anchor_ratings = []
            for t in anchor_trials:
                if t.same_person is not None:
                    anchor_ratings.append(t.same_person)
                if t.accent_shift is not None:
                    anchor_ratings.append(t.accent_shift)
                if t.naturalness is not None:
                    anchor_ratings.append(t.naturalness)
            if len(anchor_ratings) >= 2:
                cv = np.std(anchor_ratings) / (np.mean(anchor_ratings) + 1e-10)
                anchor_agreement = max(0.0, 1.0 - cv)

        # Identity catch trial d-prime
        catch_trials = [t for t in rater_trials if t.stimulus.is_identity_catch]
        hit_rate = 0.5
        dprime = np.nan
        if catch_trials:
            hits = sum(
                1 for t in catch_trials
                if t.same_person is not None and t.same_person >= 3
            )
            misses = sum(
                1 for t in catch_trials
                if t.same_person is not None and t.same_person < 3
            )
            total_same = sum(
                1 for t in catch_trials
                if "same" in t.stimulus.condition.lower()
            )
            total_diff = len(catch_trials) - total_same

            hits = max(hits, 0)
            misses = max(misses, 0)
            false_alarms = max(total_diff - (len(catch_trials) - hits - misses), 0)
            correct_rejections = max(total_diff - false_alarms, 0)

            if hits + misses + false_alarms + correct_rejections > 0:
                dprime = compute_dprime(
                    hits=hits, misses=misses,
                    false_alarms=false_alarms, correct_rejections=correct_rejections,
                )
            hit_rate = hits / max(hits + misses, 1)

        return {
            "anchor_agreement": anchor_agreement,
            "identity_catch_hit_rate": hit_rate,
            "dprime": dprime,
            "n_trials_completed": len(rater_trials),
            "n_anchor_trials": len(anchor_trials),
            "n_catch_trials": len(catch_trials),
        }

    def screen_raters(self) -> Dict[str, dict]:
        """Apply pre-registered exclusion rules to identify unreliable raters."""
        results = {}
        eligible = self.panel.get_eligible_raters()

        for rater in eligible:
            reliability = self.compute_rater_reliability(rater.rater_id)
            exclusion_reasons = []

            for rule_def in self.config.exclusion_rules:
                rule_name = rule_def["name"]

                if rule_name == "low_agreement":
                    if reliability["anchor_agreement"] < 0.3:
                        exclusion_reasons.append(
                            f"low_agreement: {reliability['anchor_agreement']:.2f}"
                        )

                elif rule_name == "random_responding":
                    hr = reliability["identity_catch_hit_rate"]
                    if hr < 0.4 or hr > 0.95:
                        exclusion_reasons.append(
                            f"random_responding: hit_rate={hr:.2f}"
                        )

                elif rule_name == "insufficient_trials":
                    if reliability["n_trials_completed"] < self.config.min_trials_per_rater:
                        exclusion_reasons.append(
                            f"insufficient_trials: {reliability['n_trials_completed']}"
                        )

            excluded = len(exclusion_reasons) > 0
            if excluded:
                self.panel.remove_rater(rater.rater_id, "; ".join(exclusion_reasons))

            results[rater.rater_id] = {
                "excluded": excluded,
                "reasons": exclusion_reasons,
                "reliability": reliability,
            }

        n_excluded = sum(1 for r in results.values() if r["excluded"])
        logger.info(
            f"Rater screening: {len(results)} checked, "
            f"{n_excluded} excluded, {len(results) - n_excluded} retained"
        )
        return results

    def export_results(self, path: Path) -> dict:
        """Export all study results to JSON for analysis."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "study_id": self.study_id,
            "config": self.config.to_dict(),
            "stimulus_set": self.stimulus_set.to_dict(),
            "panel": self.panel.to_dict(),
            "trials": [t.to_dict() for t in self.trials],
            "assignments": self._assignments,
            "n_trials": len(self.trials),
            "exported_at": datetime.now(timezone.utc).isoformat(),
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Results exported: {path} ({len(self.trials)} trials)")
        return data

    def import_results(self, path: Path) -> dict:
        """Import study results from JSON."""
        with open(path) as f:
            data = json.load(f)

        if data.get("stimulus_set"):
            self.stimulus_set = StimulusSet.from_dict(data["stimulus_set"])
        if data.get("panel"):
            self.panel = ListeningPanel.from_dict(data["panel"])

        self.trials = [ListeningTrial.from_dict(t) for t in data.get("trials", [])]
        self._assignments = data.get("assignments", {})

        logger.info(
            f"Results imported: {path} ({len(self.trials)} trials, "
            f"{self.panel.total_raters} raters)"
        )
        return data
