# Phase 0 — Target & Contract Feasibility: Comprehensive Code Analysis

> **Date:** 2026-08-27  
> **Scope:** `docs/phase0/*.md` + `src/accentedge/phase0/*.py`  
> **Audience:** Engineering team evaluating whether to advance to Phase 1

---

## Table of Contents

1. [What TGFP v2 Is Trying to Prove](#1-what-tgfp-v2-is-trying-to-prove)
2. [Experiment Flow: Step 0 Through All Gates](#2-experiment-flow-step-0-through-all-gates)
3. [Module Interfaces (17 Modules)](#3-module-interfaces)
4. [Data Formats Used](#4-data-formats-used)
5. [Target Generation Strategies A/B/C](#5-target-generation-strategies-abc)
6. [How Identity Preservation Is Measured](#6-how-identity-preservation-is-measured)
7. [Key Observations & Risks](#7-key-observations--risks)

---

## 1. What TGFP v2 Is Trying to Prove

TGFP v2 (Target Generation Feasibility Protocol v2) is the **single decisive experiment** that determines whether AccentEdge proceeds to Phase 1 (streaming/real-time model training) or stops entirely.

### The Core Question

> Can we produce a target waveform that sounds like the **same human speaking differently**, not like a different voice or impression?

TGFP v1 was exploratory. TGFP v2 is precise: it specifies one experiment (Step 0) whose result determines go/no-go.

### What Phase 0 Is NOT Trying to Prove

Per `PHASE_0_SPEC.md`:

| Question | Phase 0 answers it? |
|---|---|
| Can it run on CPU? | No |
| Which streaming encoder? | No |
| Should we use DDSP or HiFi-GAN? | No |
| What will the Windows client look like? | No |
| Does WebRTC work? | No |
| What BPO dashboard do we need? | No |
| How do we price it? | No |
| **Can we create a valid target?** | **YES** |
| **Does it still sound like the same person?** | **YES** |
| **Did we preserve words exactly?** | **YES** |
| **What pronunciation dimensions should change?** | **YES** |
| **What should remain unchanged?** | **YES** |
| **How much timing movement is natural?** | **YES** |
| **How much speaker-embedding movement is natural?** | **YES** |
| **Can synthetic targets approximate human cross-accent speech?** | **YES** |
| **Full conversion or sparse repair?** | **YES** |

### The Product Thesis (from `LLMForge Runtime Thesis` / spec)

Phase 0 exists to answer one question:

> Can we define and produce an offline speech target that represents what AccentEdge should eventually generate, while preserving the speaker and meaning well enough that we would deliberately train a weaker causal real-time model to imitate it?

Everything else is downstream.

### The Linguistic Contract

The v1 linguistic contract (`PHASE_0_SPEC.md` § Linguistic Contract v1) specifies exactly what AccentEdge will and will not change:

**MUST change (accent dimensions):**

| Dimension | Direction |
|---|---|
| /t/ aspiration | Target US aspirated |
| /d/ retroflexion | Target dental |
| /ʈʂ/ → /tʃ/ | Transform |
| Rhoticity | Add where absent |
| Flapping | Transform where needed |
| /æ/ raising | Target raised |
| /ɪ/ tensing | Target tensed |
| /oʊ/ monophthongization | Target diphthong |
| /v/–/w/ | Transform where needed |
| Intervocalic /t/ flapping | Transform |
| Cluster epenthesis | Transform |
| Lexical stress | Bounded correction |
| Phone duration | Allowed to change |

**MUST preserve:**

| Dimension | Direction |
|---|---|
| Word duration | Bounded empirically |
| Phrase timing | Prefer preservation |
| Global rhythm | Out of v1 |
| Global intonation | Out of v1 |
| Timbre | Preserve |
| Voice quality | Preserve |
| Pitch range | Preserve broadly |
| Emotion | Preserve |
| Words | Preserve exactly |

### TGFP v2 Pass Criteria (Step 0 / Strategy B)

All four must be met simultaneously:

1. Same speaker >= 4/5
2. Accent shift >= 3/5
3. Naturalness >= 3/5
4. Content preserved = Yes

### Phase 0 Decision Outcomes

| Outcome | Meaning |
|---|---|
| FULL-S2S PASS | Good whole-speech targets can be created; proceed toward causal direct S2S |
| SPARSE-REPAIR PASS | Full transformation damages identity, but controlled pronunciation repair works |
| TEACHER FAIL / GOLD PASS | Humans demonstrate the desired transformation, but our synthetic supervision cannot reproduce it yet |
| FUNDAMENTAL FAIL | Even natural same-speaker cross-accent behavior cannot satisfy the intended product contract |

---

## 2. Experiment Flow: Step 0 Through All Gates

### Step 0 — Single-Speaker Feasibility Test

**Setup (from TGFP_V2.md):**
- **One speaker:** Indian English, self-identifies as having a noticeable Indian accent
- **One sentence:** Contains at least one phoneme contrast between en-IN and en-US (e.g., /t/ flapping, /æ/ raising)
- **One strategy:** Strategy B (hand-built target)
- **Repetitions:** 5 independent conversions per strategy
- **Evaluators:** >= 3 native US English speakers trained in accent evaluation

**Procedure:**
1. Record source sentence in Indian-English pronunciation
2. Expert (native US English speaker) produces a target pronunciation guide:
   - Phone-level transcription of US-neutral realization
   - Word-level stress pattern
   - Phrase-level intonation contour
3. Source speaker attempts to produce the target pronunciation while maintaining their own voice
4. Record multiple takes
5. Select best take as `strategy_b/candidate.wav`
6. Listeners rate on 4 dimensions (1–5 scale except content)
7. Apply pass criteria (all four must be met)

**Decision tree (from TGFP_V2.md):**
```
STEP 0 RESULT
       │
   ┌───┴
   │
   ▼
[Pass criteria met?]
   │
   ├── YES → Gate 1A: generate teacher targets
   │
   └── NO
       │
       ├── Strategy B failed but human gold passes?
       │   └── YES → TEACHER FAIL / GOLD PASS
       │
       ├── Sparse repair looks promising?
       │   └── YES → Gate 0: strategy C one-shot test
       │
       └── NO
           └── FUNDAMENTAL FAIL
```

### Full Gate Sequence (from PHASE_0_SPEC.md)

```
PHASE 0
├── Gate -1A    Source annotation validity
├── Gate -1B    Measurement/probe validity
├── Gate 0      One-shot target sanity check
├── Gate 1A     Generate candidate teacher targets
├── Gate 2      Natural cross-accent calibration
├── Contract checkpoint    Is our v1 linguistic contract human-realistic?
├── Gate 1B     Adjudicate generated targets using gold-derived criteria
└── Phase-0 Decision
    ├── FULL-S2S PASS
    ├── SPARSE-REPAIR PASS
    ├── TEACHER FAIL / GOLD PASS
    └── FUNDAMENTAL FAIL
```

#### Gate -1A — Source Validity

> Do we actually know what was pronounced, where it was pronounced, and whether each targeted token needed correction in the first place?

Procedure:
1. Raw speech → exact human transcript
2. Initial forced alignment
3. Manual boundary correction
4. Canonical target phones
5. Observed source realization
6. Target-feature labels
7. ALREADY-TARGET / DEVIANT / AMBIGUOUS

#### Gate -1B — Measurement Validity

Before using automated accent metrics, verify those metrics can actually distinguish what they claim to measure.

Probe-validity test:
- Known natural US token → probe → should look target-like
- Known Indian-English token → probe → should look substitute-like
- Blind known tokens → verify probe accuracy >= 80%

#### Gate 0 — One-Shot Target Sanity Check

Quick smoke test on a few utterances before committing to full evaluation. Validates that the generation pipeline produces audible output that meets basic quality thresholds.

#### Gate 1A — Generate Candidate Teacher Targets

Run all strategies (A, B, C) on the full source set. Produce WAV files and provenance records. This is the main generation phase.

#### Gate 2 — Natural Cross-Accent Calibration

Collect gold-standard cross-accent speech from the same speaker (or a matched US-English speaker) to establish what the transformation should sound like naturally.

#### Contract Checkpoint

> Is our v1 linguistic contract human-realistic?

Compare the linguistic contract's predicted changes against what humans actually do when they naturally shift accent. If the contract asks for changes that humans don't naturally make, revise it.

#### Gate 1B — Adjudicate Generated Targets

Rated by listeners using blind stimulus sets:
- **Same person?** 1–5
- **Accent shift?** 1–5
- **Naturalness?** 1–5
- **Content preserved?** Yes/No/Partial

Pass criteria (all must be met):
- Same speaker >= 4/5
- Accent shift >= 3/5
- Naturalness >= 3/5
- Content preserved = Yes

#### Phase-0 Decision (from `experiment.py` gate outcomes)

| Outcome | Trigger |
|---|---|
| FULL-S2S PASS | Gate 1B passes for whole-speech strategy |
| SPARSE-REPAIR PASS | Gate 1B passes only for strategy C |
| TEACHER FAIL / GOLD PASS | Gold human recording passes but all synthetic strategies fail |
| FUNDAMENTAL FAIL | Even human gold doesn't pass the contract |

---

## 3. Module Interfaces

### 3.1 experiment.py

**Purpose:** Top-level experiment controller and runner.

**Key Classes:**

```python
@dataclass
class ExperimentConfig:
    description: str
    speakers: list[str]
    strategies: list[str]
    output_sample_rate: int
    time_limit_weeks: int
    # + other experiment parameters
```

```python
class Experiment:
    # Constructor: Experiment(config: ExperimentConfig, output_dir: Path)
    
    def register_source(utterance_id, audio_path) -> AudioInfo
    def register_gold(utterance_id, audio_path) -> AudioInfo
    def register_target(utterance_id, strategy, output_path, provenance) -> AudioInfo
    def get_provenance(utterance_id, strategy) -> Optional[ProvenanceRecord]
    def verify_provenance() -> list[str]  # Returns list of problems
    def get_summary() -> dict
```

```python
class ExperimentRunner:
    # Constructor: ExperimentRunner(output_dir)
    def run_gate(gate_name, experiment, config) -> dict
    def build_gate_artifact_table(experiment) -> Path
```

**Responsibilities:** Registers sources/golds/targets, verifies provenance chains, executes gates, builds the gate artifact table JSON.

**Dependencies:** annotations, audio_io, degradation, evaluation, provenance, target_generation, listening, reporting, stats.

---

### 3.2 annotations.py

**Purpose:** Data structures for per-token annotations, utterances, and alignment.

**Key Types:**

```python
class TokenLabel(IntEnum):
    ALREADY_TARGET = 0    # Realization is acceptably target-like
    DEVIANT = 1           # Differs meaningfully from target
    AMBIGUOUS = 2         # Annotator cannot confidently decide

LABEL_NAMES = {0: "ALREADY_TARGET", 1: "DEVIANT", 2: "AMBIGUOUS"}

@dataclass
class TokenAnnotation:
    word: str
    word_index: int
    phone: str
    target_dimension: str   # e.g. "T_ASPIRATION", "FLAPPING"
    label: TokenLabel
    notes: str = ""
    confidence: float = 1.0
    annotator_id: str = ""
    created_at: str = ""

@dataclass
class Alignment:
    phones: list[tuple[str, float, float]]   # (phone, start_ms, end_ms)
    words: list[dict]                         # word-level alignment info

@dataclass
class Utterance:
    utterance_id: str
    speaker_id: str
    transcript: str
    target_realization: str                   # US-neutral target transcription
    audio_path: str
    duration_seconds: float
    tokens: list[TokenAnnotation]
    alignment: Optional[Alignment]
    metadata: dict = field(default_factory=dict)
```

```python
class AnnotationDB:
    # Collection of utterances for the experiment
    
    def add(u: Utterance) -> None
    def get(utterance_id) -> Optional[Utterance]
    def by_speaker(speaker_id) -> list[Utterance]
    def to_dict() -> dict
    @classmethod
    def from_dict(cls, data) -> AnnotationDB
    
    @property
    def speaker_count(self) -> int
    @property
    def utterance_count(self) -> int
```

```python
class AnnotationVersion:
    """Immutable snapshot of an annotation state for history."""
```

**Responsibilities:** Defines the annotation schema, manages the utterance database, provides per-token labeling (ALREADY_TARGET/DEVIANT/AMBIGUOUS), handles import/export.

---

### 3.3 audio_io.py

**Purpose:** Load, save, and validate audio files. No real-time concerns — all offline.

**Key Types and Functions:**

```python
@dataclass
class AudioInfo:
    path: str
    duration_seconds: float
    sample_rate: int
    channels: int
    samples: int
    dtype: str
    file_hash: str
    format: str

def compute_file_hash(path: Path) -> str       # SHA-256, first 16 hex chars
def load_audio(path, expected_sr=None) -> tuple[np.ndarray, AudioInfo]
def save_audio(path, waveform, sample_rate) -> None
def validate_audio(waveform, sample_rate) -> list[str]  # Returns warnings list
def duration_seconds(waveform, sample_rate) -> float
```

**Responsibilities:** Audio file I/O, SHA-256 provenance hashing, amplitude/NaN/Inf validation.

---

### 3.4 degradation.py

**Purpose:** Apply realistic BPO-channel degradations to source audio for subtest conditions.

**Key Types:**

```python
@dataclass
class DegradationConfig:
    nb_sample_rate: int = 8000
    apply_mulaw: bool = True
    mulaw_bit_depth: int = 8
    babble_snr_db: Optional[float] = None
    babble_source: Optional[np.ndarray] = None
    apply_headset_eq: bool = False
    output_sample_rate: int = 22050

def apply_degradation(waveform: np.ndarray, config: DegradationConfig, 
                       input_sr: int) -> np.ndarray
```

**Presets:**

| Preset | Description |
|---|---|
| `clean` | No degradation, 22050 Hz output |
| `NB` | Narrowband: 8 kHz resample + G.711 μ-law encode/decode |
| `noisy` | Babble noise at 10 dB SNR, 22050 Hz |
| `NB+noisy` | Narrowband + babble noise |

**Responsibilities:** Simulates USB headset + telephony path. Used to test target generation on degraded audio matching real BPO deployment conditions.

---

### 3.5 evaluation.py

**Purpose:** Compute evaluation metrics for generated targets. Research probes only, not production metrics.

**Key Classes:**

```python
@dataclass
class EvaluationResult:
    utterance_id: str
    strategy: str
    critical_entities_correct: Optional[bool]
    word_error_rate: Optional[float]
    accent_shift_score: Optional[float]
    correction_rate: Optional[float]
    damage_rate: Optional[float]
    identity_score: Optional[float]
    timing_score: Optional[float]
    naturalness_score: Optional[float]
    metadata: dict = field(default_factory=dict)
```

```python
class AccentShiftEvaluator:
    """Probe-based accent shift scoring."""
    
    def __init__(self, probes_dir=None)
    def evaluate(self, audio_path, transcript, target_realization) -> AccentShiftResult
```

```python
class IdentityEvaluator:
    """Speaker-embedding identity preservation."""
    
    def compare(self, audio1_path, audio2_path) -> IdentityResult
    def compare_arrays(self, embedding1, embedding2) -> IdentityResult
```

```python
class TimingEvaluator:
    """Duration and timing distribution comparison."""
    
    def compare(self, source_path, target_path, alignment=None) -> TimingResult
```

```python
class ContentEvaluator:
    """Word Error Rate via WhisperX ASR."""
    
    def __init__(self, model_name="large-v3", device="cpu")
    def evaluate(self, audio_path, reference_transcript) -> ContentResult
    def evaluate_entities(self, reference, hypothesis) -> dict
```

**Responsibilities:** Computes WER, accent shift scores via probes, identity preservation via speaker embeddings, timing comparison, and critical entity preservation.

---

### 3.6 target_generation.py

**Purpose:** Three target generation strategies (A, B, C).

**Key Classes:**

```python
class TargetStrategy(ABC):
    name: str
    
    @abstractmethod
    def generate(self, source_audio, source_sr, transcript, target_realization,
                 strength=1.0, token_annotations=None, speaker_id="unknown",
                 utterance_id="", output_path=None) -> Tuple[np.ndarray, dict]
```

```python
class StrategyA(TargetStrategy):   # source-conditioned native synthesis
class StrategyB(TargetStrategy):   # native realization first, identity second
class StrategyC(TargetStrategy):   # sparse control-domain repair
```

```python
def get_strategy(name: str) -> TargetStrategy
```

**Detailed interfaces in Section 5 below.**

---

### 3.7 listening.py

**Purpose:** Gate 1B listening study framework — controlled listening experiment for adjudicating generated targets.

**Key Classes:**

```python
@dataclass
class Stimulus:
    """One trial audio with blinding code."""
    stimulus_id: str
    audio_path: str
    condition: str          # "source", "target_a", "target_b", "gold", etc.
    utterance_id: str
    blinding_code: str      # Opaque code like "Sample A7K2"
    duration_seconds: float
    speaker_id: str

class StimulusSet:
    """Collection of stimuli for a listening study."""
    
    def add(stimulus: Stimulus) -> None
    def get_by_code(code: str) -> Optional[Stimulus]
    def shuffle(self, seed=None) -> None
    def export(self, path) -> dict
    @classmethod
    def import_set(cls, path) -> StimulusSet

@dataclass
class Rater:
    rater_id: str
    native_language: str
    accent_training: bool
    screening_score: float
    excluded: bool
    exclusion_reason: str

class ListeningPanel:
    """Rater management and eligibility screening."""
    
    def add_rater(rater: Rater) -> None
    def get_eligible_raters() -> list[Rater]
    def mark_excluded(rater_id, reason) -> None

@dataclass
class ListeningTrial:
    """Individual rater response."""
    trial_id: str
    rater_id: str
    stimulus_id: str
    same_person_rating: int       # 1-5
    accent_shift_rating: int      # 1-5
    naturalness_rating: int       # 1-5
    content_preserved: str        # "yes" / "no" / "partial"
    response_time_seconds: float
    timestamp: str

class ListeningStudy:
    """Full study orchestration."""
    
    def __init__(self, config, stimulus_set, panel)
    def assign_trials(raters_per_stimulus=3) -> dict
    def record_trial(trial: ListeningTrial) -> None
    def compute_detailed_results() -> dict
    def export_results(self, path) -> dict
    def import_results(self, path) -> dict
```

**Responsibilities:** Blinded stimulus presentation, rater screening, trial recording, result computation, import/export.

---

### 3.8 realization_labels.py

**Purpose:** Per-token realization labeling workflow for Gate -1A.

**Key Classes:**

```python
@dataclass
class LabelingSession:
    """Manages labeling workflow for one utterance."""
    
    utterance_id: str
    annotator_id: str
    label_overrides: dict        # token_index -> TokenLabel
    started_at: str
    completed_at: Optional[str]
    
    def apply_label(token_index, label, notes="") -> None
    def finalize() -> None

@dataclass
class CorrectionDamageReport:
    correction_rate: float
    damage_rate: float
    corrected_count: int
    total_deviant: int
    damaged_count: int
    total_already_target: int
    details: dict

def compute_correction_damage(utterance, token_annotations, 
                               target_dimension=None) -> CorrectionDamageReport
```

**Formula (from spec):**
- Correction rate = deviant tokens moved toward target / all deviant tokens
- Damage rate = already-correct tokens made worse / all already-correct tokens

**Responsibilities:** Manages the labeling workflow, applies per-token ALREADY_TARGET/DEVIANT/AMBIGUOUS labels, computes correction and damage rates.

---

### 3.9 study_config.py

**Purpose:** Study configuration and pre-registration for Gate 1B.

**Key Types:**

```python
class RatingScale(IntEnum):
    SAME_PERSON = 1        # 1 = different person, 5 = same person
    ACCENT_SHIFT = 1       # 1 = still Indian, 5 = neutral US
    NATURALNESS = 1        # 1 = robotic, 5 = natural
    CONTENT_PRESERVED = 0  # categorical: yes/no/partial

@dataclass
class ExclusionRule:
    """Pre-registered exclusion criterion."""
    name: str
    condition: str
    action: str           # "exclude_rater", "exclude_trial", "flag"
    description: str

@dataclass
class StudyConfig:
    study_name: str
    rating_scales: dict
    stimuli_per_rater: int
    min_raters_per_stimulus: int
    inclusion_criteria: list[str]
    exclusion_rules: list[ExclusionRule]
    blinding_scheme: str
    randomization_seed: Optional[int]

    @classmethod
    def default_gate_1b(cls) -> StudyConfig

@dataclass
class PreRegistration:
    """Frozen pre-registration to prevent p-hacking."""
    prereg_id: str
    config: StudyConfig
    frozen: bool
    frozen_at: str
    created_at: str
    version: str
    procedures: str
    speaker_rules: str
    metrics_section: str
    listening_design: str
    exclusion_rules_section: list
    analysis_plan: str
    sample_size_justification: str
    stopping_rules: str
    other_notes: str
    
    def freeze(self) -> None
    def is_frozen(self) -> bool
```

**Responsibilities:** Encapsulates study design, pre-registration (frozen before data collection), exclusion rules, blinding scheme, stopping rules.

---

### 3.10 provenance.py

**Purpose:** Complete provenance tracking for every generated target.

**Key Types:**

```python
@dataclass
class ProvenanceRecord:
    experiment_id: str
    utterance_id: str
    speaker_id: str
    strategy: str           # "strategy_a", "strategy_b", "strategy_c"
    conversion_strength: float
    source_path: str
    source_hash: str
    output_path: str
    output_hash: str
    config: dict
    software_version: str
    created_at: str
    git_commit: Optional[str]
    notes: str = ""

def create_experiment_id(prefix="accentedge") -> str
def compute_audio_hash(path: Path) -> str
def verify_provenance_chain(record: ProvenanceRecord) -> list[str]

class ProvenanceChain:
    """Full lineage tracking across multiple steps."""
    
    def add_step(step_name, record) -> None
    def verify(self) -> list[str]
    def to_dict() -> dict
    @classmethod
    def from_dict(cls, data) -> ProvenanceChain

def provenance_diff(a: ProvenanceRecord, b: ProvenanceRecord) -> dict
```

**Responsibilities:** Records exactly how each WAV was created (source hash, output hash, config, git commit, timestamp). Supports chain verification and diff between two records.

---

### 3.11 transcription.py

**Purpose:** Audio transcription interface (WhisperX scaffold).

**Key Types:**

```python
@dataclass
class WordSegment:
    word: str
    start: float        # seconds
    end: float          # seconds
    confidence: Optional[float] = None

@dataclass
class TranscriptionResult:
    utterance_id: str
    text: str
    words: list[WordSegment]
    source: str         # "whisperx", "mock", etc.
    language: str = "en"

def transcribe_audio(audio_path, model_name="large-v3", device="cpu") -> TranscriptionResult
def export_transcript(result: TranscriptionResult, path: Path) -> None
def import_transcript(path: Path) -> TranscriptionResult
```

**Responsibilities:** Transcribes audio to word-level timestamps. The `transcribe_audio` function is a scaffold — it has a mock fallback when WhisperX is unavailable. Used by ContentEvaluator for WER computation.

---

### 3.12 alignment.py

**Purpose:** Forced alignment — phone-level and word-level timing.

**Key Types:**

```python
@dataclass
class ForcedAlignmentResult:
    utterance_id: str
    phones: list[tuple[str, float, float]]   # (phone, start_ms, end_ms)
    words: list[dict]                         # word-level info
    source: str                               # "whisperx", "mfa", "precomputed", "none"

@dataclass
class AlignmentCorrector:
    """Tracks original vs corrected alignments."""
    
    utterance_id: str
    original: ForcedAlignmentResult
    corrected: ForcedAlignmentResult
    corrections: list[dict]
    
    def apply_correction(word_index, new_start, new_end) -> None
    def diff(self) -> list[dict]
    def to_dict() -> dict
    @classmethod
    def from_dict(cls, data) -> AlignmentCorrector

def validate_alignment(result: ForcedAlignmentResult) -> list[str]
def align_audio(audio_path, transcript, method="auto") -> ForcedAlignmentResult
```

**Responsibilities:** Provides phone-level timing. Supports WhisperX, Montreal Forced Aligner (MFA), and pre-computed alignments. `AlignmentCorrector` tracks manual corrections for auditability.

---

### 3.13 identity_transfer.py

**Purpose:** Pluggable identity transfer methods for preserving speaker timbre during accent transformation.

**Key Types:**

```python
@dataclass
class TransferResult:
    waveform: np.ndarray
    sample_rate: int
    accent_leaked: bool
    leak_confidence: float
    modifications: list[str]
    source_features: Optional[np.ndarray]
    target_features: Optional[np.ndarray]

class IdentityTransfer(ABC):
    """Abstract interface for identity transfer methods."""
    
    @abstractmethod
    def transfer(self, source_audio, source_sr, target_audio, target_sr,
                 strength=1.0) -> TransferResult
    
    @abstractmethod
    def extract_speaker_embedding(self, audio, sr) -> np.ndarray

class SimpleVoiceConversionTransfer(IdentityTransfer):
    """Spectral envelope matching via librosa."""
    
    def transfer(self, source_audio, source_sr, target_audio, target_sr,
                 strength=1.0) -> TransferResult

class SpeakerEmbeddingTransfer(IdentityTransfer):
    """Speaker-embedding conditioning with optional Seed-VC backend."""
    
    def __init__(self, seed_vc_wrapper=None)
    def transfer(self, source_audio, source_sr, target_audio, target_sr,
                 strength=1.0) -> TransferResult
```

**Responsibilities:** Transfers speaker identity from source audio onto a target pronunciation (US-realized) audio while detecting whether accent information leaked through. Used internally by Strategy B and Strategy A.

---

### 3.14 identity.py

**Purpose:** Speaker identity encoders for evaluation — verifies that accent transformation preserves speaker identity.

**Key Types:**

```python
@dataclass
class SpeakerEncoderResult:
    encoder_names: list[str]
    distances: list[float]
    similarities: list[float]
    mean_distance: float
    mean_similarity: float
    all_passed: bool
    threshold: float

class SpeakerEncoder:
    """Unified interface to multiple speaker encoders."""
    
    def __init__(self, sample_rate=16000, encoders=None, threshold=0.5)
    def compare(self, audio1, audio2) -> SpeakerEncoderResult
    def encode(self, audio) -> np.ndarray

# Available encoders (lazy-loaded):
# - ECAPA-TDNN (speechbrain)
# - WavLM-based verifier (transformers)
# - Resemblyzer-style encoder (optional)

def compare_speaker_identity(audio1_path, audio2_path, sample_rate=16000,
                              encoders=None) -> SpeakerEncoderResult

def _cosine_distance(a, b) -> float
def _cosine_similarity(a, b) -> float
```

**Responsibilities:** Computes cosine distance/similarity across multiple speaker encoders. `SpeakerEncoderResult.all_passed` returns True if all encoder distances are below threshold (default 0.5). Used by `IdentityEvaluator` in evaluation.py.

---

### 3.15 reporting.py

**Purpose:** Gate artifacts, markdown reports, and decision memo generation.

**Key Types:**

```python
class GateOutcome:
    PENDING = "PENDING"
    PASS = "PASS"
    FAIL = "FAIL"
    FULL_S2S_PASS = "FULL-S2S PASS"
    SPARSE_REPAIR_PASS = "SPARSE-REPAIR PASS"
    TEACHER_FAIL_GOLD_PASS = "TEACHER FAIL / GOLD PASS"
    FUNDAMENTAL_FAIL = "FUNDAMENTAL FAIL"
    ALL = [FULL_S2S_PASS, SPARSE_REPAIR_PASS, TEACHER_FAIL_GOLD_PASS, FUNDAMENTAL_FAIL]

OUTCOME_DESCRIPTIONS = { ... }  # Human-readable descriptions for each outcome

@dataclass
class GateArtifact:
    """One gate's required artifact (spec section 56)."""
    gate_name: str
    outcome: str
    artifacts: dict
    elapsed_seconds: float
    notes: str = ""

class Phase0Report:
    """Accumulates all gate artifacts into the final report."""
    
    def add_gate_artifact(artifact: GateArtifact) -> None
    def generate_markdown() -> str
    def generate_decision_memo() -> str
    def export(self, output_dir) -> Path
```

**Responsibilities:** Collects gate results, generates markdown reports, creates the final decision memo, exports per-gate JSON artifacts and the full report.

---

### 3.16 stats.py

**Purpose:** Statistical analysis for Gate 1B listening study.

**Key Functions and Classes:**

```python
def compute_cohens_kappa(rater1_labels, rater2_labels, weights=None) -> float
def compute_icc(ratings, targets, raters) -> float
def compute_d_prime(hits, false_alarms, n_possible, n_foil) -> float
def bootstrap_ci(data, statistic, n_samples=10000, alpha=0.05) -> tuple[float, float]
def summarize_by_condition(trials, condition_field, score_field) -> dict
def compute_speaker_reference(trials, speaker_id, score_field) -> SpeakerReference

@dataclass
class SpeakerReference:
    speaker_id: str
    n_samples: int
    identity_mean: float
    identity_std: float
    timing_mean: float
    timing_std: float
    naturalness_mean: float
    naturalness_std: float
    raw_identity_scores: list
    raw_timing_scores: list
    raw_naturalness_scores: list
```

**Responsibilities:** Inter-rater reliability (Cohen's kappa, ICC), signal detection metrics (d-prime), bootstrap confidence intervals, per-condition summarization, per-speaker reference distributions. Uses scipy when available, falls back to pure numpy.

---

### 3.17 probes.py

**Purpose:** Accent pronunciation probes for Gate -1B measurement validity.

**Key Types:**

```python
class ProbeDimension(Enum):
    RHO   = "rhoticity"           # Presence/absence of rhotic /r/
    FLAP  = "flapping"            # Intervocalic /t/ → [ɾ]
    TH    = "th_alveolarization"  # /θ/ → [t]/[s]
    ASP   = "t_aspiration"        # /t/ aspiration strength
    RET   = "retroflexion"        # Retroflex vs dental
    VW    = "v_w_substitution"    # /v/ → [w] or vice versa
    RED   = "reduction"           # Schwa reduction patterns
    STR   = "lexical_stress"      # Word-level stress timing

@dataclass
class ProbeSet:
    """Collection of pronunciation probes."""
    probes: dict[str, AccentProbe]  # dimension -> probe

@dataclass
class ProbeResult:
    dimension: str
    audio_path: str
    is_target_like: bool
    confidence: float
    features: np.ndarray
    metadata: dict

class AccentProbe:
    """Single pronunciation dimension probe."""
    
    def __init__(self, dimension, reference_us=None, reference_in=None)
    def classify(self, audio_path) -> ProbeResult
    def classify_segment(self, audio, sr, start_ms, end_ms) -> ProbeResult

@dataclass
class ProbeValidationResult:
    dimension: str
    us_correct: int
    in_correct: int
    total_us: int
    total_in: int
    accuracy: float
    # ...detailed breakdowns...

def validate_probes(probes: ProbeSet, test_cases) -> ProbeValidationResult
def compute_probe_accuracy(probes, test_cases) -> dict
```

**Responsibilities:** Each probe targets a specific pronunciation dimension and classifies audio as target-like (US-neutral) or substitute-like (Indian-English) using self-supervised embeddings. Validated against known natural US tokens and known Indian-English tokens with >= 80% accuracy target.

---

## 4. Data Formats Used

### 4.1 Audio Format

| Property | Value |
|---|---|
| Primary format | WAV (PCM float32) |
| Primary sample rate | 22050 Hz |
| Degraded sample rate | 8000 Hz (narrowband) |
| Codec (NB mode) | G.711 μ-law 8-bit |
| Channels | Mono (1) |
| Bit depth | 32-bit float |
| Validation | SHA-256 hash (first 16 hex chars stored) |

**Validation checks** (from `audio_io.py:validate_audio`):
- dtype must be float32
- No NaN values
- No Inf values
- Peak amplitude <= 1.0
- Peak amplitude >= 0.01 (warns if too quiet)
- Non-empty waveform

### 4.2 Annotations Format

**In-memory (dataclasses):**
- `TokenAnnotation`: word, word_index, phone, target_dimension, label (TokenLabel enum), notes, confidence, annotator_id, timestamps
- `Utterance`: utterance_id, speaker_id, transcript, target_realization, audio_path, duration, tokens list, alignment, metadata
- `Alignment`: phones list of (phone, start_ms, end_ms), words list of dicts

**Serialized (JSON):**
```json
{
  "_meta": {
    "version": "1.0",
    "created_at": "2026-08-25T...",
    "speaker_count": 1,
    "utterance_count": 5
  },
  "speakers": ["speaker_001"],
  "utterances": [
    {
      "utterance_id": "utt_001",
      "speaker_id": "speaker_001",
      "transcript": "I can see a charge of thirty dollars...",
      "target_realization": "US-neutral transcript...",
      "audio_path": "data/raw/speaker_001/utt_001.wav",
      "duration_seconds": 4.52,
      "tokens": [...],
      "alignment": {...},
      "metadata": {}
    }
  ],
  "versions": [...]
}
```

### 4.3 Manifest / Experiment Config Format

**YAML/JSON** (`ExperimentConfig` serialized):
```yaml
description: "Phase 0 single-speaker feasibility"
speakers:
  - speaker_001
strategies:
  - strategy_a
  - strategy_b
  - strategy_c
output_sample_rate: 22050
time_limit_weeks: 10
```

### 4.4 Provenance Format

```json
{
  "experiment_id": "accentedge-20260825-abc123",
  "utterance_id": "utt_001",
  "speaker_id": "speaker_001",
  "strategy": "strategy_b",
  "conversion_strength": 0.8,
  "source_path": "data/raw/speaker_001/utt_001.wav",
  "source_hash": "a1b2c3d4e5f6...",
  "output_path": "results/strategy_b/utt_001.wav",
  "output_hash": "f6e5d4c3b2a1...",
  "config": {...},
  "software_version": "0.1.0-phase0",
  "created_at": "2026-08-25T10:30:00Z",
  "git_commit": "c8adcbd",
  "notes": "..."
}
```

### 4.5 Transcription Format

```json
{
  "utterance_id": "utt_001",
  "text": "I can see a charge of thirty dollars posted on the thirteenth of August.",
  "words": [
    {"word": "I", "start": 0.0, "end": 0.15, "confidence": 0.98},
    {"word": "can", "start": 0.16, "end": 0.35, "confidence": 0.95},
    ...
  ],
  "source": "whisperx",
  "language": "en"
}
```

### 4.6 Alignment Format

```json
{
  "utterance_id": "utt_001",
  "phones": [
    ["AH", 0, 120],
    ["K", 120, 180],
    ...
  ],
  "words": [
    {"word": "I", "start_ms": 0, "end_ms": 150, "phone_indices": [0, 1]},
    ...
  ],
  "source": "whisperx"
}
```

### 4.7 Gate Artifact Table

```json
{
  "experiment_id": "accentedge-20260825-abc123",
  "gates": [
    {
      "gate": "gate_minus_1a",
      "outcome": "PASS",
      "artifacts": {"annotation_db": "results/gate_minus_1a/annotations.json"},
      "elapsed_seconds": 45
    },
    ...
  ]
}
```

### 4.8 Listening Study Results

```json
{
  "study_id": "gate_1b_strategy_compare",
  "stimulus_set": {...},
  "panel": {...},
  "trials": [
    {
      "trial_id": "trial_001",
      "rater_id": "rater_1",
      "stimulus_id": "stim_007",
      "same_person_rating": 4,
      "accent_shift_rating": 3,
      "naturalness_rating": 4,
      "content_preserved": "yes",
      "response_time_seconds": 12.3,
      "timestamp": "2026-08-25T14:00:00Z"
    }
  ],
  "assignments": {...}
}
```

---

## 5. Target Generation Strategies A/B/C

All three strategies share the abstract interface:

```python
class TargetStrategy(ABC):
    def generate(
        self,
        source_audio: np.ndarray,
        source_sr: int,
        transcript: str,
        target_realization: str,
        strength: float = 1.0,
        token_annotations: Optional[list] = None,
        speaker_id: str = "unknown",
        utterance_id: str = "",
        output_path: Optional[str] = None,
    ) -> Tuple[np.ndarray, dict]:
```

All return `(waveform, metadata)` where metadata includes:
- `changed_regions`: list of modified time regions
- `accent_leaked`: bool
- `leak_confidence`: float
- `modifications`: list of applied modifications
- `strategy_success`: bool
- `error`: str (empty if success)
- `strength`: float
- `provenance`: ProvenanceRecord dict

Factory: `get_strategy(name) -> TargetStrategy` where name is `"strategy_a"`, `"strategy_b"`, or `"strategy_c"`.

### Strategy A — Source-Conditioned Native Synthesis

**Goal:** Synthesize a US-neutral realization of the transcript while conditioning on the source speaker's identity.

**Algorithm:**
1. Extract source speaker embedding (spectral fingerprint or Seed-VC embedding)
2. Extract F0 statistics (mean, std) from source
3. Synthesize US-neutral pronunciation from transcript using a native US TTS engine (scaffold — actual TTS backend not yet wired)
4. Apply source speaker conditioning: match spectral envelope and F0 stats
5. Blend between source (`strength=0`) and conditioned output (`strength=1`)
6. Detect accent leakage (whether source accent features persist)

**Use case:** Full whole-speech conversion. Most aggressive transformation.

**Current limitation:** The TTS backend is scaffolded — no actual synthesis engine is wired in yet. The `_synthesize_us_neutral` function returns a modified version of the source audio as a placeholder.

### Strategy B — Native Realization First, Identity Second (Step 0)

**Goal:** Produce a hand-built target where US pronunciation comes first, then speaker identity is overlaid.

**Algorithm:**
1. Compute US-realized version of source audio via spectral envelope shift (`_compute_envelope_shift_toward_us`)
2. Apply identity/timbre transfer from source onto US-realized audio:
   - Uses `SimpleVoiceConversionTransfer` (spectral envelope matching) or `SpeakerEmbeddingTransfer` (Seed-VC backend)
3. Blend between source (`strength=0`) and US+identity (`strength=1`)
4. Detect accent leakage
5. Generate provenance

**Key internal functions:**
- `_compute_envelope_shift_toward_us(source, sr, strength)`: spectral envelope manipulation to shift formants toward US-neutral
- `_compute_accent_leakage(source, output, sr)`: post-hoc check whether Indian accent features remain

**Use case:** Step 0 primary strategy. Hand-built quality target. Most carefully engineered.

### Strategy C — Sparse Control-Domain Repair

**Goal:** Only modify tokens marked DEVIANT, leaving ALREADY_TARGET and AMBIGUOUS regions untouched.

**Algorithm:**
1. Iterate over `token_annotations` (the Gate -1A labeling output)
2. For each DEVIANT token:
   a. Compute the target pronunciation correction (spectral envelope shift for that time region)
   b. Measure change magnitude (RMSE between source region and corrected region)
   c. If change_magnitude < 0.001: mark as "false fire" (token was already close enough)
   d. Otherwise: apply the correction to that time region in the output
3. For ALREADY_TARGET tokens: check for damage (RMSE > 0.02 = damaged)
4. For AMBIGUOUS tokens: skip entirely
5. Overall accent leakage check
6. Generate provenance

**Key metadata fields:**
- `deviant_count`: number of DEVIANT tokens processed
- `already_target_count`: number of ALREADY_TARGET tokens
- `ambiguous_count`: number of AMBIGUOUS tokens
- `false_fires`: list of tokens that were already close to target
- `total_corrected_tokens`: count of actually modified tokens
- `total_damaged_tokens`: count of ALREADY_TARGET tokens damaged

**Use case:** Conservative approach. Minimizes identity damage by only touching phonemes that need fixing. Preferred if Strategies A/B damage speaker identity.

### Strategy Comparison Matrix

| Property | Strategy A | Strategy B | Strategy C |
|---|---|---|---|
| Scope | Whole utterance | Whole utterance | Token-level repair |
| Identity preservation | Moderate (blend) | High (transfer) | Highest (minimal changes) |
| Accent shift | Strong | Strong | Targeted |
| Damage risk | Moderate | Low | Lowest |
| Identity leak risk | Moderate | Low | Low |
| Complexity | High | Medium | Medium |
| Current implementation status | Scaffolded TTS | Functional (spectral+VC) | Functional |
| Step 0 primary | No | **Yes** | Fallback |

---

## 6. How Identity Preservation Is Measured

### 6.1 Speaker Encoders (`identity.py`)

The primary identity metric uses **multiple speaker encoders** with cosine distance:

```python
class SpeakerEncoder:
    def __init__(self, sample_rate=16000, encoders=None, threshold=0.5)
    def compare(self, audio1, audio2) -> SpeakerEncoderResult
```

**Available encoders (lazy-loaded):**

| Encoder | Library | Notes |
|---|---|---|
| ECAPA-TDNN | speechbrain >= 1.0 | Primary speaker encoder |
| WavLM | transformers >= 4.30 | Speaker verification model |
| Resemblyzer | resemblyzer (optional) | Third encoder, may be absent |

**Scoring:**
- Cosine distance: `1.0 - cosine_similarity(embedding1, embedding2)`
- Lower distance = more similar = better identity preservation
- `SpeakerEncoderResult.all_passed`: True if **all** encoder distances < threshold (default 0.5)
- `SpeakerEncoderResult.mean_distance`: average across all encoders

**Usage in evaluation pipeline:**
```python
# In evaluation.py IdentityEvaluator:
def compare(self, audio1_path, audio2_path) -> IdentityResult:
    result = compare_speaker_identity(audio1_path, audio2_path)
    return IdentityResult(
        distance=result.mean_distance,
        similarity=result.mean_similarity,
        passed=result.all_passed,
        encoder_results={...}
    )
```

The `EvaluationResult.identity_score` field stores this distance (lower = better).

### 6.2 Identity Evaluator (`evaluation.py`)

```python
class IdentityEvaluator:
    def compare(self, audio1_path, audio2_path) -> IdentityResult
    
    def compare_to_speaker_references(self, audio_path, reference_paths) -> SpeakerReferenceResult
```

`compare_to_speaker_references` compares a target against multiple reference recordings from the same speaker to establish a within-speaker baseline.

### 6.3 Accent Leakage Detection (`target_generation.py`)

After every strategy generates output, `_compute_accent_leakage(source, output, sr)` checks whether source accent features persist in the output:

```python
def _compute_accent_leakage(source_audio, output_audio, sr) -> Tuple[bool, float, dict]
```

Returns `(leaked: bool, confidence: float, features: dict)`. This is a secondary identity check that runs inside every strategy's `generate()` method.

### 6.4 Listening Panel Identity Ratings (`listening.py`)

In the Gate 1B listening study, raters score "Same person?" on a 1–5 scale:
- 1 = different person
- 5 = same person

Pass criterion: average >= 4/5 across >= 3 native US English raters.

This is the **human-validated** identity preservation metric, distinct from the automated encoder metrics.

### 6.5 Strategy C Damage Rate (`realization_labels.py`)

For Strategy C, identity preservation is measured via **damage rate**:

```
Damage rate = already-correct tokens made worse / all already-correct tokens
```

A token is "damaged" if the RMSE between source and output in that token's time region exceeds 0.02. This ensures the sparse repair doesn't accidentally alter tokens that were already correctly pronounced.

### 6.6 Timing Preservation (`evaluation.py` TimingEvaluator)

Identity also includes timing characteristics:

```python
class TimingEvaluator:
    def compare(self, source_path, target_path, alignment=None) -> TimingResult
```

Measures whether word and phone duration distributions are preserved within natural bounds.

---

## 7. Key Observations & Risks

### 7.1 What the Forensic Audit Found (FORENSIC_AUDIT_2026-08-25.md)

| Finding | Severity | Status |
|---|---|---|
| 5 successful full-utterance offline conversions | Positive | Verified (runs 028, 030, 038, 039, 040) |
| Never demonstrated streaming | Risk | Confirmed gap |
| Never used MPS despite availability | Risk | CPU-only execution |
| Never validated accent shift vs. speaker copying | Risk | Core thesis unvalidated |
| Seed-VC is zero-shot VC, NOT accent converter | Risk | May copy reference voice instead of transforming accent |
| "P0 complete" was never accurate | Process | Infrastructure exists, runtime validation absent |

**Estimated timeline to call-centre-quality prototype (from audit):**
- 1-2 weeks: accent validation (human listening test)
- 2-4 weeks: chunked inference
- 4-8 weeks: streaming pipeline wiring
- 2-4 weeks: latency optimization
- 2-4 weeks: stability testing
- **Total: 3-4 months minimum for basic prototype, 6-12 months for production**

### 7.2 Codebase Architecture Strengths

1. **Clean separation of concerns** — 17 focused modules with clear interfaces
2. **Provenance-first design** — every generated target carries full lineage
3. **Pre-registration system** — prevents p-hacking in listening studies
4. **Blinded stimulus presentation** — prevents response bias
5. **Multiple fallback paths** — scipy optional, librosa optional, WhisperX scaffolded
6. **Correction/damage rate formula** — provides quantitative measure of strategy safety

### 7.3 Implementation Gaps

1. **Strategy A TTS backend is scaffolded** — `_synthesize_us_neutral` returns modified source audio, not actual synthesis
2. **Probes depend on speechbrain/transformers** — gate -1B cannot run without these heavy dependencies
3. **Strategy B identity transfer is spectral-level** — not a learned voice conversion; may not preserve fine-grained timbre
4. **No actual Seed-VC integration in the phase0 package** — Seed-VC exists in separate `seed-vc/` modules, not wired into the strategy interface
5. **Listening study is framework-only** — no actual human data collected yet
6. **Gate runners are stubs** — `ExperimentRunner.run_gate` returns placeholder dicts, not actual gate logic

### 7.4 Risk Summary

| Risk | Mitigation in Codebase |
|---|---|
| Seed-VC copies reference voice, not accent | Strategy B uses spectral envelope shift first, then transfers identity — order reduces this risk |
| Sparse repair over-corrects | Damage rate metric + 0.02 RMSE threshold |
| Probes don't actually measure accent | Gate -1B validation test with known US/IN tokens |
| Listening study bias | Blinding codes + pre-registration + exclusion rules |
| Accent leakage undetected | `_compute_accent_leakage` post-hoc check in every strategy |
| No streaming validation | Out of Phase 0 scope — deliberately deferred |

---

## Summary

Phase 0 is a **feasibility protocol, not a model**. It answers a single question: can we define what the ideal output should sound like, and can we produce it? The codebase implements:

- **17 modules** covering experiment control, annotation, audio I/O, degradation, evaluation, three generation strategies, listening studies, labeling, study configuration, provenance, transcription, alignment, identity transfer, speaker encoders, reporting, statistics, and pronunciation probes
- **3 strategies** (A: source-conditioned synthesis, B: hand-built target, C: sparse repair) with a shared abstract interface and factory
- **7 gates** (-1A through 1B) with explicit pass/fail criteria
- **4 decision outcomes** (FULL-S2S PASS, SPARSE-REPAIR PASS, TEACHER FAIL/GOLD PASS, FUNDAMENTAL FAIL)
- **Multiple identity metrics**: speaker encoder cosine distance, accent leakage detection, listening panel ratings, and Strategy C damage rate

The forensic audit confirms the infrastructure is solid but the core thesis (accent shift with identity preservation) remains unvalidated by human listening tests. The codebase is well-structured to support that validation once the experimental protocol is executed.
