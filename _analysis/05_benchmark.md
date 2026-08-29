# AccentEdge Benchmark / Evaluation Instrument — Deep Analysis

## 1. BenchmarkRunner Flow

### 1.1 Entry Points

The benchmark has two parallel orchestration layers:

- **`BenchmarkRunner`** (`runner/benchmark.py`) — single-item runner used for full offline evaluation against a pre-computed manifest.
- **`Phase1BenchmarkAdapter`** (`adapter.py`) — streaming-oriented runner that processes items chunk-by-chunk through a `StreamingCandidate` and collects latency/RTF alongside quality metrics.

Both paths consume a list of `DatasetItem` objects, but they differ in *when* candidate inference happens:

| Aspect | `BenchmarkRunner` | `Phase1BenchmarkAdapter` |
|---|---|---|
| Candidate interface | `CandidateAdapter.process(audio, sr, ctx)` | `StreamingCandidate.create_session()/process_chunk()` |
| Input | Full utterance audio (loaded from `canonical_path`) | Chunked audio via `_chunk_audio()` |
| Metrics collected | Pass/fail per item + raw output | Per-utterance: content_error_rate, identity_preservation, timing_error_ms, latency_ms, rtf |
| Sweep support | No | Yes (via `ChunkSweep`, `LookaheadSweep`, `CombinedSweep`) |

### 1.2 Data Loading

1. **Manifest loading** (`dataset/manifest.py`): reads a Parquet file via `pd.read_parquet()` and converts each row to a `DatasetItem` Pydantic model. Fields include `utterance_id`, `speaker_id`, `partition`, `family`, `transcript_verbatim`, `transcript_normalized`, `audio_sha256`, `canonical_path`, etc.

2. **Audio loading** (`audio/io.py`): `load_audio()` uses `soundfile` to read WAV, applies optional mono conversion and resampling via `librosa`. Returns `(waveform_float32, sr)`.

3. **Validation** (`dataset/validator.py`): validates dataset integrity:
   - Duplicate utterance IDs
   - Speaker-partition leakage (no speaker in >1 partition)
   - Duplicate audio hashes across items
   - File existence and SHA-256 match

### 1.3 Candidate Execution

In `BenchmarkRunner.run()`:
1. Filters manifest by partition (`self.split`, default `"dev"`).
2. For each item:
   - Loads audio from `item.canonical_path`
   - Builds `BenchmarkContext(target_accent, conversion_strength, utterance_id, speaker_id)`
   - Calls `self.candidate.process(audio, sr, ctx)`
   - On success: records `"utterance_id", "status": "ok"` and appends `CandidateOutput`
   - On exception: classifies error via `failures.py` (`ErrorCategory`), records `FailureRecord`, appends `"failed"` result.

Returns a summary dict with `total_items`, `succeeded`, `failed`, and per-item results.

### 1.4 Run Manifest & Lineage

`runner/run_manifest.py` creates a `RunManifest` record capturing:
- `run_id`: `<timestamp>_<candidate_name>`
- `benchmark_version`: `"1.0.0"`
- `dataset_hash`: SHA-256 of the manifest file
- `candidate_hash`: SHA-256 of candidate artifact
- `config_hash`: SHA-256 of benchmark config file
- `git_commit`: short SHA from `git rev-parse HEAD`
- `python_version`
- `timestamp`, `split`, `condition`, `conversion_strength`

### 1.5 Resume / Restart

`runner/resume.py` supports restart from failure:
- Reads `completed_items.jsonl` from the run directory
- Skipping items already completed
- `save_completed_item()` appends each completed item + metadata to the JSONL state file

### 1.6 Failure Collection

`runner/failures.py` provides:
- `ErrorCategory` enum: `INPUT_AUDIO_ERROR`, `CANDIDATE_ERROR`, `INVALID_OUTPUT`, `ALIGNMENT_ERROR`, `ASR_ERROR`, `SPEAKER_EVAL_ERROR`, `PRONUNCIATION_PROBE_ERROR`, `METRIC_ERROR`
- `FailureRecord` dataclass (utterance_id, candidate_name, error_category, error_message, stack_trace, timestamp, run_id, metadata)
- `classify_error()` heuristic based on exception message + context string
- `FailureCollector` with `to_jsonl()` and `by_category()` aggregation

---

## 2. CandidateAdapter Protocol & Built-in Candidates

### 2.1 Protocol

```python
class CandidateAdapter(ABC):
    @property
    @abstractmethod
    def metadata(self) -> CandidateMetadata: ...

    @abstractmethod
    def process(self, audio: np.ndarray, sample_rate: int,
                context: BenchmarkContext) -> CandidateOutput: ...

    def prepare(self) -> None: ...   # optional one-time setup
    def close(self) -> None: ...    # optional cleanup
```

Supporting types:
- **`CandidateMetadata`**: `name`, `version`, `description`, `target_accent`, `supports_conversion_strength`, `artifact_hash`, `configuration`
- **`BenchmarkContext`**: `target_accent`, `conversion_strength`, `utterance_id`, `speaker_id`, `metadata`
- **`CandidateOutput`**: `audio` (np.ndarray), `sample_rate`, `metadata`

### 2.2 Built-in Candidates

#### PassthroughAdapter (`candidates/passthrough.py`)

- **Name**: `"passthrough"`
- **Behavior**: Returns `audio.copy()` — no transformation. Used as a null baseline.
- **Metadata**: `target_accent = "source"`, `supports_conversion_strength = False`

#### FileOutputAdapter (`candidates/file_output.py`)

- **Name**: `"file_output"`
- **Behavior**: Loads a pre-generated WAV from `<output_dir>/<utterance_id>.wav` using `load_audio()`.
- **Requires**: `utterance_id` in context; raises `FileNotFoundError` if missing.
- **Use case**: Comparing a previously-run model without re-running inference.

### 2.3 Registry

`candidates/registry.py` provides:
- `_REGISTRY`: dict mapping `"passthrough"` → `PassthroughAdapter`, `"file_output"` → `FileOutputAdapter`
- `register(name, cls)` / `get(name)` / `available()` for extensibility

---

## 3. Evaluation Dimensions

The system defines eight evaluation dimensions:

| Dimension | Evaluator | Key Metrics / Output | Config key |
|---|---|---|---|
| **Acoustic** | `ArtifactEvaluator` | `snr_db`, `clipping_ratio`, `artifact_flags` (nan_inf, clipping, silence_insertion, duration_anomaly, dc_offset, rms_collapse, rms_explosion, spectral_discontinuity) | `artifacts.checks`, `artifacts.duration_bounds_ms` |
| **Content** | `ContentEvaluator` | `wer`, `cer`, `word_count`, `errors` via `jiwer` | `content.asr_backend`, `content.compute_wer/cer` |
| **Identity** | `IdentityEvaluator` | `source_output_distance`, `within_range` (default threshold 0.5), optional `same_session_distance`, `different_session_distance`, `cross_accent_distance` | `identity.evaluators` (wavlm, ecapa), `identity.calibration_path` |
| **Pronunciation** | `PronunciationEvaluator` | `correction_rate`, `damage_rate` per feature; counts: `corrected`, `eligible_correction`, `damaged`, `eligible_damage`, `off_target`, `ambiguous` | `pronunciation.probe_registry`, `pronunciation.calibration_path` |
| **Prosody** | `ProsodyEvaluator` | `f0_mean`, `f0_range_hz`, `energy_mean`, `speech_rate` (via `librosa.piptrack` for f0, RMS for energy) | `prosody.f0_method`, `prosody.energy_method` |
| **Timing** | `TimingEvaluator` | `duration_ratio`, `duration_delta_ms`, `within_bounds` (default ±50 ms tolerance) | `timing.duration_tolerance_ms` |
| **Robustness** | `RobustnessEvaluator` | Sliced `content_wer`, `content_cer`, `identity_distance`, `timing_ratio`, `artifact_errors` per degradation condition | Degradation conditions: `clean`, `nb`, `noisy`, `nb_noisy` |
| **Naturalness** | `NaturalnessEvaluator` | `predicted_mos` (×5 scale from `microsoft/distill-mos`), `human_mos`, `evaluator_name`, `is_auto` | `naturalness.auto_evaluator`, `naturalness.human_mos_required` |

Additionally, **entity-level accuracy** is handled by `EntityEvaluator` (`evaluation/entities.py`), which compares critical entities (NUMBER, MONEY, DATE, TIME, PERSON_NAME, ADDRESS, etc.) in the recognized transcript against reference annotations using type-specific normalization.

---

## 4. Statistical Analysis

### 4.1 Speaker-Level Aggregation (`statistics/aggregation.py`)

- **`SpeakerMetric`**: `speaker_id`, `metric_name`, `value`, `partition`
- **`aggregate_by_speaker()`**: computes `point_estimate` (mean), `std`, `speaker_count`, `item_count` for a given metric name, optionally sliced.
- **`aggregate_all()`**: applies the above for every unique metric name.

### 4.2 Bootstrap Confidence Intervals (`statistics/bootstrap.py`)

- **`speaker_bootstrap(speaker_metrics, metric_fn, n_replicates=10000, confidence_level=0.95, seed=42)`**
  - Resamples **speakers** (with replacement), including all their utterances.
  - Computes the bootstrap distribution of `metric_fn` over the sample.
  - Returns `BootstrapResult`: `point_estimate`, `ci_lower`, `ci_upper`, `replicates`, `n_speakers`, `n_replicates`.

### 4.3 Paired Comparison (`statistics/paired.py`)

- **`paired_bootstrap(metrics_a, metrics_b, delta_fn, ...)`**
  - `metrics_a/b`: dicts `{speaker_id: metric_value}`
  - Computes deltas on **common speakers only** (intersection).
  - Resamples speaker pairs with replacement, applies `delta_fn` to each replicate.
  - Significance test: **CI does not include zero** (no p-value calculation; two-sided CI-based test).
  - Returns `PairedResult`: `delta_mean`, `delta_ci_lower`, `delta_ci_upper`, `p_significant`, `n_replicates`, `confidence_level`.

### 4.4 Pareto Frontier (`comparison.py`)

- `SweepResult` objects (from `sweeps.py`) carry `candidate_id`, `chunk_size_ms`, `lookahead_ms`, `metrics` (content, identity, state_size_bytes), `latency_ms`, `rtf`.
- `compute_pareto_frontier()` classifies results as non-dominated vs. dominated.
- `_dominates()`: A dominates B if A ≥ B on all quality axes (`content`, `identity`) AND A ≤ B on all cost axes (`latency_ms`, `rtf`, `state_size_bytes`) with strict improvement on at least one cost axis.
- `generate_frontier_report()` produces per-candidate best/worst stats plus the Pareto table.

---

## 5. Reporting System

### 5.1 JSON Report (`reporting/json_report.py`)

- `generate_json_report(results, output_path, run_manifest)`
- Writes a structured JSON with `benchmark_version`, `run_manifest`, and `summary`.

### 5.2 HTML Report (`reporting/html.py`)

- `generate_html_report(summary, slices, failures, run_manifest, output_path)`
- Uses a **Jinja2** template (`HTML_TEMPLATE`) with:
  - Summary table: metric, value, CI lower, CI upper
  - Robustness slices table: slice, WER, CER, damage
  - Failures table: utterance, category, error
  - Pass/fail styling (`.pass` / `.fail` CSS classes)
  - Run metadata (run ID, candidate, split, condition, timestamp)

### 5.3 Markdown Tables (`reporting/tables.py`)

- `summary_table(results)`: markdown table with Metric | Value | CI Lower | CI Upper | Count
- `robustness_table(slice_metrics)`: markdown table with Slice | WER | CER | Identity | Damage

---

## 6. Dataset Management

### 6.1 Manifest Format

`DatasetItem` (Pydantic model) in `schemas/__init__.py`:

```python
class DatasetItem(BaseModel):
    utterance_id: str         # pattern: ^[a-z0-9_]+$
    speaker_id: str
    partition: Partition      # DEV | LOCKED_TEST | CALIBRATION
    family: Family            # BPO_SCRIPTED, CRITICAL_ENTITY, PRONUNCIATION_CONTRAST,
                              # ALREADY_TARGET, BPO_SPONTANEOUS, GENERAL_SPONTANEOUS
    prompt_id: str | None
    raw_path: str | None
    canonical_path: str
    sample_rate: int          # 8000–48000
    duration_ms: float
    accent_strength: float    # 0.0–1.0
    bpo_experience: bool
    transcript_verbatim: str
    transcript_normalized: str
    audio_sha256: str
    annotation_version: str = "1.0.0"
    has_critical_entities: bool
    has_target_features: bool
    l1_category: str
    metadata: dict
```

Auto-flags via `@model_validator`: if `family == CRITICAL_ENTITY` → `has_critical_entities = True`; if `family == PRONUNCIATION_CONTRAST` → `has_target_features = True`.

### 6.2 Splits (`dataset/splits.py`)

- **`build_splits(items, speaker_metadata, dev_count=24, locked_test_count=24, seed=42, ...)`**
  1. Optionally excludes speakers already in `CALIBRATION` partition.
  2. Stratifies available speakers by configurable factors (`l1_category`, `accent_strength`, `bpo_experience`) using deterministic binning and shuffling.
  3. Assigns speakers to `dev` / `locked_test` proportionally from each stratification bucket.
  4. Fills shortfall from unassigned speakers randomly.
  5. **Hard constraint**: `assert len(set(dev) & set(test)) == 0`
- **`validate_splits()`**: checks for speaker overlap.

`configs/splits.yaml` enforces:
- `speaker_disjoint: true` (hard requirement)
- `no_duplicate_audio: true` (same hash not in multiple splits)
- `deterministic: true` (same seed → same output)
- `seed: 42`

### 6.3 Dataset Registry (`dataset/registry.py`)

- `DatasetRegistry` loads datasets by name from `data/manifests/<name>.parquet`
- Caches in `_datasets` dict
- `validate(name)` runs full dataset validation

### 6.4 Lineage

- `RunManifest` records `dataset_hash` (SHA-256 of manifest file) and `candidate_hash`
- `BenchmarkSettings` (from `config.py`) provides canonical paths and parameters
- `run_manifest.run_id` embeds UTC timestamp + candidate name

---

## 7. Degradation System for Robustness Testing

### 7.1 Conditions (`degradations/__init__.py` + `configs/degradations.yaml`)

Four conditions:

| Condition | Description | Implementation |
|---|---|---|
| `clean` | Canonical 16 kHz mono PCM, no degradation | Returns audio unchanged |
| `nb` | Narrowband telephony simulation | Resample to 8 kHz, then back to 16 kHz; optional μ-law |
| `noisy` | Controlled BPO-like babble/noise | Add zero-mean Gaussian noise at 15 dB SNR; noise types: babble, office, street; clip to ±1.0 |
| `nb_noisy` | Combined telephony + noise | Resample (as `nb`) then add noise (as `noisy`) |

### 7.2 Determinism

- `apply_degradation(audio, sr, condition, seed=42, ...)` uses `np.random.RandomState(seed)` for reproducible noise.
- `_resample()` tries `librosa.resample` first; falls back to linear interpolation.
- All parameters (`nb_target_sr`, `noisy_snr_db`) are configurable.

### 7.3 Robustness Evaluation

`RobustnessEvaluator` (`evaluation/robustness.py`):
- Collects per-item results into condition-keyed buckets.
- `report()` computes average WER per condition from `ContentResult.wer`.
- `RobustnessSlice` dataclass captures condition, WER, CER, identity_distance, timing_ratio, artifact_errors, n_items.

---

## 8. Identity Calibration

### 8.1 Architecture

`identity/base.py`:
- **`SpeakerEmbedder`** (ABC): `embed(audio, sr) → np.ndarray`, `distance(a, b) → float`, `name` property.
- **`IdentityEvaluator`**: takes optional `SpeakerEmbedder`; computes `source_output_distance`. Default threshold `within_range = dist < 0.5`.

### 8.2 Calibration Loading

`identity/calibration.py`:
- `load_calibration(path)` reads a Parquet file produced in Phase 0.
- Computes mean distances for three conditions:
  - `same_session_mean` (default: 0.92)
  - `different_session_mean` (default: 0.72)
  - `cross_accent_mean` (default: 0.55)
- These calibration means are presumably used to set per-evaluator thresholds dynamically instead of the hard-coded 0.5.

### 8.3 Probe Calibration

`probes/calibration.py`:
- `ProbeCalibration` dataclass stores: `probe_name`, `reference_corpus_version`, `embedding_model`, `layer`, `centroid_version`, `feature_definition`, `noise_floor`, `known_limitations`.
- `to_dict()` serializes for reporting.

### 8.4 Identity Registry

`identity/registry.py`:
- `IdentityRegistry` holds named `SpeakerEmbedder` implementations.
- Two configured in `evaluators.yaml`: `wavlm` (`microsoft/wavlm-base-plus-sv`) and `ecapa` (`speechbrain/spkrec-ecapa-voxceleb`), both lazy-loaded.

### 8.5 Calibration Flow Summary

1. Phase 0 generates `data/calibration/identity.parquet` with known-speaker distances across conditions.
2. `load_calibration()` ingests this to derive baseline means.
3. During benchmark runs, `IdentityEvaluator` compares source vs. output embeddings.
4. The distance is evaluated against calibration-derived thresholds (or the fallback 0.5) to determine `within_range`.

---

## Summary of Key Files

| Component | Key File(s) |
|---|---|
| Runner | `runner/benchmark.py`, `runner/run_manifest.py`, `runner/failures.py`, `runner/resume.py` |
| Adapter | `adapter.py` (Phase1BenchmarkAdapter), `candidates/base.py`, `candidates/passthrough.py`, `candidates/file_output.py` |
| Schemas | `schemas/__init__.py` |
| Evaluations | `evaluation/content.py`, `evaluation/identity.py`, `evaluation/pronunciation.py`, `evaluation/prosody.py`, `evaluation/timing.py`, `evaluation/naturalness.py`, `evaluation/robustness.py`, `evaluation/entities.py`, `evaluation/artifacts.py` |
| Statistics | `statistics/paired.py`, `statistics/bootstrap.py`, `statistics/aggregation.py` |
| Reporting | `reporting/html.py`, `reporting/json_report.py`, `reporting/tables.py` |
| Dataset | `dataset/manifest.py`, `dataset/splits.py`, `dataset/registry.py`, `dataset/validator.py` |
| Degradations | `degradations/canonicalize.py`, `degradations/__init__.py` |
| Identity | `identity/base.py`, `identity/calibration.py`, `identity/registry.py`, `probes/calibration.py` |
| Config | `configs/benchmark.yaml`, `configs/evaluators.yaml`, `configs/splits.yaml`, `configs/degradations.yaml` |
| Config loading | `config.py` |
