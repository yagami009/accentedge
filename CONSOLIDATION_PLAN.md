# AccentEdge Monorepo Consolidation Plan

**Date:** 2026-08-27
**Source projects:** accent-voice-api, accentedge, accentedge-benchmark, accentedge-model-lab
**CosyAccent:** Stays separate (different paper, different authors)
**Target:** Single repo at ~/accentedge/
**Status:** Draft -- awaiting approval

---

## Target Layout

```
accentedge/
├── pyproject.toml                    # Single dependency lock (union of all)
├── README.md                        # Merged from all READMEs
├── docs/                            # All .md docs merged here
│   ├── phase0/TGFP_V2.md
│   ├── phase0/PHASE_0_SPEC.md
│   ├── phase0/FORENSIC_AUDIT_2026-08-25.md
│   ├── phase1/IMPLEMENTATION.md
│   ├── phase1/STATUS.md
│   ├── phase1/ARCHITECTURE_DECISIONS.md
│   ├── phase1/UPSTREAM_AUDIT.md
│   ├── phase1/PAPER_VERIFICATION.md
│   ├── phase1/PHONE_VOCAB_RECONCILIATION.md
│   ├── phase1/TRANSCRIPT_DEPENDENCY.md
│   ├── phase1/ZC2_CONTRACT.md
│   └── phase1/CONSISTENCY_AUDIT_REPORT.md
│
├── FAcodec/                         # Existing ~/FAcodec/ -- git submodule or symlink
├── Amphion/                         # Git submodule (Amphion fork for Colab)
│
├── src/accentedge/
│   ├── __init__.py
│   │
│   ├── phase0/                      # From accent-voice-api + accentedge_phase0
│   │   ├── __init__.py
│   │   ├── experiment.py            # Experiment controller
│   │   ├── annotations.py           # Phone-level annotation DB
│   │   ├── audio_io.py              # Audio I/O (replaces duplicate in src/audio/)
│   │   ├── degradation.py            # Audio degradation for conditions
│   │   ├── evaluation.py            # Gate evaluation logic
│   │   ├── target_generation.py     # Strategy A/B/C target generation
│   │   ├── listening.py             # Listener study framework
│   │   ├── realization_labels.py    # DEVIANT/ALREADY-TARGET/AMBIGUOUS
│   │   ├── study_config.py          # Study configuration
│   │   └── provenance.py            # Experiment provenance tracking
│   │
│   ├── codec/                       # From accentedge/src/accentedge/codec/
│   │   ├── __init__.py
│   │   ├── interfaces.py            # FactorizedLatents, FactorizedSpeechCodec
│   │   ├── facodec.py               # FACodecAdapter (wraps ~/FAcodec/)
│   │   └── swap.py                  # Content-codebook swap logic (moved from codec/)
│   │
│   ├── phase1/                      # From accentedge/src/accentedge/phase1/
│   │   ├── __init__.py
│   │   ├── converter.py             # FAC-FACodec converter (top-level orchestrator)
│   │   ├── denoiser.py              # Paper-faithful DDPM denoiser
│   │   ├── diffusion.py             # DDPM/DDIM schedule, q_sample, p_sample
│   │   ├── strength.py              # Strength control (0..1 mapping)
│   │   ├── phoneme_pipeline.py      # Audio → phone IDs, forced alignment
│   │   ├── zc2_recompute.py         # Content-codebook recomputation
│   │   └── target_accent.py         # Target-accent phone prior (from Phase 0)
│   │
│   ├── evaluation/                  # MERGED from all three projects
│   │   ├── __init__.py
│   │   ├── acoustic.py              # From accentedge (mel L1, STOI, PESQ)
│   │   ├── content.py               # From benchmark (WER, CER, ASR backend)
│   │   ├── identity.py              # From benchmark (ECAPA-TDNN, speaker distance)
│   │   ├── phonemes.py              # From accentedge (phone error rate)
│   │   ├── prosody.py               # From benchmark (duration, energy, pitch)
│   │   ├── degradation.py           # From benchmark (artifact scoring)
│   │   ├── metrics.py               # From accent-voice-api (STOI, PESQ, MCD)
│   │   ├── system.py                # From accent-voice-api (GPU, RAM, RTF)
│   │   └── latency.py               # From accent-voice-api (per-stage timing)
│   │
│   ├── training/                    # From accentedge/src/accentedge/training/
│   │   ├── __init__.py
│   │   ├── dataset.py               # Training dataset loader
│   │   ├── overfit.py               # Tiny overfit smoke test
│   │   ├── checkpoint.py            # Checkpoint save/load
│   │   └── loss.py                  # Training losses (from model-lab training/)
│   │
│   ├── models/                      # From accentedge-model-lab/src/accentedge_lab/models/
│   │   ├── __init__.py
│   │   ├── interfaces.py            # StreamingCandidate, CandidateMetadata
│   │   ├── registry.py              # ModelRegistry (register/get/create)
│   │   ├── candidate_a.py           # Full-forward encoder-decoder
│   │   ├── candidate_b.py           # Cascaded encoder + lightweight decoder
│   │   ├── candidate_c.py           # Token translation (LSTM + FiLM)
│   │   ├── candidate_d.py           # Minimal hybrid (SELECTED)
│   │   ├── sparse_repair.py         # Sparse repair scaffold
│   │   ├── articulatory_ddsp.py     # Articulatory DDSP (backup)
│   │   └── adapters.py              # Candidate adapters (from benchmark)
│   │
│   ├── streaming/                   # From accentedge-model-lab/src/accentedge_lab/streaming/
│   │   ├── __init__.py
│   │   ├── chunker.py               # Chunk-based streaming
│   │   ├── session.py               # StreamingSession state management
│   │   ├── simulator.py             # Virtual-time streaming simulator
│   │   ├── continuity.py            # Cross-chunk continuity
│   │   ├── timeline.py              # Processing timeline tracker
│   │   ├── state_growth.py          # State growth analysis
│   │   └── latency.py               # Latency measurement (merged)
│   │
│   ├── benchmark/                   # From accentedge-benchmark/src/accentedge_benchmark/
│   │   ├── __init__.py
│   │   ├── runner/benchmark.py      # BenchmarkRunner
│   │   ├── runner/run_manifest.py   # Run manifest generation
│   │   ├── runner/failures.py       # Failure tracking
│   │   ├── runner/reporting.py      # Report generation
│   │   ├── candidates/base.py       # CandidateAdapter base
│   │   ├── candidates/registry.py   # Candidate registry
│   │   ├── candidates/loaders.py    # Candidate loader
│   │   ├── candidates/comparison.py # Cross-candidate comparison
│   │   ├── candidates/sweeps.py     # Parameter sweeps
│   │   ├── candidates/integration.py # Integration helpers
│   │   ├── schemas.py               # Pydantic schemas (DatasetItem, RunManifest)
│   │   ├── config/loader.py         # Config loader
│   │   ├── config/schema.py         # Config schema
│   │   ├── data/lineage.py          # Data lineage tracking
│   │   ├── data/schema.py           # Data schema
│   │   ├── data/validation.py       # Data validation
│   │   ├── audio/hashing.py         # Audio hashing
│   │   ├── audio/io.py              # Audio I/O
│   │   ├── audio/validate.py        # Audio validation
│   │   ├── alignment/importer.py    # Alignment importer
│   │   ├── alignment/textgrid.py    # TextGrid parsing
│   │   ├── alignment/validator.py   # Alignment validator
│   │   ├── annotations/entities.py  # Annotation entities
│   │   ├── annotations/pronunciation.py # Pronunciation annotations
│   │   ├── annotations/validation.py # Annotation validation
│   │   ├── evaluation/runner.py     # Evaluation runner
│   │   ├── evaluation/schemas.py    # Evaluation schemas
│   │   ├── evaluation/statistics.py # Statistical analysis
│   │   ├── evaluation/stats.py      # Stats helpers
│   │   ├── metrics/stoi.py          # STOI metric
│   │   ├── metrics/pesq.py          # PESQ metric
│   │   ├── metrics/speaker.py       # Speaker encoder metric
│   │   ├── metrics/asr.py           # ASR metric
│   │   ├── metrics/mos.py           # MOS predictor
│   │   ├── metrics/degradations.py  # Degradation detectors
│   │   ├── splits/stratified.py     # Stratified splits
│   │   ├── splits/speaker.py        # Speaker splits
│   │   ├── cli.py                   # CLI entry point
│   │   └── version.py               # Version info
│   │
│   ├── audio/                       # From accent-voice-api/src/audio/ (keep as shared utils)
│   │   ├── __init__.py
│   │   ├── buffer.py                # RingBuffer
│   │   ├── capture.py               # AudioCapture
│   │   ├── playback.py              # AudioPlayback
│   │   ├── vad.py                   # VoiceActivityDetector
│   │   └── realtime_factor.py       # Real-time factor calculator
│   │
│   ├── api/                         # From accent-voice-api/src/api/ (Phase 0 API)
│   │   ├── __init__.py
│   │   ├── app.py                   # FastAPI REST endpoints
│   │   ├── websocket_handler.py     # WebSocket handler
│   │   └── models.py                # API request/response models
│   │
│   ├── experiments/                 # From accentedge-model-lab/experiments/
│   │   ├── registry.py              # Experiment registry
│   │   └── runner.py                # Experiment runner
│   │
│   └── shared/                      # NEW: cross-phase shared utilities
│       ├── __init__.py
│       ├── config.py                # Unified config loader
│       ├── logging.py               # Logging setup
│       └── paths.py                 # Path constants for data/checkpoints/results
│
├── configs/                          # All YAML configs merged
│   ├── phase0/                      # From accent-voice-api/configs/
│   ├── phase1/                      # From accentedge/configs/
│   ├── benchmark/                   # From accentedge-benchmark/configs/
│   └── experiments/                 # From accentedge-model-lab/experiments/ (if any)
│
├── data/                             # Unified data directory
│   ├── raw/                         # Original WAV files
│   ├── gold/                        # Gold standard recordings
│   ├── reference/                   # Reference accent audio
│   ├── manifests/                   # Dataset manifests
│   ├── annotations/                 # Phone-level annotations
│   └── processed/                   # Preprocessed audio
│
├── checkpoints/                      # Unified model weights
│   ├── phase0/                      # Seed-VC checkpoints (moved from results/)
│   ├── phase1/                      # FACodec, Phase 1 checkpoints
│   └── phase2/                      # Phase 2 best_model.pt, final_model.pt
│
├── results/                          # All results unified
│   ├── offline/                     # From accent-voice-api/results/offline/
│   ├── evaluation/                  # From accent-voice-api/results/evaluation/
│   ├── streaming/                   # From accent-voice-api/results/streaming/
│   ├── phase1/                      # From accentedge/artifacts/
│   ├── phase2/                      # From accentedge-model-lab/results/
│   └── benchmark/                   # From accentedge-benchmark/runs/
│
├── scripts/                          # All scripts merged
│   ├── phase0/                      # Phase 0 scripts (run_offline_conversion.py, etc.)
│   ├── phase1/                      # Phase 1 scripts (train_phase1.py, etc.)
│   ├── phase2/                      # Phase 2 scripts
│   ├── benchmark/                   # Benchmark scripts
│   └── utils/                       # Shared utility scripts
│
├── tests/                            # All tests merged
│   ├── phase0/                      # From accent-voice-api/tests/
│   ├── phase1/                      # From accentedge/tests/
│   ├── phase2/                      # From accentedge-model-lab/tests/
│   ├── benchmark/                   # From accentedge-benchmark/tests/
│   └── conftest.py                  # Shared test fixtures
│
├── samples/                          # From accent-voice-api/samples/
├── demo/                             # From accent-voice-api/demo/
└── .gitignore
```

---

## Phase 1: pyproject.toml

Create a single pyproject.toml that is the union of all four existing ones.

### Merged dependencies (union of all projects)

```toml
[project]
name = "accentedge"
version = "0.1.0"
description = "Identity-preserving speech-to-speech accent conversion"
requires-python = ">=3.11"
license = { text = "MIT" }

dependencies = [
    # Core
    "torch>=2.2",
    "numpy>=1.24",
    "scipy>=1.11",
    "soundfile>=0.12",
    "librosa>=0.10",
    "pyyaml>=6.0",
    "einops>=0.7",
    "pydantic>=2.0",
    "jiwer>=3.0",
    "huggingface-hub>=0.20",
    "phonemizer>=3.0",
    "speechbrain>=0.5.16",
    "faster-whisper>=1.0",
    "transformers>=4.30.0",
    "torchaudio>=2.2",
    "audioread>=3.0",
    "pandas>=2.0",
    "matplotlib>=3.7",
    "typer>=0.9",
    "jinja2>=3.1",
    "tqdm>=4.66",
    "psutil>=5.9",
]

[project.optional-dependencies]
dev = ["pytest>=8.0", "pytest-cov>=4.1", "pytest-xdist>=3.3"]
runtime = ["distill-mos"]
asr = ["faster-whisper"]

[project.scripts]
ae-convert = "accentedge.phase0.scripts.convert:main"
ae-train = "accentedge.phase1.training.trainer:main"
ae-bench = "accentedge.benchmark.cli:app"

[tool.setuptools.packages.find]
where = ["src"]

[tool.setuptools.package-dir]
"" = "src"

[tool.pytest.ini_options]
testpaths = ["tests"]
```

---

## Phase 2: Directory Restructuring

### Step 2.1 -- Move docs into docs/

```bash
cd ~/accentedge
mkdir -p docs/phase0 docs/phase1

# Phase 0 docs
cp ~/accent-voice-api/TGFP_V2.md docs/phase0/
cp ~/accent-voice-api/PHASE_0_SPEC.md docs/phase0/
cp ~/accent-voice-api/FORENSIC_AUDIT_2026-08-25.md docs/phase0/

# Phase 1 docs
cp ~/accentedge/IMPLEMENTATION.md docs/phase1/
cp ~/accentedge/STATUS.md docs/phase1/
cp ~/accentedge/ARCHITECTURE_DECISIONS.md docs/phase1/
cp ~/accentedge/docs/phase1/*.md docs/phase1/
```

### Step 2.2 -- Move phase0 source

```bash
mkdir -p src/accentedge/phase0

# From accent-voice-api/accentedge_phase0/ (the research harness)
cp -r ~/accent-voice-api/accentedge_phase0/* src/accentedge/phase0/

# From accent-voice-api/src/ (Seed-VC adapter, metrics, audio)
cp ~/accent-voice-api/src/audio/buffer.py   src/accentedge/audio/
cp ~/accent-voice-api/src/audio/capture.py  src/accentedge/audio/
cp ~/accent-voice-api/src/audio/playback.py src/accentedge/audio/
cp ~/accent-voice-api/src/audio/vad.py      src/accentedge/audio/
cp ~/accent-voice-api/src/audio/realtime_factor.py src/accentedge/audio/
cp ~/accent-voice-api/src/metrics/latency.py src/accentedge/phase0/latency.py
cp ~/accent-voice-api/src/metrics/system.py  src/accentedge/phase0/system.py
cp ~/accent-voice-api/src/metrics/realtime_factor.py src/accentedge/phase0/realtime_factor.py
cp ~/accent-voice-api/src/metrics/metrics.py  src/accentedge/evaluation/
cp ~/accent-voice-api/src/evaluation/evaluator.py src/accentedge/evaluation/
cp ~/accent-voice-api/src/evaluation/metrics.py  src/accentedge/evaluation/
cp ~/accent-voice-api/src/conversion/seedvc.py   src/accentedge/phase0/
cp ~/accent-voice-api/src/conversion/engine.py   src/accentedge/phase0/
cp ~/accent-voice-api/src/api/app.py             src/accentedge/phase0/api/
cp ~/accent-voice-api/src/api/websocket_handler.py src/accentedge/phase0/api/
```

### Step 2.3 -- Move phase1 source

```bash
# From accentedge/src/accentedge/ (already has clean package structure)
cp -r ~/accentedge/src/accentedge/codec     src/accentedge/
cp -r ~/accentedge/src/accentedge/phase1    src/accentedge/
cp -r ~/accentedge/src/accentedge/evaluation/*.py  src/accentedge/evaluation/
cp -r ~/accentedge/src/accentedge/training  src/accentedge/
```

### Step 2.4 -- Move phase2 source

```bash
# From accentedge-model-lab/src/accentedge_lab/
cp -r ~/accentedge-model-lab/src/accentedge_lab/models    src/accentedge/models/
cp -r ~/accentedge-model-lab/src/accentedge_lab/streaming src/accentedge/streaming/
cp -r ~/accentedge-model-lab/src/accentedge_lab/training  src/accentedge/training/
cp -r ~/accentedge-model-lab/src/accentedge_lab/experiments src/accentedge/experiments/
cp -r ~/accentedge-model-lab/src/accentedge_lab/config    src/accentedge/config/
cp -r ~/accentedge-model-lab/src/accentedge_lab/data      src/accentedge/data_schemas/
```

### Step 2.5 -- Move benchmark source

```bash
mkdir -p src/accentedge/benchmark
cp -r ~/accentedge-benchmark/src/accentedge_benchmark/* src/accentedge/benchmark/
```

### Step 2.6 -- Move scripts and tests

```bash
mkdir -p scripts/phase0 scripts/phase1 scripts/phase2 scripts/benchmark
cp -r ~/accent-voice-api/scripts/*               scripts/phase0/
cp -r ~/accentedge/scripts/*                     scripts/phase1/
cp -r ~/accentedge-model-lab/scripts/*           scripts/phase2/
cp -r ~/accentedge-benchmark/scripts/*           scripts/benchmark/

mkdir -p tests/phase0 tests/phase1 tests/phase2 tests/benchmark
cp -r ~/accent-voice-api/tests/*                 tests/phase0/
cp -r ~/accentedge/tests/*                       tests/phase1/
cp -r ~/accentedge-model-lab/tests/*             tests/phase2/
cp -r ~/accentedge-benchmark/tests/*             tests/benchmark/
```

---

## Phase 3: Import Rewrites

This is the mechanical work. Every `from accentedge.codec.facodec import ...` stays the same. Every `from accentedge_lab.models...` becomes `from accentedge.models...`. Every `from accentedge_benchmark...` becomes `from accentedge.benchmark...`.

### Files that need import changes

| Old import | New import |
|---|---|
| `from accentedge_lab.models.interfaces` | `from accentedge.models.interfaces` |
| `from accentedge_lab.streaming.simulator` | `from accentedge.streaming.simulator` |
| `from accentedge_lab.config.loader` | `from accentedge.config.loader` |
| `from accentedge_lab.data.lineage` | `from accentedge.data_schemas.lineage` |
| `from accentedge_lab.training.trainer` | `from accentedge.training.trainer` |
| `from accentedge_lab.experiments.registry` | `from accentedge.experiments.registry` |
| `from accentedge_benchmark.schemas` | `from accentedge.benchmark.schemas` |
| `from accentedge_benchmark.runner` | `from accentedge.benchmark.runner` |
| `from accentedge_benchmark.metrics` | `from accentedge.benchmark.metrics` |
| `from accentedge_benchmark.alignment` | `from accentedge.benchmark.alignment` |
| `from accentedge_benchmark.annotations` | `from accentedge.benchmark.annotations` |
| `from accentedge_benchmark.audio` | `from accentedge.benchmark.audio` |
| `from accentedge_benchmark.evaluation` | `from accentedge.benchmark.evaluation` |
| `from accentedge_benchmark.candidates` | `from accentedge.benchmark.candidates` |
| `from accentedge_benchmark.splits` | `from accentedge.benchmark.splits` |
| `from accentedge_benchmark.cli` | `from accentedge.benchmark.cli` |
| `from accentedge_phase0.annotations` | `from accentedge.phase0.annotations` |
| `from accentedge_phase0.audio_io` | `from accentedge.phase0.audio_io` |
| `from accentedge_phase0.target_generation` | `from accentedge.phase0.target_generation` |
| `from accentedge_phase0.listening` | `from accentedge.phase0.listening` |
| `from accentedge_phase0.evaluation` | `from accentedge.phase0.evaluation` |
| `from accentedge_phase0.experiment` | `from accentedge.phase0.experiment` |
| `from accentedge.codec.facodec` | `from accentedge.codec.facodec` (unchanged) |
| `from accentedge.codec.interfaces` | `from accentedge.codec.interfaces` (unchanged) |
| `from accentedge.phase1.denoiser` | `from accentedge.phase1.denoiser` (unchanged) |
| `from accentedge.phase1.diffusion` | `from accentedge.phase1.diffusion` (unchanged) |

---

## Phase 4: Resolve Conflicts and Duplicates

### 4.1 -- Duplicate evaluation modules

These exist in multiple projects with different implementations:

| Module | accent-voice-api | accentedge | benchmark | Resolution |
|---|---|---|---|---|
| content.py (WER/CER) | evaluator.py uses jiwer | evaluation/content.py | evaluation/content.py | Use benchmark version (has ASR backend protocol) |
| identity.py | evaluator.py uses nothing | evaluation/identity.py (ECAPA-TDNN) | evaluation/identity.py (protocol) | Merge: benchmark protocol + accentedge ECAPA impl |
| acoustic.py | Not present | evaluation/acoustic.py (mel L1, STOI, PESQ) | Not present | Use accentedge version |
| metrics.py | evaluation/metrics.py | Not present | metrics/ (per-metric files) | Use benchmark per-metric files + accentedge composite |
| latency.py | metrics/latency.py | Not present | streaming/latency.py | Merge into accentedge.evaluation.latency |

### 4.2 -- Duplicate audio I/O

- `accent-voice-api/src/audio/` has buffer/capture/playback/vad
- `accentedge_phase0/audio_io.py` has separate audio I/O
- `accentedge-benchmark/src/audio/` has io.py, validate.py, hashing.py
- **Resolution**: Keep `accentedge/audio/` (RingBuffer, VAD, capture) + merge benchmark's audio io into `accentedge/evaluation/audio_io.py`

### 4.3 -- Duplicate seedvc.py

- `accent-voice-api/src/conversion/seedvc.py` is the Seed-VC wrapper
- **Resolution**: Move to `src/accentedge/phase0/seedvc.py` and deprecate

---

## Phase 5: FAcodec and Amphion

### FAcodec

~/FAcodec/ already exists. Two options:

1. **Git submodule**: `git submodule add https://github.com/Plachtaa/FAcodec.git FAcodec`
2. **Symlink**: `ln -s ~/FAcodec ./FAcodec`

Recommendation: Git submodule, since it's a real upstream dependency.

### Amphion

Currently cloned in Colab only. For local development:

1. **Git submodule**: `git submodule add https://github.com/open-mmlab/Amphion.git Amphion`
2. **Conditional dependency**: Only needed for Colab, document in setup guide

Recommendation: Git submodule with lazy loading. FACodecAdapter should work with Plachta's FAcodec locally and only reference Amphion's variant when explicitly requested.

---

## Phase 6: Data Migration

```bash
# Audio files
cp -r ~/accent-voice-api/data/raw/      data/raw/
cp -r ~/accent-voice-api/data/gold/     data/gold/
cp -r ~/accent-voice-api/data/reference/ data/reference/

# Checkpoints
cp -r ~/accentedge/checkpoints/*        checkpoints/phase1/
cp -r ~/accentedge-model-lab/stage_a/results/*.pt checkpoints/phase2/
cp -r ~/accent-voice-api/seed-vc/*.bin  checkpoints/phase0/

# Results
cp -r ~/accent-voice-api/results/       results/phase0/
cp -r ~/accentedge/artifacts/           results/phase1/
cp -r ~/accentedge-model-lab/results/   results/phase2/
cp -r ~/accentedge-benchmark/runs/      results/benchmark/

# Configs
cp -r ~/accentedge/configs/             configs/phase1/
cp -r ~/accent-voice-api/configs/       configs/phase0/
cp -r ~/accentedge-benchmark/configs/   configs/benchmark/
```

---

## Phase 7: Validation

After consolidation, validate with:

1. `pip install -e ".[dev]"` -- single install works
2. `pytest tests/phase0/` -- Phase 0 tests pass
3. `pytest tests/phase1/` -- Phase 1 tests pass
4. `pytest tests/phase2/` -- Phase 2 tests pass
5. `pytest tests/benchmark/` -- Benchmark tests pass
6. `python -c "import accentedge; print(accentedge.__version__)"` -- package imports
7. `ae-convert --help` -- CLI entry point works
8. `ae-train --help` -- CLI entry point works
9. `ae-bench --help` -- CLI entry point works

---

## What Gets Deleted

After successful consolidation and validation:

| Path | Action |
|---|---|
| ~/accent-voice-api/ | Delete (all code moved) |
| ~/accentedge-benchmark/ | Delete (all code moved) |
| ~/accentedge-model-lab/ | Delete (all code moved) |
| ~/accent-voice-api/.venv/ | Delete (use single venv) |
| ~/accent-voice-api/seed-vc/ | Delete (submodule replaces) |
| ~/accent-voice-api/src/ (the duplicate with VAD/streaming) | Delete (merged into phase0/) |
| ~/accent-voice-api/accentedge_phase0/ | Delete (merged into phase0/) |
| ~/accentedge/src/accentedge/ | Delete (merged into src/accentedge/) |
| ~/accentedge/colab/ | Keep (Colab notebooks reference AccentEdge directly) |
| ~/accentedge/data/ | Delete (merged into ./data/) |
| ~/accentedge/artifacts/ | Delete (merged into ./results/) |
| ~/CosyAccent/ | KEEP (separate project) |

---

## Risk Assessment

| Risk | Mitigation |
|---|---|
| Import rewrites break things | Rewrite per-phase, test after each phase |
| Hidden circular imports between phases | Run `pytest --collect-only` to surface |
| FAcodec path resolution | Keep existing ~/FAcodec/ working during transition |
| Data loss | Copy, don't move; validate before deleting originals |
| Git history fragmentation | Each project has its own git history; monorepo starts fresh |

---

## Estimated Effort

| Phase | Work | Time |
|---|---|---|
| 1: pyproject.toml | Write unified config | 15 min |
| 2: Directory restructuring | Copy files to new layout | 30 min |
| 3: Import rewrites | sed/grep across all .py files | 45 min |
| 4: Conflict resolution | Manual merge of duplicates | 1 hour |
| 5: FAcodec/Amphion | Git submodule setup | 20 min |
| 6: Data migration | Copy data/checkpoints/results | 20 min |
| 7: Validation | Run all tests, fix breakage | 1-2 hours |
| **Total** | | **~4 hours** |
